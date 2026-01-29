# Tenstorrent L1 메모리 오버플로우 해결 가이드

## 목차
1. [L1 메모리 구조 이해](#1-l1-메모리-구조-이해)
2. [오버플로우 원인 분석](#2-오버플로우-원인-분석)
3. [해결 방법](#3-해결-방법)
4. [메모리 디버깅 도구](#4-메모리-디버깅-도구)
5. [모델별 최적화 예시](#5-모델별-최적화-예시)
6. [고급 최적화 기법](#6-고급-최적화-기법)
7. [트러블슈팅 체크리스트](#7-트러블슈팅-체크리스트)

---

## 1. L1 메모리 구조 이해

### 1.1 Tensix 코어 L1 SRAM 레이아웃

각 Tensix 코어는 약 **1,464KB (~1.5MB)**의 L1 SRAM을 가지고 있습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              Tensix 코어 L1 SRAM (~1,464 KB)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Reserved Region (예약 영역)                             │   │
│  │  ├── 펌웨어 코드                                         │   │
│  │  ├── 커널 바이너리                                       │   │
│  │  ├── 런타임 메타데이터                                   │   │
│  │  └── 약 100-200 KB (고정)                               │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                          │   │
│  │  Circular Buffers (순환 버퍼)                            │   │
│  │  ├── CB0: 입력 버퍼 A                                    │   │
│  │  ├── CB1: 입력 버퍼 B                                    │   │
│  │  ├── CB16: 출력 버퍼                                     │   │
│  │  └── 기타 중간 버퍼들                                    │   │
│  │                                                          │   │
│  │      ↑↑↑ 낮은 주소에서 높은 주소로 성장 ↑↑↑              │   │
│  │                                                          │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                          │   │
│  │           ╔═══════════════════════════╗                  │   │
│  │           ║    사용 가능 영역          ║                  │   │
│  │           ║  (Free Space)             ║                  │   │
│  │           ╚═══════════════════════════╝                  │   │
│  │                                                          │   │
│  │  ⚠️ 이 영역이 0이 되면 Out of Memory!                    │   │
│  │                                                          │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                          │   │
│  │      ↓↓↓ 높은 주소에서 낮은 주소로 성장 ↓↓↓              │   │
│  │                                                          │   │
│  │  L1 Buffers (텐서 버퍼)                                  │   │
│  │  ├── 모델 가중치 (샤딩된 경우)                           │   │
│  │  ├── 활성화 텐서                                         │   │
│  │  └── 중간 연산 결과                                      │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 아키텍처별 L1 메모리 비교

| 아키텍처 | L1 SRAM/코어 | 컴퓨팅 코어 수 | 총 L1 용량 |
|----------|-------------|---------------|-----------|
| Wormhole N150 | 1,464 KB | 64 (8×8) | ~91 MB |
| Blackhole p100 | 1,464 KB + 캐시 | 130 (13×10) | ~186 MB |

### 1.3 메모리 할당 방향

```
Reserved (고정)
      ↓
┌─────────────────┐ ← 낮은 주소
│ Circular Buffers│ ↓ 아래로 성장
├─────────────────┤
│                 │
│   Free Space    │ ← 오버플로우 시 사라짐!
│                 │
├─────────────────┤
│  L1 Buffers     │ ↑ 위로 성장
└─────────────────┘ ← 높은 주소
```

**중요**: 순환 버퍼와 L1 버퍼가 서로를 향해 성장하다 만나면 **Out of Memory** 오류 발생!

---

## 2. 오버플로우 원인 분석

### 2.1 일반적인 원인

| 원인 | 설명 | 증상 |
|------|------|------|
| **큰 텐서** | 단일 코어에 너무 큰 텐서 할당 | 즉시 OOM |
| **많은 순환 버퍼** | 복잡한 연산으로 버퍼 다수 생성 | 연산 중 OOM |
| **큰 배치 크기** | 배치당 활성화 메모리 증가 | 배치 증가 시 OOM |
| **비효율적 샤딩** | 코어당 할당량 불균형 | 특정 코어에서 OOM |
| **중간 결과 누적** | 그래프 최적화 부재 | 연산 체인에서 OOM |

### 2.2 오류 메시지 예시

```
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/allocator.cpp:123:
Out of Memory: Not enough space to allocate 2097152 bytes in L1

RuntimeError: Circular buffer allocation failed:
requested 65536 bytes but only 32768 bytes available

RuntimeError: L1 buffer overlap detected between CB and tensor buffer
```

---

## 3. 해결 방법

### 3.1 방법 1: `l1_small_size` 조정

디바이스 초기화 시 L1 메모리 할당 크기를 조정합니다.

```python
import ttnn

# 기본값 확인
print(f"Default L1 small size: {ttnn.device.DEFAULT_L1_SMALL_SIZE}")

# 메모리 절약이 필요한 경우 (작은 모델)
device = ttnn.open_device(
    device_id=0,
    l1_small_size=8192  # 8KB
)

# 복잡한 모델의 경우 (CNN, Attention 등)
device = ttnn.open_device(
    device_id=0,
    l1_small_size=32768  # 32KB
)

# 매우 복잡한 모델
device = ttnn.open_device(
    device_id=0,
    l1_small_size=65536  # 64KB
)

# 작업 완료 후 반드시 닫기
ttnn.close_device(device)
```

**권장 설정:**

| 모델 유형 | 권장 `l1_small_size` |
|----------|---------------------|
| 간단한 MLP | 8,192 (8KB) |
| 기본 CNN | 16,384 (16KB) |
| 복잡한 CNN | 24,576 (24KB) |
| Transformer | 32,768 (32KB) |
| 대형 모델 | 65,536 (64KB) |

---

### 3.2 방법 2: DRAM 메모리 사용

L1 대신 더 큰 DRAM에 텐서를 저장합니다.

```python
import ttnn

# ❌ L1에 저장 (메모리 부족 가능)
tensor_l1 = ttnn.to_device(
    tensor,
    device,
    memory_config=ttnn.L1_MEMORY_CONFIG
)

# ✅ DRAM에 저장 (더 많은 용량)
tensor_dram = ttnn.to_device(
    tensor,
    device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG
)
```

**L1 vs DRAM 비교:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    메모리 유형 비교                              │
├───────────────┬─────────────────┬─────────────────┬─────────────┤
│     특성      │    L1 SRAM      │     DRAM        │   선택 기준 │
├───────────────┼─────────────────┼─────────────────┼─────────────┤
│ 용량 (p100)   │ ~1.5MB/코어     │ ~32GB 총        │ DRAM > L1   │
│ 접근 속도     │ ~10-20 사이클   │ ~100-200 사이클 │ L1 > DRAM   │
│ 대역폭        │ 높음            │ 중간            │ L1 > DRAM   │
│ 사용 사례     │ 활성화, 임시    │ 가중치, 큰 텐서 │             │
└───────────────┴─────────────────┴─────────────────┴─────────────┘
```

**혼합 전략:**

```python
# 자주 접근하는 작은 텐서 → L1
activations = ttnn.to_device(act, device, memory_config=ttnn.L1_MEMORY_CONFIG)

# 큰 가중치 → DRAM
weights = ttnn.to_device(w, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

---

### 3.3 방법 3: 샤딩 (Sharding) 사용

큰 텐서를 여러 코어에 분산하여 코어당 메모리 사용량을 줄입니다.

#### 3.3.1 샤딩 개념

```
┌─────────────────────────────────────────────────────────────────┐
│                    샤딩 vs 인터리빙                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Interleaved (기본값):                                          │
│  ┌─────────────────────────────────────────┐                   │
│  │ 전체 텐서가 모든 코어에 라운드로빈 분산   │                   │
│  │                                          │                   │
│  │  Core0: Page0, Page4, Page8...          │                   │
│  │  Core1: Page1, Page5, Page9...          │                   │
│  │  Core2: Page2, Page6, Page10...         │                   │
│  │  Core3: Page3, Page7, Page11...         │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                 │
│  Sharded (샤딩):                                                │
│  ┌─────────────────────────────────────────┐                   │
│  │ 텐서가 논리적으로 분할되어 각 코어에 할당 │                   │
│  │                                          │                   │
│  │  Core0: Shard[0:H/4, 0:W/4]             │                   │
│  │  Core1: Shard[0:H/4, W/4:W/2]           │                   │
│  │  Core2: Shard[H/4:H/2, 0:W/4]           │                   │
│  │  Core3: Shard[H/4:H/2, W/4:W/2]         │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.3.2 샤딩 유형

```python
import ttnn

# 1. HEIGHT_SHARDED: 높이 방향으로 분할
#    - 시퀀스 처리, RNN에 적합
height_shard_spec = ttnn.ShardSpec(
    core_grid=ttnn.CoreGrid(y=8, x=1),  # 8개 코어 세로 배치
    shard_shape=(64, 512),               # 코어당 64×512
    shard_orientation=ttnn.ShardOrientation.ROW_MAJOR
)

# 2. WIDTH_SHARDED: 너비 방향으로 분할
#    - 채널 분산, 1x1 Conv에 적합
width_shard_spec = ttnn.ShardSpec(
    core_grid=ttnn.CoreGrid(y=1, x=8),  # 8개 코어 가로 배치
    shard_shape=(512, 64),               # 코어당 512×64
    shard_orientation=ttnn.ShardOrientation.COL_MAJOR
)

# 3. BLOCK_SHARDED: 2D 블록으로 분할
#    - 행렬 연산, Attention에 적합
block_shard_spec = ttnn.ShardSpec(
    core_grid=ttnn.CoreGrid(y=4, x=4),  # 4×4 = 16개 코어
    shard_shape=(128, 128),              # 코어당 128×128
    shard_orientation=ttnn.ShardOrientation.ROW_MAJOR
)
```

#### 3.3.3 샤딩 적용 예제

```python
import ttnn
import torch

# 큰 텐서 생성
large_tensor = torch.randn(1, 1, 512, 512)

# TTNN 텐서로 변환
ttnn_tensor = ttnn.from_torch(large_tensor, dtype=ttnn.bfloat16)

# 샤딩 설정
shard_spec = ttnn.ShardSpec(
    core_grid=ttnn.CoreGrid(y=8, x=8),   # 64개 코어 사용
    shard_shape=(64, 64),                 # 코어당 64×64 = 8KB
    shard_orientation=ttnn.ShardOrientation.ROW_MAJOR
)

memory_config = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    buffer_type=ttnn.BufferType.L1,
    shard_spec=shard_spec
)

# 샤딩된 텐서를 디바이스로 전송
sharded_tensor = ttnn.to_device(ttnn_tensor, device, memory_config=memory_config)

# 이제 각 코어는 64×64 = 4096 elements × 2 bytes = 8KB만 사용
```

#### 3.3.4 샤딩 크기 계산

```python
def calculate_shard_memory(tensor_shape, core_grid, dtype_size=2):
    """
    코어당 메모리 사용량 계산

    Args:
        tensor_shape: (batch, channel, height, width) 또는 (height, width)
        core_grid: (y, x) 코어 그리드 크기
        dtype_size: 데이터 타입 크기 (bfloat16=2, float32=4)

    Returns:
        코어당 메모리 사용량 (bytes)
    """
    if len(tensor_shape) == 4:
        _, _, h, w = tensor_shape
    else:
        h, w = tensor_shape

    grid_y, grid_x = core_grid

    shard_h = (h + grid_y - 1) // grid_y  # 올림 나눗셈
    shard_w = (w + grid_x - 1) // grid_x

    memory_per_core = shard_h * shard_w * dtype_size

    print(f"텐서 크기: {h}×{w}")
    print(f"코어 그리드: {grid_y}×{grid_x} = {grid_y * grid_x}개 코어")
    print(f"샤드 크기: {shard_h}×{shard_w}")
    print(f"코어당 메모리: {memory_per_core:,} bytes ({memory_per_core/1024:.1f} KB)")

    return memory_per_core

# 예시
calculate_shard_memory((1, 1, 512, 512), (8, 8))
# 출력:
# 텐서 크기: 512×512
# 코어 그리드: 8×8 = 64개 코어
# 샤드 크기: 64×64
# 코어당 메모리: 8,192 bytes (8.0 KB)
```

---

### 3.4 방법 4: 배치 크기 줄이기

```python
# ❌ 큰 배치 (메모리 부족)
batch_size = 32
output = model(large_batch)  # OOM!

# ✅ 작은 배치로 분할 처리
batch_size = 8
results = []

for i in range(0, total_samples, batch_size):
    mini_batch = data[i:i+batch_size]
    mini_batch_device = ttnn.to_device(mini_batch, device)

    result = model(mini_batch_device)

    # 결과를 호스트로 이동하여 디바이스 메모리 해제
    result_cpu = ttnn.to_torch(result)
    results.append(result_cpu)

    # 명시적 메모리 해제 (선택적)
    ttnn.deallocate(mini_batch_device)

# 결과 합치기
final_result = torch.cat(results, dim=0)
```

**배치 크기 가이드:**

| 모델 크기 | L1 여유 | 권장 배치 |
|----------|---------|----------|
| 소형 (< 100MB) | 많음 | 32+ |
| 중형 (100-500MB) | 중간 | 8-16 |
| 대형 (> 500MB) | 적음 | 1-4 |

---

### 3.5 방법 5: 순환 버퍼 크기 최적화

커널 레벨에서 순환 버퍼 크기를 조정합니다.

```cpp
// 커널 코드에서 순환 버퍼 설정

// ❌ 큰 순환 버퍼 (더블 버퍼링, 4 타일)
constexpr uint32_t tiles_per_cb = 4;  // 4 × 2KB = 8KB

// ✅ 최소 크기 (싱글 버퍼링, 1 타일)
constexpr uint32_t tiles_per_cb = 1;  // 1 × 2KB = 2KB

// ✅ 또는 더블 버퍼링 최소 (2 타일)
constexpr uint32_t tiles_per_cb = 2;  // 2 × 2KB = 4KB
```

**호스트 코드에서 순환 버퍼 크기 지정:**

```cpp
constexpr uint32_t tile_size = 32 * 32 * sizeof(bfloat16);  // 2KB
constexpr uint32_t num_tiles = 2;  // 더블 버퍼링

CreateCircularBuffer(
    program,
    core,
    CircularBufferConfig(
        num_tiles * tile_size,  // 총 크기
        {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}}
    ).set_page_size(tt::CBIndex::c_0, tile_size)
);
```

---

### 3.6 방법 6: 텐서 수명 관리

사용하지 않는 텐서를 명시적으로 해제합니다.

```python
import ttnn

def process_with_memory_management(input_tensor, device):
    # 입력을 디바이스로
    x = ttnn.to_device(input_tensor, device)

    # 연산 1
    y = ttnn.relu(x)
    ttnn.deallocate(x)  # x 더 이상 필요 없음

    # 연산 2
    z = ttnn.linear(y, weights)
    ttnn.deallocate(y)  # y 더 이상 필요 없음

    # 연산 3
    output = ttnn.softmax(z)
    ttnn.deallocate(z)  # z 더 이상 필요 없음

    return output

# 또는 컨텍스트 매니저 사용
with ttnn.manage_device(device_id=0) as device:
    output = model(input_tensor)
    # 컨텍스트 종료 시 자동 정리
```

---

## 4. 메모리 디버깅 도구

### 4.1 메모리 리포트 활성화

```python
import ttnn

# 메모리 리포트 활성화
ttnn.device.EnableMemoryReports()

# 모델 실행
device = ttnn.open_device(device_id=0)
output = model(input_tensor)

# 리포트 비활성화
ttnn.device.DisableMemoryReports()

ttnn.close_device(device)
```

**생성되는 파일 (in `$TT_METAL_HOME/generated/`):**

| 파일 | 내용 |
|------|------|
| `l1_usage_summary.csv` | 프로그램별 최소 L1 여유 공간 |
| `memory_usage_summary.csv` | DRAM/SRAM 뱅크별 사용량 요약 |
| `detailed_memory_usage.csv` | 모든 메모리 블록 상세 정보 |

### 4.2 특정 시점 메모리 덤프

```python
# 디버그 체크포인트에서 메모리 상태 덤프
ttnn.device.dump_device_memory_state(device, prefix="before_attention")

# Attention 레이어 실행
output = attention(query, key, value)

# 이후 상태 덤프
ttnn.device.dump_device_memory_state(device, prefix="after_attention")
```

### 4.3 C++ API

```cpp
#include "tt_metal/host_api.hpp"

// 메모리 상태 덤프
DumpDeviceMemoryState(device, "checkpoint_1");

// 메모리 뷰 가져오기
auto l1_view = GetMemoryView(device, BufferType::L1);
auto dram_view = GetMemoryView(device, BufferType::DRAM);

// 리포트 활성화/비활성화
EnableMemoryReports();
DisableMemoryReports();
```

### 4.4 리포트 분석 예시

```csv
# l1_usage_summary.csv 예시
program_id,min_free_l1_block,max_interleaved_buffer_size
0,524288,1048576
1,262144,524288
2,65536,131072  ← 여유 공간 작음, OOM 위험!
```

---

## 5. 모델별 최적화 예시

### 5.1 Whisper (STT)

```python
import ttnn

# Whisper는 Encoder/Decoder 구조로 L1 사용량이 높음
device = ttnn.open_device(
    device_id=0,
    l1_small_size=24576  # 24KB 권장
)

# 또는 환경변수로 설정
# export WHISPER_L1_SMALL_SIZE=24576
```

### 5.2 ViT (Vision Transformer)

```python
import ttnn

# 이미지 해상도에 따라 조정
image_size = 224  # 또는 384, 512

if image_size <= 224:
    l1_small_size = 16384  # 16KB
elif image_size <= 384:
    l1_small_size = 24576  # 24KB
else:
    l1_small_size = 32768  # 32KB

device = ttnn.open_device(device_id=0, l1_small_size=l1_small_size)
```

### 5.3 LLM (대형 언어 모델)

```python
import ttnn

# LLM은 대부분의 가중치를 DRAM에 저장
def create_llm_memory_config(seq_len, hidden_dim):
    """LLM용 메모리 설정 생성"""

    # KV 캐시는 L1에 샤딩
    kv_shard_spec = ttnn.ShardSpec(
        core_grid=ttnn.CoreGrid(y=8, x=8),
        shard_shape=(seq_len // 64, hidden_dim // 64),
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    kv_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=kv_shard_spec
    )

    # 가중치는 DRAM에 인터리빙
    weight_memory_config = ttnn.DRAM_MEMORY_CONFIG

    return kv_memory_config, weight_memory_config
```

### 5.4 CNN

```python
import ttnn

# CNN은 sliding window 연산에 L1 필요
device = ttnn.open_device(
    device_id=0,
    l1_small_size=8192  # 기본 CNN: 8KB
)

# 깊은 네트워크의 경우
device = ttnn.open_device(
    device_id=0,
    l1_small_size=16384  # ResNet, VGG 등: 16KB
)
```

---

## 6. 고급 최적화 기법

### 6.1 연산 융합 (Operation Fusion)

```python
# ❌ 비융합: 각 연산마다 중간 텐서 생성
x = ttnn.linear(input, w1)     # 중간 텐서 1
x = ttnn.relu(x)               # 중간 텐서 2
x = ttnn.linear(x, w2)         # 중간 텐서 3

# ✅ 융합: 중간 텐서 최소화
# TTNN이 자동으로 융합 가능한 경우 처리
# 또는 명시적 융합 연산 사용
x = ttnn.linear(input, w1, activation="relu")  # relu 융합
x = ttnn.linear(x, w2)
```

### 6.2 그래디언트 체크포인팅 (학습 시)

```python
# 메모리 절약을 위해 중간 활성화를 재계산
def forward_with_checkpointing(x, layers):
    for i, layer in enumerate(layers):
        if i % 2 == 0:  # 매 2번째 레이어만 저장
            x = checkpoint(layer, x)
        else:
            x = layer(x)
    return x
```

### 6.3 동적 메모리 할당

```python
def adaptive_batch_size(model, input_shape, device, max_batch=32):
    """OOM 없이 가능한 최대 배치 크기 찾기"""

    for batch_size in [max_batch, max_batch//2, max_batch//4, 1]:
        try:
            test_input = torch.randn(batch_size, *input_shape)
            test_tensor = ttnn.from_torch(test_input)
            test_tensor = ttnn.to_device(test_tensor, device)

            # 테스트 실행
            _ = model(test_tensor)

            ttnn.deallocate(test_tensor)
            print(f"성공: batch_size={batch_size}")
            return batch_size

        except RuntimeError as e:
            if "Out of Memory" in str(e):
                print(f"OOM: batch_size={batch_size}, 더 작은 크기 시도...")
                continue
            raise

    raise RuntimeError("batch_size=1에서도 OOM 발생")
```

---

## 7. 트러블슈팅 체크리스트

L1 메모리 오버플로우 발생 시 다음 순서로 시도:

### 빠른 해결

- [ ] **1. DRAM 사용**
  ```python
  tensor = ttnn.to_device(tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
  ```

- [ ] **2. l1_small_size 조정**
  ```python
  device = ttnn.open_device(device_id=0, l1_small_size=8192)
  ```

- [ ] **3. 배치 크기 줄이기**
  ```python
  batch_size = batch_size // 2
  ```

### 중급 해결

- [ ] **4. 샤딩 적용**
  - 텐서를 여러 코어에 분산
  - `BLOCK_SHARDED` 또는 `HEIGHT_SHARDED` 시도

- [ ] **5. 메모리 리포트 분석**
  ```python
  ttnn.device.EnableMemoryReports()
  ```

- [ ] **6. 텐서 수명 관리**
  ```python
  ttnn.deallocate(unused_tensor)
  ```

### 고급 해결

- [ ] **7. 순환 버퍼 크기 최적화** (커널 수정 필요)

- [ ] **8. 연산 융합**

- [ ] **9. 모델 분할** (여러 디바이스 또는 순차 실행)

---

## 참고 자료

- [Memory Allocator Tech Report](../../tech_reports/memory/allocator.md)
- [Tensor Layouts Tech Report](../../tech_reports/tensor_layouts/tensor_layouts.md)
- [Tensor Sharding Tech Report](../../tech_reports/tensor_sharding/tensor_sharding.md)
- [TTNN API Reference](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html)

---

*문서 버전: 1.0*
*작성일: 2026-01-29*
*대상: Wormhole (n150, n300) 및 Blackhole (p100, p150)*
