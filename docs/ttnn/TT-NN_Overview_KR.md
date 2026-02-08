# TT-NN (Tenstorrent Neural Network Library) 종합 설명 자료

## 목차

1. [개요](#1-개요)
2. [아키텍처 구조](#2-아키텍처-구조)
3. [핵심 구성요소](#3-핵심-구성요소)
4. [텐서 시스템](#4-텐서-시스템)
5. [연산(Operations) 체계](#5-연산operations-체계)
6. [디바이스 관리](#6-디바이스-관리)
7. [메모리 관리](#7-메모리-관리)
8. [멀티 디바이스 및 분산 처리](#8-멀티-디바이스-및-분산-처리)
9. [Python-C++ 바인딩 구조](#9-python-c-바인딩-구조)
10. [프로파일링 및 디버깅](#10-프로파일링-및-디버깅)
11. [사용 예시](#11-사용-예시)
12. [디렉토리 구조 레퍼런스](#12-디렉토리-구조-레퍼런스)

---

## 1. 개요

### TT-NN이란?

TT-NN은 Tenstorrent AI 가속기 위에서 신경망 연산을 수행하기 위한 **오픈소스 C++/Python 라이브러리**입니다. TT-Metalium 프로그래밍 모델 위에 구축되어 있으며, PyTorch와 유사한 인터페이스를 제공하여 ML 워크로드를 Tenstorrent 하드웨어에서 실행할 수 있게 합니다.

### 핵심 목적

| 목적 | 설명 |
|------|------|
| **ML 프레임워크 지원** | PyTorch, JAX, TensorFlow 등 기존 프레임워크의 모델을 Tenstorrent HW에서 실행하기 위한 빌딩 블록 제공 |
| **수동 최적화** | 데이터 포맷(mixed-precision), 텐서 레이아웃, 분산 배치, Op 퓨전 등 하드웨어 특화 최적화를 세밀하게 제어 |
| **고수준 API** | 하드웨어 복잡성을 추상화하면서도 성능 튜닝이 가능한 인터페이스 |

### 지원 하드웨어

- **Wormhole B0** - Tenstorrent의 주력 AI 가속기
- **Blackhole** - 차세대 아키텍처
- **멀티 디바이스 클러스터** - 여러 디바이스를 하나의 논리 유닛으로 가상화

---

## 2. 아키텍처 구조

### 계층 구조

```
┌─────────────────────────────────────────────────┐
│              Python API (ttnn)                   │
│    PyTorch-like 인터페이스, 연산자 오버로딩         │
├─────────────────────────────────────────────────┤
│          Nanobind 바인딩 계층                     │
│    C++ ↔ Python 타입 변환, 메모리 관리            │
├─────────────────────────────────────────────────┤
│           C++ 연산 라이브러리                      │
│    200+ 최적화된 연산 구현                        │
├─────────────────────────────────────────────────┤
│            TT-Metalium                          │
│    디바이스 프로그래밍, 커널 실행, 메모리 관리       │
├─────────────────────────────────────────────────┤
│         Tenstorrent Hardware                    │
│    Wormhole B0 / Blackhole / Multi-Device       │
└─────────────────────────────────────────────────┘
```

### 연산 구현 패턴

```
Operation (C++ 구현)
    ↓
Nanobind Binding (C++ → Python 브릿지)
    ↓
Python Wrapper (사용자 API)
    ↓
Golden Function (PyTorch 참조 구현 - 검증용)
```

각 연산은 위 4개 계층을 따르며, Golden Function은 PyTorch의 동등한 연산을 참조 구현으로 등록하여 정확도 검증(PCC 비교)에 사용됩니다.

---

## 3. 핵심 구성요소

### 3.1 Python 패키지 (`ttnn/ttnn/`)

| 모듈 | 역할 |
|------|------|
| `__init__.py` | 진입점. 모든 타입/함수 import, 연산자 오버로딩 설정, C++ 연산 자동 등록 |
| `types.py` | 데이터 타입, 메모리 설정, 레이아웃, 샤딩 등 타입 정의 |
| `device.py` | 디바이스 라이프사이클 관리 (open/close/sync) |
| `core.py` | 텐서 검사 유틸리티, 메모리 관리, 디버깅 |
| `decorators.py` | 연산 등록 시스템, Hook 메커니즘 |
| `database.py` | 프로파일링/트레이싱 데이터베이스 |
| `operations/` | 26개 Python 연산 모듈 |

### 3.2 C++ 구현 (`ttnn/cpp/`)

| 디렉토리 | 역할 |
|----------|------|
| `ttnn/operations/` | 32개 카테고리, 203개 하위 디렉토리의 최적화된 C++ 연산 |
| `ttnn-nanobind/` | Python 바인딩 인프라 (nanobind 기반) |
| `ttnn/kernel/` | RISC-V 컴퓨트 및 데이터 이동 커널 |

### 3.3 Core 계층 (`ttnn/core/`)

| 디렉토리 | 역할 |
|----------|------|
| `tensor/` | 텐서 핵심 구현 (C++) |
| `distributed/` | 분산 처리 |
| `graph/` | 그래프 트레이싱 인프라 |

---

## 4. 텐서 시스템

### 4.1 Tensor 타입

`ttnn.Tensor`는 Tenstorrent 하드웨어와 호스트 간에 다차원 데이터를 관리하는 핵심 추상화입니다.

```python
# 텐서 속성
tensor.shape              # 논리적 형상
tensor.padded_shape       # 물리적 패딩된 형상
tensor.layout             # ROW_MAJOR 또는 TILE
tensor.dtype              # 데이터 타입
tensor.device             # 텐서가 위치한 디바이스
tensor.memory_config()    # 메모리 설정
tensor.storage_type()     # HOST 또는 DEVICE
tensor.is_sharded()       # 샤딩 여부
tensor.is_allocated()     # 할당 여부
```

### 4.2 지원 데이터 타입

| 타입 | 설명 |
|------|------|
| `float32` | 32비트 부동소수점 |
| `bfloat16` | Brain Floating Point 16 (가장 많이 사용) |
| `bfloat8_b` | 8비트 Brain Float (높은 처리량) |
| `bfloat4_b` | 4비트 Brain Float (최대 처리량) |
| `int32` / `uint32` | 32비트 정수 |
| `uint16` / `uint8` | 16/8비트 부호없는 정수 |

### 4.3 메모리 레이아웃

| 레이아웃 | 설명 |
|----------|------|
| `ROW_MAJOR_LAYOUT` | 선형 메모리 배치. 일반적인 CPU/GPU 방식 |
| `TILE_LAYOUT` | **32x32 타일** 기반 배치. Tenstorrent HW의 연산 효율 극대화 |

> **중요**: 대부분의 연산은 `TILE_LAYOUT`에서 가장 효율적입니다. 텐서는 자동으로 32x32 타일 경계에 맞게 패딩됩니다.

### 4.4 PyTorch 연동

```python
import torch
import ttnn

# PyTorch → TT-NN
torch_tensor = torch.randn(1, 32, 32)
ttnn_tensor = ttnn.from_torch(torch_tensor, device=device, layout=ttnn.TILE_LAYOUT)

# TT-NN → PyTorch
result_torch = ttnn.to_torch(ttnn_tensor)
```

### 4.5 텐서 생성

```python
# 기본 생성
zeros = ttnn.zeros((2, 3), device=device, dtype=ttnn.bfloat16)
ones = ttnn.ones((4, 4), device=device)
full = ttnn.full((3, 3), fill_value=1.0, device=device)
arange = ttnn.arange(10, device=device)

# PyTorch-like 인덱싱
result = tensor[0]
result = tensor[0:2]
result = tensor[:, 1:3, ::2]
```

---

## 5. 연산(Operations) 체계

### 5.1 연산 카테고리 전체 맵

TT-NN은 **200개 이상의 최적화된 연산**을 제공합니다.

#### Element-wise 연산

| 카테고리 | 주요 연산 | 개수 |
|----------|----------|------|
| **Unary** | `relu`, `sigmoid`, `tanh`, `sqrt`, `exp`, `log`, `gelu`, `silu`, `abs`, `neg`, `reciprocal` 등 | 50+ |
| **Binary** | `add`, `subtract`, `multiply`, `divide`, `power`, `max`, `min`, `logical_and/or/xor` 등 | 70+ |
| **Ternary** | `where`, `clamp`, `lerp` | 3+ |
| **Comparison** | `eq`, `ne`, `lt`, `le`, `gt`, `ge` | 6 |

#### 선형대수

| 연산 | 설명 | 특징 |
|------|------|------|
| `matmul` | 행렬 곱셈 | 5개 이상의 프로그램 설정 (MultiCore, Reuse, MultiCast, DRAM-Sharded 등) |
| `mm` / `bmm` | 행렬/배치 행렬 곱셈 | matmul의 단축 API |

**Matmul 프로그램 설정 옵션:**
- `MatmulMultiCoreReuseProgramConfig` - 멀티코어 재사용
- `MatmulMultiCoreReuseMultiCastProgramConfig` - 멀티코어 멀티캐스트
- `MatmulMultiCoreReuseMultiCast1DProgramConfig` - 1D 멀티캐스트
- `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` - DRAM 샤딩

#### 리덕션

| 연산 | 설명 |
|------|------|
| `sum`, `mean`, `max`, `min` | 기본 리덕션 |
| `prod`, `std`, `var` | 통계 리덕션 |
| `argmax`, `argmin` | 인덱스 리덕션 |
| `topk` | Top-K 선택 |

#### 정규화 (Normalization)

| 연산 | 설명 |
|------|------|
| `softmax` | Softmax (기본, 샤딩 멀티코어 설정 지원) |
| `layer_norm` | Layer Normalization |
| `rmsnorm` | RMSNorm (LLM에서 주로 사용) |
| `batch_norm` | Batch Normalization |
| `group_norm` | Group Normalization |

#### Transformer 연산

| 연산 | 설명 |
|------|------|
| `scaled_dot_product_attention` | SDPA (Flash Attention 스타일) |
| `sdpa_decode` | KV-cache 인식 SDPA (LLM 추론용) |
| `sdpa_windowed` | 윈도우 기반 SDPA |
| `attention_softmax` | Attention용 특화 Softmax |
| `split_query_key_value_and_split_heads` | Q/K/V 분할 |
| `concatenate_heads` | Multi-head 합치기 |
| `rotary_embedding` | RoPE (위치 임베딩) |

#### 합성곱 및 풀링

| 연산 | 설명 |
|------|------|
| `conv2d` | 2D 합성곱 (다중 구현) |
| `conv1d` | 1D 합성곱 |
| `conv_transpose2d` | 전치 합성곱 |
| `max_pool2d` | 최대 풀링 |
| `avg_pool2d` | 평균 풀링 |
| `global_avg_pool` | 전역 평균 풀링 |

#### 데이터 이동

| 연산 | 설명 |
|------|------|
| `reshape`, `permute`, `transpose` | 형상 변환 |
| `pad`, `slice` | 패딩, 슬라이싱 |
| `concat`, `split`, `chunk` | 결합, 분할 |
| `gather`, `scatter` | 인덱스 기반 데이터 이동 |
| `repeat`, `expand` | 반복, 브로드캐스트 |

#### Collective Communication (CCL)

| 연산 | 설명 |
|------|------|
| `all_gather` | 모든 디바이스에서 데이터 수집 |
| `all_reduce` | 모든 디바이스의 데이터 리듀스 |
| `reduce_scatter` | 리듀스 후 분산 |
| `broadcast` | 한 디바이스에서 전체로 전송 |

#### Backward 연산 (학습용)

- `binary_backward` - Binary 연산의 역전파
- `unary_backward` - Unary 연산의 역전파
- `ternary_backward` - Ternary 연산의 역전파
- `complex_binary_backward`, `complex_unary_backward` - 복소수 역전파

#### 활성화 함수 (37+)

`relu`, `relu6`, `sigmoid`, `tanh`, `swish`, `mish`, `gelu`, `silu`, `softplus`, `softsign`, `elu`, `celu`, `hardtanh`, `hardswish`, `hardshrink` 등

### 5.2 연산 등록 시스템

TT-NN은 데코레이터 기반의 연산 등록 시스템을 사용합니다:

```python
# C++ 연산 자동 등록
auto_register_ttnn_cpp_operations(ttnn._ttnn)

# Python 연산 수동 등록
@ttnn.register_python_operation(name="ttnn.my_op")
def my_op(tensor, param1):
    return ttnn._ttnn.my_op(tensor, param1)

# Golden Function 등록 (PyTorch 참조 구현)
def _golden_function(input_a, input_b, **kwargs):
    import torch
    return input_a + input_b

ttnn.attach_golden_function(ttnn.add, golden_function=_golden_function)
```

### 5.3 연산자 오버로딩

PyTorch와 동일한 방식으로 연산자를 사용할 수 있습니다:

```python
c = a + b        # ttnn.add(a, b)
c = a - b        # ttnn.subtract(a, b)
c = a * b        # ttnn.multiply(a, b)
c = a / b        # ttnn.divide(a, b)
c = a == b       # ttnn.eq(a, b)
c = a > b        # ttnn.gt(a, b)
result = tensor[0:2]  # ttnn.operations.core.__getitem__
```

---

## 6. 디바이스 관리

### 6.1 디바이스 열기/닫기

```python
import ttnn

# 방법 1: Context Manager (권장)
with ttnn.manage_device(device_id=0) as device:
    # 디바이스 사용
    tensor = ttnn.zeros((32, 32), device=device)
    # 블록 종료 시 자동 정리

# 방법 2: 수동 관리
device = ttnn.open_device(device_id=0)
try:
    tensor = ttnn.zeros((32, 32), device=device)
finally:
    ttnn.close_device(device)
```

### 6.2 디바이스 정보 확인

```python
# 사용 가능한 디바이스 수
num_devices = ttnn.GetNumAvailableDevices()
num_pcie = ttnn.GetNumPCIeDevices()

# 하드웨어 타입 확인
is_wh = ttnn.is_wormhole_b0(device)    # Wormhole B0
is_bh = ttnn.is_blackhole(device)       # Blackhole

# 아키텍처 이름
arch = ttnn.get_arch_name()

# 코어 그리드 정보
core_grid = device.core_grid
```

### 6.3 디바이스 동기화

```python
# 디바이스 연산 완료 대기
ttnn.synchronize_device(device)

# 특정 커맨드 큐에서 연산 실행
with ttnn.command_queue(queue_id=1):
    result = ttnn.add(a, b)  # 큐 1에서 실행
```

---

## 7. 메모리 관리

### 7.1 메모리 타입

| 타입 | 설명 | 특성 |
|------|------|------|
| **DRAM** | Device DRAM | 대용량, 상대적 높은 지연 |
| **L1** | 코어 로컬 SRAM | 소용량, 낮은 지연, 컴퓨트 코어에 근접 |

### 7.2 메모리 설정

```python
# 기본 메모리 설정 (pre-defined)
ttnn.DRAM_MEMORY_CONFIG        # DRAM에 인터리브
ttnn.L1_MEMORY_CONFIG          # L1에 인터리브
ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG     # L1 블록 샤딩
ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG    # L1 높이 샤딩
ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG     # L1 너비 샤딩
```

### 7.3 텐서 샤딩(Sharding)

샤딩은 텐서 데이터를 여러 컴퓨트 코어에 분산 배치하여 병렬 처리를 극대화하는 기법입니다.

```
┌──────────────────────────────────────┐
│           HEIGHT Sharding            │
│  Core0: [rows 0-7]                  │
│  Core1: [rows 8-15]                 │
│  Core2: [rows 16-23]                │
│  Core3: [rows 24-31]                │
├──────────────────────────────────────┤
│           WIDTH Sharding             │
│  Core0: [cols 0-7]                  │
│  Core1: [cols 8-15]                 │
│  Core2: [cols 16-23]                │
│  Core3: [cols 24-31]                │
├──────────────────────────────────────┤
│           BLOCK Sharding             │
│  Core0: [rows 0-15, cols 0-15]      │
│  Core1: [rows 0-15, cols 16-31]     │
│  Core2: [rows 16-31, cols 0-15]     │
│  Core3: [rows 16-31, cols 16-31]    │
└──────────────────────────────────────┘
```

```python
# 샤딩 메모리 설정 생성
config = ttnn.create_sharded_memory_config(
    shape=[8, 8],
    core_grid=device.core_grid,
    strategy=ttnn.ShardStrategy.WIDTH
)

# 텐서에 메모리 설정 적용
tensor = ttnn.to_memory_config(tensor, config)
```

### 7.4 메모리 이동

```python
# 호스트 → 디바이스
tensor = ttnn.to_device(tensor, device)

# 레이아웃 변환
tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)

# 데이터 타입 변환
tensor = ttnn.to_dtype(tensor, ttnn.float32)

# 메모리 설정 변경
tensor = ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)
```

---

## 8. 멀티 디바이스 및 분산 처리

### 8.1 Mesh 개념

TT-NN은 여러 디바이스를 **Mesh**로 추상화하여 단일 논리 유닛처럼 사용할 수 있습니다.

```
┌──────────────────────────────────┐
│          Mesh (2x2)              │
│  ┌─────────┐  ┌─────────┐       │
│  │ Device0 │  │ Device1 │       │
│  │ (0,0)   │  │ (0,1)   │       │
│  └─────────┘  └─────────┘       │
│  ┌─────────┐  ┌─────────┐       │
│  │ Device2 │  │ Device3 │       │
│  │ (1,0)   │  │ (1,1)   │       │
│  └─────────┘  └─────────┘       │
└──────────────────────────────────┘
```

### 8.2 텐서 분산

```python
# 모든 디바이스에 복제
distributed = ttnn.replicate_tensor_to_mesh_mapper(tensor, mesh_shape, device=mesh_device)

# 디바이스 간 샤딩
distributed = ttnn.shard_tensor_to_mesh_mapper(tensor, mesh_shape, device=mesh_device)

# 디바이스 텐서 결합
combined = ttnn.combine_device_tensors(device_tensors, mesh_shape)
```

### 8.3 Collective Communication

```python
# All-Reduce: 모든 디바이스의 결과를 합산
result = ttnn.all_reduce(tensor, topology=...)

# All-Gather: 모든 디바이스의 데이터 수집
result = ttnn.all_gather(tensor, topology=...)

# Reduce-Scatter: 리듀스 후 분산
result = ttnn.reduce_scatter(tensor, topology=...)

# Broadcast: 한 디바이스에서 전체로
result = ttnn.broadcast(tensor, topology=...)
```

### 8.4 분산 컨텍스트 (멀티 프로세스)

```python
# 분산 환경 초기화
ttnn.init_distributed_context(rank=0, size=4)

# 상태 확인
rank = ttnn.distributed_context_get_rank()
size = ttnn.distributed_context_get_size()

# 동기화 배리어
ttnn.distributed_context_barrier()
```

---

## 9. Python-C++ 바인딩 구조

### 9.1 Nanobind 아키텍처

TT-NN은 [nanobind](https://github.com/wjakob/nanobind)를 사용하여 C++ 구현을 Python에 노출합니다.

```
ttnn/cpp/ttnn-nanobind/
├── __init__.cpp           # 메인 모듈 초기화
├── tensor.hpp/cpp         # 텐서 바인딩
├── pytensor.cpp           # Python 텐서 래퍼
├── activation.cpp         # 활성화 타입
├── bfloat16_type_caster.hpp  # bfloat16 타입 캐스터
├── ndarray_helper.hpp     # NumPy 배열 상호운용
├── cluster.cpp            # 클러스터 타입
└── mesh_socket.cpp        # 메시 소켓
```

### 9.2 연산별 바인딩 패턴

각 C++ 연산은 다음 파일 구조를 따릅니다:

```
operation_name/
├── operation_name.hpp           # C++ 공개 인터페이스
├── operation_name.cpp           # C++ 구현
├── operation_name_nanobind.hpp  # Nanobind 바인딩 선언
├── operation_name_nanobind.cpp  # Nanobind 바인딩 구현
└── device/
    ├── device_operation.hpp/cpp # 디바이스 연산
    ├── program_factory.hpp      # 프로그램 팩토리
    └── kernels/
        ├── compute/             # RISC-V 컴퓨트 커널
        └── dataflow/            # 데이터 이동 커널
```

### 9.3 자동 등록

`__init__.py`에서 C++ 연산이 자동으로 Python 네임스페이스에 등록됩니다:

```python
def auto_register_ttnn_cpp_operations(module):
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if hasattr(attribute, "__ttnn_operation__") and attribute.__ttnn_operation__ is None:
            full_name = attribute.python_fully_qualified_name
            module_path, _, func_name = full_name.rpartition(".")
            target_module = create_module_if_not_exists(module_path)
            register_cpp_operation(target_module, func_name, attribute)
        elif isinstance(attribute, ModuleType):
            auto_register_ttnn_cpp_operations(attribute)

auto_register_ttnn_cpp_operations(ttnn._ttnn)
```

---

## 10. 프로파일링 및 디버깅

### 10.1 Tracy 프로파일러

```python
# Tracy 프로파일링 존
ttnn.start_tracy_zone("my_operation")
# 코드 실행
ttnn.stop_tracy_zone()

ttnn.tracy_message("Processing batch")
ttnn.tracy_frame()
```

### 10.2 Trace 기반 실행

연산 시퀀스를 캡처하여 반복 실행할 수 있습니다 (오버헤드 최소화):

```python
# 트레이스 캡처 시작
trace_id = ttnn.begin_trace_capture(device, command_queue)

# 연산 실행 (캡처됨)
result = ttnn.add(a, b)
result = ttnn.relu(result)

# 캡처 종료
ttnn.end_trace_capture(device, command_queue, trace_id)

# 캡처된 트레이스 반복 실행 (낮은 오버헤드)
for _ in range(100):
    ttnn.execute_trace(device, command_queue, trace_id)

# 트레이스 해제
ttnn.release_trace(device, trace_id)
```

### 10.3 성능 데이터

```python
# 최근 프로그램 성능 데이터
perf_data = ttnn.get_latest_programs_perf_data()

# 전체 프로그램 성능 데이터
all_perf_data = ttnn.get_all_programs_perf_data()
```

### 10.4 메모리 리포트

```python
# 메모리 리포트 활성화/비활성화
ttnn.EnableMemoryReports()
ttnn.DisableMemoryReports()

# 디바이스 메모리 상태 덤프
ttnn.dump_device_memory_state(device, prefix="debug_")
```

### 10.5 환경 변수를 이용한 디버깅

```bash
# 연산 로그 출력
export TT_LOGGER_TYPES=Op
export TT_LOGGER_LEVEL=DEBUG

# TT-NN 연산 로깅 (모든 연산을 블로킹으로 실행)
export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}'
```

### 10.6 관련 도구

| 도구 | 설명 |
|------|------|
| **TT-Triage** | ARC, NOC, L1, RISC-V 코어 건강 상태 진단 |
| **TT-NN Visualizer** | 모델 실행 시각화, 메모리 플롯, 연산 흐름 그래프 |
| **TT-Exalens** | 저수준 하드웨어 디버깅 |
| **TT-SMI** | 디바이스 텔레메트리, 펌웨어 정보 |
| **Model Explorer** | 모델 그래프 계층적 시각화 |
| **Kernel Print (DPRINT)** | 커널 내 변수/주소/CB 데이터 출력 |
| **Watcher** | 펌웨어/커널 오류 모니터링 |

---

## 11. 사용 예시

### 11.1 기본 텐서 연산

```python
import ttnn

# 디바이스 열기
device = ttnn.open_device(device_id=0)

# 텐서 생성
a = ttnn.full([5, 5, 5], fill_value=1.0, dtype=ttnn.bfloat16,
              layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.full([1], fill_value=2.0, dtype=ttnn.bfloat16,
              layout=ttnn.TILE_LAYOUT, device=device)

# 연산 실행
c = a * b  # element-wise 곱셈
print(c)

# 디바이스 정리
ttnn.close_device(device)
```

### 11.2 PyTorch 모델 변환 예시

```python
import torch
import ttnn

# PyTorch에서 텐서 준비
x = torch.randn(1, 1, 32, 32)
weight = torch.randn(32, 32)
bias = torch.randn(32)

# 디바이스 열기
device = ttnn.open_device(device_id=0)

# TT-NN으로 변환
x_tt = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
w_tt = ttnn.from_torch(weight, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
b_tt = ttnn.from_torch(bias, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

# Linear 연산 (matmul + bias)
out = ttnn.matmul(x_tt, w_tt)
out = ttnn.add(out, b_tt)
out = ttnn.relu(out)

# 결과를 PyTorch로 변환
result = ttnn.to_torch(out)

ttnn.close_device(device)
```

### 11.3 메모리 최적화된 연산

```python
# L1 샤딩을 사용한 최적화
memory_config = ttnn.create_sharded_memory_config(
    shape=[32, 32],
    core_grid=device.core_grid,
    strategy=ttnn.ShardStrategy.BLOCK
)

# 샤딩된 메모리에서 연산
tensor = ttnn.to_memory_config(tensor, memory_config)
result = ttnn.matmul(tensor, weights, memory_config=memory_config)
```

### 11.4 Transformer Attention

```python
# Scaled Dot-Product Attention
output = ttnn.transformer.scaled_dot_product_attention(
    query,    # [batch, heads, seq_len, head_dim]
    key,      # [batch, heads, seq_len, head_dim]
    value,    # [batch, heads, seq_len, head_dim]
    is_causal=True
)

# LLM 추론용 SDPA Decode (KV-cache 활용)
output = ttnn.transformer.sdpa_decode(
    query,    # [batch, heads, 1, head_dim] (현재 토큰만)
    key,      # [batch, heads, kv_len, head_dim] (캐시된 K)
    value     # [batch, heads, kv_len, head_dim] (캐시된 V)
)
```

---

## 12. 디렉토리 구조 레퍼런스

```
ttnn/
├── CMakeLists.txt                # 빌드 설정 (C++ 빌드, nanobind, 커널 컴파일)
├── README.md                     # 영문 README
│
├── api/                          # 공개 C++ API 헤더
│   └── ttnn/                     # 핵심 API (async_runtime, config, operation, device_operation)
│
├── core/                         # 핵심 C++ 구현
│   ├── tensor/                   # 텐서 핵심 구현
│   ├── distributed/              # 분산 처리
│   ├── graph/                    # 그래프 트레이싱
│   ├── async_runtime.cpp         # 비동기 런타임
│   ├── cluster.cpp               # 클러스터 관리
│   ├── device.cpp                # 디바이스 관리
│   └── operation.cpp             # 기본 연산 인터페이스
│
├── cpp/                          # C++ 연산 및 바인딩
│   ├── ttnn/
│   │   ├── operations/           # 32개 카테고리, 200+ 연산 구현
│   │   │   ├── eltwise/          # Element-wise (unary, binary, ternary, backward)
│   │   │   ├── matmul/           # 행렬 곱셈
│   │   │   ├── transformer/      # SDPA, Attention, RoPE
│   │   │   ├── normalization/    # Softmax, LayerNorm, RMSNorm
│   │   │   ├── conv/             # 합성곱 (conv1d, conv2d, conv_transpose2d)
│   │   │   ├── pool/             # 풀링
│   │   │   ├── reduction/        # 리덕션
│   │   │   ├── data_movement/    # 데이터 이동 (35+ 연산)
│   │   │   ├── ccl/              # Collective Communication
│   │   │   ├── embedding/        # 임베딩
│   │   │   ├── loss/             # 손실 함수
│   │   │   ├── moreh/            # 특화 연산 (34개)
│   │   │   ├── kv_cache/         # KV 캐시
│   │   │   ├── experimental/     # 실험적 연산 (23개)
│   │   │   └── ...
│   │   └── kernel/               # RISC-V 커널 (compute, dataflow)
│   └── ttnn-nanobind/            # Python 바인딩 (nanobind)
│
├── ttnn/                         # Python 패키지
│   ├── __init__.py               # 메인 진입점 (487줄)
│   ├── types.py                  # 타입 정의 (DataType, Layout, Memory 등)
│   ├── device.py                 # 디바이스 관리 (249줄)
│   ├── core.py                   # 코어 유틸리티 (363줄)
│   ├── decorators.py             # 연산 등록 시스템 (1074줄)
│   ├── database.py               # 프로파일링 DB (868줄)
│   ├── graph.py                  # 그래프 트레이싱
│   ├── tracer.py                 # 트레이서
│   ├── distributed/              # 분산 처리 래퍼
│   └── operations/               # 26개 Python 연산 모듈
│       ├── binary.py             # Binary 연산 (509줄)
│       ├── unary.py              # Unary 연산 (785줄)
│       ├── matmul.py             # 행렬 곱셈
│       ├── conv2d.py             # 합성곱
│       ├── normalization.py      # 정규화
│       ├── transformer.py        # Transformer
│       ├── data_movement.py      # 데이터 이동
│       ├── reduction.py          # 리덕션
│       ├── creation.py           # 텐서 생성
│       ├── activations.py        # 활성화
│       ├── ccl.py                # CCL
│       ├── *_backward.py         # 역전파 연산들
│       └── ...
│
├── tutorials/                    # 튜토리얼 (Jupyter notebooks)
├── examples/                     # C++ 예제
├── test/                         # 테스트 스위트
└── tt_lib/                       # 레거시 호환 라이브러리
```

---

## 부록: 주요 설정 옵션

| 설정 | 설명 | 기본값 |
|------|------|--------|
| `enable_fast_runtime_mode` | Op 검증 건너뛰기 (프로덕션용) | `True` |
| `enable_logging` | 연산 로깅 (모든 연산 블로킹) | `False` |
| `throw_exception_on_fallback` | 폴백 시 예외 발생 | `False` |
| `enable_comparison_mode` | Golden 비교 모드 | `False` |
| `comparison_mode_pcc` | PCC 비교 임계값 | `0.9999` |

```python
# 설정 변경
ttnn.CONFIG.enable_logging = True

# 임시 설정 변경
with ttnn.manage_config("enable_logging", True):
    result = ttnn.add(a, b)  # 로깅 활성화

# 환경 변수로 설정
# export TTNN_CONFIG_OVERRIDES='{"enable_logging": true}'
```

---

*이 문서는 tenstorrent/tt-metal 리포지토리의 TT-NN 모듈에 대한 종합 설명 자료입니다.*
*최종 업데이트: 2026-02-08*
