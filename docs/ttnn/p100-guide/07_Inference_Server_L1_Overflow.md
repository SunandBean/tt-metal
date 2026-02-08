# tt-inference-server (vLLM) P100 L1 오버플로 분석

## 핵심 문제: Demo는 되는데 Inference Server에서는 L1 오버플로

데모(`simple_text_demo.py`)와 inference server(`tt-inference-server`의 `run.py` → vLLM backend)는 **디바이스 초기화 방식이 근본적으로 다릅니다**. 이 차이가 P100에서 L1 오버플로를 유발합니다.

## Demo vs Inference Server 비교

### 디바이스 초기화 파라미터

| 파라미터 | Demo (pytest) | Inference Server (vLLM) |
|----------|-------------|----------------------|
| `trace_region_size` | **50,000,000** (50MB, 명시적) | **0** (기본값, `DEFAULT_TRACE_REGION_SIZE`) |
| `fabric_config` | **True** (명시적) | 설정 안 함 (기본값) |
| `num_command_queues` | **1** (명시적) | 설정 안 함 (기본값) |
| `l1_small_size` | 기본값 (0) | 기본값 (0) |

**증거 - Demo:**
```python
# simple_text_demo.py:751-752
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
```

**증거 - 기본값:**
```cpp
// tt_metal/hostdevcommon/api/hostdevcommon/common_values.hpp:15-16
constexpr static std::size_t DEFAULT_L1_SMALL_SIZE = 0;
constexpr static std::size_t DEFAULT_TRACE_REGION_SIZE = 0;
```

### trace_region_size의 중요성

Demo는 `trace_region_config.py`에서 모델/디바이스별로 최적의 `trace_region_size`를 선택합니다:

```python
# trace_region_config.py:82-87
"Llama-3.1-8B": {
    "N150": 25000000,    # 25 MB
    "N300": 38000000,    # 38 MB
    "T3K": 50000000,     # 50 MB
    "TG": 50000000,      # 50 MB
    # P100: 없음! → None 반환
}
```

**문제**: P100 엔트리 자체가 없고, inference server는 이 config를 참조하지도 않음.

### max_seq_len 검증 누락

```python
# generator_vllm.py:383-393
# LlamaForCausalLM.initialize_vllm_model()
if (
    ("3.1-8B" in hf_model_name or "3.2-11B" in hf_model_name)
    and mesh_device.get_num_devices() == 1
    and is_wormhole_b0()          # ← Wormhole만 체크!
):
    MAX_PROMPT_LEN = 32768
    if max_seq_len > MAX_PROMPT_LEN:
        raise ValueError(...)
# P100 (Blackhole) 체크 없음!
# → P100에서 너무 큰 max_seq_len이 그대로 전달됨
```

**vLLM에서 `--max_model_len=131072` 같은 큰 값을 넣으면** → P100의 28GB DRAM에서 KV cache 할당 실패 또는 matmul config에서 L1 오버플로 발생.

## L1 오버플로 발생 경로 상세

### 경로 1: trace_region_size 기본값 문제

```
vLLM run.py 시작
    ↓
ttnn.open_device(trace_region_size=0)  ← 기본값
    ↓
Trace capture 시 DRAM에서 trace region 할당
    ↓
trace_region_size=0이면 시스템이 자동 결정
    ↓
P100의 28GB DRAM에서 과다 할당 가능
    ↓
나머지 weight + KV cache 공간 부족
    ↓
weight를 L1에 spill하거나 sharding 변경
    ↓
L1 오버플로!
```

### 경로 2: max_seq_len 검증 누락

```
vLLM --max_model_len=131072 (기본값)
    ↓
LlamaForCausalLM.initialize_vllm_model()
    ↓
is_wormhole_b0() = False (P100은 Blackhole)
    ↓
max_seq_len 검증 건너뜀!
    ↓
ModelArgs(max_seq_len=131072)
    ↓
KV cache 할당: 32 layers × 2(K,V) × 131072 × 128 × 8heads
    ↓
DRAM 과다 사용 → weight DRAM sharding 불가
    ↓
Matmul이 L1 기반으로 전환 → L1 오버플로!
```

### 경로 3: prefill chunk size 미설정

```
vLLM에서 긴 프롬프트 (>1024 토큰) 수신
    ↓
Generator.prefill_forward_text() 호출
    ↓
model_args.max_prefill_chunk_size 참조
    ↓
MAX_PREFILL_CHUNK_SIZES에 P100 없음
    ↓
기본값 4 * 1024 = 4096으로 폴백 (경고만 출력)
    ↓
4096 토큰 prefill 시도
    ↓
P100 prefill_len_cutoff = 512 → 8개 chunk
    ↓
각 chunk에서 QKV matmul per_core_M=7 사용
    ↓
하지만 다른 연산(MLP, LM head)은 P100 최적화 안 됨
    ↓
L1 오버플로!
```

## 구체적 수정 방안

### 수정 1: vLLM 모델 초기화에 P100 검증 추가

**파일**: `models/tt_transformers/tt/generator_vllm.py`

```python
# 현재 코드 (line 383-393):
if (
    ("3.1-8B" in hf_model_name or "3.2-11B" in hf_model_name)
    and mesh_device.get_num_devices() == 1
    and is_wormhole_b0()
):
    MAX_PROMPT_LEN = 32768
    ...

# 수정 제안:
from models.common.utility_functions import is_blackhole
if (
    ("3.1-8B" in hf_model_name or "3.2-11B" in hf_model_name)
    and mesh_device.get_num_devices() == 1
):
    if is_wormhole_b0():
        MAX_PROMPT_LEN = 32768
    elif is_blackhole():
        # P100: DRAM 28GB, seq_len > 1024 미지원 (Issue #33991)
        MAX_PROMPT_LEN = 1024
    else:
        MAX_PROMPT_LEN = 32768

    if max_seq_len > MAX_PROMPT_LEN:
        raise ValueError(
            f"Llama 8B does not support max_model_len > {MAX_PROMPT_LEN} "
            f"on this device. Set --max_model_len={MAX_PROMPT_LEN} in vLLM."
        )
```

### 수정 2: tt-inference-server의 run.py에서 device_params 전달

tt-inference-server (별도 리포지토리)에서 디바이스를 열 때 P100 전용 파라미터를 전달해야 합니다:

```python
# tt-inference-server의 device 초기화 코드에 추가:
device_params = {}
if is_blackhole():
    dram_grid = mesh_device.dram_grid_size()
    if dram_grid.x == 7:  # P100
        device_params = {
            "trace_region_size": 25000000,   # 25MB
            "num_command_queues": 1,
            "fabric_config": False,           # P100은 이더넷 없음
        }
    else:  # P150
        device_params = {
            "trace_region_size": 50000000,
            "num_command_queues": 1,
            "fabric_config": True,
        }
```

### 수정 3: vLLM 시작 시 환경변수로 제한

tt-inference-server의 `run.py`에서 P100일 때:

```bash
# P100에서 Llama 3.1 8B 실행 시 권장 설정
export MESH_DEVICE=P100
export MAX_PREFILL_CHUNK_SIZE=4

# vLLM 시작
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max_model_len 1024 \           # P100 제한!
    --max_num_seqs 1 \               # 배치 1
    --block_size 32 \
    --device tt
```

### 수정 4: model_config.py에 P100 config 완성

```python
# MAX_PREFILL_CHUNK_SIZES (line ~547)
"Llama-3.1-8B": {
    "P100": 4,          # 추가!
    "N150": 4,
    "N300": 64,
    ...
},

# trace_region_config.py (line ~82)
"Llama-3.1-8B": {
    "P100": 25000000,   # 추가!
    "N150": 25000000,
    ...
},
```

## Demo가 작동하는 이유 (비교)

| 항목 | Demo | Inference Server |
|------|------|-----------------|
| `trace_region_size` | 50MB (명시적) | 0 (기본값) |
| `max_seq_len` | 1024 (테스트별 고정) | 131072 (vLLM 기본값) |
| P100 config | pytest 파라미터화 | 없음 |
| `fabric_config` | True | 미설정 |
| Prefill chunk | trace_region_config 참조 | 미참조 |
| Warmup | `warmup_model_prefill()` 호출 | 호출 여부 불확실 |

## 디버깅 순서

### Step 1: vLLM 시작 시 로그 확인

```bash
export TT_LOGGER_TYPES=Op
export TT_LOGGER_LEVEL=INFO
# vLLM 시작 후 첫 번째 요청에서 에러 로그 확인
```

### Step 2: max_model_len 축소 테스트

```bash
# 가장 보수적인 설정부터 시작
python -m vllm ... --max_model_len 1024 --max_num_seqs 1
# 성공하면 점진적으로 늘리기
```

### Step 3: device_params 직접 확인

```python
# vLLM의 tt_model_runner.py에서 디바이스 열기 전에
import ttnn
print(f"DRAM grid: {mesh_device.dram_grid_size()}")
print(f"L1 per core: {mesh_device.l1_size_per_core()}")
print(f"Num devices: {mesh_device.get_num_devices()}")
```

### Step 4: L1 사용량 모니터링

```python
# 연산 실행 전후로 L1 상태 확인
ttnn.dump_device_memory_state(device, prefix="before_qkv_")
result = ttnn.matmul(...)  # QKV prefill
ttnn.dump_device_memory_state(device, prefix="after_qkv_")
```

## 근본 해결을 위한 장기 과제

1. **tt-inference-server에 디바이스별 config 시스템 추가**
   - `trace_region_config.py`와 유사한 구조를 inference server에도 적용
   - DRAM grid size 기반 자동 감지

2. **vLLM TT backend에 Blackhole/P100 분기 추가**
   - `generator_vllm.py`의 `initialize_vllm_model()`에 `is_blackhole()` 체크
   - `max_seq_len` 자동 제한

3. **디바이스 자동 프로파일링**
   - 디바이스 열 때 사용 가능한 L1/DRAM 확인
   - 자동으로 최적 config 선택

4. **Issue #33991 해결**
   - P100 seq_len > 1024 지원
   - per_core_M/N 동적 계산 개선
