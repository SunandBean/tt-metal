# P100 작업 플랜 (실행 가이드)

이 문서는 P100에서 Llama 3.1 8B를 동작시키고, 이후 Whisper/VLM으로 확장하기 위한 **단계별 실행 가이드**입니다.

---

## Phase 1: Llama 3.1 8B Demo 동작 확인 (먼저 확인)

Demo가 이미 동작하는지부터 확인합니다. Demo는 `device_params`가 명시적으로 설정되어 있으므로 동작할 가능성이 높습니다.

### Step 1-1: 환경 설정

```bash
# P100 디바이스 확인
tt-smi  # 디바이스 상태 확인

# 환경 변수 설정
export MESH_DEVICE=P100
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
export WH_ARCH_YAML=blackhole  # Blackhole 아키텍처
```

### Step 1-2: 가장 간단한 Demo 테스트

```bash
# batch-1, ci-1 테스트 (가장 보수적)
pytest models/tt_transformers/demo/simple_text_demo.py::test_demo_text[ci-1-performance] \
    -svv \
    --max_seq_len 1024 \
    --batch_size 1 \
    --max_generated_tokens 32
```

**성공하면** → Phase 2로 진행
**L1 오버플로 발생하면** → Step 1-3으로

### Step 1-3: L1 오버플로 디버깅 (Demo에서 실패한 경우)

```bash
# 디버깅 로그 활성화
export TT_LOGGER_TYPES=Op
export TT_LOGGER_LEVEL=DEBUG
export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}'

# 재실행 후 에러 메시지에서 확인할 것:
# 1. "Statically allocated circular buffers ... grow to N B which is beyond max L1 size"
#    → per_core_M 또는 per_core_N이 너무 큼
# 2. "Statically allocated circular buffers ... clash with L1 buffers"
#    → sharded tensor와 CB가 충돌
# 3. "Out of Memory: Not enough space to allocate"
#    → DRAM 또는 L1 할당 실패
```

---

## Phase 2: Config 엔트리 추가 (핵심 코드 수정)

### Step 2-1: MAX_PREFILL_CHUNK_SIZES에 P100 추가

**파일**: `models/tt_transformers/tt/model_config.py` (line 547)

```python
# 수정 전:
"Llama-3.1-8B": {"N150": 4, "N300": 64, "T3K": 128, "TG": 128, "P150x4": 128},

# 수정 후:
"Llama-3.1-8B": {"P100": 4, "P150": 4, "N150": 4, "N300": 64, "T3K": 128, "TG": 128, "P150x4": 128},
```

> P100과 P150 모두 추가합니다. 값 `4`는 N150과 동일 (보수적 시작).

### Step 2-2: trace_region_size에 P100 추가

**파일**: `models/tt_transformers/demo/trace_region_config.py` (line 82)

```python
# 수정 전:
"Llama-3.1-8B": {
    "N150": 25000000,
    "N300": 38000000,
    "T3K": 50000000,
    "TG": 50000000,
},

# 수정 후:
"Llama-3.1-8B": {
    "P100": 25000000,
    "N150": 25000000,
    "N300": 38000000,
    "T3K": 50000000,
    "TG": 50000000,
},
```

### Step 2-3: 수정 후 Demo 재테스트

```bash
# 128 토큰 prefill
pytest models/tt_transformers/demo/simple_text_demo.py::test_demo_text[ci-1-performance] \
    -svv --max_seq_len 1024 --batch_size 1

# 성공하면 1024 토큰 prefill
pytest models/tt_transformers/demo/simple_text_demo.py::test_demo_text[batch-1-performance] \
    -svv --max_seq_len 1024 --batch_size 1 --max_generated_tokens 128
```

---

## Phase 3: tt-inference-server (vLLM) 동작시키기

### Step 3-1: generator_vllm.py에 Blackhole/P100 검증 추가

**파일**: `models/tt_transformers/tt/generator_vllm.py` (line 382-393)

```python
# 수정 전 (line 382-393):
hf_model_name = hf_config._name_or_path
if (
    ("3.1-8B" in hf_model_name or "3.2-11B" in hf_model_name)
    and mesh_device.get_num_devices() == 1
    and is_wormhole_b0()
):
    MAX_PROMPT_LEN = 32768
    if max_seq_len > MAX_PROMPT_LEN:
        raise ValueError(
            f"TT-LLama8B and TT-Llama11B do not support max_model_len greater than {MAX_PROMPT_LEN} on N150 "
            f"(received {max_seq_len}). Set --max_model_len to {MAX_PROMPT_LEN} or lower in vLLM."
        )

# 수정 후:
hf_model_name = hf_config._name_or_path
if (
    ("3.1-8B" in hf_model_name or "3.2-11B" in hf_model_name)
    and mesh_device.get_num_devices() == 1
):
    if is_wormhole_b0():
        MAX_PROMPT_LEN = 32768
    elif is_blackhole():
        MAX_PROMPT_LEN = 1024  # P100/P150 제한 (Issue #33991)
    else:
        MAX_PROMPT_LEN = 32768

    if max_seq_len > MAX_PROMPT_LEN:
        device_name = "P100/P150" if is_blackhole() else "N150"
        raise ValueError(
            f"Llama 8B does not support max_model_len > {MAX_PROMPT_LEN} on {device_name} "
            f"(received {max_seq_len}). Set --max_model_len to {MAX_PROMPT_LEN} or lower in vLLM."
        )
```

**import도 추가** (line 35 근처):
```python
from models.common.utility_functions import is_wormhole_b0, nearest_32
# 아래 추가:
from models.common.utility_functions import is_blackhole
```

### Step 3-2: tt-inference-server에서 device_params 전달

tt-inference-server (별도 리포지토리)의 `run.py`에서 디바이스를 열 때 아래와 유사한 파라미터가 전달되어야 합니다.

**확인할 것**: tt-inference-server 코드에서 `ttnn.open_device()` 또는 `CreateDevice()` 호출 부분을 찾아서 아래 파라미터가 전달되는지 확인:

```python
# P100 필수 파라미터:
trace_region_size = 25000000  # 25MB
num_command_queues = 1
```

**만약 직접 전달이 어려우면**, 환경변수로 우회:

```bash
# tt-inference-server 실행 전에
export TT_METAL_TRACE_REGION_SIZE=25000000  # 이 환경변수가 지원되는지 확인 필요
```

### Step 3-3: vLLM 시작 옵션 (P100 전용)

```bash
# P100에서 Llama 3.1 8B 실행
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --device tt \
    --max_model_len 1024 \
    --max_num_seqs 1 \
    --block_size 32 \
    --dtype bfloat8
```

> **주의**: `--max_model_len`을 반드시 1024 이하로 설정

### Step 3-4: API 호출 테스트

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "Hello, how are you?",
        "max_tokens": 32,
        "temperature": 0
    }'
```

---

## Phase 4: L1 오버플로 추가 대응 (Phase 2-3에서 실패할 경우)

### Step 4-1: per_core_M 추가 축소

**파일**: `models/tt_transformers/tt/model_config.py` (line 927)

```python
# 현재: per_core_M=7
# 7에서도 실패하면 6으로:
per_core_M=6
if self.device_name == "P100"
```

### Step 4-2: l1_small_size 축소 테스트

디바이스 열 때 `l1_small_size=16384`로 설정하여 dispatch 영역을 줄입니다.

**Demo에서 테스트**:
```python
# conftest.py의 device_params에서:
device_params = {"trace_region_size": 25000000, "l1_small_size": 16384, "num_command_queues": 1}
```

### Step 4-3: FP32 누산 비활성화 재확인

```python
# model_config.py:965-969 확인
# fp32_dest_acc_en이 반드시 False인지:
fp32_dest_acc_en=False  # True면 intermediate CB 2배 → L1 초과
```

### Step 4-4: LM Head 최적화 (필요 시)

**파일**: `models/tt_transformers/tt/model_config.py` (line 903-908)

```python
# 현재 코드 (Blackhole 전체 하드코딩):
self.max_columns_per_device_lm_head = (
    128256 // 8 if is_blackhole() else 668 * self.lm_head_core_grid.num_cores
)

# P100 전용으로 변경:
if is_blackhole():
    dram_cores = self.dram_grid_size.x  # P100: 7, P150: 8
    self.max_columns_per_device_lm_head = 128256 // (8 if dram_cores == 8 else 9)
else:
    self.max_columns_per_device_lm_head = 668 * self.lm_head_core_grid.num_cores
```

---

## Phase 5: Whisper on P100

### Step 5-1: 기존 Whisper 테스트 실행

```bash
export MESH_DEVICE=P100

# 모듈별 테스트
pytest models/demos/audio/whisper/tests/test_whisper_modules.py -v --timeout 600

# 실패하면 개별 모듈 테스트
pytest models/demos/audio/whisper/tests/test_whisper_modules.py -v -k "encoder" --timeout 600
pytest models/demos/audio/whisper/tests/test_whisper_modules.py -v -k "decoder" --timeout 600
```

### Step 5-2: Whisper Demo 실행

```bash
pytest models/demos/audio/whisper/demo/demo.py -v --timeout 600
```

### Step 5-3: L1 오버플로 시 (Whisper)

Whisper는 모델이 작으므로 (~1.5B) 큰 문제가 없을 것으로 예상하지만, 만약 실패하면:

1. Encoder seq_len 확인 (30초 오디오 → ~1500 토큰)
2. `prefill_len_cutoff = 512` (Blackhole) 내에서 chunking이 잘 되는지 확인
3. Conv1D → Conv2D 내부 변환이 P100 코어 그리드에서 정상인지 확인

---

## Phase 6: VLM on P100 (Qwen2.5-VL-3B)

### Step 6-1: Config 추가

**파일**: `models/tt_transformers/tt/model_config.py` (line 554)

```python
# 수정 전:
"Qwen2.5-VL-3B": {"N150": 128, "N300": 128, "T3K": None, "TG": None, "P150x4": None},

# 수정 후:
"Qwen2.5-VL-3B": {"P100": 128, "P150": 128, "N150": 128, "N300": 128, "T3K": None, "TG": None, "P150x4": None},
```

### Step 6-2: Qwen2.5-VL 테스트

```bash
export MESH_DEVICE=P100
export HF_MODEL=Qwen/Qwen2.5-VL-3B-Instruct

pytest models/demos/qwen25_vl/ -v --timeout 600
```

### Step 6-3: 이미지 해상도 제한

P100에서 224×224 이미지 기준 vision encoder가 196 patches → prefill_len_cutoff(512) 이내.
고해상도 이미지 테스트 시 `max_image_size=224`로 제한 필요.

---

## 수정 파일 요약 (체크리스트)

| # | 파일 | 수정 내용 | Phase |
|---|------|----------|-------|
| 1 | `models/tt_transformers/tt/model_config.py:547` | MAX_PREFILL_CHUNK_SIZES에 `"P100": 4` 추가 | 2 |
| 2 | `models/tt_transformers/demo/trace_region_config.py:82` | trace_region_size에 `"P100": 25000000` 추가 | 2 |
| 3 | `models/tt_transformers/tt/generator_vllm.py:382-393` | `is_blackhole()` 체크 추가, max_seq_len 검증 | 3 |
| 4 | `models/tt_transformers/tt/generator_vllm.py:35` | `from ... import is_blackhole` 추가 | 3 |
| 5 | `models/tt_transformers/tt/model_config.py:927` | per_core_M=7→6 (L1 오버플로 지속 시) | 4 |
| 6 | `models/tt_transformers/tt/model_config.py:903-908` | LM head max_columns P100 최적화 | 4 |
| 7 | `models/tt_transformers/tt/model_config.py:554` | Qwen VL에 `"P100": 128` 추가 | 6 |
| 8 | tt-inference-server `run.py` | device_params에 trace_region_size 전달 | 3 |

---

## 성공 기준

### Llama 3.1 8B
- [ ] Demo `ci-1-performance` 테스트 통과 (batch=1, seq=128)
- [ ] Demo `batch-1-performance` 테스트 통과 (batch=1, seq=1024)
- [ ] Decode 속도 ≥ 25 tok/s/u (목표: 29.5)
- [ ] vLLM `--max_model_len 1024` 로 API 서빙 정상 동작
- [ ] vLLM에서 10회 연속 요청 처리 (안정성)

### Whisper
- [ ] Encoder forward pass 통과
- [ ] Decoder forward pass 통과 (cross-attention 포함)
- [ ] 30초 오디오 end-to-end 변환 성공
- [ ] WER(Word Error Rate) 참조 모델 대비 5% 이내

### VLM (Qwen2.5-VL-3B)
- [ ] Vision encoder (224×224) forward pass 통과
- [ ] Language model forward pass 통과
- [ ] 이미지 + 텍스트 end-to-end 추론 성공

---

## 트러블슈팅 참조

| 에러 메시지 | 원인 | 해결 |
|------------|------|------|
| `circular buffers grow to N B beyond max L1` | per_core_M/N 너무 큼 | per_core_M 축소 (7→6→4) |
| `circular buffers clash with L1 buffers` | sharded tensor + CB 충돌 | DRAM_MEMORY_CONFIG 사용 |
| `Out of Memory: Not enough space` | DRAM/L1 할당 실패 | max_seq_len 축소, trace_region_size 축소 |
| `max_model_len greater than` | vLLM seq_len 초과 | `--max_model_len 1024` |
| `P100 runs OOM in L1 with 8 per_core_M` | QKV matmul L1 초과 | 이미 per_core_M=7로 패치됨 |
| `Unknown model ... setting MAX_PREFILL_CHUNK_SIZE to 4` | P100 config 없음 | Step 2-1 적용 |
| `KeyError: 'P100'` | config dict에 P100 없음 | 해당 dict에 P100 엔트리 추가 |
