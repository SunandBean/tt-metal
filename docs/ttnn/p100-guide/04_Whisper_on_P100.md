# Whisper on P100: 현황 및 포팅 가이드

## 현재 구현 상태: 이미 구현됨

Whisper 모델은 `models/demos/audio/whisper/`에 이미 구현되어 있습니다.

| 파일 | 역할 |
|------|------|
| `tt/ttnn_optimized_functional_whisper.py` | TTNN 최적화 구현 (인코더 + 디코더) |
| `tt/whisper_generator.py` | 텍스트 생성 로직 |
| `demo/demo.py` | 데모 실행 스크립트 |
| `tests/test_whisper_modules.py` | 모듈별 테스트 |

### 지원 모델

- `distil-whisper/distil-large-v3` (주력)
- `openai/whisper-base` (테스트)
- 조건부 생성, 오디오 분류, 번역 지원

## Whisper 아키텍처와 TT-NN 매핑

```
Audio Input
    ↓
[CPU] Mel Spectrogram (HuggingFace AutoFeatureExtractor)
    ↓
[TT Device] Conv1D Feature Extraction
    ↓
[TT Device] Encoder Self-Attention (×N layers)
    ↓
[TT Device] Decoder Self-Attention + Cross-Attention (×N layers)
    ↓
[TT Device] LM Head
    ↓
Text Output
```

### 필요한 연산과 구현 상태

| 연산 | TT-NN 구현 | 위치 |
|------|-----------|------|
| **Conv1D** | 구현됨 (Conv2D 래퍼) | `ttnn/cpp/ttnn/operations/conv/conv1d/` |
| **Self-Attention** | 구현됨 | `ttnn/cpp/ttnn/operations/transformer/sdpa/` |
| **Cross-Attention** | 구현됨 | `models/demos/audio/whisper/tt/` |
| **Layer Norm** | 구현됨 | `ttnn/cpp/ttnn/operations/normalization/layernorm/` |
| **GELU** | 구현됨 | `ttnn/cpp/ttnn/operations/eltwise/unary/` |
| **Matmul (Linear)** | 구현됨 | `ttnn/cpp/ttnn/operations/matmul/` |
| **KV Cache** | 구현됨 | Whisper generator에서 관리 |
| **Mel Spectrogram** | CPU 전처리 | HuggingFace 라이브러리 |

## P100에서 돌리기 위한 수정 사항

### 1. 디바이스 호환성 확인

Whisper 구현은 Wormhole와 Blackhole 모두를 대상으로 하지만, P100 전용 config는 없습니다.

**확인 필요 사항:**
- Conv1D → Conv2D 변환이 P100의 코어 그리드에서 올바르게 동작하는지
- Encoder self-attention의 core grid 설정이 P100(8x10 Blackhole)에 맞는지
- Cross-attention KV cache의 메모리 설정이 P100 DRAM grid(7개)에 맞는지

### 2. L1 메모리 영향 분석

Whisper는 Llama 3.1 8B보다 **훨씬 작은 모델**입니다:

| 비교 항목 | Whisper Large-v3 | Llama 3.1 8B |
|-----------|-----------------|--------------|
| 파라미터 | ~1.5B | 8B |
| Hidden dim | 1280 | 4096 |
| FFN dim | 5120 | 14336 |
| Attention heads | 20 | 32 |
| Encoder layers | 32 | - |
| Decoder layers | 32 | 32 |

**L1 영향**: 모델 크기가 작으므로 per_core_M, per_core_N이 훨씬 작게 계산됨 → **L1 오버플로 위험 낮음**

### 3. Encoder Sequence Length

Whisper encoder는 30초 오디오에 대해 ~1500 토큰을 처리합니다.

```
P100 prefill_len_cutoff: 512 (Blackhole)
Whisper encoder seq: ~1500
→ Chunking 필요 (512 토큰씩 나누어 처리)
```

**주의**: `prefill_len_cutoff`가 512로 제한되어 있으므로, encoder prefill을 3-4개 chunk로 나눠야 합니다. 기존 Whisper 구현에서 이미 chunking을 지원하는지 확인 필요.

### 4. Conv1D 동작 확인

```cpp
// ttnn/cpp/ttnn/operations/conv/conv1d/conv1d.cpp
// Conv1D는 내부적으로 4D reshape → Conv2D 호출
// [batch, channels, length] → [batch, channels, 1, length] → Conv2D
```

Conv2D가 P100에서 정상 작동하면 Conv1D도 자동으로 동작합니다.

### 5. Trace 최적화

Whisper 구현은 `ttnn.begin_trace_capture()` / `ttnn.execute_trace()`를 사용합니다:

```python
# whisper_generator.py
# 첫 번째 decoder iteration: 일반 실행
# 이후 iterations: trace 기반 최적화 실행
trace_id = ttnn.begin_trace_capture(device, command_queue)
# ... decoder forward pass ...
ttnn.end_trace_capture(device, command_queue, trace_id)
# 반복 실행
ttnn.execute_trace(device, command_queue, trace_id)
```

P100에서 trace_region_size를 적절히 설정해야 합니다.

## 구체적 수정 계획

### Step 1: 기본 동작 테스트

```bash
# Whisper 기본 테스트 실행
pytest models/demos/audio/whisper/tests/test_whisper_modules.py -v
```

### Step 2: P100 config 추가 (필요 시)

```python
# model_config 또는 whisper config에 P100 설정 추가
# (Whisper가 별도 config 시스템을 사용하면 해당 파일에)
"whisper-large-v3": {
    "P100": {
        "prefill_chunk_size": 512,
        "trace_region_size": 10000000,  # 10MB (모델이 작으므로)
    }
}
```

### Step 3: Encoder prefill chunking 확인

1500 토큰 encoder prefill을 512 토큰 chunk로 나눌 수 있는지 확인.
기존 구현에서 이미 지원하면 OK, 아니면 chunking 로직 추가.

### Step 4: Cross-attention cache DRAM 설정

```python
# Cross-attention KV cache는 encoder 출력을 저장
# DRAM에 배치하면 L1 부담 없음
cross_attn_kv_cache_config = ttnn.DRAM_MEMORY_CONFIG
```

## 예상 난이도

**낮음 ~ 중간**

- 모델이 Llama보다 훨씬 작아서 L1 문제가 적을 것
- 핵심 연산이 모두 구현됨
- 주요 작업: P100 device config 설정 + encoder chunking 확인

## Mel Spectrogram (on-device 처리 불가)

현재 Mel spectrogram은 CPU에서 HuggingFace `AutoFeatureExtractor`로 처리합니다.

On-device 처리를 위해서는 다음이 필요하지만, **현재 미구현이며 우선순위 낮음:**
- FFT/STFT 커널
- Mel 필터뱅크 행렬 곱
- 로그 스케일링

실용적으로는 CPU 전처리로 충분합니다 (30초 오디오의 spectrogram 변환은 CPU에서 수 ms).
