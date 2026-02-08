# VLM (Vision-Language Model) on P100: 현황 및 포팅 가이드

## 이미 구현된 VLM 모델들

tt-metal에는 다양한 VLM이 이미 구현되어 있습니다:

| 모델 | 위치 | Vision Encoder | Language Model | 크기 |
|------|------|---------------|---------------|------|
| **Gemma-3-4B Vision** | `models/demos/multimodal/gemma3/` | SigLIP (27 블록) | Gemma 2B | ~4B |
| **Qwen2.5-VL-3B** | `models/demos/qwen25_vl/` | Vision Attention | Qwen 3B | ~3B |
| **Qwen2.5-VL-7B** | `models/demos/qwen25_vl/` | Vision Attention | Qwen 7B | ~7B |
| **Qwen3-VL-32B** | `models/demos/qwen3_vl/` | DeepStack Merger | Qwen 32B | ~32B |
| **Llama 3.2 Vision 11B** | `models/tt_transformers/tt/multimodal/` | Conv2D + ViT | Llama 8B | ~11B |
| **PI0** | `models/experimental/pi0/` | SigLIP | Gemma 2B | ~3B |
| **ViT** | `models/demos/vision/classification/vit/` | - | - | ~300M |

## P100에 적합한 VLM 후보

P100은 싱글 디바이스(32GB DRAM)이므로, **8B 이하 모델**이 적합합니다:

### 1순위: Qwen2.5-VL-3B (가장 현실적)

```python
# 이미 N150 지원 (MAX_PREFILL_CHUNK_SIZES)
"Qwen2.5-VL-3B": {"N150": 128, "N300": 128, ...}
```

- **장점**: 모델이 작고, N150(12GB DRAM, 80 코어) 지원이 이미 됨
- **P100 이점**: N150보다 DRAM 2.6배 크고, 코어 1.75배 많음
- **필요 작업**: P100 config 엔트리 추가만으로 가능성 높음

### 2순위: Gemma-3-4B Vision

- **장점**: SigLIP vision encoder + 작은 language model
- **구현 완성도**: 20+ 테스트 모듈
- **필요 작업**: P100 config 추가 + L1 프로파일링

### 3순위: Qwen2.5-VL-7B

```python
# N150/N300 지원
"Qwen2.5-VL-7B": {"N150": 64, "N300": 128, ...}
```

- **주의**: 7B 모델이므로 P100에서 L1 압박 가능
- **필요 작업**: Llama 8B와 유사한 수준의 L1 최적화

## VLM 아키텍처 공통 구조

```
Image Input
    ↓
[TT Device] Patch Embedding (Conv2D)
    ↓
[TT Device] Vision Encoder (ViT/SigLIP Self-Attention × N)
    ↓
[TT Device] Multimodal Projector (Linear)
    ↓
[TT Device] Language Model (Cross-Attention + Self-Attention × N)
    ↓
Text Output
```

### 필요 연산과 구현 상태

| 연산 | 상태 | 비고 |
|------|------|------|
| **Conv2D (Patch Embedding)** | 구현됨 | 이미지 → 패치 변환 |
| **Vision Self-Attention** | 구현됨 | SigLIP, ViT 등 |
| **Position Embedding** | 구현됨 | 학습된 위치 임베딩 |
| **Cross-Attention** | 구현됨 | Vision → Language |
| **Multimodal Projector** | 구현됨 | Linear 변환 |
| **RMSNorm** | 구현됨 | 정규화 |
| **Rotary Embedding** | 구현됨 | RoPE |
| **SDPA** | 구현됨 | Scaled Dot-Product Attention |
| **Image Preprocessing** | CPU | Resize, Normalize 등 |

## P100에서 VLM을 돌리기 위한 구체적 작업

### 작업 1: P100 Config 엔트리 추가

**파일**: `models/tt_transformers/tt/model_config.py`

```python
# MAX_PREFILL_CHUNK_SIZES에 P100 추가
"Qwen2.5-VL-3B": {"P100": 128, "N150": 128, "N300": 128, ...},
"Qwen2.5-VL-7B": {"P100": 4, "N150": 64, "N300": 128, ...},
```

### 작업 2: DRAM Grid Width 맞추기

VLM의 weight matmul에서 P100의 DRAM 코어 7개에 맞게 per_core_N 계산:

```python
# 기존 로직 활용 (이미 model_config.py에 있음)
dram_shard_grid_width = self.dram_grid_size.x  # P100: 7
per_core_N = math.ceil(n / (tile_size * dram_shard_grid_width))
```

### 작업 3: Vision Encoder L1 프로파일링

Vision encoder의 self-attention은 이미지 토큰 수에 비례하는 L1 사용:

```
Image: 224×224, Patch: 16×16
→ 196 patches = 196 tokens (seq_len)
→ P100 prefill_len_cutoff: 512
→ 196 < 512 → 단일 chunk로 처리 가능!
```

**좋은 소식**: 일반적인 이미지(224×224)에서 vision encoder의 seq_len은 196으로, P100의 512 cutoff 내에 여유 있게 들어갑니다.

### 작업 4: Cross-Attention 메모리 설정

```python
# Vision encoder 출력을 DRAM에 캐시
vision_cache_config = ttnn.DRAM_MEMORY_CONFIG  # L1 부담 없음

# Cross-attention KV는 DRAM에서 읽기
cross_attn_memory_config = ttnn.create_sharded_memory_config(
    shape=(...),
    core_grid=device.core_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
)
```

### 작업 5: Trace Region 설정

```python
# trace_region_config.py
"Qwen2.5-VL-3B": {
    "P100": 10000000,    # 10MB
},
"Qwen2.5-VL-7B": {
    "P100": 15000000,    # 15MB
},
```

## Vision 전용 연산 고려사항

### 고해상도 이미지 지원

일부 VLM은 고해상도(예: 448×448, 896×896)를 지원합니다:

```
448×448 / 16×16 = 784 patches → 784 tokens
→ 784 > 512 (prefill cutoff) → Chunking 필요!

896×896 / 16×16 = 3136 patches → 3136 tokens
→ 심각한 L1 압박 + 다중 chunk 필요
```

**권장**: P100에서는 224×224 또는 336×336 해상도를 사용하여 prefill cutoff 내에서 동작하도록 제한.

### Dynamic Resolution (Qwen-VL 특화)

Qwen-VL 시리즈는 동적 해상도를 지원하지만, P100에서는 고정 해상도를 사용하는 것이 안전합니다.

## 예상 난이도

| 모델 | 난이도 | 이유 |
|------|--------|------|
| **Qwen2.5-VL-3B** | **낮음** | 작은 모델, N150 이미 지원, config 추가만으로 가능 |
| **Gemma-3-4B Vision** | **중간** | 비전 인코더 추가 L1 프로파일링 필요 |
| **Qwen2.5-VL-7B** | **중간-높음** | Llama 8B 수준의 L1 최적화 필요 |
| **Llama 3.2 Vision 11B** | **높음** | 8B 이상, P100 단독으로 빠듯 |

## 장기적 추가 작업

1. **멀티 이미지 지원**: 여러 이미지 입력 시 vision encoder 반복 → DRAM 캐싱 최적화
2. **Video 입력**: 프레임별 vision encoder + 시간축 attention (현재 미구현)
3. **On-device 이미지 전처리**: Resize, Normalize를 TT 디바이스에서 (현재 CPU)
4. **양자화**: Vision encoder에 BFP4/BFP8 적용으로 추가 메모리 절감
