# P100 GPU에서 vLLM 서빙 가이드

## 개요

P100 (Pascal 아키텍처, compute capability 6.0)은 vLLM에서 공식 지원하지 않지만,
커뮤니티 패치를 통해 실험적으로 사용할 수 있습니다.

**P100 스펙:**
- VRAM: 16GB
- Compute Capability: 6.0
- FP16 지원 (bfloat16 미지원)

---

## 방법 1: Docker 사용 (권장)

가장 쉬운 방법입니다.

```bash
# Pascal 패치된 vLLM Docker 이미지 pull
docker pull ghcr.io/sasha0552/vllm:v0.10.0

# 서버 실행 (예: Llama 3 8B)
docker run --gpus all -p 8000:8000 \
    ghcr.io/sasha0552/vllm:v0.10.0 \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
```

---

## 방법 2: pip 설치

```bash
# 가상환경 생성
python -m venv vllm-pascal-env
source vllm-pascal-env/bin/activate

# Pascal 패치 패키지 인덱스 설정
export PIP_EXTRA_INDEX_URL="https://sasha0552.github.io/pascal-pkgs-ci/"

# vLLM Pascal 버전 설치
pip install vllm-pascal==0.10.0

# 필요시 패치된 Triton도 설치
pip install triton  # pascal-pkgs-ci에서 자동으로 패치 버전 설치됨
```

---

## P100 16GB에 적합한 모델 구성

인터뷰 도우미 아키텍처를 P100에 맞게 조정:

| 원래 모델 | P100 대체 모델 | VRAM 사용량 |
|-----------|----------------|-------------|
| Llama 3 8B | Llama 3 8B (Q4) | ~5GB |
| LLaVA 13B | LLaVA 7B (Q4) | ~4GB |
| Llama 3 70B | Llama 3 8B 또는 Qwen2.5-14B (Q4) | ~8GB |

### 추천 모델 조합 (총 ~14GB)

**Phase 1 (실시간):**
- STT: `openai/whisper-large-v3-turbo` (faster-whisper)
- LLM: `meta-llama/Meta-Llama-3-8B-Instruct` (Q4 양자화)

**Phase 2 (배치):**
- VLM: `llava-hf/llava-1.5-7b-hf` (Q4 양자화)
- 리포트: 동일한 8B 모델 재사용

---

## 서빙 스크립트

### serve_p100.py

```python
#!/usr/bin/env python3
"""P100 GPU용 vLLM 서빙 스크립트"""

from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import run_server

# P100 최적화 설정
MODEL_CONFIG = {
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "dtype": "float16",  # P100은 bfloat16 미지원
    "max_model_len": 4096,  # 메모리 절약
    "gpu_memory_utilization": 0.85,
    "enforce_eager": True,  # CUDA graph 비활성화 (안정성)
    "disable_custom_all_reduce": True,  # Pascal 호환성
}

def main():
    # OpenAI 호환 API 서버 실행
    import subprocess
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_CONFIG["model"],
        "--dtype", MODEL_CONFIG["dtype"],
        "--max-model-len", str(MODEL_CONFIG["max_model_len"]),
        "--gpu-memory-utilization", str(MODEL_CONFIG["gpu_memory_utilization"]),
        "--enforce-eager",
        "--host", "0.0.0.0",
        "--port", "8000",
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
```

### 실행 방법

```bash
# 환경변수 설정 (P100 최적화)
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS  # FlashAttention 대신 사용

# 서버 실행
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000
```

---

## API 테스트

```bash
# Health check
curl http://localhost:8000/health

# Chat completion (OpenAI 호환)
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
            {"role": "user", "content": "안녕하세요! 자기소개 부탁드립니다."}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }'
```

---

## 인터뷰 도우미 간소화 버전

P100에서 실행 가능한 간소화 아키텍처:

```
┌─────────────────────────────────────────────────────────┐
│                    P100 16GB                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐    ┌─────────────────────────┐    │
│  │ faster-whisper  │    │  vLLM Server            │    │
│  │ (CPU 기반)      │    │  Llama 3 8B-Instruct    │    │
│  │                 │    │  (~8GB VRAM)            │    │
│  └────────┬────────┘    └────────────┬────────────┘    │
│           │                          │                  │
│           └──────────┬───────────────┘                  │
│                      ▼                                  │
│           ┌──────────────────┐                          │
│           │  Interview App   │                          │
│           │  (Python)        │                          │
│           └──────────────────┘                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 구성 요소

1. **STT (CPU)**: faster-whisper는 CPU에서도 충분히 빠름
2. **LLM (GPU)**: vLLM으로 Llama 3 8B 서빙
3. **VLM (선택)**: 필요시 LLaVA 7B 교체 로드

---

## 트러블슈팅

### 1. CUDA out of memory
```bash
# max-model-len 줄이기
--max-model-len 2048

# 또는 양자화 모델 사용
--model TheBloke/Llama-2-7B-Chat-GPTQ
```

### 2. Triton 관련 오류
```bash
# Triton 비활성화
export VLLM_ATTENTION_BACKEND=XFORMERS
```

### 3. FlashAttention 오류
```bash
# P100은 FlashAttention 미지원, enforce-eager 사용
--enforce-eager
```

### 4. 느린 속도
- P100은 Tensor Core가 없어 V100 대비 ~2-3배 느림
- 양자화 모델 사용으로 개선 가능

---

## 대안: Ollama 사용

더 간단한 설정을 원한다면 Ollama도 P100을 지원합니다:

```bash
# Ollama 설치
curl -fsSL https://ollama.ai/install.sh | sh

# 모델 다운로드
ollama pull llama3:8b

# 서버 실행 (자동으로 GPU 사용)
ollama serve

# API 호출 (OpenAI 호환)
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3:8b",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

---

## 참고 자료

- [pascal-pkgs-ci](https://github.com/sasha0552/pascal-pkgs-ci) - Pascal GPU용 vLLM/Triton 빌드
- [vLLM 공식 문서](https://docs.vllm.ai)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - 빠른 Whisper 구현
