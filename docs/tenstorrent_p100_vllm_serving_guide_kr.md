# Tenstorrent P100에서 vLLM 서빙 가이드

## 개요

Tenstorrent P100 (Blackhole 아키텍처)에서 vLLM을 사용한 LLM 서빙 가이드입니다.

**P100 스펙:**
- 칩: Blackhole (6nm)
- Tensix 코어: 120개
- SRAM: 180MB
- DRAM: 28GB GDDR6 (448 GB/s)
- TDP: 300W
- 가격: $999

**P100에서 지원되는 모델:**
| 모델 | 성능 (t/s/u) | TTFT (ms) | Top-1 정확도 |
|------|-------------|-----------|--------------|
| Llama 3.1 8B | 29.5 | 84 | 90% |

---

## 사전 요구사항

### 1. 시스템 설정

Tenstorrent 문서에 따라 시스템을 설정합니다:
- 펌웨어 설치
- 커널 모듈 드라이버 (KMD) 설치
- Hugepages 설정

자세한 내용: https://docs.tenstorrent.com

### 2. tt-metal 설치

```bash
# tt-metal 레포 클론 (이미 되어있다면 스킵)
git clone https://github.com/tenstorrent/tt-metal.git
cd tt-metal

# 설치 가이드 참조
cat INSTALLING.md
```

### 3. HuggingFace 토큰 설정

```bash
# HuggingFace CLI 로그인
huggingface-cli login

# 또는 환경변수로 설정
export HF_TOKEN=<your_token>
```

---

## 방법 1: tt-inference-server 사용 (권장)

가장 쉬운 방법입니다. Docker 기반으로 vLLM 서버를 실행합니다.

### 설치

```bash
# tt-inference-server 클론
git clone https://github.com/tenstorrent/tt-inference-server.git
cd tt-inference-server

# P100용 Docker 이미지 사용
# (정확한 이미지 태그는 README 참조)
```

### 서버 실행

```bash
# Llama 3.1 8B 서빙
docker run --rm -it \
    --device /dev/tenstorrent \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    ghcr.io/tenstorrent/tt-inference-server:latest \
    --model meta-llama/Llama-3.1-8B-Instruct
```

---

## 방법 2: tt-metal에서 직접 실행

### 모델 테스트 (데모)

```bash
cd /home/user/tt-metal

# 환경 설정
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
export MESH_DEVICE=P100  # P100 싱글 칩 사용

# Batch-1 테스트
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"

# Batch-32 테스트
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-32"
```

### vLLM 서버로 실행

tt-metal은 Tenstorrent의 vLLM fork와 통합되어 있습니다.

```bash
# Tenstorrent vLLM fork 클론
git clone https://github.com/tenstorrent/vllm.git
cd vllm

# tt_metal 브랜치로 체크아웃
git checkout tt_metal

# 설치
pip install -e .
```

#### vLLM 서버 시작

```bash
# 환경 설정
export TT_METAL_HOME=/home/user/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH

# vLLM 서버 실행
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port 8000
```

---

## API 테스트

```bash
# Health check
curl http://localhost:8000/health

# Chat completion (OpenAI 호환 API)
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "user", "content": "안녕하세요! 자기소개 부탁드립니다."}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }'
```

---

## 인터뷰 도우미 아키텍처 적용

문서에서 본 인터뷰 도우미 시스템을 P100에 맞게 구성합니다.

### P100에서 가능한 구성

```
┌─────────────────────────────────────────────────────────────┐
│                 Tenstorrent P100 (28GB)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────────┐    │
│  │ faster-whisper      │    │  vLLM Server            │    │
│  │ (CPU 기반 STT)      │    │  Llama 3.1 8B-Instruct  │    │
│  │                     │    │  29.5 t/s/u             │    │
│  └──────────┬──────────┘    └────────────┬────────────┘    │
│             │                            │                  │
│             └────────────┬───────────────┘                  │
│                          ▼                                  │
│               ┌──────────────────┐                          │
│               │  Interview App   │                          │
│               │  (Python)        │                          │
│               └──────────────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 모델 매핑

| 원래 모델 (문서) | P100 대체 모델 | 비고 |
|------------------|----------------|------|
| Llama 3 8B (팔로업) | Llama 3.1 8B-Instruct | 동일 |
| LLaVA 13B (비언어) | 미지원 | 추후 지원 예정 |
| Llama 3 70B (리포트) | Llama 3.1 8B | 단일 칩 제한 |

### 권장 사항

1. **Phase 1 (실시간)**:
   - STT: `faster-whisper` (CPU에서 실행)
   - LLM: `Llama 3.1 8B-Instruct` (P100에서 실행)

2. **Phase 2 (배치 분석)**:
   - 현재 P100은 싱글 칩이라 70B 모델 불가
   - 8B 모델로 대체하거나 QuietBox/LoudBox 사용 권장

---

## 성능 최적화

### 배치 크기 조정

```bash
# 단일 사용자 (낮은 지연시간)
pytest models/tt_transformers/demo/simple_text_demo.py \
    -k "performance and batch-1"

# 다중 사용자 (높은 처리량)
pytest models/tt_transformers/demo/simple_text_demo.py \
    -k "performance and batch-32"
```

### 정밀도 설정

```bash
# 성능 우선 (bfp4/bfp8)
pytest models/tt_transformers/demo/simple_text_demo.py \
    -k "performance and batch-1"

# 정확도 우선 (bfp8/bf16)
pytest models/tt_transformers/demo/simple_text_demo.py \
    -k "accuracy and batch-1"
```

---

## 추가 지원 모델

P100에서 실험적으로 지원되는 추가 모델들:

| 모델 | 상태 |
|------|------|
| Llama 3.1 8B | ✅ 검증됨 |
| Whisper Large V3 | ✅ 지원 |
| 기타 8B 모델 | 🧪 실험적 |

---

## 트러블슈팅

### 1. 디바이스 인식 안됨
```bash
# 디바이스 확인
ls /dev/tenstorrent*

# KMD 상태 확인
dmesg | grep tenstorrent
```

### 2. 메모리 부족
```bash
# 최대 시퀀스 길이 줄이기
--max-model-len 2048
```

### 3. 느린 첫 응답
- 첫 실행 시 weight cache 생성에 시간이 걸림
- 이후 실행은 빨라짐

---

## 참고 자료

- [tt-metal GitHub](https://github.com/tenstorrent/tt-metal)
- [tt-inference-server](https://github.com/tenstorrent/tt-inference-server)
- [Tenstorrent vLLM Fork](https://github.com/tenstorrent/vllm)
- [Tenstorrent 공식 문서](https://docs.tenstorrent.com)
- [Blackhole 스펙](https://tenstorrent.com/hardware/blackhole)
