#!/bin/bash
# Tenstorrent P100용 LLM 서빙 스크립트

set -e

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 기본값 설정
MODEL="${HF_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

# tt-metal 경로
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(dirname "$SCRIPT_DIR")}"

echo -e "${GREEN}=========================================="
echo "Tenstorrent P100 LLM 서빙"
echo -e "==========================================${NC}"
echo ""
echo "모델: $MODEL"
echo "포트: $PORT"
echo "배치 크기: $BATCH_SIZE"
echo "최대 시퀀스 길이: $MAX_SEQ_LEN"
echo ""

# 디바이스 확인
echo -e "${YELLOW}[1/3] Tenstorrent 디바이스 확인 중...${NC}"
if [ ! -e /dev/tenstorrent ]; then
    echo -e "${RED}[ERROR] Tenstorrent 디바이스를 찾을 수 없습니다.${NC}"
    echo "KMD가 설치되어 있는지 확인하세요."
    exit 1
fi
echo -e "${GREEN}[OK] Tenstorrent 디바이스 발견${NC}"

# 환경 설정
echo -e "${YELLOW}[2/3] 환경 설정 중...${NC}"
export TT_METAL_HOME="$TT_METAL_HOME"
export PYTHONPATH="$TT_METAL_HOME:$PYTHONPATH"
export HF_MODEL="$MODEL"
export MESH_DEVICE="P100"

# HuggingFace 토큰 확인
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}[WARNING] HF_TOKEN이 설정되지 않았습니다.${NC}"
    echo "일부 모델은 인증이 필요할 수 있습니다."
    echo "export HF_TOKEN=<your_token> 으로 설정하세요."
fi

echo -e "${GREEN}[OK] 환경 설정 완료${NC}"

# 서버 실행
echo -e "${YELLOW}[3/3] vLLM 서버 시작 중...${NC}"
echo ""

# vLLM이 설치되어 있는지 확인
if python -c "import vllm" 2>/dev/null; then
    echo "vLLM 서버 모드로 실행..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --max-model-len "$MAX_SEQ_LEN" \
        --host 0.0.0.0 \
        --port "$PORT"
else
    echo "vLLM이 설치되지 않았습니다. 데모 모드로 실행..."
    echo ""
    echo "vLLM 서버를 사용하려면:"
    echo "  1. git clone https://github.com/tenstorrent/vllm.git"
    echo "  2. cd vllm && git checkout tt_metal"
    echo "  3. pip install -e ."
    echo ""
    echo "대신 데모를 실행합니다..."
    cd "$TT_METAL_HOME"
    pytest models/tt_transformers/demo/simple_text_demo.py \
        -k "performance and batch-1" \
        --batch_size "$BATCH_SIZE" \
        --max_seq_len "$MAX_SEQ_LEN" \
        -v
fi
