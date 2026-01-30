#!/bin/bash
# P100 GPU용 vLLM 설치 스크립트

set -e

echo "=========================================="
echo "P100 GPU용 vLLM 설치 시작"
echo "=========================================="

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# GPU 확인
echo -e "\n${YELLOW}[1/5] GPU 확인 중...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${RED}nvidia-smi를 찾을 수 없습니다. CUDA가 설치되어 있는지 확인하세요.${NC}"
    exit 1
fi

# Python 버전 확인
echo -e "\n${YELLOW}[2/5] Python 버전 확인 중...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python 버전: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.9" ]]; then
    echo -e "${RED}Python 3.9 이상이 필요합니다.${NC}"
    exit 1
fi

# 가상환경 생성
echo -e "\n${YELLOW}[3/5] 가상환경 생성 중...${NC}"
VENV_DIR="vllm-pascal-env"

if [ -d "$VENV_DIR" ]; then
    echo "기존 가상환경 발견: $VENV_DIR"
    read -p "삭제하고 새로 만들까요? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "기존 가상환경 사용"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# Pascal 패치 패키지 설치
echo -e "\n${YELLOW}[4/5] vLLM Pascal 버전 설치 중...${NC}"
export PIP_EXTRA_INDEX_URL="https://sasha0552.github.io/pascal-pkgs-ci/"

# PyTorch 설치 (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# vLLM Pascal 버전 설치
pip install vllm-pascal==0.10.0

# 추가 의존성
echo -e "\n${YELLOW}[5/5] 추가 패키지 설치 중...${NC}"
pip install faster-whisper openai requests

echo -e "\n${GREEN}=========================================="
echo "설치 완료!"
echo "==========================================${NC}"

echo -e "\n사용법:"
echo "  1. 가상환경 활성화:"
echo "     source $VENV_DIR/bin/activate"
echo ""
echo "  2. 서버 실행:"
echo "     python -m vllm.entrypoints.openai.api_server \\"
echo "         --model meta-llama/Meta-Llama-3-8B-Instruct \\"
echo "         --dtype float16 \\"
echo "         --max-model-len 4096 \\"
echo "         --enforce-eager \\"
echo "         --host 0.0.0.0 --port 8000"
echo ""
echo "  또는 간단히:"
echo "     ./run_vllm_server.sh"
