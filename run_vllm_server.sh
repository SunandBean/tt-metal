#!/bin/bash
# P100 GPU용 vLLM 서버 실행 스크립트

set -e

# 기본값 설정
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.85}"

# 가상환경 활성화
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/vllm-pascal-env"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "가상환경을 찾을 수 없습니다. 먼저 setup_p100_vllm.sh를 실행하세요."
    exit 1
fi

# P100 최적화 환경변수
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_ATTENTION_BACKEND=XFORMERS

echo "=========================================="
echo "vLLM 서버 시작 (P100 최적화)"
echo "=========================================="
echo "모델: $MODEL"
echo "포트: $PORT"
echo "최대 컨텍스트: $MAX_MODEL_LEN"
echo "GPU 메모리 사용률: $GPU_UTIL"
echo "=========================================="

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype float16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enforce-eager \
    --disable-custom-all-reduce \
    --host 0.0.0.0 \
    --port "$PORT"
