#!/usr/bin/env python3
"""
Tenstorrent P100용 인터뷰 도우미

이 스크립트는 Tenstorrent P100 (Blackhole)에서 실행되는
인터뷰 도우미 시스템입니다.

사전 요구사항:
    1. Tenstorrent P100 하드웨어 및 드라이버 설치
    2. tt-metal 설치
    3. vLLM 서버 실행 (선택사항)

사용법:
    # vLLM 서버가 실행 중인 경우
    python p100_interview_assistant.py

    # 직접 tt-metal 사용 (서버 없이)
    python p100_interview_assistant.py --direct
"""

import argparse
import os
import sys
from typing import Optional

# vLLM 서버 설정
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# 시스템 프롬프트
INTERVIEW_SYSTEM_PROMPT = """당신은 인터뷰 진행을 돕는 전문 어시스턴트입니다.
지원자의 답변을 듣고 면접관이 물어볼 수 있는 좋은 팔로업 질문을 제안합니다.

규칙:
1. 답변의 핵심 내용을 파악합니다
2. 깊이 있는 탐구가 가능한 팔로업 질문을 2-3개 제안합니다
3. 질문은 구체적이고 열린 형태여야 합니다
4. 한국어로 응답합니다"""


class InterviewAssistant:
    """Tenstorrent P100에서 실행되는 인터뷰 도우미"""

    def __init__(self, use_vllm_server: bool = True):
        self.use_vllm_server = use_vllm_server

        if use_vllm_server:
            self._init_vllm_client()
        else:
            self._init_direct_model()

    def _init_vllm_client(self):
        """vLLM 서버 클라이언트 초기화"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=VLLM_BASE_URL,
                api_key="not-needed"
            )
            print(f"[OK] vLLM 서버 연결: {VLLM_BASE_URL}")
        except ImportError:
            print("[ERROR] openai 패키지가 필요합니다: pip install openai")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] vLLM 서버 연결 실패: {e}")
            sys.exit(1)

    def _init_direct_model(self):
        """tt-metal 직접 사용 초기화"""
        try:
            # tt-metal 환경 확인
            tt_metal_home = os.getenv("TT_METAL_HOME")
            if not tt_metal_home:
                print("[WARNING] TT_METAL_HOME이 설정되지 않았습니다.")
                print("export TT_METAL_HOME=/path/to/tt-metal")

            print("[INFO] 직접 모델 로드는 데모 모드에서 지원됩니다.")
            print("vLLM 서버 사용을 권장합니다.")

        except Exception as e:
            print(f"[ERROR] 모델 초기화 실패: {e}")
            sys.exit(1)

    def generate_followup(self, answer: str) -> str:
        """지원자 답변에 대한 팔로업 질문 생성"""
        if self.use_vllm_server:
            return self._generate_via_vllm(answer)
        else:
            return self._generate_direct(answer)

    def _generate_via_vllm(self, answer: str) -> str:
        """vLLM 서버를 통한 생성"""
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": INTERVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": f"지원자 답변:\n{answer}\n\n이 답변에 대한 팔로업 질문을 제안해주세요."}
                ],
                max_tokens=256,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[오류] 팔로업 질문 생성 실패: {e}"

    def _generate_direct(self, answer: str) -> str:
        """tt-metal 직접 사용 생성 (미구현)"""
        return "[INFO] 직접 모드는 추후 지원 예정입니다. vLLM 서버를 사용해주세요."

    def run_interactive(self):
        """대화형 모드 실행"""
        print("=" * 60)
        print("Tenstorrent P100 인터뷰 도우미")
        print("=" * 60)
        print(f"모델: {MODEL_NAME}")
        print("지원자의 답변을 입력하면 팔로업 질문을 제안합니다.")
        print("'quit' 또는 'exit'로 종료")
        print("=" * 60)
        print()

        while True:
            try:
                answer = input("[지원자 답변] >>> ")

                if answer.lower() in ['quit', 'exit', 'q']:
                    print("종료합니다.")
                    break

                if not answer.strip():
                    continue

                print("\n[분석 중...]")
                followup = self.generate_followup(answer)
                print(f"\n[팔로업 질문 제안]\n{followup}")
                print("-" * 60 + "\n")

            except KeyboardInterrupt:
                print("\n종료합니다.")
                break


def check_vllm_server() -> bool:
    """vLLM 서버 연결 확인"""
    try:
        import requests
        response = requests.get(
            f"{VLLM_BASE_URL.replace('/v1', '')}/health",
            timeout=5
        )
        return response.status_code == 200
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Tenstorrent P100용 인터뷰 도우미"
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="vLLM 서버 없이 직접 tt-metal 사용"
    )
    parser.add_argument(
        "--server-url",
        default=VLLM_BASE_URL,
        help=f"vLLM 서버 URL (기본값: {VLLM_BASE_URL})"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Tenstorrent P100 인터뷰 도우미")
    print("=" * 60)
    print()

    if args.direct:
        print("[INFO] 직접 모드로 실행...")
        assistant = InterviewAssistant(use_vllm_server=False)
    else:
        print("[1/2] vLLM 서버 확인 중...")
        if not check_vllm_server():
            print(f"[WARNING] vLLM 서버에 연결할 수 없습니다: {args.server_url}")
            print()
            print("vLLM 서버를 먼저 실행하세요:")
            print("  ./scripts/p100_serve_llm.sh")
            print()
            print("또는 --direct 옵션으로 직접 모드 사용:")
            print("  python p100_interview_assistant.py --direct")
            print()
            sys.exit(1)
        print("[OK] vLLM 서버 연결 성공")

        print("[2/2] 인터뷰 도우미 초기화 중...")
        assistant = InterviewAssistant(use_vllm_server=True)

    print()
    assistant.run_interactive()


if __name__ == "__main__":
    main()
