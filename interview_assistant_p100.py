#!/usr/bin/env python3
"""
P100 GPU용 인터뷰 도우미 간소화 버전

이 스크립트는 P100 16GB에서 실행 가능한 인터뷰 도우미입니다.
- STT: faster-whisper (CPU)
- LLM: vLLM (Llama 3 8B)

사용 전 vLLM 서버를 먼저 실행하세요:
    ./run_vllm_server.sh
"""

import os
import sys
import time
import queue
import threading
from typing import Optional

# vLLM 서버 설정
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

# 시스템 프롬프트
INTERVIEW_SYSTEM_PROMPT = """당신은 인터뷰 진행을 돕는 전문 어시스턴트입니다.
지원자의 답변을 듣고 면접관이 물어볼 수 있는 좋은 팔로업 질문을 제안합니다.

규칙:
1. 답변의 핵심 내용을 파악합니다
2. 깊이 있는 탐구가 가능한 팔로업 질문을 2-3개 제안합니다
3. 질문은 구체적이고 열린 형태여야 합니다
4. 한국어로 응답합니다"""


class InterviewAssistant:
    """P100에서 실행 가능한 간소화된 인터뷰 도우미"""

    def __init__(self):
        self.transcript_queue = queue.Queue()
        self.running = False

        # OpenAI 클라이언트 초기화 (vLLM 호환)
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=VLLM_BASE_URL,
                api_key="not-needed"  # vLLM은 API 키 불필요
            )
            print(f"[OK] vLLM 서버 연결: {VLLM_BASE_URL}")
        except Exception as e:
            print(f"[ERROR] OpenAI 클라이언트 초기화 실패: {e}")
            print("pip install openai 를 실행하세요.")
            sys.exit(1)

    def generate_followup(self, answer: str) -> str:
        """지원자 답변에 대한 팔로업 질문 생성"""
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

    def process_audio_realtime(self):
        """실시간 음성 처리 (faster-whisper 사용)"""
        try:
            from faster_whisper import WhisperModel
            from RealtimeSTT import AudioToTextRecorder
        except ImportError:
            print("[WARNING] 실시간 STT 패키지가 없습니다.")
            print("pip install faster-whisper RealtimeSTT pyaudio 를 실행하세요.")
            return

        print("[INFO] 실시간 음성 인식 시작...")
        print("말씀하시면 자동으로 텍스트로 변환됩니다.")
        print("Ctrl+C로 종료\n")

        def on_transcription(text: str):
            if text.strip():
                print(f"\n[지원자] {text}")
                self.transcript_queue.put(text)

        # CPU에서 faster-whisper 실행
        recorder = AudioToTextRecorder(
            model="tiny",  # CPU 사용시 작은 모델 권장
            language="ko",
            spinner=False,
            on_transcription_complete=on_transcription,
        )

        self.running = True
        try:
            while self.running:
                recorder.text()
        except KeyboardInterrupt:
            self.running = False

    def process_followup_loop(self):
        """팔로업 질문 생성 루프"""
        buffer = []
        last_process_time = time.time()

        while self.running:
            try:
                # 큐에서 텍스트 가져오기 (3초 타임아웃)
                text = self.transcript_queue.get(timeout=3.0)
                buffer.append(text)

                # 버퍼에 충분한 내용이 있거나 5초 지났으면 처리
                current_time = time.time()
                if len(buffer) >= 2 or (current_time - last_process_time > 5 and buffer):
                    combined_answer = " ".join(buffer)

                    print("\n[분석 중...]")
                    followup = self.generate_followup(combined_answer)
                    print(f"\n[팔로업 질문 제안]\n{followup}\n")
                    print("-" * 50)

                    buffer = []
                    last_process_time = current_time

            except queue.Empty:
                continue

    def run_interactive(self):
        """대화형 모드 실행 (마이크 없이 텍스트 입력)"""
        print("=" * 50)
        print("인터뷰 도우미 (대화형 모드)")
        print("=" * 50)
        print("지원자의 답변을 입력하면 팔로업 질문을 제안합니다.")
        print("'quit' 또는 'exit'로 종료\n")

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
                print("-" * 50 + "\n")

            except KeyboardInterrupt:
                print("\n종료합니다.")
                break

    def run_realtime(self):
        """실시간 모드 실행 (마이크 입력)"""
        print("=" * 50)
        print("인터뷰 도우미 (실시간 모드)")
        print("=" * 50)

        # 팔로업 생성 스레드 시작
        followup_thread = threading.Thread(target=self.process_followup_loop)
        followup_thread.daemon = True
        followup_thread.start()

        # 음성 인식 시작
        self.process_audio_realtime()


def check_vllm_server():
    """vLLM 서버 연결 확인"""
    import requests
    try:
        response = requests.get(f"{VLLM_BASE_URL.replace('/v1', '')}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    print("=" * 50)
    print("P100 GPU용 인터뷰 도우미")
    print("=" * 50)

    # vLLM 서버 확인
    print("\n[1/2] vLLM 서버 확인 중...")
    if not check_vllm_server():
        print("[WARNING] vLLM 서버에 연결할 수 없습니다.")
        print(f"서버 주소: {VLLM_BASE_URL}")
        print("\n먼저 vLLM 서버를 실행하세요:")
        print("  ./run_vllm_server.sh")
        print("\n또는 Ollama 사용:")
        print("  ollama serve")
        sys.exit(1)
    print("[OK] vLLM 서버 연결 성공")

    # 어시스턴트 초기화
    print("[2/2] 인터뷰 도우미 초기화 중...")
    assistant = InterviewAssistant()

    # 모드 선택
    print("\n실행 모드를 선택하세요:")
    print("  1. 대화형 (텍스트 입력)")
    print("  2. 실시간 (마이크 입력)")

    choice = input("\n선택 (1/2) [기본: 1]: ").strip() or "1"

    if choice == "2":
        assistant.run_realtime()
    else:
        assistant.run_interactive()


if __name__ == "__main__":
    main()
