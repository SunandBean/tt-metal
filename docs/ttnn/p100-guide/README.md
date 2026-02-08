# P100 모델 포팅 가이드

Tenstorrent P100 디바이스에서 Llama 3.1 8B, Whisper, VLM을 실행하기 위한 종합 분석 자료입니다.

## 문서 구성

| 번호 | 문서 | 내용 |
|------|------|------|
| 01 | [P100 하드웨어 제약 사항](01_P100_Hardware_Constraints.md) | P100 스펙, P150과의 차이, DRAM 코어 7개 문제 |
| 02 | [L1 버퍼 오버플로 분석](02_L1_Buffer_Overflow_Analysis.md) | L1 메모리 구조, CB 계산, 에러 패턴, 튜닝 레버 |
| 03 | [Llama 3.1 8B on P100](03_Llama31_8B_on_P100.md) | 현재 상태, 누락된 config, 구체적 수정 방안 |
| 04 | [Whisper on P100](04_Whisper_on_P100.md) | 기존 구현 현황, 필요 작업, Conv1D/Cross-Attention |
| 05 | [VLM on P100](05_VLM_on_P100.md) | 구현된 VLM 목록, 적합한 모델, 포팅 가이드 |
| 06 | [액션 플랜 요약](06_Action_Plan_Summary.md) | 수정 우선순위, 핵심 파일, 검증 체크리스트 |

## 핵심 요약

### P100 = Blackhole (DRAM 7코어, 이더넷 없음)

P100은 P150과 동일한 Blackhole 칩이지만:
- **DRAM 코어 7개** (P150은 8개) → matmul per_core_N 계산이 달라짐
- **이더넷 비활성** → 싱글 디바이스만 가능

### L1 오버플로 근본 원인

DRAM 코어가 7개라서 per_core_N이 P150보다 크게 계산됨 → CB 크기 증가 → L1 초과. 이미 QKV prefill에서 `per_core_M=7`로 패치되어 있지만, 여러 config 엔트리에 P100이 누락되어 있음.

### 수정 필요 범위

대부분의 연산과 모델 구현은 이미 존재함. **config 파일에 P100 엔트리를 추가**하는 것이 핵심 작업.

---

*작성일: 2026-02-08*
*기반 커밋: upstream tenstorrent/tt-metal main (2026-02-08 sync)*
