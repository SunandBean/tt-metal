# Tenstorrent Blackhole p100 멀티모델 파이프라인 가이드

## 목차
1. [개요](#1-개요)
2. [Blackhole p100 하드웨어 아키텍처](#2-blackhole-p100-하드웨어-아키텍처)
3. [p100에서 지원되는 모델](#3-p100에서-지원되는-모델)
4. [멀티모델 순차 파이프라인 구현](#4-멀티모델-순차-파이프라인-구현)
5. [메모리 관리 및 최적화](#5-메모리-관리-및-최적화)
6. [실행 예제](#6-실행-예제)
7. [트러블슈팅](#7-트러블슈팅)

---

## 1. 개요

이 문서는 Tenstorrent Blackhole p100 가속기에서 **VLM(Vision Language Model)**, **LLM(Large Language Model)**, **STT(Speech-to-Text)** 모델을 순차적으로 호출하는 멀티모델 파이프라인을 구성하는 방법을 설명합니다.

### 1.1 사용 사례

```
[오디오 입력] → [STT: Whisper] → [텍스트]
                                    ↓
[이미지 입력] → [Vision: ViT] → [비전 피처] → [LLM] → [최종 응답]
```

이러한 파이프라인은 다음과 같은 애플리케이션에 활용됩니다:
- 멀티모달 AI 어시스턴트
- 비디오/오디오 분석 시스템
- 자동 캡셔닝 및 설명 생성

---

## 2. Blackhole p100 하드웨어 아키텍처

### 2.1 칩 레벨 개요

Blackhole은 Tenstorrent의 최신 AI 가속기 아키텍처입니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Blackhole p100 SoC Layout                        │
│                         (17 x 12 Grid)                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   [E][E][E][E][E][E][E]        [E][E][E][E][E][E][E]               │
│                                                                     │
│   [D]                    [D]                    [D]                 │
│                                                                     │
│   [T][T][T][T][T][T][T] [R] [T][T][T][T][T][T][T]   ← Tensix Grid  │
│   [T][T][T][T][T][T][T] [R] [T][T][T][T][T][T][T]     (13 x 10)    │
│   [T][T][T][T][T][T][T] [R] [T][T][T][T][T][T][T]                   │
│   [T][T][T][T][T][T][T] [R] [T][T][T][T][T][T][T]                   │
│   [T][T][T][T][T][T][T] [R] [T][T][T][T][T][T][T]                   │
│   [T][T][T][T][T][T][T] [R] [T][T][T][T][T][T][T]                   │
│   [T][T][T][T][T][T][T] [R] [T][T][T][T][T][T][T]                   │
│   [T][T][T][T][T][T][T] [R] [T][T][T][T][T][T][T]                   │
│   [T][T][T][T][T][T][T] [R] [T][T][T][T][T][T][T]                   │
│   [T][T][T][T][T][T][T] [R] [T][T][T][T][T][T][T]                   │
│                                                                     │
│   [D]                    [D]                    [D]                 │
│                                                                     │
│   [P]        [A]         [D]         [D]        [P]                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

범례: T=Tensix Core, D=DRAM Controller, E=Ethernet, R=Router, P=PCIe, A=ARC
```

### 2.2 Wormhole vs Blackhole 비교

| 특성 | Wormhole N150 | Blackhole p100/p150 |
|------|---------------|---------------------|
| **Tensix 코어** | 8×10 (80개) | 14×10 (140개) |
| **사용 가능 컴퓨팅 코어** | 8×8 (64개) | 13×10 (130개) |
| **코어당 L1 SRAM** | 1,464 KB | 1,464 KB + 데이터 캐시 |
| **DRAM 뱅크** | 12개 @ 1GB | 8개 @ ~4GB |
| **총 DRAM** | ~12 GB | ~32 GB |
| **이더넷 코어** | 16개 (1×RISC-V) | 14개 (2×RISC-V) |
| **DRAM 읽기 정렬** | 32B | 64B |
| **멀티캐스트** | 직사각형만 | 직사각형, 스트라이드, L자형 |

### 2.3 Tensix 코어 아키텍처 (심층 분석)

각 Tensix 코어는 완전한 데이터 처리 유닛입니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                     TENSIX CORE 상세 구조                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    L1 SRAM (1.5 MB)                      │   │
│  │  ┌─────────────┬─────────────┬─────────────────────┐    │   │
│  │  │  Circular   │  Circular   │     Tile Storage    │    │   │
│  │  │  Buffer 0   │  Buffer 1   │    (32×32 tiles)    │    │   │
│  │  │  (입력 A)   │  (입력 B)   │                     │    │   │
│  │  ├─────────────┼─────────────┼─────────────────────┤    │   │
│  │  │  Circular   │   Kernel    │     Stack/Heap      │    │   │
│  │  │  Buffer 16  │   Code      │     (RISC-V용)      │    │   │
│  │  │  (출력)     │             │                     │    │   │
│  │  └─────────────┴─────────────┴─────────────────────┘    │   │
│  │                                                          │   │
│  │  [NEW] L1 Data Cache: 4 × 16B = 64B (Write-Through)     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│           ┌──────────────────┼──────────────────┐              │
│           ▼                  ▼                  ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  RISC-V #0  │    │  RISC-V #1  │    │  RISC-V #2  │        │
│  │ (Data Move) │    │ (Data Move) │    │  (Unpack)   │        │
│  │   Reader    │    │   Writer    │    │             │        │
│  │   NoC 0     │    │   NoC 1     │    │             │        │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│         │                  │                  │                │
│         ▼                  ▼                  ▼                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │  RISC-V #3  │    │  RISC-V #4  │    │   UNPACKER      │    │
│  │   (Math)    │    │   (Pack)    │    │  L1→SrcA/SrcB   │    │
│  └──────┬──────┘    └──────┬──────┘    └────────┬────────┘    │
│         │                  │                    │              │
│         ▼                  ▼                    ▼              │
│  ┌──────────────────────────────────────────────────────┐     │
│  │                  COMPUTE ENGINES                      │     │
│  │  ┌────────────────────┐  ┌────────────────────────┐  │     │
│  │  │   FPU (Matrix)     │  │    SFPU (Vector)       │  │     │
│  │  │                    │  │                        │  │     │
│  │  │  • 32×32 타일 연산  │  │  • 32-wide 벡터 연산   │  │     │
│  │  │  • GEMM 최적화     │  │  • 32-bit 부동소수점   │  │     │
│  │  │  • TF32 정밀도     │  │  • 초월함수 (sin,exp) │  │     │
│  │  │                    │  │                        │  │     │
│  │  │  SrcA ─┐           │  │  Dst ◄────────────────┤  │     │
│  │  │  SrcB ─┼──► Dst    │  │       │               │  │     │
│  │  └────────┴───────────┘  └───────┴───────────────┘  │     │
│  └──────────────────────────────────────────────────────┘     │
│                              │                                 │
│                              ▼                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │                      PACKER                           │     │
│  │                   Dst → L1 SRAM                       │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │              NETWORK-ON-CHIP (NoC) 인터페이스          │     │
│  │                                                        │     │
│  │  NoC 0 (Read Path)  ◄───────────────────►  DRAM/Other │     │
│  │  NoC 1 (Write Path) ◄───────────────────►  Cores      │     │
│  │                                                        │     │
│  │  • 2D Torus 토폴로지 (Wraparound)                     │     │
│  │  • 반대 방향으로 동작 (전이중 통신)                    │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 레지스터 파일 구조

컴퓨트 엔진 내부의 레지스터 구조:

```
┌─────────────────────────────────────────────────────────────┐
│                    레지스터 데이터플로우                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   L1 SRAM                                                   │
│      │                                                      │
│      ▼ (Unpacker)                                          │
│   ┌──────┐    ┌──────┐                                     │
│   │ SrcA │    │ SrcB │  ← 소스 레지스터 (FPU 입력)          │
│   │      │    │      │    각 32×32 타일 저장               │
│   └──┬───┘    └──┬───┘                                     │
│      │           │                                          │
│      └─────┬─────┘                                          │
│            ▼                                                │
│   ┌─────────────────┐                                      │
│   │   FPU (Matrix)  │  행렬 연산                           │
│   │   A × B = C     │                                      │
│   └────────┬────────┘                                      │
│            │                                                │
│            ▼                                                │
│   ┌─────────────────┐                                      │
│   │    Dst 레지스터  │  ← 목적지 레지스터                    │
│   │                 │    16-bit 모드: 8/16 타일            │
│   │  (FPU 출력 &    │    32-bit 모드: 4/8 타일             │
│   │   SFPU 입/출력)  │                                      │
│   └────────┬────────┘                                      │
│            │                                                │
│            ▼                                                │
│   ┌─────────────────┐                                      │
│   │  SFPU (Vector)  │  벡터 연산 (활성화 함수 등)           │
│   │  sin, cos, exp  │                                      │
│   │  relu, sigmoid  │                                      │
│   └────────┬────────┘                                      │
│            │                                                │
│            ▼                                                │
│   ┌─────────────────┐                                      │
│   │   LReg (내부)   │  ← SFPU 내부 상태 레지스터            │
│   │   32×32-bit     │                                      │
│   └────────┬────────┘                                      │
│            │                                                │
│            ▼ (Packer)                                      │
│      L1 SRAM                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.5 메모리 계층 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                     BLACKHOLE 메모리 계층                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   레지스터 파일                           │   │
│  │                                                           │   │
│  │   • SrcA/SrcB: FPU 소스 (즉시 접근, <1 사이클)            │   │
│  │   • Dst: 8-16 타일 (더블 버퍼링 시 절반)                  │   │
│  │   • LReg: SFPU 내부 상태                                  │   │
│  │                                                           │   │
│  │   레이턴시: ~1 사이클                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   L1 SRAM (1.5 MB/코어)                   │   │
│  │                                                           │   │
│  │   • 총 용량: 130 코어 × 1.5MB = ~195 MB (온칩)           │   │
│  │   • 스크래치패드 (명시적 관리, 캐시 아님)                 │   │
│  │   • 순환 버퍼로 커널 간 데이터 교환                       │   │
│  │   • [신규] 64B 데이터 캐시 (Write-Through)               │   │
│  │                                                           │   │
│  │   레이턴시: ~10-20 사이클                                 │   │
│  │   대역폭: 읽기 16B, 쓰기 16B (NoC 정렬)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                        NoC (Network-on-Chip)                    │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   DRAM (~32 GB 총용량)                    │   │
│  │                                                           │   │
│  │   • 8개 DRAM 컨트롤러, 각 ~4 GB                          │   │
│  │   • Interleaved 모드: 자동 분산                          │   │
│  │   • Sharded 모드: 토폴로지 기반 배치                     │   │
│  │                                                           │   │
│  │   레이턴시: ~100-200 사이클                              │   │
│  │   대역폭: 읽기 64B 정렬 (Blackhole 개선점)               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                         PCIe Gen4                               │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    호스트 메모리 (CPU)                     │   │
│  │                                                           │   │
│  │   레이턴시: ~1000+ 사이클                                 │   │
│  │   대역폭: 읽기 64B 정렬                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.6 데이터 흐름 (3-커널 모델)

```
┌─────────────────────────────────────────────────────────────────┐
│                   TENSIX 데이터 흐름 파이프라인                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐        ┌──────────────────────┐        ┌──────┐ │
│   │   DRAM   │ ─NoC0─►│     Reader Kernel     │───────►│ CB0  │ │
│   │          │        │   (RISC-V #0)         │        │      │ │
│   └──────────┘        │   • noc_async_read    │        │ CB1  │ │
│                       │   • 타일 단위 전송     │        │      │ │
│                       └──────────────────────┘        └──┬───┘ │
│                                                          │     │
│   ┌──────────────────────────────────────────────────────┘     │
│   │  순환버퍼: cb_push_back() / cb_wait_front()                │
│   ▼                                                             │
│   ┌──────────────────────────────────────────────────────┐     │
│   │                  Compute Kernel                       │     │
│   │            (RISC-V #2, #3, #4 협력 실행)              │     │
│   │                                                       │     │
│   │   Unpack Core (#2):                                   │     │
│   │     • cb_wait_front(cb_in0, 1)                       │     │
│   │     • L1 → SrcA/SrcB 레지스터 이동                   │     │
│   │                                                       │     │
│   │   Math Core (#3):                                     │     │
│   │     • tile_regs_acquire()                            │     │
│   │     • add_tiles() / matmul_tiles()                   │     │
│   │     • tile_regs_commit()                             │     │
│   │                                                       │     │
│   │   Pack Core (#4):                                     │     │
│   │     • tile_regs_wait()                               │     │
│   │     • pack_tile(dst_reg, cb_out)                     │     │
│   │     • tile_regs_release()                            │     │
│   │                                                       │     │
│   └──────────────────────────────────────────────────────┘     │
│                              │                                  │
│   ┌──────────────────────────┘                                 │
│   │  순환버퍼: cb_push_back() / cb_wait_front()                │
│   ▼                                                             │
│   ┌──────┐        ┌──────────────────────┐        ┌──────────┐ │
│   │CB16  │───────►│    Writer Kernel     │─NoC1──►│   DRAM   │ │
│   │      │        │   (RISC-V #1)        │        │          │ │
│   └──────┘        │   • noc_async_write  │        └──────────┘ │
│                   └──────────────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. p100에서 지원되는 모델

### 3.1 현재 Blackhole 공식 지원 모델

| 모델 유형 | 모델명 | 지원 상태 | 설명 |
|----------|--------|----------|------|
| **STT** | Whisper (distil-large-v3) | ✅ 완전 지원 | p100, p150 명시 지원 |
| **Vision** | ViT (Base/16-224) | ✅ 완전 지원 | ~3700 FPS (p150) |
| **Vision** | ViT High-Res | ✅ 완전 지원 | Batch=1, 고해상도 |
| **Vision** | ResNet50 | ✅ 지원 | 이미지 분류 |
| **NLP** | Sentence BERT | ✅ 지원 | 텍스트 임베딩 |
| **Vision** | VGG-UNet | ✅ 지원 | 세그멘테이션 |
| **Vision** | Stable Diffusion | ✅ 지원 | 이미지 생성 |
| **VLM** | Qwen2.5-VL | 🔄 개발중 | 현재 Wormhole/T3K |
| **LLM** | Llama/DeepSeek | 🔄 개발중 | 현재 Galaxy/T3K |

### 3.2 p100 멀티모델 파이프라인 구성

현재 p100에서 구현 가능한 파이프라인:

```
┌─────────────────────────────────────────────────────────────────┐
│                p100 멀티모델 파이프라인 옵션                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  옵션 1: STT → Embedding 파이프라인                             │
│  ┌──────────┐    ┌─────────────────┐    ┌────────────────┐     │
│  │ 오디오   │───►│ Whisper (STT)   │───►│ Sentence BERT  │     │
│  │ 입력     │    │ p100 지원 ✅    │    │ 텍스트 임베딩   │     │
│  └──────────┘    └─────────────────┘    └────────────────┘     │
│                                                                 │
│  옵션 2: Vision → Embedding 파이프라인                          │
│  ┌──────────┐    ┌─────────────────┐    ┌────────────────┐     │
│  │ 이미지   │───►│   ViT/ResNet    │───►│ Sentence BERT  │     │
│  │ 입력     │    │ p100 지원 ✅    │    │ 멀티모달 융합   │     │
│  └──────────┘    └─────────────────┘    └────────────────┘     │
│                                                                 │
│  옵션 3: 멀티모달 인코더 파이프라인 (권장)                       │
│  ┌──────────┐    ┌─────────────────┐                           │
│  │ 오디오   │───►│ Whisper (STT)   │──┐                        │
│  └──────────┘    └─────────────────┘  │   ┌────────────────┐   │
│                                        ├──►│ Feature Fusion │   │
│  ┌──────────┐    ┌─────────────────┐  │   │ (호스트 또는   │   │
│  │ 이미지   │───►│      ViT        │──┘   │  외부 LLM)     │   │
│  └──────────┘    └─────────────────┘      └────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 멀티모델 순차 파이프라인 구현

### 4.1 하드웨어 수준 실행 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│           멀티모델 순차 실행 시 하드웨어 상태 변화                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  시간 ──────────────────────────────────────────────────────►   │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │   Model A (STT) │  │  Model B (ViT)  │  │ Model C (BERT) │  │
│  └────────┬────────┘  └────────┬────────┘  └───────┬────────┘  │
│           │                    │                    │           │
│  ┌────────▼────────────────────────────────────────────────┐   │
│  │  Phase 1: Model A 로드 및 실행                           │   │
│  │                                                          │   │
│  │  DRAM 상태:                                              │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │ [Model A Weights] [Input Buffer] [Output Buffer] │   │   │
│  │  │      ~500MB           ~10MB          ~10MB       │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │                                                          │   │
│  │  L1 SRAM 상태 (각 활성 코어):                            │   │
│  │  ┌─────────────────────────────────────────┐            │   │
│  │  │ [Kernel Code] [CB0] [CB1] [CB16] [...]  │            │   │
│  │  │    ~100KB     ~64KB ~64KB ~64KB         │            │   │
│  │  └─────────────────────────────────────────┘            │   │
│  │                                                          │   │
│  │  NoC 트래픽: DRAM→L1 (읽기), L1→DRAM (쓰기)             │   │
│  └──────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼  모델 A 결과 저장, 리소스 해제                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Phase 2: Model B 로드 및 실행                           │   │
│  │                                                          │   │
│  │  DRAM 상태:                                              │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │ [Model B Weights] [A Output→B Input] [B Output]  │   │   │
│  │  │      ~350MB            ~10MB            ~10MB     │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │                                                          │   │
│  │  주의: Model A 가중치 메모리 해제됨                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼  모델 B 결과 저장, 리소스 해제                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Phase 3: Model C 로드 및 실행                           │   │
│  │                                                          │   │
│  │  DRAM 상태:                                              │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │ [Model C Weights] [A+B Features] [Final Output]  │   │   │
│  │  │      ~110MB          ~20MB           ~10MB       │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 구현 코드: Python API (TTNN)

```python
#!/usr/bin/env python3
"""
Blackhole p100 멀티모델 순차 파이프라인 예제
STT (Whisper) → Vision (ViT) → Text Embedding (Sentence BERT)
"""

import torch
import ttnn
from pathlib import Path


class MultiModelPipeline:
    """
    p100에서 여러 모델을 순차적으로 실행하는 파이프라인

    하드웨어 고려사항:
    - 각 모델은 DRAM에서 가중치 로드 → L1 SRAM에서 연산 → DRAM에 결과 저장
    - 모델 간 전환 시 이전 모델의 가중치를 명시적으로 해제해야 함
    - p100의 ~32GB DRAM으로 대부분의 단일 모델 가중치 수용 가능
    """

    def __init__(self, device_id: int = 0):
        """
        디바이스 초기화

        하드웨어 수준 동작:
        1. PCIe를 통해 디바이스와 연결
        2. DRAM 컨트롤러 초기화 (8개 뱅크)
        3. 130개 Tensix 코어 초기화
        4. Command Queue 설정 (Fast Dispatch)
        """
        # 단일 디바이스를 1x1 메시로 취급
        self.device = ttnn.open_device(device_id=device_id)

        # 디바이스 정보 출력
        print(f"Device: {self.device}")
        print(f"Compute Grid: {self.device.compute_with_storage_grid_size()}")

        # 모델 캐시 딕셔너리
        self.models = {}

    def __del__(self):
        """디바이스 리소스 해제"""
        if hasattr(self, 'device'):
            ttnn.close_device(self.device)

    def load_whisper(self, model_name: str = "distil-whisper/distil-large-v3"):
        """
        Whisper STT 모델 로드

        DRAM 레이아웃:
        ┌────────────────────────────────────────────────────┐
        │ Encoder Weights │ Decoder Weights │ KV Cache      │
        │    ~300MB       │    ~200MB       │  동적 할당    │
        └────────────────────────────────────────────────────┘

        L1 SRAM 사용 (코어당):
        - Attention 연산: ~500KB
        - FFN 연산: ~300KB
        - 순환 버퍼: ~200KB
        """
        from models.demos.whisper.tt import ttnn_optimized_functional_whisper

        # 모델 가중치를 DRAM에 로드
        # Interleaved 모드: 8개 DRAM 뱅크에 자동 분산
        model = ttnn_optimized_functional_whisper.load_model(
            self.device,
            model_name=model_name
        )

        self.models['whisper'] = model
        return model

    def load_vit(self, model_name: str = "google/vit-base-patch16-224"):
        """
        ViT 모델 로드

        DRAM 레이아웃:
        ┌────────────────────────────────────────────────────┐
        │ Patch Embed │ Transformer Blocks │ Classification │
        │   ~10MB     │     ~300MB         │     ~3MB       │
        └────────────────────────────────────────────────────┘

        연산 특성:
        - 입력: 224×224 이미지 → 196개 패치 (14×14)
        - 각 패치: 768차원 임베딩
        - 12개 Transformer 레이어
        """
        from models.demos.blackhole.vit.tt import ttnn_optimized_sharded_vit_bh

        model = ttnn_optimized_sharded_vit_bh.load_model(
            self.device,
            model_name=model_name
        )

        self.models['vit'] = model
        return model

    def load_sentence_bert(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Sentence BERT 모델 로드

        특성:
        - 6개 Transformer 레이어
        - 384차원 임베딩 출력
        - 텍스트 시퀀스 → 고정 크기 벡터
        """
        from models.demos.blackhole.sentence_bert.tt import sentence_bert

        model = sentence_bert.load_model(
            self.device,
            model_name=model_name
        )

        self.models['sentence_bert'] = model
        return model

    def unload_model(self, model_name: str):
        """
        모델 언로드 및 DRAM 메모리 해제

        하드웨어 동작:
        1. DRAM의 가중치 버퍼 해제
        2. L1 SRAM의 캐시된 데이터 무효화
        3. 할당된 Tensix 코어 해제
        """
        if model_name in self.models:
            model = self.models.pop(model_name)
            # TTNN의 명시적 메모리 해제
            ttnn.deallocate(model)

            # 동기화 대기 (모든 작업 완료 보장)
            ttnn.synchronize_device(self.device)

            print(f"Model '{model_name}' unloaded, DRAM freed")

    def run_stt(self, audio_input: torch.Tensor) -> str:
        """
        Speech-to-Text 실행

        데이터 흐름:
        1. 오디오 → Mel Spectrogram (호스트 전처리)
        2. Mel → DRAM 전송 (PCIe)
        3. Encoder 실행 (Tensix 코어)
        4. Decoder 실행 (자동회귀)
        5. 토큰 → 텍스트 (호스트 후처리)

        NoC 트래픽 패턴:
        - Encoder: DRAM→L1 (가중치), L1→L1 (어텐션)
        - Decoder: 각 토큰마다 전체 모델 순회
        """
        whisper = self.models.get('whisper')
        if whisper is None:
            raise RuntimeError("Whisper model not loaded")

        # 호스트에서 전처리 (Mel Spectrogram)
        mel_features = self._preprocess_audio(audio_input)

        # 텐서를 디바이스로 전송
        # PCIe를 통해 호스트 → DRAM
        mel_tt = ttnn.from_torch(
            mel_features,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT  # 32×32 타일 포맷
        )

        # 모델 실행
        output = whisper(mel_tt)

        # 결과를 호스트로 전송
        output_torch = ttnn.to_torch(output)

        # 토큰 디코딩
        text = self._decode_tokens(output_torch)

        return text

    def run_vision(self, image_input: torch.Tensor) -> torch.Tensor:
        """
        Vision Encoder 실행

        ViT 연산 분해 (하드웨어 관점):

        1. Patch Embedding:
           - 16×16 Conv2D → 196개 패치
           - FPU: 행렬 곱셈

        2. Position Embedding:
           - 196 + 1 (CLS) 토큰에 위치 정보 추가
           - SFPU: 벡터 덧셈

        3. Transformer Block (×12):
           - Multi-Head Attention:
             * Q, K, V 투영: FPU matmul
             * Attention Score: FPU matmul (Q @ K^T)
             * Softmax: SFPU exp, div
             * Value 집계: FPU matmul
           - FFN:
             * Linear1: FPU matmul
             * GELU: SFPU 활성화
             * Linear2: FPU matmul

        코어 활용:
        - 130개 코어에 배치/토큰 분산
        - 각 코어: 일부 어텐션 헤드 담당
        """
        vit = self.models.get('vit')
        if vit is None:
            raise RuntimeError("ViT model not loaded")

        # 이미지 전처리 (호스트)
        # 224×224 정규화, 패치화
        processed = self._preprocess_image(image_input)

        # 디바이스로 전송
        img_tt = ttnn.from_torch(
            processed,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT
        )

        # ViT 실행
        features = vit(img_tt)

        # CLS 토큰 (첫 번째 토큰)을 이미지 표현으로 사용
        cls_token = features[:, 0, :]

        return ttnn.to_torch(cls_token)

    def run_text_embedding(self, text: str) -> torch.Tensor:
        """
        텍스트 임베딩 생성

        Sentence BERT 연산:
        1. 토큰화 (호스트)
        2. BERT Forward:
           - 임베딩 레이어
           - 6개 Transformer 블록
        3. Mean Pooling (모든 토큰 평균)
        """
        bert = self.models.get('sentence_bert')
        if bert is None:
            raise RuntimeError("Sentence BERT model not loaded")

        # 토큰화
        tokens = self._tokenize_text(text)

        # 디바이스 전송
        tokens_tt = ttnn.from_torch(
            tokens,
            device=self.device,
            dtype=ttnn.uint32
        )

        # BERT 실행
        embeddings = bert(tokens_tt)

        # Mean pooling
        pooled = ttnn.mean(embeddings, dim=1)

        return ttnn.to_torch(pooled)

    def run_pipeline(
        self,
        audio_input: torch.Tensor = None,
        image_input: torch.Tensor = None
    ) -> dict:
        """
        전체 멀티모델 파이프라인 실행

        실행 순서 (메모리 효율 최적화):
        1. STT 로드 → 실행 → 언로드
        2. ViT 로드 → 실행 → 언로드
        3. BERT 로드 → 실행 → 언로드

        이 순차 로드/언로드 방식의 이유:
        - p100 DRAM (~32GB)에 모든 모델 동시 로드 가능하지만
        - 순차 방식이 L1 캐시 효율성 향상
        - 모델 간 NoC 충돌 방지
        """
        results = {}

        # Phase 1: Speech-to-Text
        if audio_input is not None:
            print("Phase 1: Running Whisper STT...")
            self.load_whisper()

            text = self.run_stt(audio_input)
            results['stt_output'] = text

            self.unload_model('whisper')
            print(f"  STT Result: {text[:100]}...")

        # Phase 2: Vision Encoding
        if image_input is not None:
            print("Phase 2: Running ViT...")
            self.load_vit()

            vision_features = self.run_vision(image_input)
            results['vision_features'] = vision_features

            self.unload_model('vit')
            print(f"  Vision Feature Shape: {vision_features.shape}")

        # Phase 3: Text/Feature Embedding
        if 'stt_output' in results:
            print("Phase 3: Running Sentence BERT...")
            self.load_sentence_bert()

            text_embedding = self.run_text_embedding(results['stt_output'])
            results['text_embedding'] = text_embedding

            self.unload_model('sentence_bert')
            print(f"  Text Embedding Shape: {text_embedding.shape}")

        # Feature Fusion (호스트에서 수행)
        if 'vision_features' in results and 'text_embedding' in results:
            # 간단한 연결 융합
            fused = torch.cat([
                results['vision_features'],
                results['text_embedding']
            ], dim=-1)
            results['fused_features'] = fused
            print(f"  Fused Feature Shape: {fused.shape}")

        return results

    # --- 유틸리티 메서드 (실제 구현은 모델별로 다름) ---

    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """오디오 → Mel Spectrogram 변환"""
        # 실제 구현은 Whisper preprocessor 사용
        import torchaudio
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=80
        )(audio)
        return mel

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """이미지 전처리 (224×224 정규화)"""
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(image)

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """텍스트 토큰화"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        tokens = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        return tokens['input_ids']

    def _decode_tokens(self, tokens: torch.Tensor) -> str:
        """토큰 → 텍스트 디코딩"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "distil-whisper/distil-large-v3"
        )
        return tokenizer.decode(tokens[0], skip_special_tokens=True)


# 사용 예제
if __name__ == "__main__":
    import torchaudio
    from PIL import Image
    import torchvision.transforms as transforms

    # 파이프라인 초기화
    pipeline = MultiModelPipeline(device_id=0)

    # 테스트 입력 준비
    # 오디오: 16kHz 샘플
    audio, sr = torchaudio.load("test_audio.wav")
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)

    # 이미지: PIL → Tensor
    image = Image.open("test_image.jpg")
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)

    # 파이프라인 실행
    results = pipeline.run_pipeline(
        audio_input=audio,
        image_input=image_tensor
    )

    print("\n=== Pipeline Results ===")
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: Tensor{value.shape}")
        else:
            print(f"{key}: {value[:100]}..." if len(str(value)) > 100 else f"{key}: {value}")
```

### 4.3 저수준 Metalium API (C++)

더 세밀한 하드웨어 제어가 필요한 경우:

```cpp
/**
 * Metalium 저수준 API를 사용한 멀티모델 파이프라인
 *
 * 이 예제는 모델 전환 시 리소스 관리를 명시적으로 보여줍니다.
 */

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/bfloat16.hpp"

using namespace tt;
using namespace tt::tt_metal;

class MetaliumMultiModelPipeline {
public:
    MetaliumMultiModelPipeline(int device_id = 0) {
        // 디바이스를 1x1 메시로 열기
        device_ = MeshDevice::create_unit_mesh(device_id);
        cq_ = &device_->mesh_command_queue(0);

        // 컴퓨트 그리드 크기 확인
        // Blackhole: 13x10 = 130 코어
        grid_size_ = device_->compute_with_storage_grid_size();

        std::cout << "Initialized device with grid: "
                  << grid_size_.x << "x" << grid_size_.y << std::endl;
    }

    ~MetaliumMultiModelPipeline() {
        device_.reset();  // 디바이스 닫기
    }

    /**
     * DRAM 버퍼 할당
     *
     * 하드웨어 동작:
     * - Interleaved 모드: 8개 DRAM 뱅크에 라운드로빈 분산
     * - 페이지 크기: 타일 크기 (32x32 x sizeof(bfloat16) = 2KB)
     */
    std::shared_ptr<MeshBuffer> allocate_buffer(size_t size_bytes) {
        DeviceLocalBufferConfig dram_config{
            .page_size = TILE_SIZE_BYTES,
            .buffer_type = BufferType::DRAM
        };

        ReplicatedBufferConfig buffer_config{
            .size = size_bytes
        };

        return MeshBuffer::create(buffer_config, dram_config, device_.get());
    }

    /**
     * 순환 버퍼 설정
     *
     * 순환 버퍼는 커널 간 데이터 동기화의 핵심:
     * - Reader → Compute: cb_in0, cb_in1
     * - Compute → Writer: cb_out (c_16)
     *
     * 버퍼 크기 결정 요인:
     * - 더블 버퍼링 필요 시 최소 2 타일
     * - L1 용량 제한 (1.5MB/코어)
     * - 파이프라인 깊이
     */
    void setup_circular_buffers(
        Program& program,
        CoreCoord core,
        uint32_t tiles_per_cb = 2
    ) {
        constexpr uint32_t tile_size = 32 * 32 * sizeof(bfloat16);  // 2KB

        // 입력 버퍼 0
        CreateCircularBuffer(
            program, core,
            CircularBufferConfig(
                tiles_per_cb * tile_size,
                {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}}
            ).set_page_size(tt::CBIndex::c_0, tile_size)
        );

        // 입력 버퍼 1
        CreateCircularBuffer(
            program, core,
            CircularBufferConfig(
                tiles_per_cb * tile_size,
                {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}}
            ).set_page_size(tt::CBIndex::c_1, tile_size)
        );

        // 출력 버퍼
        CreateCircularBuffer(
            program, core,
            CircularBufferConfig(
                tiles_per_cb * tile_size,
                {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}}
            ).set_page_size(tt::CBIndex::c_16, tile_size)
        );
    }

    /**
     * 프로그램 실행
     *
     * Fast Dispatch 모드:
     * 1. 커맨드 큐에 작업 추가 (비동기)
     * 2. 디바이스가 독립적으로 실행
     * 3. blocking=true면 완료까지 대기
     */
    void execute_program(Program& program, bool blocking = true) {
        MeshWorkload workload;
        workload.add_program(
            MeshCoordinateRange(device_->shape()),
            std::move(program)
        );
        EnqueueMeshWorkload(*cq_, workload, blocking);
    }

    /**
     * 모델 전환 시 동기화
     *
     * 중요: 다음 모델 로드 전 현재 작업 완료 보장
     * - 모든 NoC 트랜잭션 완료
     * - DRAM 쓰기 플러시
     * - L1 캐시 무효화 (Blackhole)
     */
    void synchronize() {
        // 커맨드 큐 작업 완료 대기
        Finish(*cq_);

        // Blackhole의 L1 데이터 캐시 무효화
        // (환경변수 TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS 사용 시)
        // device_->invalidate_l1_cache();
    }

private:
    std::unique_ptr<MeshDevice> device_;
    CommandQueue* cq_;
    CoreCoord grid_size_;

    static constexpr uint32_t TILE_SIZE_BYTES = 32 * 32 * sizeof(bfloat16);
};
```

---

## 5. 메모리 관리 및 최적화

### 5.1 DRAM 메모리 레이아웃 전략

```
┌─────────────────────────────────────────────────────────────────┐
│                   p100 DRAM 메모리 관리 전략                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  전략 1: 순차 로드/언로드 (메모리 효율 최대화)                    │
│  ═══════════════════════════════════════════════                │
│                                                                 │
│  시간 →                                                         │
│  ┌────────────┐                                                │
│  │ Model A    │ 실행 완료 후 해제                               │
│  └────────────┘                                                │
│               ┌────────────┐                                   │
│               │ Model B    │ 실행 완료 후 해제                  │
│               └────────────┘                                   │
│                            ┌────────────┐                      │
│                            │ Model C    │                      │
│                            └────────────┘                      │
│                                                                 │
│  장점: 메모리 사용량 최소화                                      │
│  단점: 모델 로드 오버헤드                                        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  전략 2: 동시 로드 (레이턴시 최소화)                              │
│  ═══════════════════════════════════════════════                │
│                                                                 │
│  DRAM 레이아웃:                                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ Bank 0-1 │ Bank 2-3 │ Bank 4-5 │ Bank 6-7 │ I/O   │       │
│  │ Model A  │ Model B  │ Model C  │ Reserved │Buffer │       │
│  │ ~8GB     │ ~8GB     │ ~8GB     │ ~4GB     │~4GB   │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
│  장점: 모델 전환 시 재로드 불필요                                │
│  단점: 대형 모델에는 부적합                                      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  권장: 하이브리드 전략                                           │
│  ════════════════════════════════════════════                   │
│                                                                 │
│  • 작은 모델 (< 4GB): DRAM에 상주                               │
│  • 큰 모델: 필요 시 로드/언로드                                  │
│  • 중간 결과: Interleaved 버퍼에 저장                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 L1 SRAM 최적화

```
┌─────────────────────────────────────────────────────────────────┐
│                  코어당 L1 SRAM 예산 (1.5MB)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  일반적인 할당 예시 (Transformer 레이어):                        │
│                                                                 │
│  ┌───────────────────────────────────────────────┐             │
│  │              L1 SRAM (1,464 KB)               │             │
│  ├───────────────────────────────────────────────┤             │
│  │                                               │             │
│  │  Kernel Code & Stack     │    ~100 KB        │             │
│  │  ─────────────────────────────────────        │             │
│  │                                               │             │
│  │  Circular Buffer 0 (Q)   │    ~128 KB        │             │
│  │  Circular Buffer 1 (K)   │    ~128 KB        │             │
│  │  Circular Buffer 2 (V)   │    ~128 KB        │             │
│  │  ─────────────────────────────────────        │             │
│  │                                               │             │
│  │  Circular Buffer 16 (Out)│    ~128 KB        │             │
│  │  ─────────────────────────────────────        │             │
│  │                                               │             │
│  │  Intermediate Tiles      │    ~500 KB        │             │
│  │  (Attention Scores,      │                   │             │
│  │   FFN intermediates)     │                   │             │
│  │  ─────────────────────────────────────        │             │
│  │                                               │             │
│  │  Reserved/Headroom       │    ~352 KB        │             │
│  │                                               │             │
│  └───────────────────────────────────────────────┘             │
│                                                                 │
│  최적화 팁:                                                     │
│  • 더블 버퍼링: 데이터 로드와 연산 오버랩                        │
│  • 타일 재사용: 브로드캐스트로 동일 데이터 공유                  │
│  • 융합 연산: 중간 결과를 L1에 유지                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Blackhole L1 데이터 캐시 활용

```cpp
/**
 * Blackhole의 새로운 L1 데이터 캐시 (64B) 활용
 *
 * Write-Through 정책:
 * - 쓰기는 즉시 L1 SRAM에 반영
 * - 다른 코어에서 읽기 전 무효화 필요
 */

// 커널 내에서 캐시 활성화
void kernel_main() {
    // 캐시 활성화
    set_l1_data_cache<true>();

    // ... 연산 수행 ...

    // 다른 코어와 데이터 공유 전 무효화
    invalidate_l1_cache();

    // 커널 종료 전 캐시 비활성화
    set_l1_data_cache<false>();
}

// 또는 환경변수로 글로벌 활성화
// export TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS=BR,NC,TR,ER
```

---

## 6. 실행 예제

### 6.1 환경 설정

```bash
# tt-metal 환경 설정
source /path/to/tt-metal/build/python_env/bin/activate

# Blackhole 디바이스 확인
tt-smi

# 환경 변수 설정
export ARCH_NAME=blackhole
export TT_METAL_HOME=/path/to/tt-metal

# (선택) L1 데이터 캐시 활성화
export TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS=BR,NC,TR,ER
```

### 6.2 개별 모델 테스트

```bash
# 1. Whisper STT 테스트
pytest --disable-warnings \
    models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation_dataset

# 2. ViT 테스트 (Blackhole 전용)
pytest --disable-warnings \
    models/demos/blackhole/vit/tests/test_ttnn_optimized_sharded_vit_bh.py

# 3. Sentence BERT 테스트
pytest --disable-warnings \
    models/demos/blackhole/sentence_bert/tests/test_sentence_bert.py
```

### 6.3 성능 프로파일링

```bash
# Tracy Profiler를 사용한 성능 분석
./tools/tracy/profile_this.py -n multi_model -c \
    "python your_pipeline_script.py"

# 디바이스 연산 분석
pytest models/demos/blackhole/vit/tests/test_vit_device_perf.py::test_vit_perf_device
```

---

## 7. 트러블슈팅

### 7.1 일반적인 문제와 해결책

| 문제 | 증상 | 해결책 |
|------|------|--------|
| 메모리 부족 | `OutOfMemory` 에러 | 순차 로드/언로드 사용, 배치 크기 감소 |
| NoC 타임아웃 | 행 또는 ND 실패 | `noc_async_*_barrier()` 추가, Watcher 활성화 |
| L1 캐시 불일치 | 잘못된 결과 | `invalidate_l1_cache()` 호출 |
| 정렬 오류 | DRAM 읽기 실패 | 64B 정렬 확인 (Blackhole) |

### 7.2 디버깅 도구

```bash
# Watcher 활성화 (커널 오류 감지)
export TT_METAL_WATCHER=1

# Slow Dispatch 모드 (디버깅용)
export TT_METAL_SLOW_DISPATCH_MODE=1

# DPRINT 활성화 (커널 내 printf)
export TT_METAL_DPRINT_CORES="0,0"
export TT_METAL_DPRINT_FILE="dprint_log.txt"
```

### 7.3 성능 최적화 체크리스트

- [ ] Fast Dispatch 모드 사용 중인가?
- [ ] 순환 버퍼 크기가 적절한가? (더블 버퍼링)
- [ ] DRAM 접근이 64B 정렬되어 있는가?
- [ ] NoC 트래픽이 균형 잡혀 있는가?
- [ ] L1 데이터 캐시를 효과적으로 사용하고 있는가?
- [ ] 불필요한 호스트-디바이스 전송이 없는가?

---

## 참고 자료

- [TT-Metalium Programming Guide](../../METALIUM_GUIDE.md)
- [Blackhole Bring-Up Guide](../../tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md)
- [Whisper Demo](../models/demos/whisper/README.md)
- [ViT Demo (Blackhole)](../models/demos/blackhole/vit/README.md)
- [Memory Allocator Tech Report](../../tech_reports/memory/allocator.md)
- [Tensor Layouts Tech Report](../../tech_reports/tensor_layouts/tensor_layouts.md)

---

*문서 버전: 1.0*
*작성일: 2026-01-29*
*대상 하드웨어: Tenstorrent Blackhole p100/p150*
