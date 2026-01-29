# Tenstorrent 아키텍처별 커널 개발 가이드

## 목차
1. [개요](#1-개요)
2. [코드 구조](#2-코드-구조)
3. [개발 추상화 레벨](#3-개발-추상화-레벨)
4. [아키텍처 분기 패턴](#4-아키텍처-분기-패턴)
5. [Wormhole vs Blackhole 핵심 차이점](#5-wormhole-vs-blackhole-핵심-차이점)
6. [커널 개발 시 고려사항](#6-커널-개발-시-고려사항)
7. [개발 워크플로우](#7-개발-워크플로우)
8. [실전 예제](#8-실전-예제)
9. [빌드 및 테스트](#9-빌드-및-테스트)

---

## 1. 개요

Tenstorrent는 여러 세대의 AI 가속기를 출시하고 있으며, 각 아키텍처는 하드웨어 특성이 다릅니다. 이 문서는 **Wormhole**과 **Blackhole** 아키텍처를 모두 지원하는 커널을 개발하는 방법을 설명합니다.

### 1.1 왜 아키텍처별 코드가 필요한가?

GPU(CUDA)와 달리 Tenstorrent의 TT-Metalium은 **저수준 하드웨어 제어**를 제공합니다:

```
GPU 방식:     코드 → CUDA Runtime → [자동 최적화] → 하드웨어
Tenstorrent:  코드 → Metalium    → [수동 최적화] → 하드웨어
                                        ↑
                          아키텍처별 커널 코드 필요
```

이로 인해 최대 성능을 얻을 수 있지만, 아키텍처 변경 시 커널 수정이 필요합니다.

### 1.2 지원 아키텍처

| 아키텍처 | 제품 | 출시 | 상태 |
|----------|------|------|------|
| Grayskull | e75, e150 | 2022 | 레거시 |
| **Wormhole** | n150, n300, Galaxy | 2023 | 프로덕션 |
| **Blackhole** | p100, p150 | 2024 | 개발 중 |

---

## 2. 코드 구조

### 2.1 디렉토리 레이아웃

```
tt_metal/hw/
├── ckernels/                      # 컴퓨트 커널 (SFPU/FPU 연산)
│   ├── blackhole/
│   │   └── metal/llk_api/
│   │       ├── llk_math_*.h       # FPU 연산
│   │       ├── llk_pack_api.h     # Packer
│   │       └── llk_sfpu/          # SFPU 연산
│   │           ├── ckernel_sfpu_trigonometry.h
│   │           ├── ckernel_sfpu_exp.h
│   │           ├── ckernel_sfpu_relu.h
│   │           └── ...
│   │
│   └── wormhole_b0/
│       └── metal/llk_api/
│           └── llk_sfpu/          # Wormhole SFPU 구현
│               └── ...            # (동일한 파일 구조)
│
├── inc/
│   ├── api/
│   │   ├── dataflow/
│   │   │   └── dataflow_api.h     # 데이터 이동 API
│   │   └── compute/
│   │       └── compute_api.h      # 컴퓨트 API
│   │
│   └── internal/
│       └── tt-2xx/
│           └── risc_common.h      # RISC-V 공통 코드
│
├── firmware/                      # 펌웨어
│   └── src/
│       └── tt-1xx/
│           ├── erisc.cc           # Ethernet RISC
│           ├── ncrisck.cc         # NCRISC
│           └── trisc.cc           # Tensix RISC
│
└── toolchain/
    └── main.ld                    # 링커 스크립트
```

### 2.2 빌드 시 파일 선택

```cmake
# 빌드 시스템이 ARCH_NAME에 따라 올바른 디렉토리 선택
if(ARCH_NAME STREQUAL "blackhole")
    include_directories(tt_metal/hw/ckernels/blackhole/metal)
elseif(ARCH_NAME STREQUAL "wormhole_b0")
    include_directories(tt_metal/hw/ckernels/wormhole_b0/metal)
endif()
```

---

## 3. 개발 추상화 레벨

### 3.1 세 가지 레벨 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                    커널 개발 추상화 레벨                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Level 3: TTNN (Python)                                         │
│  ═══════════════════════                                        │
│  • 가장 쉬움, 대부분의 경우 사용                                  │
│  • 아키텍처 자동 감지                                            │
│  • 모델 개발자용                                                 │
│                                                                 │
│  예시:                                                          │
│    output = ttnn.sin(input_tensor)                              │
│    output = ttnn.matmul(a, b)                                   │
│                                                                 │
│  ───────────────────────────────────────────────────────────── │
│                                                                 │
│  Level 2: Compute API (C++)                                     │
│  ══════════════════════════                                     │
│  • 중간 레벨                                                     │
│  • 컴파일 시 아키텍처별 구현 자동 선택                           │
│  • 새 연산자 구현 시 사용                                        │
│                                                                 │
│  예시:                                                          │
│    sin_tile(dst_reg);                                           │
│    add_tiles(cb_in0, cb_in1, 0, 0, dst_reg);                   │
│    matmul_tiles(cb_a, cb_b, 0, 0, dst_reg, false);             │
│                                                                 │
│  ───────────────────────────────────────────────────────────── │
│                                                                 │
│  Level 1: LLK API (C++)                                         │
│  ══════════════════════                                         │
│  • 최저 레벨, 직접 하드웨어 제어                                 │
│  • 명시적 아키텍처 분기 필요                                     │
│  • NoC, 메모리, 레지스터 직접 조작                               │
│                                                                 │
│  예시:                                                          │
│    #ifdef ARCH_BLACKHOLE                                        │
│        // Blackhole 전용 코드                                   │
│        noc_async_read_barrier();                                │
│        invalidate_l1_cache();                                   │
│    #else                                                        │
│        // Wormhole 코드                                         │
│    #endif                                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 레벨 선택 가이드

| 작업 | 권장 레벨 | 이유 |
|------|----------|------|
| 모델 추론/학습 | TTNN | 아키텍처 추상화, 빠른 개발 |
| 새 연산자 추가 | Compute API | 이식성 유지, 적절한 제어 |
| NoC 최적화 | LLK | 세밀한 타이밍 제어 필요 |
| 메모리 레이아웃 최적화 | LLK | 하드웨어별 정렬 요구사항 |
| 커스텀 데이터 타입 | LLK | 하드웨어 레지스터 직접 접근 |

---

## 4. 아키텍처 분기 패턴

### 4.1 패턴 A: 전처리기 조건부 컴파일

가장 일반적인 패턴으로, 저수준 API에서 사용됩니다.

```cpp
// dataflow_api.h 예시

template <bool posted = false>
FORCE_INLINE void noc_inline_dw_write_with_state(...) {

#ifdef ARCH_BLACKHOLE
    // Issue #28758: Blackhole에서는 fabric router hang 방지를 위해
    // 항상 카운터 업데이트 필요
    constexpr bool update_counter_in_callee = true;
#else
    // Wormhole에서는 선택적
    constexpr bool update_counter_in_callee = update_counter;
#endif

    noc_fast_write_dw_inline_with_state<
        noc_mode,
        update_addr_lo,
        update_addr_hi,
        update_val,
        posted,
        update_counter_in_callee,  // ← 아키텍처에 따라 다른 값
        dst_type>(noc, cmd_buf, val, addr);
}
```

### 4.2 패턴 B: 별도 파일 분리

컴퓨트 커널에서 주로 사용됩니다. 동일한 함수 시그니처, 다른 구현.

```cpp
// blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h
namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sine() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0] * FRAC_1_PI;
        vInt whole_v = float_to_int16(v, 0);  // Blackhole: float_to_int16 사용 가능
        v -= int32_to_float(whole_v, 0);
        v = sfpu_sinpi<APPROXIMATION_MODE>(v);

        v_if(whole_v & 1) { v = -v; }
        v_endif;
        dst_reg[0] = v;
        dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

```cpp
// wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h
// (동일한 네임스페이스, 동일한 함수명, 다른 구현 가능)
```

### 4.3 패턴 C: 템플릿 파라미터 차이

SFPU 벡터 너비 차이로 인한 반복 횟수 변경:

```cpp
// 아키텍처별 ITERATIONS 정의
// Wormhole:  SFPU 64-wide → ITERATIONS = 4 (64/16 = 4)
// Blackhole: SFPU 32-wide → ITERATIONS = 8 (32/4 = 8)

// 상위 레벨에서 호출 시:
#ifdef ARCH_BLACKHOLE
    constexpr int ITERATIONS = 8;
#else
    constexpr int ITERATIONS = 4;
#endif

calculate_sine<APPROX_MODE, ITERATIONS>();
```

### 4.4 패턴 D: 런타임 분기 (호스트 코드)

```cpp
// 호스트 C++ 코드에서
#include "tt_metal/impl/device/device.hpp"

void configure_for_arch(Device* device) {
    auto arch = device->arch();

    if (arch == tt::ARCH::BLACKHOLE) {
        // Blackhole 설정
        alignment = 64;
        core_grid = {13, 10};
    } else if (arch == tt::ARCH::WORMHOLE_B0) {
        // Wormhole 설정
        alignment = 32;
        core_grid = {8, 8};
    }
}
```

---

## 5. Wormhole vs Blackhole 핵심 차이점

### 5.1 하드웨어 스펙 비교

```
┌────────────────────────┬─────────────────┬─────────────────┐
│         항목           │  Wormhole N150  │    Blackhole    │
├────────────────────────┼─────────────────┼─────────────────┤
│ Tensix 코어 (총)       │     8×10 (80)   │   14×10 (140)   │
│ 컴퓨팅 코어            │     8×8 (64)    │   13×10 (130)   │
│ L1 SRAM/코어           │    1,464 KB     │ 1,464 KB + 캐시 │
│ DRAM 뱅크              │    12 × 1GB     │    8 × ~4GB     │
│ 총 DRAM                │     ~12 GB      │     ~32 GB      │
├────────────────────────┼─────────────────┼─────────────────┤
│ DRAM 읽기 정렬         │      32B        │      64B ❗     │
│ DRAM 쓰기 정렬         │      16B        │      16B        │
│ PCIe 읽기 정렬         │      32B        │      64B ❗     │
│ L1 읽기 정렬           │      16B        │      16B        │
├────────────────────────┼─────────────────┼─────────────────┤
│ 이더넷 RISC-V          │       1개       │       2개       │
│ 이더넷 L1              │     256 KB      │     512 KB      │
│ DRAM RISC-V            │       N/A       │       1개       │
├────────────────────────┼─────────────────┼─────────────────┤
│ L1 데이터 캐시         │       없음      │   64B (신규)    │
│ 멀티캐스트 패턴        │   직사각형만    │ 직사각형+L자형  │
│ SFPU 벡터 너비         │     64-wide     │     32-wide     │
└────────────────────────┴─────────────────┴─────────────────┘
```

### 5.2 주요 코드 변경 사항

#### (1) DRAM 정렬

```cpp
// Wormhole: 32B 정렬 OK
noc_async_read(dram_addr, l1_addr, 32);

// Blackhole: 64B 정렬 필수!
noc_async_read(dram_addr, l1_addr, 64);

// 크로스 아키텍처 코드:
#ifdef ARCH_BLACKHOLE
    constexpr uint32_t DRAM_ALIGNMENT = 64;
#else
    constexpr uint32_t DRAM_ALIGNMENT = 32;
#endif
```

#### (2) NoC Barrier 요구사항

```cpp
// Blackhole의 RISC-V가 더 빨라서 명시적 barrier 필수!
// Wormhole에서는 타이밍 여유로 생략 가능했던 것이 Blackhole에서 필수

noc_async_read(src_addr, dst_addr, size);

// ⚠️ Blackhole: 데이터 사용 전 barrier 필수!
noc_async_read_barrier();

// 이제 안전하게 데이터 사용 가능
process_data(dst_addr);
```

**Blackhole Bring-Up Guide에서 인용:**
> "Previous architectures did not need this because of higher RISC to L1 latency compared to NoC latency."

#### (3) L1 데이터 캐시

```cpp
// Blackhole에만 있는 64B L1 캐시
// Write-through 정책: 쓰기는 즉시 L1에 반영

// 캐시 활성화 (커널 내)
set_l1_data_cache<true>();

// ... 연산 수행 ...

// 다른 코어와 데이터 공유 전 무효화 필수!
invalidate_l1_cache();

// 커널 종료 전 캐시 비활성화
set_l1_data_cache<false>();
```

또는 환경변수로 글로벌 활성화:
```bash
export TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS=BR,NC,TR,ER
```

#### (4) 코어 그리드 크기

```cpp
// 호스트 코드에서 코어 범위 설정
#ifdef ARCH_BLACKHOLE
    CoreRange cores = {{0, 0}, {12, 9}};  // 13×10 그리드
#else
    CoreRange cores = {{0, 0}, {7, 7}};   // 8×8 그리드
#endif

// 또는 런타임에 쿼리
auto grid = device->compute_with_storage_grid_size();
CoreRange cores = {{0, 0}, {grid.x - 1, grid.y - 1}};
```

---

## 6. 커널 개발 시 고려사항

### 6.1 데이터 이동 커널

```cpp
// reader_kernel.cpp

#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_l1_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    // 아키텍처별 정렬
    #ifdef ARCH_BLACKHOLE
        constexpr uint32_t ALIGNMENT = 64;
    #else
        constexpr uint32_t ALIGNMENT = 32;
    #endif

    constexpr uint32_t TILE_SIZE = 32 * 32 * 2;  // bfloat16

    // 타일 사이즈가 정렬 요구사항 만족하는지 확인
    static_assert(TILE_SIZE % 64 == 0, "Tile size must be 64B aligned for Blackhole");

    for (uint32_t i = 0; i < num_tiles; i++) {
        // 순환 버퍼 공간 예약
        cb_reserve_back(cb_id, 1);

        uint32_t l1_write_addr = get_write_ptr(cb_id);
        uint64_t src_noc_addr = get_noc_addr(src_dram_addr + i * TILE_SIZE);

        // 비동기 읽기
        noc_async_read(src_noc_addr, l1_write_addr, TILE_SIZE);

        // ⚠️ Barrier: Blackhole에서 필수, Wormhole에서도 권장
        noc_async_read_barrier();

        // 순환 버퍼에 푸시
        cb_push_back(cb_id, 1);
    }
}
```

### 6.2 컴퓨트 커널

```cpp
// compute_kernel.cpp

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // 초기화 (아키텍처별 자동 처리)
    unary_op_init_common(cb_in, cb_out);

    for (uint32_t i = 0; i < num_tiles; i++) {
        // 입력 대기
        cb_wait_front(cb_in, 1);

        // 출력 공간 예약
        cb_reserve_back(cb_out, 1);

        // 레지스터 획득
        tile_regs_acquire();

        // 타일 복사 및 연산 (Compute API가 아키텍처 처리)
        copy_tile(cb_in, 0, 0);

        // SFPU 연산 예: sin
        // ※ 내부적으로 ITERATIONS가 아키텍처별로 다름
        //    Wormhole: 4, Blackhole: 8
        sin_tile(0);

        // 레지스터 커밋
        tile_regs_commit();

        // 입력 팝
        cb_pop_front(cb_in, 1);

        // 패킹 대기 및 실행
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        // 출력 푸시
        cb_push_back(cb_out, 1);
    }
}
}  // namespace NAMESPACE
```

### 6.3 Writer 커널

```cpp
// writer_kernel.cpp

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t TILE_SIZE = 32 * 32 * 2;
    constexpr auto cb_out = tt::CBIndex::c_16;

    for (uint32_t i = 0; i < num_tiles; i++) {
        // 데이터 대기
        cb_wait_front(cb_out, 1);

        uint32_t l1_read_addr = get_read_ptr(cb_out);
        uint64_t dst_noc_addr = get_noc_addr(dst_dram_addr + i * TILE_SIZE);

        // 비동기 쓰기
        noc_async_write(l1_read_addr, dst_noc_addr, TILE_SIZE);

        // Barrier
        noc_async_write_barrier();

        // 순환 버퍼에서 팝
        cb_pop_front(cb_out, 1);
    }
}
```

---

## 7. 개발 워크플로우

### 7.1 권장 개발 순서

```
┌─────────────────────────────────────────────────────────────────┐
│              아키텍처별 커널 개발 워크플로우                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Wormhole에서 먼저 개발 & 테스트                         │
│  ────────────────────────────────────────                       │
│  • 더 많은 참조 코드 존재                                        │
│  • 안정적인 CI/CD 환경                                           │
│  • 문서화가 잘 되어 있음                                         │
│                                                                 │
│                         ↓                                       │
│                                                                 │
│  Step 2: Blackhole 포팅 체크리스트                               │
│  ────────────────────────────────                               │
│  □ DRAM 정렬 64B로 변경                                         │
│  □ NoC barrier 추가 (noc_async_*_barrier)                       │
│  □ 코어 그리드 크기 조정 (8×8 → 13×10)                          │
│  □ L1 캐시 무효화 추가 (필요시)                                  │
│  □ ITERATIONS 파라미터 확인 (4 → 8)                             │
│                                                                 │
│                         ↓                                       │
│                                                                 │
│  Step 3: 테스트                                                  │
│  ───────────                                                    │
│  • Watcher 활성화하여 오류 감지                                  │
│  • 단위 테스트 실행                                              │
│  • 정확도 검증 (golden 비교)                                    │
│  • 성능 프로파일링                                               │
│                                                                 │
│                         ↓                                       │
│                                                                 │
│  Step 4: 조건부 컴파일로 통합                                    │
│  ────────────────────────────                                   │
│  • #ifdef ARCH_BLACKHOLE / #else / #endif                       │
│  • 또는 별도 파일 분리                                           │
│  • 공통 인터페이스 유지                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 디버깅 도구

```bash
# 1. Watcher 활성화 (커널 오류 감지)
export TT_METAL_WATCHER=1

# 2. Slow Dispatch 모드 (디버깅용)
export TT_METAL_SLOW_DISPATCH_MODE=1

# 3. DPRINT 활성화 (커널 내 printf)
export TT_METAL_DPRINT_CORES="0,0"
export TT_METAL_DPRINT_FILE="dprint_log.txt"

# 4. L1 캐시 비활성화 (캐시 문제 분리)
# (환경변수 설정 안 하면 기본 비활성화)

# 5. 하드웨어 캐시 무효화 타임아웃 활성화
export TT_METAL_ENABLE_HW_CACHE_INVALIDATION=1
```

---

## 8. 실전 예제

### 8.1 완전한 크로스 아키텍처 커널 세트

**호스트 프로그램 (C++)**

```cpp
// host_program.cpp

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/bfloat16.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main() {
    // 디바이스 열기
    auto device = MeshDevice::create_unit_mesh(0);
    auto& cq = device->mesh_command_queue(0);

    // 아키텍처 감지
    auto arch = device->arch();
    bool is_blackhole = (arch == tt::ARCH::BLACKHOLE);

    // 아키텍처별 설정
    uint32_t alignment = is_blackhole ? 64 : 32;
    CoreCoord grid_size = device->compute_with_storage_grid_size();

    std::cout << "Architecture: " << (is_blackhole ? "Blackhole" : "Wormhole") << std::endl;
    std::cout << "Grid size: " << grid_size.x << "x" << grid_size.y << std::endl;

    // 프로그램 생성
    Program program = CreateProgram();

    // 버퍼 설정
    constexpr uint32_t num_tiles = 64;
    constexpr uint32_t tile_size = 32 * 32 * sizeof(bfloat16);

    DeviceLocalBufferConfig dram_config{
        .page_size = tile_size,
        .buffer_type = BufferType::DRAM
    };

    ReplicatedBufferConfig buffer_config{
        .size = num_tiles * tile_size
    };

    auto input_buffer = MeshBuffer::create(buffer_config, dram_config, device.get());
    auto output_buffer = MeshBuffer::create(buffer_config, dram_config, device.get());

    // 단일 코어에서 실행
    CoreCoord core{0, 0};

    // 순환 버퍼 설정
    constexpr uint32_t cb_tiles = 2;  // 더블 버퍼링

    CreateCircularBuffer(program, core,
        CircularBufferConfig(cb_tiles * tile_size,
            {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, tile_size));

    CreateCircularBuffer(program, core,
        CircularBufferConfig(cb_tiles * tile_size,
            {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_16, tile_size));

    // 커널 생성
    auto reader = CreateKernel(
        program,
        "kernels/reader_kernel.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        });

    auto compute = CreateKernel(
        program,
        "kernels/compute_kernel.cpp",
        core,
        ComputeConfig{});

    auto writer = CreateKernel(
        program,
        "kernels/writer_kernel.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default
        });

    // 런타임 인자 설정
    SetRuntimeArgs(program, reader, core, {
        input_buffer->address(),
        num_tiles
    });

    SetRuntimeArgs(program, compute, core, {
        num_tiles
    });

    SetRuntimeArgs(program, writer, core, {
        output_buffer->address(),
        num_tiles
    });

    // 입력 데이터 준비 및 전송
    std::vector<bfloat16> input_data(num_tiles * 32 * 32);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = bfloat16(static_cast<float>(i) * 0.01f);
    }
    EnqueueWriteMeshBuffer(cq, input_buffer, input_data, false);

    // 프로그램 실행
    MeshWorkload workload;
    workload.add_program(MeshCoordinateRange(device->shape()), std::move(program));
    EnqueueMeshWorkload(cq, workload, true);

    // 결과 읽기
    std::vector<bfloat16> output_data(num_tiles * 32 * 32);
    EnqueueReadMeshBuffer(cq, output_buffer, output_data, true);

    // 검증
    std::cout << "First 5 outputs: ";
    for (int i = 0; i < 5; i++) {
        std::cout << output_data[i].to_float() << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

---

## 9. 빌드 및 테스트

### 9.1 환경 설정

```bash
# 아키텍처 지정
export ARCH_NAME=blackhole  # 또는 wormhole_b0

# TT-Metal 홈
export TT_METAL_HOME=/path/to/tt-metal

# Python 환경 활성화
source $TT_METAL_HOME/build/python_env/bin/activate
```

### 9.2 빌드

```bash
# CMake 설정
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DTT_METAL_BUILD_TESTS=ON

# 빌드
cmake --build build -j$(nproc)
```

### 9.3 테스트 실행

```bash
# 특정 아키텍처 테스트
pytest tests/tt_metal/tt_metal/test_kernels.py -v

# Watcher와 함께 실행 (디버깅)
TT_METAL_WATCHER=1 pytest tests/... -v

# 특정 테스트만
pytest tests/tt_metal/... -k "test_my_kernel" -v
```

### 9.4 성능 프로파일링

```bash
# Tracy 프로파일러 사용
./tools/tracy/profile_this.py -n my_kernel_profile -c \
    "python my_kernel_test.py"

# 디바이스 성능 테스트
pytest tests/tt_metal/.../test_device_perf.py -v
```

---

## 부록: 아키텍처 전처리 매크로

| 매크로 | 설명 |
|--------|------|
| `ARCH_BLACKHOLE` | Blackhole 타겟 빌드 |
| `ARCH_WORMHOLE` | Wormhole 타겟 빌드 |
| `ARCH_WORMHOLE_B0` | Wormhole B0 리비전 |
| `COMPILE_FOR_NCRISC` | NCRISC 컴파일 |
| `COMPILE_FOR_BRISC` | BRISC 컴파일 |
| `COMPILE_FOR_TRISC` | TRISC 컴파일 |
| `COMPILE_FOR_ERISC` | ERISC 컴파일 |

---

## 참고 자료

- [TT-Metalium Programming Guide](../../METALIUM_GUIDE.md)
- [Blackhole Bring-Up Guide](../../tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md)
- [Tensor Layouts](../../tech_reports/tensor_layouts/tensor_layouts.md)
- [Data Formats](../../tech_reports/data_formats/data_formats.md)
- [Low Level Kernels Documentation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/sfpu/llk.html)

---

*문서 버전: 1.0*
*작성일: 2026-01-29*
*대상: Wormhole (n150, n300) 및 Blackhole (p100, p150)*
