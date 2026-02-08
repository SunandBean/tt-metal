# L1 버퍼 오버플로 분석 및 해결 가이드

## L1 메모리 구조

### 코어당 L1 메모리 레이아웃

```
┌─────────────────────────────────────────┐  0x00000
│  Firmware (BRISC, NCRISC, TRISC0-2)     │
│  Mailbox (12.8 KB)                      │
│  LLK Debug (1 KB)                       │
│  Zeros Section (512 B)                  │
├─────────────────────────────────────────┤  base_allocator_addr
│                                         │
│  Circular Buffers (CB)                  │
│  - src0 CB (input A)                    │
│  - src1 CB (input B)                    │
│  - output CB                            │
│  - intermediate CB                      │
│  - bias CB                              │
│                                         │
├─────────────────────────────────────────┤  cb_region_end
│                                         │
│  Dynamic L1 Buffers                     │
│  (sharded tensors, etc.)                │
│                                         │
├─────────────────────────────────────────┤  lowest_occupied_l1_address
│  Free Space                             │
└─────────────────────────────────────────┘  MEM_L1_SIZE (1,536 KB)
```

### 핵심 상수

```cpp
// Blackhole: tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h
#define MEM_L1_SIZE (1536 * 1024)  // 1.5 MB per core
```

## L1 오버플로가 발생하는 메커니즘

### Circular Buffer 크기 계산

Matmul 연산의 CB 크기 추정 (핵심 함수):

```cpp
// ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp:15-59
uint32_t get_estimated_size_of_cbs(...) {
    // MCAST_INPUT_BUFFERING_DEPTH = 2 (더블 버퍼링)
    uint32_t in0_size = per_core_M * in0_block_w * 2 * in0_tile_size;   // Input A CB
    uint32_t in1_size = per_core_N * in0_block_w * 2 * in1_tile_size;   // Input B CB
    uint32_t out_size = per_core_M * per_core_N * output_tile_size;      // Output CB
    uint32_t interm_size = per_core_M * per_core_N * interm_tile_size;   // Intermediate CB
    uint32_t bias_size = in0_block_w * bias_tile_size;                   // Bias CB
    return in0_size + in1_size + out_size + interm_size + bias_size;
}
```

### L1 사용 가능 공간 계산

```cpp
// matmul_utilities.cpp:74-80
uint32_t get_max_l1_space(const Tensor& input_tensor_a) {
    auto* device = input_tensor_a.device();
    auto lowest_address = device->lowest_occupied_compute_l1_address();
    uint32_t max_l1_space = lowest_address.has_value()
        ? lowest_address.value()
        : device->l1_size_per_core();
    max_l1_space -= device->allocator()->get_base_allocator_addr(HalMemType::L1);
    return max_l1_space;
}
```

### CB가 L1에 들어가는지 확인

```cpp
// matmul_program_config.cpp:179-202
bool can_cbs_fit_in_l1(...) {
    uint32_t max_l1_space = utilities::get_max_l1_space(input_tensor_a);
    uint32_t size = utilities::get_estimated_size_of_cbs(
        per_core_M, per_core_N, in0_block_w, ...);
    return size < max_l1_space;
}
```

## 에러 메시지 종류

### 1. CB가 최대 L1 크기 초과

```
"Statically allocated circular buffers on core range {} grow to {} B
 which is beyond max L1 size of {} B"
```
- **파일**: `tt_metal/impl/program/program.cpp:927-932`
- **원인**: per_core_M * per_core_N이 너무 큼

### 2. CB가 동적 L1 버퍼와 충돌

```
"Statically allocated circular buffers in program {} clash with L1 buffers
 on core range {}. L1 buffer allocated at {} and static circular buffer
 region ends at {}"
```
- **파일**: `tt_metal/impl/program/program.cpp:935-941`
- **원인**: sharded tensor가 L1을 차지하고 있는데 CB도 L1 필요

### 3. Out of Memory

```
"Out of Memory: Not enough space to allocate {} B {} buffer across {} banks,
 where each bank needs to store {} B, but bank size is {} B
 (allocated: {} B, free: {} B, largest free block: {} B)"
```
- **파일**: `tt_metal/impl/allocator/bank_manager.cpp:423-434`
- **원인**: DRAM 또는 L1 할당기에서 공간 부족

### 4. 워크로드 초과

```
"Workload of Tiles {} at Tile Size {} (times 2 for output) exceeds L1 capacity {}"
```
- **파일**: `ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads/...`
- **원인**: QKV head 생성 시 L1 부족

## 오버플로 발생 시나리오 (P100 특화)

### 시나리오 1: QKV Prefill Matmul

```
입력: [batch, seq_len, 4096] x [4096, 6144]
컴퓨트 그리드: (8, 10) = 80 코어

per_core_M = 8 일 때 CB 크기:
  src0: 8 * 1 * 2 * 1024 = 16 KB (bfloat16 tile)
  src1: 28 * 1 * 2 * 1024 = 56 KB (P100 per_core_N=28)
  output: 8 * 28 * 1024 = 224 KB
  interm: 8 * 28 * 1024 = 224 KB (bfloat16)
  ────────────────────────────
  합계: ~520 KB → L1 사용 가능 공간 내에 있을 수 있지만
        sharded tensor + firmware 오버헤드 고려 시 초과!

per_core_M = 7 일 때:
  output: 7 * 28 * 1024 = 196 KB
  interm: 7 * 28 * 1024 = 196 KB
  ────────────────────────────
  합계: ~464 KB → 통과 (P100은 이 값 사용)
```

**→ P100은 per_core_M=7로 이미 패치됨 (model_config.py:927-928)**

### 시나리오 2: FP32 누산 활성화 시

```
fp32_dest_acc_en = True 일 때:
  interm_tile_size = 4096 (Float32)  vs  2048 (Float16)
  → intermediate CB 크기 2배 → 오버플로!

현재 설정: fp32_dest_acc_en=False (model_config.py:968)
→ 안전
```

### 시나리오 3: DRAM shard width 불일치

```python
# P100: dram_grid_size.x = 7
# per_core_N 계산이 7 기준
# 만약 8 기준으로 잘못 계산되면 → silent PCC failure (정확도 오류)
```

## L1 제어 레버 (튜닝 가능 파라미터)

| 레버 | 위치 | 효과 | 트레이드오프 |
|------|------|------|-------------|
| **per_core_M 축소** | matmul config | Output/Interm CB 축소 | 더 많은 코어 필요 |
| **per_core_N 축소** | matmul config | Output/Interm/src1 CB 축소 | 더 많은 코어 필요 |
| **in0_block_w=1** | matmul config | Input CB 50% 절감 | 재사용 효율 감소 |
| **fp32_dest_acc_en=False** | compute config | Interm CB 50% 절감 | 정밀도 감소 |
| **DRAM sharding** | memory config | L1 사용 최소화 | DRAM 대역폭 소비 |
| **l1_small_size 축소** | device open | Dispatch 영역 축소 | Command queue 제한 |
| **trace_region_size 축소** | device config | Trace 메모리 절감 | Trace 가능 연산 수 감소 |

## 자동 폴백 메커니즘

```
1차 시도: Multicast Config (per_core_M=8, per_core_N=계산값)
   ↓ can_cbs_fit_in_l1() == false
2차 시도: per_core_M 축소 (8 → 4 → 2 → 1)
   ↓ still false
3차 시도: Simple Matmul Config (단일 코어 per 블록)
   ↓ still false
최종: MatmulMultiCoreProgramConfig (최소 설정)
```

코드 위치: `matmul_program_config.cpp:429-450`

## 디버깅 방법

### 1. 환경 변수로 로깅 활성화

```bash
export TT_LOGGER_TYPES=Op
export TT_LOGGER_LEVEL=DEBUG
export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}'
```

### 2. 메모리 리포트 확인

```python
ttnn.dump_device_memory_state(device, prefix="debug_")
```

### 3. 사용 가능한 L1 확인

```python
# Python에서 간접적으로 확인
max_l1 = ttnn.get_max_worker_l1_unreserved_size(device)
```
