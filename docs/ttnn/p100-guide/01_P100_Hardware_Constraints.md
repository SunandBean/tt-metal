# P100 하드웨어 제약 사항 분석

## P100이란?

P100은 Tenstorrent **Blackhole** 아키텍처 기반의 싱글 카드 가속기로, 이더넷이 하베스트(비활성화)되어 비용이 절감된 모델입니다.

- **ClusterType**: `P100` (cluster.hpp에서 ClusterType = 6)
- **아키텍처**: Blackhole
- **특징**: 이더넷 완전 비활성화 → 싱글 디바이스 전용

## 하드웨어 스펙

| 항목 | P100 | P150 | N150 (Wormhole) |
|------|------|------|-----------------|
| **아키텍처** | Blackhole | Blackhole | Wormhole B0 |
| **컴퓨트 코어** | 140개 | 140개 | 80개 |
| **코어 그리드** | 17x12 | 17x12 | 10x12 |
| **L1/코어** | 1,536 KB | 1,536 KB | 1,464 KB |
| **DRAM 뱅크** | **7개** (1개 하베스트) | **8개** | 12개 |
| **총 DRAM** | **~28 GB** (7 × 4GB) | ~32 GB (8 × 4GB) | ~12 GB |
| **이더넷** | 비활성(harvested) | 활성 | 활성 |
| **멀티디바이스** | 불가 | 가능 | 가능 |

## P100 vs P150: 핵심 차이점

**P100과 P150은 동일한 Blackhole 칩이지만 핵심 차이가 있습니다:**

1. **DRAM 뱅크: 7개 vs 8개** → 총 DRAM **28GB vs 32GB**, matmul per_core_N 계산이 달라짐
2. **이더넷: 비활성 vs 활성** → 싱글 디바이스만 가능

### DRAM 하베스팅 상세

```cpp
// tools/scaleout/factory_system_descriptor/utils.cpp:867-868
if (board_type_enum == BoardType::P100) {
    dram_harvesting_mask = 8;  // binary 0b1000 → 1개 DRAM 뱅크 비활성화
}

// blackhole dev_mem_map.h:39
#define MEM_DRAM_SIZE (4177920 * 1024U)  // 뱅크당 ~3.99 GB
// P100: 7 뱅크 × ~4 GB = ~28 GB
// P150: 8 뱅크 × ~4 GB = ~32 GB
```

### DRAM 뱅크 7개의 영향

DRAM-sharded matmul에서 `per_core_N` 계산이 모두 달라집니다:

```python
# model_config.py:3354
1: "P100" if dram_grid_size and dram_grid_size.x == 7 else "P150"
# → P100 판별 기준: DRAM grid x축이 7

# model_config.py:819
dram_shard_grid_width = 8 if is_wormhole_b0() else self.dram_grid_size.x
# → P100: 7, P150: 8
```

**per_core_N 계산 영향 예시 (QKV matmul):**
```
P100: ceil(6144 / 1 / 32 / 7) = ceil(27.4) = 28
P150: ceil(6144 / 1 / 32 / 8) = ceil(24.0) = 24
```

per_core_N이 더 크면 → Circular Buffer 크기 증가 → L1 overflow 위험 증가

## Harvesting 설정

```yaml
# p100_cluster_desc.yaml
harvesting:
  noc_translation: true
  harvest_mask: 0              # 컴퓨트 코어 하베스트 없음
  dram_harvesting_mask: 0      # DRAM 하베스트 없음
  eth_harvesting_mask: 288     # 이더넷 하베스트 (비활성)
  pcie_harvesting_mask: 2      # 최소 PCIe 하베스트
```

## 디바이스 감지 코드

```python
# model_config.py:3332-3358
def determine_device_name(mesh_device):
    dram_grid_size = mesh_device.dram_grid_size()
    if is_blackhole():
        dict_device_names = {
            1: "P100" if dram_grid_size and dram_grid_size.x == 7 else "P150",
            2: "P300",
            ...
        }
```

## CCL (Collective Communication) 제한

```python
# ccl.py
link_dict = {
    "P100": (0, 0),   # 이더넷 없음 → CCL 불가
    "P150": (0, 0),   # 싱글도 CCL 불필요
    "N300": (1, 1),   # 2-chip 연결
    "T3K":  (1, 1),   # 8-chip 연결
}
```

## 관련 파일

| 파일 | 역할 |
|------|------|
| `tt_metal/api/tt-metalium/cluster.hpp` | ClusterType 정의 |
| `tt_metal/soc_descriptors/blackhole_140_arch.yaml` | SOC 디스크립터 |
| `tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/p100_cluster_desc.yaml` | 클러스터 디스크립터 |
| `models/tt_transformers/tt/model_config.py:3332-3374` | 디바이스 이름 결정 로직 |
| `models/tt_transformers/demo/trace_region_config.py` | 트레이스 리전 설정 |
