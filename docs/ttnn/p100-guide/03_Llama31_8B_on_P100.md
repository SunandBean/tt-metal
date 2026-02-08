# Llama 3.1 8B on P100: 문제점과 해결 방안

## 현재 상태

PERF.md에 P100 성능 목표가 이미 존재합니다:

| 디바이스 | 속도 (tok/s/u) | TTFT (ms) | Top-1 정확도 |
|----------|---------------|-----------|-------------|
| N150 | 28.3 | 104 | 90% |
| **P100** | **29.5** | **84** | **90%** |
| P150 | 33.6 | 76 | 90% |
| N300 | 44.2 | 67 | 90% |

## 모델 스펙 (Llama 3.1 8B)

| 파라미터 | 값 |
|----------|-----|
| Hidden dim | 4096 |
| 레이어 수 | 32 |
| Attention Heads | 32 |
| KV Heads | 8 (GQA) |
| Head dim | 128 |
| FFN hidden dim | 14336 |
| Vocab size | 128,256 |
| Context window | 8,192 (base) |

## P100 전용 설정 (이미 존재하는 것)

### 1. QKV Prefill - per_core_M=7 고정

```python
# model_config.py:927-934
per_core_M=7
if self.device_name == "P100"
else (max(1, 8 if seq_len >= ... else ...))
# NOTE: P100 runs OOM in L1 with 8 per_core_M
```

**이유**: P100은 DRAM 코어 7개 → per_core_N이 P150보다 크게 계산됨 → per_core_M=8이면 CB가 L1 초과

### 2. DRAM shard width = 7

```python
# model_config.py:819
dram_shard_grid_width = 8 if is_wormhole_b0() else self.dram_grid_size.x
# P100: 7, P150: 8
```

### 3. 시퀀스 길이 제한 (128, 1024만 지원)

```python
# model_config.py:1459
"P100": [128, 1024],  # 다른 디바이스는 최대 8192
```

### 4. QKV Decode Shard (Blackhole 전용)

```python
# model_config.py:946-956
ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, self.head_dim),  # (32, 128)
    core_grid=ttnn.CoreGrid(y=4, x=8),      # 32 코어
    strategy=ttnn.ShardStrategy.HEIGHT,
)
```

## 누락된 P100 Config (수정 필요)

### 문제 1: MAX_PREFILL_CHUNK_SIZES에 P100 없음

**파일**: `model_config.py:544-547`

```python
# 현재 코드:
"Llama-3.1-8B": {"N150": 4, "N300": 64, "T3K": 128, "TG": 128, "P150x4": 128},
#                              ^^^ P100 없음!

# 수정 방안:
"Llama-3.1-8B": {"N150": 4, "P100": 4, "P150": 4, "N300": 64, "T3K": 128, "TG": 128, "P150x4": 128},
```

**영향**: P100이 없으면 KeyError → 기본값 4로 폴백 + 경고 출력. 동작은 하지만 명시적이지 않아서 다른 코드 경로에서 예상 못한 config가 사용될 수 있음.

### 문제 2: Trace Region Size에 P100 없음

**파일**: `trace_region_config.py:82-87`

```python
# 현재 코드:
"Llama-3.1-8B": {
    "N150": 25000000,
    "N300": 38000000,
    "T3K": 50000000,
    "TG": 50000000,
    # P100 없음! → None 반환 → 기본 trace_region_size 사용
}

# 수정 방안:
"Llama-3.1-8B": {
    "P100": 25000000,    # N150과 동일하게 시작
    "N150": 25000000,
    ...
}
```

**영향**: Trace region size가 None이면 기본값이 사용되는데, 이 기본값이 P100의 DRAM 크기에 비해 너무 크면 OOM 발생 가능.

### 문제 3: LM Head 하드코딩

**파일**: `model_config.py:903-908`

```python
# 현재 코드:
self.max_columns_per_device_lm_head = (
    128256 // 8 if is_blackhole() else 668 * self.lm_head_core_grid.num_cores
)
# FIXME: Update blackhole figure to be per-core as well.
```

P100/P150 구분 없이 `128256 // 8 = 16,032`로 하드코딩. 7개 DRAM 코어에 맞는 값으로 조정 필요할 수 있음.

### 문제 4: prefill_len_cutoff 명시적 설정 없음

**파일**: `model_config.py:593-598`

```python
# 현재 코드: N150만 명시, P100은 is_blackhole()로 자동 512
if (
    self.base_model_name in ["Llama-3.1-8B", ...]
    and self.device_name == "N150"
):
    self.prefill_len_cutoff = 512

# P100은 이미 is_blackhole() 체크로 512 (line 482)이므로
# 동작에 문제는 없지만 명시적이지 않음
```

### 문제 5: Issue #33991 (seq_len > 1024 제한)

```python
# model_config.py:1436-1442
# TODO: https://github.com/tenstorrent/tt-metal/issues/33991
if self.base_model_name == "Llama-3.1-8B" and self.device_name == "P100":
    for seq_len in to_warmup_seq_lens:
        if seq_len > 1024:
            to_warmup_seq_lens = to_warmup_seq_lens[:to_warmup_seq_lens.index(seq_len)]
            break
```

**원인**: 1024 이상 시퀀스에서 assert 발생. QKV prefill matmul의 per_core_M/per_core_N 계산이 더 긴 시퀀스에 대응하지 못함.

## 수정 가이드

### Step 1: Config 엔트리 추가 (필수)

```python
# model_config.py - MAX_PREFILL_CHUNK_SIZES
"Llama-3.1-8B": {
    "P100": 4,           # 추가
    "N150": 4,
    "N300": 64,
    "T3K": 128,
    "TG": 128,
    "P150x4": 128
},
```

### Step 2: Trace Region 추가 (필수)

```python
# trace_region_config.py
"Llama-3.1-8B": {
    "P100": 25000000,    # 추가
    "N150": 25000000,
    ...
},
```

### Step 3: L1 추가 축소 (오버플로 지속 시)

```python
# model_config.py - QKV prefill
per_core_M=6  # 7에서 6으로 추가 축소
if self.device_name == "P100"
```

### Step 4: seq_len > 1024 지원 (Issue #33991)

per_core_M의 동적 계산이 P100의 DRAM grid width=7에 맞게 조정되어야 함:

```python
# 제안:
per_core_M = max(1, min(7, math.ceil(seq_len / self.tile_size / 10)))
# 10은 Blackhole 컴퓨트 그리드 y축 크기
```

## L1 오버플로 시 추가 디버깅

### 환경 변수 설정

```bash
export TT_LOGGER_TYPES=Op
export TT_LOGGER_LEVEL=DEBUG
export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}'
```

### 디바이스 열 때 L1 조정

```python
device = ttnn.open_device(
    device_id=0,
    l1_small_size=16384,         # 기본값보다 축소
    trace_region_size=25000000,  # P100에 맞게 설정
)
```

## 성능 최적화 방향

| 최적화 | 방법 | 예상 효과 |
|--------|------|----------|
| per_core_M 최적화 | 7 → L1에 맞는 최대값 탐색 | 처리량 향상 |
| Prefill chunk 확대 | 4K → 8K (L1 허용 범위 내) | TTFT 개선 |
| DRAM sharding 최적화 | LM head의 per-core 계산 | L1 사용 효율화 |
| out_subblock 튜닝 | (1,1) → 더 큰 값 탐색 | 연산 효율 향상 |

## 관련 이슈

- [#33991](https://github.com/tenstorrent/tt-metal/issues/33991) - P100 seq_len > 1024 제한
- model_config.py:905 FIXME - Blackhole LM head per-core 최적화 미완
- model_config.py:924 FIXME - Prefill config 최적화 미완
