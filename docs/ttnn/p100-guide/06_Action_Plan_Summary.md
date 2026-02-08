# P100 모델 포팅 액션 플랜 요약

## 수정 우선순위 전체 맵

```
 우선순위     작업                               난이도    예상 효과
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [P0] 필수   Llama 3.1 8B Config 보완            낮음     P100에서 기본 동작
 [P1] 높음   L1 오버플로 추가 대응                중간     안정성 확보
 [P2] 중간   Whisper P100 테스트/Config          낮음     새 모달리티 지원
 [P3] 중간   Qwen2.5-VL-3B P100 Config          낮음     VLM 지원
 [P4] 낮음   seq_len > 1024 지원 (Issue #33991)  높음     긴 컨텍스트 지원
 [P5] 낮음   고해상도 VLM 최적화                  높음     다양한 이미지 크기
```

---

## [P0] Llama 3.1 8B Config 보완 (필수)

### 수정 파일 1: `models/tt_transformers/tt/model_config.py`

**Line ~547**: MAX_PREFILL_CHUNK_SIZES에 P100 추가
```python
# Before:
"Llama-3.1-8B": {"N150": 4, "N300": 64, "T3K": 128, "TG": 128, "P150x4": 128},
# After:
"Llama-3.1-8B": {"P100": 4, "N150": 4, "N300": 64, "T3K": 128, "TG": 128, "P150x4": 128},
```

### 수정 파일 2: `models/tt_transformers/demo/trace_region_config.py`

**Line ~82**: Trace region size에 P100 추가
```python
# Before:
"Llama-3.1-8B": {
    "N150": 25000000,
    "N300": 38000000,
    "T3K": 50000000,
    "TG": 50000000,
},
# After:
"Llama-3.1-8B": {
    "P100": 25000000,
    "N150": 25000000,
    "N300": 38000000,
    "T3K": 50000000,
    "TG": 50000000,
},
```

---

## [P1] L1 오버플로 추가 대응

### per_core_M 추가 축소 (필요 시)

**파일**: `models/tt_transformers/tt/model_config.py:927`
```python
# 현재: per_core_M=7 for P100
# 오버플로 지속 시: 7 → 6 또는 4로 축소
per_core_M=6 if self.device_name == "P100" else ...
```

### 디바이스 l1_small_size 축소

```python
device = ttnn.open_device(
    device_id=0,
    l1_small_size=16384,  # 기본값보다 축소
)
```

### FP32 누산 비활성화 확인

```python
# model_config.py:965-969 - 이미 False로 설정되어 있음을 확인
fp32_dest_acc_en=False  # 이것이 True면 intermediate CB 2배
```

---

## [P2] Whisper P100 포팅

### 테스트 실행
```bash
pytest models/demos/audio/whisper/tests/test_whisper_modules.py -v
```

### Config 추가 (필요 시)
- Encoder seq_len ~1500을 512 chunk로 분할 가능한지 확인
- trace_region_size 설정

### 예상: 모델이 작아서 큰 수정 없이 동작할 가능성 높음

---

## [P3] VLM (Qwen2.5-VL-3B) P100 포팅

### Config 추가
```python
# model_config.py
"Qwen2.5-VL-3B": {"P100": 128, "N150": 128, "N300": 128, ...},
```

### 224×224 이미지 기준 Vision Encoder
- 196 patches < 512 cutoff → OK
- L1 프로파일링 필요

---

## [P4] seq_len > 1024 지원

### Issue: #33991
### 원인: QKV prefill matmul에서 per_core_M/per_core_N 동적 계산 필요
### 난이도: 높음 (matmul config 로직 수정 필요)

---

## 핵심 파일 레퍼런스

| 파일 | 수정 내용 |
|------|----------|
| `models/tt_transformers/tt/model_config.py` | P100 config 추가, per_core_M 튜닝 |
| `models/tt_transformers/demo/trace_region_config.py` | P100 trace region 추가 |
| `ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp` | L1 계산 로직 이해 |
| `ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp` | CB fit 체크 로직 |
| `models/demos/audio/whisper/tt/ttnn_optimized_functional_whisper.py` | Whisper 구현 |
| `models/demos/qwen25_vl/` | Qwen VLM 구현 |

---

## 검증 체크리스트

### Llama 3.1 8B
- [ ] P100 config 엔트리 추가
- [ ] trace_region_size 추가
- [ ] decode 모드 (1 토큰 생성) 동작 확인
- [ ] prefill 모드 (128 토큰) 동작 확인
- [ ] prefill 모드 (1024 토큰) 동작 확인
- [ ] 정확도 확인 (PCC > 0.99)
- [ ] 성능 확인 (목표: 29.5 tok/s/u)

### Whisper
- [ ] 기본 테스트 실행
- [ ] P100에서 encoder forward pass
- [ ] P100에서 decoder forward pass (cross-attention 포함)
- [ ] 30초 오디오 end-to-end 변환
- [ ] 정확도 확인 (WER 비교)

### VLM (Qwen2.5-VL-3B)
- [ ] P100 config 추가
- [ ] Vision encoder (224×224) forward pass
- [ ] Language model forward pass
- [ ] 이미지 + 텍스트 end-to-end 추론
- [ ] 생성 품질 확인
