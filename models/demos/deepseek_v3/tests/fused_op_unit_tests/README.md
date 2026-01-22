# DeepSeek v3 fused-op unit tests

This table enumerates pytest tests under this folder and the cases they cover.

| Module | Operation | Unit test exists | Decode | Prefill 128 | Prefill 1k | Prefill 8k | Prefill 32k | Prefill 128k | PCC ATOL | E2E Perf | Device Perf | Single device test | Random weights | Program_cache on/off | Tracing on/off | 100 iterations | Added to CI | Perf shown in superset | Path to test file | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| moe | ds_moe_op | ✅ | ✅ | ✅ | ✅ | ❌ | ⛔️ | ☑️ | ✅ | ✅ | ✅ | ☑️ | ✅ | ✅ | ✅ | ⛔️ | ✅ |  | `models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_op.py` | Prefill 131072 gated by `DEEPSEEK_V3_LONG_SEQ_TESTS`; prefill 8192 is failing; no 32k prefill; single-device variants exist but are always skipped (`_skip_single_device_ccl`). |
| moe | ds_moe_all_gather | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ☑️ | ✅ | ✅ | ✅ | ☑️ | ✅ | ✅ | ✅ | ⛔️ | ✅ |  | `models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_all_gather.py` | Prefill 131072 gated by `DEEPSEEK_V3_LONG_SEQ_TESTS`; 32k prefill added but not verified; current failures include cache out-of-space during test runs; single-device variants exist but are always skipped (`_skip_single_device_ccl`). |
| moe | ds_moe_repeat_permute_expert_weights | ✅ | ✅ | ✅ | ✅ | ✅ | ☑️ | ☑️ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  | `models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe_repeat_permute_expert_weights.py` | Prefill 32768/131072 gated by `DEEPSEEK_V3_LONG_SEQ_TESTS`; perf baselines TODO. |
| mla | paged_update_cache | ✅ | ✅ | ☑️ | ☑️ | ☑️ | ⛔️ | ☑️ | ✅ | ✅ | ✅ | ⛔️ | ⛔️ | ✅ | ✅ | ⛔️ | ✅ |  | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_paged_update_cache.py` | Prefill cases are explicitly skipped (op used only in decode); long seq gated by `DEEPSEEK_V3_LONG_SEQ_TESTS`; device-perf `test_path` points to a non-existent file (missing `/mla/`). |
| mla | ds_mla_norm_and_rope | ✅ | ✅ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ✅ | ✅ | ✅ | ⛔️ | ⛔️ | ✅ | ✅ | ⛔️ | ✅ |  | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_mla_norm_and_rope.py` | Decode-only op; device-perf `test_path` points to a non-existent file (missing `/mla/`). |
| mla | ds_ag_reshape | ✅ | ✅ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ✅ | ✅ | ✅ | ⛔️ | ⛔️ | ✅ | ✅ | ⛔️ | ✅ |  | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_ag_reshape.py` | Decode-only op; device-perf `test_path` points to a non-existent file (missing `/mla/`). |
| mla | ds_mla_all_to_all_before_flash_mla | ✅ | ✅ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ✅ | ✅ | ✅ | ⛔️ | ⛔️ | ✅ | ✅ | ⛔️ | ✅ |  | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_mla_all_to_all_before_flash_mla.py` | Decode-only op; device-perf `test_path` points to a non-existent file (missing `/mla/`). |
| mla | ds_flash_mla | ✅ | ✅ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ✅ | ✅ | ✅ | ⛔️ | ⛔️ | ✅ | ✅ | ⛔️ | ✅ |  | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_flash_mla.py` | Decode-only op; device-perf `test_path` points to a non-existent file (missing `/mla/`). |
| mla | ds_wo | ✅ | ✅ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ✅ | ✅ | ✅ | ✅ | ⛔️ | ✅ | ✅ | ⛔️ | ✅ |  | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_wo.py` | Decode-only op; single-device tests exist (one uses a submesh); device-perf `test_path` points to a non-existent file (missing `/mla/`). |
| mla | ds_fused_wqkva | ✅ | ✅ | ✅ | ✅ | ✅ | ⛔️ | ☑️ | ✅ | ✅ | ✅ | ☑️ | ⛔️ | ✅ | ✅ | ⛔️ | ✅ |  | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_fused_wqkva.py` | Prefill 131072 gated by `DEEPSEEK_V3_LONG_SEQ_TESTS`; no 32k prefill; single-device variants exist but are always skipped (`_skip_single_device_ccl`); device-perf `test_path` points to a non-existent file (missing `/mla/` and `test_` prefix). |
| mla | ds_linear_with_input_dim | ✅ | ✅ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ✅ | ✅ | ✅ | ⛔️ | ⛔️ | ✅ | ✅ | ⛔️ | ✅ |  | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_linear_with_input_dim.py` | Decode-only op; device-perf `test_path` points to a non-existent file (missing `/mla/`). |
| mla | ds_fused_q_rope_nope | ✅ | ✅ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ⛔️ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⛔️ | ✅ |  | `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_fused_q_rope_nope.py` | Decode-only op; prefill TODO in test body. |

Each row referrs to a fused op and it's related unit tests.

Legend
✅: indicates the test case exists and is passing
❌: indicates the test case exists but is failing
☑️: indicates the test case exists but is skipped
⛔️: indicates the test case does not exist

Test cases / features:
- Unit test exists: whether the unit tests exists
- Decode: whether decode setting is tested
- Prefill (128,1k,8k,32k,128k): whether prefill is tested with the specified seqquence length
- PCC ATOL: whether pcc and atol is tested and asserted on
- E2E Perf: whether there is an e2e perf test preparing the e2e data to be uploaded to superset
- Device Perf: device perf test preparing the device data to be uploaded to superset
- Single deivce test: single device test that tests the first device's chunk of the workload (only for non-CCL ops)
- Random weights: random weight option available for tests
- Program_cache on/off: option available
- Tracing on/off: option available
- 100 iterations: does the test run 100 iterations when checking PCC/ATOL
- Added to CI: whether it's added to any CI pipeline
- Perf shown in superset: whether any superset dashboard cisualizes the data
- Path to test file: path to the test file

"Added to CI" reflects `pytest models/demos/deepseek_v3/tests --ignore=models/demos/deepseek_v3/tests/unit` in `.github/workflows/galaxy-deepseek-tests-impl.yaml`.

Superset dashboard: https://superset.tenstorrent.com/superset/dashboard/4fa0fef8-cced-4a8e-8819-48aeed75dcee/?permalink_key=6JbY9pNEQaZ
