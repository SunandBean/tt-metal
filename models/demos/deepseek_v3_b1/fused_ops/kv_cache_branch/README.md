# KV Cache Branch Fused Operation

## Overview

This directory contains the boilerplate infrastructure for the KV Cache Branch fused operation, modeled after the PreSDPA operation structure.

## Structure

```
kv_cache_branch/
├── __init__.py              # Package initialization
├── op.py                    # Main operation implementation (Python)
├── kernels/
│   └── kv_cache_branch_kernel.cpp  # Unified kernel (C++)
└── README.md                # This file
```

## Files

### `op.py`

Contains the `KVCacheBranch` class with two main methods:

- **`golden()`**: PyTorch reference implementation for validation
- **`op()`**: TTNN implementation using `ttnn.generic_op`

Both methods are currently stubbed out with TODO comments and `NotImplementedError`.

### `kernels/kv_cache_branch_kernel.cpp`

Unified kernel that compiles for all RISC cores (NCRISC, BRISC, TRISC). Contains:

- Compile-time role flags for core differentiation
- Separate sections for each RISC core type
- TODO comments indicating where to add operation-specific logic

## Test File

A corresponding test file has been created at:
```
tests/unit_tests/test_kv_cache_branch.py
```

The test includes:
- Grid validation logic
- Tensor creation boilerplate
- Sharding configuration
- Placeholders for operation execution and validation

## Implementation Steps

1. **Define Operation Logic**:
   - Update `KVCacheBranch.golden()` with PyTorch reference implementation
   - Define required tensors and their shapes

2. **Implement Python Op**:
   - In `op.py`, fill in tile configuration
   - Define CB indices and semaphore IDs
   - Create compile-time args for each RISC core
   - Set up circular buffer descriptors
   - Create unified kernel descriptor
   - Build program descriptor

3. **Implement C++ Kernel**:
   - In `kv_cache_branch_kernel.cpp`, include necessary operation headers
   - Define core roles (is_input_core, is_matmul_core, etc.)
   - Implement NCRISC, BRISC, TRISC logic
   - Set up sharded buffers
   - Implement operation sequence

4. **Complete Test**:
   - In `test_kv_cache_branch.py`, uncomment and fill in tensor creation
   - Add operation call
   - Implement validation logic
   - Add PCC checks

## Reference

This boilerplate was created based on the PreSDPA operation structure:
- `fused_ops/pre_sdpa/op.py`
- `fused_ops/pre_sdpa/kernels/pre_sdpa_kernel.cpp`
- `tests/unit_tests/test_pre_sdpa.py`

Refer to these files for examples of:
- Mcast/gather operations
- Multiple matmul stages
- RMSNorm integration
- Complex CB management
- Multi-stage operation chaining
