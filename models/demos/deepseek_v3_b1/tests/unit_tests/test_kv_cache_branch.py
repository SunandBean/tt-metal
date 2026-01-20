# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN KV Cache Branch Test
Tests KV cache branch fused operation
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_b1.fused_ops.kv_cache_branch.op import KVCacheBranch


@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("use_fp32", [True])
def test_kv_cache_branch(device, epsilon, use_fp32):
    """Test TTNN KV cache branch fused operation"""

    position_id = 0
    max_seq_len = 8192
    batch = 1

    # Input tensor shapes
    input_shape = (1, 7168)
    W_dkv_rope_shape = (7168, 576)

    # Rope config
    rope_head_dim = 64
    rope_num_heads = 1

    # Create input PyTorch tensors
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_W_dkv_rope = torch.randn(W_dkv_rope_shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn((1, 512), dtype=torch.bfloat16)

    # ROPE
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_head_dim, 2, dtype=torch.float32) / rope_head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)

    # Meta-style: stack [cos(t), cos(t)] interleaved
    torch_cos = torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2)  # [max_seq_len, head_dim]
    torch_sin = torch.stack((freqs.sin(), freqs.sin()), dim=-1).flatten(-2)  # [max_seq_len, head_dim]
    position_ids = torch.tensor([position_id])  # positions 0, 1, 2, ...
    position_ids_expanded = position_ids.unsqueeze(1)  # [batch, 1]

    logger.info(f"Done creating torch tensors.")

    # TT setup
    # Grid configuration
    spec_start_offset = (0, 8)  # Offset for the operation grid
    spec_grid = (9, 2)  # Grid dimensions for the operation

    spec_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(spec_start_offset[0], spec_start_offset[1]),
                ttnn.CoreCoord(spec_grid[0] + spec_start_offset[0] - 1, spec_grid[1] + spec_start_offset[1] - 1),
            )
        }
    )
    rms_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(spec_start_offset[0], spec_start_offset[1]),
                ttnn.CoreCoord(spec_start_offset[0], spec_start_offset[1]),
            )
        }
    )
    rope_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(8 + spec_start_offset[0], spec_start_offset[1]),
                ttnn.CoreCoord(8 + spec_start_offset[0], 1 + spec_start_offset[1]),
            )
        }
    )

    logger.info("hi")
    logger.info(f"spec_crs: {spec_crs}")
    logger.info(f"rms_crs: {rms_crs}")
    logger.info(f"rope_crs: {rope_crs}")

    # Validate grid fits within device
    device_grid_size = device.compute_with_storage_grid_size()
    print(f"device_grid_size: {device_grid_size}")
    print(f"device_grid_size.x: {device_grid_size.x}")
    print(f"device_grid_size.y: {device_grid_size.y}")
    assert (
        spec_grid[0] + spec_start_offset[0] <= device_grid_size.x
    ), f"spec_grid.x ({spec_grid[0]}) + spec_start_offset.x ({spec_start_offset[0]}) must be <= device_grid_size.x ({device_grid_size.x})"
    assert (
        spec_grid[1] + spec_start_offset[1] <= device_grid_size.y
    ), f"spec_grid.y ({spec_grid[1]}) + spec_start_offset.y ({spec_start_offset[1]}) must be <= device_grid_size.y ({device_grid_size.y})"

    tile = ttnn.Tile([1, 32])

    print(f"spec_crs: {spec_crs}")
    input_shard_spec = ttnn.ShardSpec(
        spec_crs,
        input_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    # Create input tensor sharded on spec grid
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile,
    )

    # DKV Matmul
    num_cores = spec_grid[0] * spec_grid[1]
    shard_width = W_dkv_rope_shape[1] // num_cores
    assert (
        W_dkv_rope_shape[1] % num_cores == 0
    ), f"W_dkv_rope_shape[1] ({W_dkv_rope_shape[1]}) must be divisible by grid size ({num_cores})"
    assert shard_width == 32, f"Expected shard width of 32, got {shard_width}"

    W_dkv_rope_shard_shape = (W_dkv_rope_shape[0], shard_width)
    W_dkv_rope_shard_spec = ttnn.ShardSpec(
        spec_crs,
        W_dkv_rope_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    W_dkv_rope_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, W_dkv_rope_shard_spec
    )

    ttnn_W_dkv_rope = ttnn.from_torch(
        torch_W_dkv_rope,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=W_dkv_rope_mem_config,
    )

    # GAMMA
    gamma_shard_spec = ttnn.ShardSpec(
        rms_crs,
        (1, 512),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gamma_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gamma_shard_spec)
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gamma_mem_config,
        tile=tile,
    )

    # ROPE
    # Cos/sin indexed by position: [1, batch, 1, head_dim]
    # Shape stays [1, 1, 1, head_dim] - broadcast multiply will use row 0
    rope_tile = ttnn.Tile((rope_num_heads, ttnn.TILE_SIZE))
    trans_tile = ttnn.Tile((ttnn.TILE_SIZE, ttnn.TILE_SIZE))
    cos_selected = torch_cos[position_ids].unsqueeze(0).unsqueeze(2)
    sin_selected = torch_sin[position_ids].unsqueeze(0).unsqueeze(2)

    # Use same tiny tile as input - data in row 0, rows 1+ are padding
    cos_sin_shard_spec = ttnn.ShardSpec(
        rope_crs,
        (rope_num_heads, rope_head_dim),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    cos_sin_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, cos_sin_shard_spec
    )

    tt_cos = ttnn.from_torch(
        cos_selected,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cos_sin_mem_config,
        tile=rope_tile,
    )
    tt_sin = ttnn.from_torch(
        sin_selected,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cos_sin_mem_config,
        tile=rope_tile,
    )
    print("hi", tt_cos.memory_config().shard_spec.grid)

    # Transformation matrix - standard 32x32 tile
    trans_mat = get_rot_transformation_mat()
    trans_mat = trans_mat.repeat(1, 1, batch, 1)

    trans_shard_spec = ttnn.ShardSpec(
        rope_crs,
        (ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    trans_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, trans_shard_spec)

    tt_trans = ttnn.from_torch(
        trans_mat,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=trans_mem_config,
        tile=trans_tile,
    )

    # Create output tensor
    output_shape = (1, 512)
    output_shard_shape = (1, 512)  # (1, 128)
    output_shard_spec = ttnn.ShardSpec(
        rms_crs,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=tile,
    )

    logger.info(f"Created tensors sharded on single core with shard shape {output_shard_shape}")
    logger.info(f"Done creating TT tensors.")
    logger.info("Running KV cache branch operation...")
    ttnn_result = KVCacheBranch.op(
        ttnn_input,
        ttnn_W_dkv_rope,
        ttnn_gamma,
        tt_cos,
        tt_sin,
        position_ids_expanded,
        ttnn_output,
    )

    # TODO: Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_result)
    print(output_torch.shape)

    # TODO: Verify output shape
    # assert output_torch.shape == expected_shape, f"Expected shape {expected_shape}, got {output_torch.shape}"

    logger.info("Running KV cache branch golden reference...")
    # TODO: Compute reference output using PyTorch
    torch_expected = KVCacheBranch.golden(
        torch_input,
        torch_W_dkv_rope,
        torch_gamma,
        torch_cos,
        torch_sin,
        position_ids_expanded,
        epsilon=epsilon,
        fp32_dest_acc_en=use_fp32,
    )

    # TODO: Check if outputs are close
    # max_diff = torch.max(torch.abs(output_torch - torch_expected)).item()
    # mean_diff = torch.mean(torch.abs(output_torch - torch_expected)).item()
    # logger.info(f"Max absolute difference: {max_diff}")
    # logger.info(f"Mean absolute difference: {mean_diff}")

    # TODO: Add PCC check
    # from models.common.utility_functions import comp_pcc
    # passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.98)
    # logger.info(pcc_message)
    # assert passing, pcc_message

    logger.info("✓ KV cache branch test setup complete (implementation pending)")
