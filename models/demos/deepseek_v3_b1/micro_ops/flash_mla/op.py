# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Flash Multi-Latent Attention (MLA) Decode operation.

This implements flash attention decode for MLA where K and V share the same tensor,
with V being the first head_dim_v elements of the KV cache.

This Python implementation matches the C++ sdpa_decode_program_factory.cpp exactly.
"""

import math

import torch

import ttnn


def pack_two_bfloat16_into_uint32(val: float) -> int:
    """Pack a float value into two bfloat16 values in a uint32."""
    import struct

    # Convert float to bfloat16 (truncate lower 16 bits of float32)
    float_bytes = struct.pack("f", val)
    float_int = struct.unpack("I", float_bytes)[0]
    bfloat16_val = float_int >> 16
    # Pack two bfloat16 values into uint32
    return (bfloat16_val << 16) | bfloat16_val


def float_to_uint32(val: float) -> int:
    """Convert float to uint32 representation."""
    import struct

    return struct.unpack("I", struct.pack("f", val))[0]


def get_interleaved_tensor_accessor_args(tensor):
    """
    Construct tensor accessor compile-time args for interleaved tensors (DRAM or L1).

    Returns [args_config] where args_config = IsDram (2) for DRAM, 0 for L1.
    """
    is_dram = tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    return [2 if is_dram else 0]


def get_tensor_accessor_args(tensor):
    """
    Construct tensor accessor compile-time args for any tensor (interleaved or sharded).

    Uses ttnn.TensorAccessorArgs to get proper args for the tensor's memory config.
    Returns list of compile-time args.
    """
    accessor_args = ttnn.TensorAccessorArgs(tensor)
    return accessor_args.get_compile_time_args()


class FlashMLADecode:
    """
    Flash Multi-Latent Attention Decode.

    This class implements flash attention decode operation optimized for
    Multi-Latent Attention (MLA) where K and V are stored together in
    a single KV cache tensor.

    Q: [1, batch, num_heads, kvpe_dim] - Query tensor
    KV Cache: [batch, 1, max_seq_len, kvpe_dim] - Combined KV cache
    V is the first head_dim_v elements of the kvpe_dim dimension.
    """

    @staticmethod
    def golden(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        position_ids: torch.Tensor,
        head_dim_v: int,
        scale: float,
    ) -> torch.Tensor:
        """
        PyTorch reference implementation for flash MLA decode.

        Args:
            q: Query tensor [1, batch_size, num_heads, kvpe_dim]
            kv_cache: KV cache tensor [batch_size, 1, max_seq_len, kvpe_dim]
            position_ids: Position indices [batch_size]
            head_dim_v: The value head dimension (first head_dim_v elements of kvpe_dim)
            scale: Attention scale factor

        Returns:
            Attention output [1, batch_size, num_heads, head_dim_v]
        """
        batch_size = q.shape[1]
        num_heads = q.shape[2]
        kvpe_dim = q.shape[3]

        # Reshape Q: [1, batch, num_heads, kvpe_dim] -> [batch, num_heads, 1, kvpe_dim]
        q = q.permute(1, 2, 0, 3)  # [batch, num_heads, 1, kvpe_dim]

        # Process each batch element with its own sequence length
        outputs = []
        for b in range(batch_size):
            seq_len = position_ids[b].item() + 1  # Attend to positions 0 to position_ids[b]

            # Get KV for this batch: [1, seq_len, kvpe_dim]
            kv = kv_cache[b, :, :seq_len, :]  # [1, seq_len, kvpe_dim]

            # Expand KV to num_heads: [num_heads, seq_len, kvpe_dim]
            kv_expanded = kv.expand(num_heads, seq_len, kvpe_dim)

            # Q for this batch: [num_heads, 1, kvpe_dim]
            q_b = q[b]  # [num_heads, 1, kvpe_dim]

            # Attention scores: Q @ K^T -> [num_heads, 1, seq_len]
            attn_scores = torch.matmul(q_b, kv_expanded.transpose(-2, -1)) * scale

            # Softmax
            attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(q.dtype)

            # V is the first head_dim_v elements of KV
            v = kv_expanded[:, :, :head_dim_v]  # [num_heads, seq_len, head_dim_v]

            # Output: attn_probs @ V -> [num_heads, 1, head_dim_v]
            out_b = torch.matmul(attn_probs, v)
            outputs.append(out_b)

        # Stack outputs: [batch, num_heads, 1, head_dim_v]
        output = torch.stack(outputs, dim=0)

        # Reshape to [1, batch, num_heads, head_dim_v]
        output = output.squeeze(2)  # [batch, num_heads, head_dim_v]
        output = output.unsqueeze(0)  # [1, batch, num_heads, head_dim_v]

        return output

    @staticmethod
    def op(
        q_tensor: ttnn.Tensor,
        kv_cache_tensor: ttnn.Tensor,
        head_dim_v: int,
        cur_pos_tensor: ttnn.Tensor,
        output_tensor: ttnn.Tensor,
        scale: float = None,
        program_config: "ttnn.SDPAProgramConfig" = None,
        compute_kernel_config: "ttnn.DeviceComputeKernelConfig" = None,
    ) -> ttnn.Tensor:
        """
        Execute flash MLA decode operation using ttnn.generic_op.

        This implementation matches the C++ sdpa_decode_program_factory.cpp exactly.

        Args:
            q_tensor: Query tensor [1, batch, num_heads, kvpe_dim] (must be height sharded)
            kv_cache_tensor: KV cache tensor [batch, 1, max_seq_len, kvpe_dim] in DRAM
            head_dim_v: The value head dimension (first head_dim_v elements of kvpe_dim)
            cur_pos_tensor: Current position tensor [batch] (int32)
            output_tensor: Pre-allocated output tensor [1, batch, num_heads, head_dim_v]
            scale: Attention scale factor (optional, computed if not provided)
            program_config: ttnn.SDPAProgramConfig with k_chunk_size and exp_approx_mode
            compute_kernel_config: ttnn.DeviceComputeKernelConfig (optional)

        Returns:
            Output tensor with attention result [1, batch, num_heads, head_dim_v]
        """
        # =========================================================================
        # Extract parameters (matching C++ lines 40-55)
        # =========================================================================
        use_mla = True  # This is the MLA variant
        input_tensor_q = q_tensor
        input_tensor_k = kv_cache_tensor
        input_tensor_v = kv_cache_tensor  # MLA: V is same as K

        device = input_tensor_q.device()

        # Scale (C++ lines 41-43)
        if scale is None:
            scale = 1.0 / math.sqrt(float(input_tensor_q.padded_shape[-1]))

        # Program config parameters
        if program_config is not None:
            k_chunk_size = program_config.k_chunk_size
            exp_approx_mode = program_config.exp_approx_mode if program_config.exp_approx_mode is not None else True
            grid_size = program_config.compute_with_storage_grid_size
        else:
            k_chunk_size = 128
            exp_approx_mode = True
            grid_size = device.compute_with_storage_grid_size()

        # Compute kernel config
        if compute_kernel_config is not None:
            math_fidelity = compute_kernel_config.math_fidelity
            math_approx_mode = compute_kernel_config.math_approx_mode
            fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en
            packer_l1_acc = compute_kernel_config.packer_l1_acc
        else:
            math_fidelity = ttnn.MathFidelity.HiFi2
            math_approx_mode = True
            fp32_dest_acc_en = False
            packer_l1_acc = False

        # =========================================================================
        # Shape extraction (matching C++ lines 70-114)
        # =========================================================================
        # Get tile dimensions from tensors (Q may use tiny tiles, K uses standard tiles)
        q_tile = input_tensor_q.get_tile()
        k_tile = input_tensor_k.get_tile()
        Q_TILE_HEIGHT = q_tile.tile_shape[0]
        K_TILE_HEIGHT = k_tile.tile_shape[0]
        TILE_WIDTH = 32  # Width is always 32

        q_shape = input_tensor_q.padded_shape
        q_shape_unpadded = input_tensor_q.shape
        k_shape = input_tensor_k.padded_shape

        # Q: [1, B, PNH, DH]
        # K: [B, NKV, S, DH]
        B = q_shape[1]
        PNH = q_shape[2]
        S = k_shape[2]
        DH = k_shape[3]

        num_kv_heads = k_shape[1]
        num_q_heads = q_shape_unpadded[2]

        # Q heads parallel factor (Q is always sharded for MLA)
        q_heads_parallel_factor = 1
        if use_mla:
            q_shard_height = input_tensor_q.memory_config().shard_spec.shape[0]
            q_heads_parallel_factor = max(1, (num_q_heads + q_shard_height - 1) // q_shard_height)
            B *= q_heads_parallel_factor  # Adjust batch size to account for Q sharding

        Bkv = k_shape[0]
        St = S // K_TILE_HEIGHT  # K/V use standard tile height
        DHt = DH // TILE_WIDTH
        vDHt = head_dim_v // TILE_WIDTH if use_mla else DHt
        PNHt = PNH // q_heads_parallel_factor // Q_TILE_HEIGHT  # Q uses its own tile height

        Sk_chunk_t = k_chunk_size // K_TILE_HEIGHT  # K chunks use K tile height

        # =========================================================================
        # Parallelization scheme (matching C++ lines 163-282)
        # =========================================================================
        num_cores_available = grid_size.x * grid_size.y

        # Core grid setup
        core_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
        )

        # Balance cores (C++ lines 197-207)
        # C++ default for max_cores_per_head_batch is 16
        max_cores_per_head_batch = 16  # Default value matching C++
        if program_config is not None and hasattr(program_config, "max_cores_per_head_batch"):
            max_cores_per_head_batch = program_config.max_cores_per_head_batch
        max_num_cores_for_compute = (
            max_cores_per_head_batch * B * num_kv_heads if program_config is not None else num_cores_available
        )
        num_cores_per_batch = min(num_cores_available, max_num_cores_for_compute) // B
        num_cores_per_head = max(1, num_cores_per_batch // num_kv_heads)
        num_heads_per_core = max(1, math.ceil(num_kv_heads / num_cores_per_batch))
        num_reducer_cores = num_kv_heads * B // num_heads_per_core
        num_output_cores = B
        num_active_cores = num_cores_per_head * num_kv_heads * B // num_heads_per_core
        # Recalculate num_cores_per_batch based on num_active_cores
        num_cores_per_batch = num_active_cores // B

        # Create core group (Q and output are always sharded)
        core_group = []
        core_group_idle = []

        reducer_idx = 0
        worker_idx = num_output_cores

        for i in range(num_cores_available):
            if i % num_cores_per_batch == 0 and reducer_idx < num_output_cores:
                core = ttnn.CoreCoord(reducer_idx % grid_size.x, reducer_idx // grid_size.x)
                reducer_idx += 1
            else:
                core = ttnn.CoreCoord(worker_idx % grid_size.x, worker_idx // grid_size.x)
                worker_idx += 1

            if i < num_active_cores:
                core_group.append(core)
            else:
                core_group_idle.append(core)

        # =========================================================================
        # CB tile counts (matching C++ lines 285-299)
        # =========================================================================
        dst_size = 4 if fp32_dest_acc_en else 8
        max_dynamic_chunk_size = dst_size
        Sk_chunk_t_cb_size = Sk_chunk_t if Sk_chunk_t > 0 else max_dynamic_chunk_size

        q_tiles = PNHt * DHt
        k_tiles = Sk_chunk_t_cb_size * DHt * 2  # double buffer
        v_tiles = Sk_chunk_t_cb_size * vDHt * 2  # double buffer
        qk_tiles = PNHt * Sk_chunk_t_cb_size
        out_im_tiles = PNHt * vDHt
        out0_t = PNHt * vDHt
        scale_tiles = 1
        statistics_tiles = PNHt

        # =========================================================================
        # Matmul configs (matching C++ lines 310-371)
        # =========================================================================
        qk_in0_block_w = DHt
        qk_num_blocks = DHt // qk_in0_block_w

        if Sk_chunk_t > 0:
            qk_out_subblock_w = min(Sk_chunk_t, dst_size)
            qk_out_subblock_h = min(PNHt, dst_size // qk_out_subblock_w) if qk_out_subblock_w == Sk_chunk_t else 1
            qk_in0_num_subblocks = PNHt // qk_out_subblock_h
            qk_in1_num_subblocks = Sk_chunk_t // qk_out_subblock_w
        else:
            qk_out_subblock_w = 0
            qk_out_subblock_h = 0
            qk_in0_num_subblocks = 0
            qk_in1_num_subblocks = 0

        if Sk_chunk_t > 0:
            out_in0_block_w = Sk_chunk_t
            out_num_blocks = Sk_chunk_t // out_in0_block_w
        else:
            out_in0_block_w = 0
            out_num_blocks = 0

        out_out_subblock_w = min(vDHt, dst_size)
        out_out_subblock_h = min(PNHt, dst_size // out_out_subblock_w) if out_out_subblock_w == vDHt else 1
        out_in0_num_subblocks = PNHt // out_out_subblock_h
        out_in1_num_subblocks = vDHt // out_out_subblock_w

        dht_granularity = min(DHt, dst_size)
        log2_dht_granularity = int(math.log2(dht_granularity)) if dht_granularity > 0 else 0
        if dht_granularity != (1 << log2_dht_granularity):
            dht_granularity = 1
            log2_dht_granularity = 0

        # =========================================================================
        # Data formats (matching C++ lines 374-426)
        # =========================================================================
        q_df = input_tensor_q.dtype
        k_df = input_tensor_k.dtype
        v_df = input_tensor_v.dtype
        out_df = output_tensor.dtype
        scalar_df = ttnn.bfloat16
        im_df = ttnn.bfloat16
        stats_df = ttnn.bfloat16

        # Create tile objects - Q uses tiny tiles, K/V use full tiles
        # Tiny tile height matches Q tile height (e.g., 8 for 8 heads per core)
        q_tiny_tile = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))
        full_tile = ttnn.Tile((K_TILE_HEIGHT, TILE_WIDTH))

        # All intermediate/stats/scalar tiles use tiny tile dimensions (same as Q)
        im_tile = q_tiny_tile
        stats_tile = q_tiny_tile
        scalar_tile = q_tiny_tile
        mask_tile = q_tiny_tile
        # K, V, col_identity use full tiles
        k_tile_obj = full_tile
        v_tile_obj = full_tile

        # Create tile descriptors for CB setup
        q_tile_descriptor = ttnn.TileDescriptor(q_tiny_tile)
        im_tile_descriptor = ttnn.TileDescriptor(im_tile)
        stats_tile_descriptor = ttnn.TileDescriptor(stats_tile)
        scalar_tile_descriptor = ttnn.TileDescriptor(scalar_tile)
        mask_tile_descriptor = ttnn.TileDescriptor(mask_tile)
        full_tile_descriptor = ttnn.TileDescriptor(full_tile)

        # Tile sizes - use tile.get_tile_size(dtype) for proper sizing
        q_tile_size = q_tiny_tile.get_tile_size(q_df)
        k_tile_size = k_tile_obj.get_tile_size(k_df)
        v_tile_size = v_tile_obj.get_tile_size(v_df)
        scalar_tile_size = scalar_tile.get_tile_size(scalar_df)
        im_tile_size = im_tile.get_tile_size(im_df)
        stats_tile_size = stats_tile.get_tile_size(stats_df)
        mask_tile_size = mask_tile.get_tile_size(im_df)
        col_identity_tile_size = full_tile.get_tile_size(scalar_df)

        # Intermediate output tiles (C++ line 427)
        intermed_output_tiles = (out0_t + 2 * PNHt) * (num_cores_per_head - 1)

        # Index stick size for position tensor (C++ lines 429-445)
        use_cur_pos_tensor = True
        index_stick_size = B * 4  # int32 = 4 bytes per element
        index_stick_size = ((index_stick_size + 31) // 32) * 32  # Align to 32 bytes

        # =========================================================================
        # Packed constants (matching C++ lines 658-670)
        # =========================================================================
        packed_identity_scalar = pack_two_bfloat16_into_uint32(1.0)
        packed_zero_scalar = pack_two_bfloat16_into_uint32(0.0)
        scale_uint32 = float_to_uint32(scale)

        # =========================================================================
        # Create reducer and output core lists (matching C++ lines 671-721)
        # =========================================================================
        reduce_core_physical_xs = []
        reduce_core_physical_ys = []

        for i in range(num_active_cores):
            core = core_group[i]
            worker_id_for_reduce = (i % num_cores_per_head) - 1
            do_reduce = worker_id_for_reduce == 0xFFFFFFFF  # -1 in unsigned
            if i % num_cores_per_head == 0:  # First core in head group is reducer
                core_physical = device.worker_core_from_logical_core(core)
                reduce_core_physical_xs.append(core_physical.x)
                reduce_core_physical_ys.append(core_physical.y)

        output_core_physical_xs = []
        output_core_physical_ys = []

        for i in range(num_active_cores):
            core = core_group[i]
            worker_id_for_output = (i % num_cores_per_batch) - 1
            do_output = worker_id_for_output == 0xFFFFFFFF  # -1 in unsigned
            if i % num_cores_per_batch == 0:  # First core in batch group is output
                core_physical = device.worker_core_from_logical_core(core)
                output_core_physical_xs.append(core_physical.x)
                output_core_physical_ys.append(core_physical.y)

        # =========================================================================
        # Compile time args (matching C++ lines 732-841)
        # =========================================================================
        tilize_q = input_tensor_q.layout == ttnn.ROW_MAJOR_LAYOUT

        # Q chunk size bytes
        q_chunk_size_bytes = q_tiles * (num_q_heads * TILE_WIDTH * 2 if tilize_q else q_tile_size)

        # Reader compile time args (simplified for sharded Q/output)
        reader_compile_time_args = [
            B,  # 0
            PNHt,  # 1
            St,  # 2
            DHt,  # 3
            vDHt,  # 4
            Sk_chunk_t,  # 5
            num_active_cores,  # 6
            num_cores_per_batch,  # 7
            k_chunk_size,  # 8
            index_stick_size,  # 9
            num_kv_heads,  # 10
            Bkv,  # 11
            q_heads_parallel_factor,  # 12
            num_cores_per_head,  # 13
            num_heads_per_core,  # 14
            num_output_cores,  # 15
            max_dynamic_chunk_size,  # 16
            1 if tilize_q else 0,  # 17
            q_chunk_size_bytes,  # 18
        ]
        # TensorAccessorArgs for K, V (KV cache - can be interleaved or height-sharded), and pos tensor
        reader_compile_time_args.extend(get_tensor_accessor_args(kv_cache_tensor))  # K
        reader_compile_time_args.extend(get_tensor_accessor_args(kv_cache_tensor))  # V (same as K for MLA)
        reader_compile_time_args.extend(
            get_interleaved_tensor_accessor_args(cur_pos_tensor)
        )  # pos (always DRAM interleaved)

        # Writer compile time args (simplified for sharded output, num_kv_heads=1)
        writer_compile_time_args = [
            B,  # 0
            PNHt,  # 1
            St,  # 2
            vDHt,  # 3
            Sk_chunk_t,  # 4
            packed_identity_scalar,  # 5
            packed_zero_scalar,  # 6
            scale_uint32,  # 7
            num_cores_per_batch,  # 8
            num_active_cores,  # 9
            0,  # 10: reducer_semaphore_id
            k_chunk_size,  # 11
            num_cores_per_head,  # 12
            num_heads_per_core,  # 13
            num_reducer_cores,  # 14
            max_dynamic_chunk_size,  # 15
            q_heads_parallel_factor,  # 16
            Q_TILE_HEIGHT,  # 17: Q tile height for tiny tile support
        ]

        # Compute compile time args (simplified)
        compute_compile_time_args = [
            St,  # 0
            DHt,  # 1
            vDHt,  # 2
            PNHt,  # 3
            Sk_chunk_t,  # 4
            qk_in0_block_w,  # 5
            qk_out_subblock_w,  # 6
            qk_out_subblock_h,  # 7
            qk_in0_num_subblocks,  # 8
            qk_in1_num_subblocks,  # 9
            qk_num_blocks,  # 10
            out_in0_block_w,  # 11
            out_out_subblock_w,  # 12
            out_out_subblock_h,  # 13
            out_in0_num_subblocks,  # 14
            out_in1_num_subblocks,  # 15
            out_num_blocks,  # 16
            num_cores_per_batch,  # 17
            k_chunk_size,  # 18
            num_cores_per_head,  # 19
            num_heads_per_core,  # 20
            max_dynamic_chunk_size,  # 21
            1 if tilize_q else 0,  # 22
            q_heads_parallel_factor,  # 23
            Q_TILE_HEIGHT,  # 24: Q tile height for vector mode selection
            scale_uint32,  # 25
        ]

        # Compute defines (C++ lines 844-892)
        compute_defines = []
        if Sk_chunk_t > 0:
            sub_exp_granularity = min(Sk_chunk_t, dst_size)
            log2_sub_exp_granularity = int(math.log2(sub_exp_granularity))
            mul_bcast_granularity = min(PNHt * Sk_chunk_t, dst_size)
            log2_mul_bcast_granularity = int(math.log2(mul_bcast_granularity))
            stats_granularity = min(Sk_chunk_t, dst_size)
            log2_stats_granularity = int(math.log2(stats_granularity))

            compute_defines.extend(
                [
                    ("SUB_EXP_GRANULARITY", str(sub_exp_granularity)),
                    ("LOG2_SUB_EXP_GRANULARITY", str(log2_sub_exp_granularity)),
                    ("MUL_BCAST_GRANULARITY", str(mul_bcast_granularity)),
                    ("LOG2_MUL_BCAST_GRANULARITY", str(log2_mul_bcast_granularity)),
                    ("STATS_GRANULARITY", str(stats_granularity)),
                    ("LOG2_STATS_GRANULARITY", str(log2_stats_granularity)),
                ]
            )
        else:
            compute_defines.append(("DYNAMIC_CHUNK_SIZE", "1"))

        compute_defines.extend(
            [
                ("EXP_APPROX_MODE", "1" if exp_approx_mode else "0"),
                ("DHT_GRANULARITY", str(dht_granularity)),
                ("LOG2_DHT_GRANULARITY", str(log2_dht_granularity)),
            ]
        )

        # =========================================================================
        # Create CB descriptors (matching C++ lines 475-655)
        # =========================================================================
        cb_descriptors = []

        # c_0: Q input (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=q_tiles * q_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(0, q_df, q_tile_size, q_tile_descriptor)],
            )
        )

        # c_1: K input (full tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=k_tiles * k_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(1, k_df, k_tile_size)],
            )
        )

        # c_2: V input (full tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=v_tiles * v_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(2, v_df, v_tile_size)],
            )
        )

        # c_3: attn_mask input (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=qk_tiles * mask_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(3, im_df, mask_tile_size, mask_tile_descriptor)],
            )
        )

        # c_5: identity scale input (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=scale_tiles * scalar_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(5, scalar_df, scalar_tile_size, scalar_tile_descriptor)],
            )
        )

        # c_6: cb_m_in (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(6, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_7: cb_l_in (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(7, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_8: cur_pos input
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=index_stick_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(8, ttnn.int32, index_stick_size)],
            )
        )

        # c_10: tilized Q input (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=q_tiles * q_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(10, q_df, q_tile_size, q_tile_descriptor)],
            )
        )

        # c_11: cb_col_identity (full tile - always 32x32)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=scale_tiles * col_identity_tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(11, scalar_df, col_identity_tile_size, full_tile_descriptor)
                ],
            )
        )

        # c_12: cb zero config (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=scale_tiles * scalar_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(12, scalar_df, scalar_tile_size, scalar_tile_descriptor)],
            )
        )

        # c_24: cb_qk_im (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=qk_tiles * im_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(24, im_df, im_tile_size, im_tile_descriptor)],
            )
        )

        # c_25: cb_out_im (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=out_im_tiles * im_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(25, im_df, im_tile_size, im_tile_descriptor)],
            )
        )

        # c_26: cb_out_accumulate_im (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=out_im_tiles * im_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(26, im_df, im_tile_size, im_tile_descriptor)],
            )
        )

        # c_27: cb_cur_max (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(27, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_28: cb_prev_max (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(28, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_29: cb_cur_sum (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(29, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_30: cb_prev_sum (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(30, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_31: cb_exp_max_diff (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(31, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_21: cb_prev_sum_2 (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(21, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_22: cb_exp_max_diff_2 (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(22, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_23: cb_out_accumulate_im_2 (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=out_im_tiles * im_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(23, im_df, im_tile_size, im_tile_descriptor)],
            )
        )

        # c_16: cb_out_o (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=out0_t * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(16, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_17: cb_out_m (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(17, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_18: cb_out_l (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(18, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_19: cb_intermed_out (tiny tile, only if intermed_output_tiles > 0)
        if intermed_output_tiles > 0:
            cb_descriptors.append(
                ttnn.CBDescriptor(
                    total_size=intermed_output_tiles * stats_tile_size,
                    core_ranges=core_grid,
                    format_descriptors=[ttnn.CBFormatDescriptor(19, stats_df, stats_tile_size, stats_tile_descriptor)],
                )
            )

        # c_20: cb_out_final (always sharded output)
        cb_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(20, output_tensor)
        cb_descriptors.append(cb_out_descriptor)

        # =========================================================================
        # Create semaphore descriptors (matching C++ lines 724-725)
        # =========================================================================
        semaphore_descriptors = [
            ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, core_grid, 0),  # reducer_semaphore
            ttnn.SemaphoreDescriptor(1, ttnn.CoreType.WORKER, core_grid, 0),  # output_semaphore
        ]

        # =========================================================================
        # Create kernel descriptors (matching C++ lines 894-918)
        # =========================================================================
        # Addresses for runtime args
        q_addr = q_tensor.buffer_address()
        k_addr = kv_cache_tensor.buffer_address()
        v_addr = k_addr  # MLA: V is same as K
        pos_addr = cur_pos_tensor.buffer_address()
        out_addr = output_tensor.buffer_address()

        # Create single RuntimeArgs objects for all cores (matching C++ pattern)
        reader_rtargs = ttnn.RuntimeArgs()
        writer_rtargs = ttnn.RuntimeArgs()
        compute_rtargs = ttnn.RuntimeArgs()

        # Add runtime args for active cores (matching C++ lines 929-1016)
        for i in range(num_active_cores):
            core = core_group[i]

            # Calculate per-core values (matching C++ lines 932-954)
            worker_id_for_reduce = (i % num_cores_per_head) - 1
            worker_id_for_output = (i % num_cores_per_batch) - 1
            # Handle unsigned comparison: -1 becomes 0xFFFFFFFF
            do_reduce = 1 if (worker_id_for_reduce & 0xFFFFFFFF) == 0xFFFFFFFF else 0
            do_output = 1 if (worker_id_for_output & 0xFFFFFFFF) == 0xFFFFFFFF else 0

            cur_head = (i % num_cores_per_batch) // num_cores_per_head
            cur_batch = i // num_cores_per_batch
            core_num_in_reduce = i % num_cores_per_head
            core_num_in_output = i % num_cores_per_batch

            cur_pos = 0xFFFFFFFF  # -1 in unsigned, means use cur_pos_tensor

            # Reader runtime args (simplified)
            reader_runtime_args = [
                q_addr,
                k_addr,
                v_addr,
                pos_addr,
                do_reduce,
                do_output,
                cur_head,
                cur_batch,
                core_num_in_reduce,
                core_num_in_output,
                cur_pos,
            ]
            reader_runtime_args.extend(output_core_physical_xs)
            reader_runtime_args.extend(output_core_physical_ys)

            # Writer runtime args (simplified, num_kv_heads=1)
            writer_runtime_args = [
                out_addr,
                worker_id_for_reduce & 0xFFFFFFFF,  # unsigned
                do_reduce,
                cur_head,
                cur_batch,
                core_num_in_reduce,
                cur_pos,
            ]
            writer_runtime_args.extend(reduce_core_physical_xs)
            writer_runtime_args.extend(reduce_core_physical_ys)

            # Compute runtime args (matching C++ lines 994-995)
            compute_runtime_args = [
                do_reduce,
                do_output,
                cur_head,
                cur_batch,
                core_num_in_reduce,
                core_num_in_output,
                cur_pos,
            ]

            # Append to single RuntimeArgs objects
            reader_rtargs.append(core, reader_runtime_args)
            writer_rtargs.append(core, writer_runtime_args)
            compute_rtargs.append(core, compute_runtime_args)

        # Add runtime args for idle cores
        for core in core_group_idle:
            # Idle cores get zero/placeholder runtime args
            idle_reader_runtime_args = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            idle_reader_runtime_args.extend([0] * len(output_core_physical_xs))
            idle_reader_runtime_args.extend([0] * len(output_core_physical_ys))

            idle_writer_runtime_args = [0, 0, 0, 0, 0, 0, 0]
            idle_writer_runtime_args.extend([0] * len(reduce_core_physical_xs))
            idle_writer_runtime_args.extend([0] * len(reduce_core_physical_ys))

            # Compute idle runtime args (do_reduce=65 signals idle)
            idle_compute_runtime_args = [65, 0, 0, 0, 0, 0, 0]

            reader_rtargs.append(core, idle_reader_runtime_args)
            writer_rtargs.append(core, idle_writer_runtime_args)
            compute_rtargs.append(core, idle_compute_runtime_args)

        # Create 3 kernel descriptors covering ALL cores (not per-core)
        kernel_descriptors = [
            # Reader kernel (use NOC_0 explicitly for DRAM reads)
            ttnn.KernelDescriptor(
                kernel_source="models/demos/deepseek_v3_b1/micro_ops/flash_mla/kernels/dataflow/reader_decode_all.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=reader_compile_time_args,
                runtime_args=reader_rtargs,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,
                    noc=ttnn.NOC.NOC_0,
                ),
            ),
            # Writer kernel (use NOC_1 to avoid conflict with reader on NOC_0)
            ttnn.KernelDescriptor(
                kernel_source="models/demos/deepseek_v3_b1/micro_ops/flash_mla/kernels/dataflow/writer_decode_all.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=writer_compile_time_args,
                runtime_args=writer_rtargs,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1,
                    noc=ttnn.NOC.NOC_1,
                ),
            ),
            # Compute kernel
            ttnn.KernelDescriptor(
                kernel_source="models/demos/deepseek_v3_b1/micro_ops/flash_mla/kernels/compute/sdpa_flash_decode.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=compute_compile_time_args,
                runtime_args=compute_rtargs,
                defines=compute_defines,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=math_fidelity,
                    fp32_dest_acc_en=fp32_dest_acc_en,
                    math_approx_mode=math_approx_mode,
                ),
            ),
        ]

        # =========================================================================
        # Create and execute program
        # =========================================================================
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=kernel_descriptors,
            cbs=cb_descriptors,
            semaphores=semaphore_descriptors,
        )

        io_tensors = [q_tensor, kv_cache_tensor, cur_pos_tensor, output_tensor]
        ttnn.generic_op(io_tensors, program_descriptor)

        return output_tensor
