// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <vector>

#include "../rt_args_common.hpp"

/******************************************************************************
 *                   Helper Functions                                          *
 ******************************************************************************/
template <uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

template <
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t barrier_threshold,
    uint32_t cb_k_in,
    uint32_t cb_v_in,
    typename KReaderType>
void read_kv_chunks(
    uint32_t k_chunk_start,
    uint32_t k_chunk_end,
    uint32_t k_start_tile_id,
    uint32_t Sk_chunk_t,
    uint32_t k_chunk_tiles,
    uint32_t v_chunk_tiles,
    const KReaderType& k_reader,
    uint32_t k_tile_bytes,
    uint32_t v_tile_bytes) {
    uint32_t barrier_count = 0;
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        // Read K chunk transposed
        uint64_t k_base_read_ptr;
        {
            DeviceZoneScopedN("reader-k-read");
            cb_reserve_back(cb_k_in, k_chunk_tiles);
            uint32_t k_write_ptr = get_write_ptr(cb_k_in);
            k_base_read_ptr = get_noc_addr(k_write_ptr);
            barrier_count = 0;
            for (uint32_t col = 0; col < DHt; ++col) {
                uint32_t k_tile_id = k_start_tile_id + col;
                for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                    noc_async_read_tile(k_tile_id, k_reader, k_write_ptr);
                    if (++barrier_count == barrier_threshold) {
                        noc_async_read_barrier();
                        barrier_count = 0;
                    }
                    k_tile_id += DHt;
                    k_write_ptr += k_tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_k_in, k_chunk_tiles);
        }

        // Read V chunk (transpose of K), from K's L1 buffer (MLA always reuses K for V)
        {
            DeviceZoneScopedN("reader-v-read");
            cb_reserve_back(cb_v_in, v_chunk_tiles);
            uint32_t v_write_ptr = get_write_ptr(cb_v_in);
            uint64_t k_read_ptr = k_base_read_ptr;
            for (uint32_t row = 0; row < Sk_chunk_t; ++row) {       // Row of V
                k_read_ptr = k_base_read_ptr + row * k_tile_bytes;  // Increment across K's Col

                for (uint32_t col = 0; col < vDHt; ++col) {  // Col of V
                    noc_async_read(k_read_ptr, v_write_ptr, v_tile_bytes);

                    v_write_ptr += v_tile_bytes;
                    k_read_ptr += Sk_chunk_t * k_tile_bytes;  // Stride across K's width
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_v_in, v_chunk_tiles);
        }

        // Update the starting tile id for next iteration
        k_start_tile_id += k_chunk_tiles;
    }
}

/******************************************************************************
 *                   Kernel Main                                               *
 ******************************************************************************/
void kernel_main() {
    /*
    Simplified Flash MLA Decode reader kernel.
    Q is always sharded, KV cache is always in DRAM interleaved.
    */
    constexpr uint32_t B = get_compile_time_arg_val(0);           // batch size
    constexpr uint32_t PNHt = get_compile_time_arg_val(1);        // padded number of heads in tiles
    constexpr uint32_t St = get_compile_time_arg_val(2);          // full sequence length of kv cache in tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(3);         // head dim
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);        // head dim of V
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);  // number of tiles in seqlen of a k/v/mask chunk
    constexpr uint32_t num_cores = get_compile_time_arg_val(6);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(7);
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(8);
    constexpr uint32_t index_stick_size_B = get_compile_time_arg_val(9);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(10);
    constexpr uint32_t Bkv = get_compile_time_arg_val(11);
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(12);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(13);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(14);
    constexpr uint32_t num_output_cores = get_compile_time_arg_val(15);
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(16);
    constexpr bool tilize_q = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t q_chunk_size_bytes = get_compile_time_arg_val(18);

    // TensorAccessorArgs for K and V (KV cache in DRAM), and pos tensor
    constexpr auto k_args = TensorAccessorArgs<19>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    uint32_t arg_idx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pos_addr = get_arg_val<uint32_t>(arg_idx++);
    const bool is_worker = get_arg_val<uint32_t>(arg_idx++) == 0;
    const bool is_output_core = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_head_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_output = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_pos_arg = get_arg_val<uint32_t>(arg_idx++);

    // idle core
    if (q_addr == 0) {
        return;
    }

    // Get cur_pos (MLA decode is always causal)
    uint32_t cur_pos;
    // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list
    if (cur_pos_arg != UINT32_MAX) {
        cur_pos = cur_pos_arg;
    } else {
        constexpr uint32_t cb_index_id = tt::CBIndex::c_8;
        cb_reserve_back(cb_index_id, 1);
        uint32_t index_cb_wr_ptr = get_write_ptr(cb_index_id);

        // Read cur_pos tensor from DRAM interleaved
        const auto pos_reader = TensorAccessor(pos_args, pos_addr, index_stick_size_B);
        uint64_t tensor_index_noc_addr = pos_reader.get_noc_addr(0);
        noc_async_read(tensor_index_noc_addr, index_cb_wr_ptr, index_stick_size_B);
        noc_async_read_barrier();

        cb_push_back(cb_index_id, 1);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);
        cur_pos = index_ptr[cur_batch / q_heads_parallel_factor];
    }

    if (cur_pos == UINT32_MAX) {
        // cur_pos of -1 indicates that the user should be skipped
        return;
    }

    auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(cur_pos);
    auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;

    // Sequence length assignment (no sliding window for MLA)
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] = get_runtime_args(
        cur_pos, cur_batch, core_num_in_reduce, num_cores_per_head, k_chunk_size_dynamic, std::nullopt);

    if (k_chunk_start == k_chunk_end) {
        return;  // early exit because no computes needs to be done
    }

    tt_l1_ptr uint32_t* all_output_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_output_cores;
    tt_l1_ptr uint32_t* all_output_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx++));

    uint32_t output_core_noc_x = all_output_noc_x[cur_batch];
    uint32_t output_core_noc_y = all_output_noc_y[cur_batch];

    constexpr uint32_t q_chunk_tiles = PNHt * DHt;
    uint32_t k_chunk_tiles = Sk_chunk_t_dynamic * DHt;
    uint32_t v_chunk_tiles = Sk_chunk_t_dynamic * vDHt;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_q_rm = tt::CBIndex::c_10;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();

    // Read Q from sharded memory (Q is always sharded)
    {
        DeviceZoneScopedN("reader-q-read");
        uint64_t q_read_addr;
        uint32_t q_write_ptr;
        if (is_output_core) {
            q_read_addr = get_noc_addr(q_addr);
        } else {
            q_read_addr = get_noc_addr(output_core_noc_x, output_core_noc_y, q_addr);
        }
        if constexpr (tilize_q) {
            cb_reserve_back(cb_q_rm, q_chunk_tiles);
            q_write_ptr = get_write_ptr(cb_q_rm);
        } else {
            cb_reserve_back(cb_q_in, q_chunk_tiles);
            q_write_ptr = get_write_ptr(cb_q_in);
        }
        // Q tensor is properly set up with tiny tiles, just read contiguously
        noc_async_read(q_read_addr, q_write_ptr, q_chunk_size_bytes);
        noc_async_read_barrier();
        // DPRINT << TileSlice(cb_q_in, 0, SliceRange{.h0 = 0, .h1 = 8, .hs = 1, .w0 = 0, .w1 = 32, .ws = 8}, true,
        // true) << ENDL();
        if constexpr (tilize_q) {
            cb_push_back(cb_q_rm, q_chunk_tiles);
        } else {
            cb_push_back(cb_q_in, q_chunk_tiles);
        }
    }

    // Create KV cache reader (DRAM interleaved)
    const auto k_reader = TensorAccessor(k_args, k_addr, k_tile_bytes);

    for (uint32_t cur_head = cur_head_group * num_heads_per_core;
         cur_head < cur_head_group * num_heads_per_core + num_heads_per_core;
         ++cur_head) {
        // Offset for current batch (non-paged attention)
        const uint32_t k_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bkv) * num_kv_heads * St * DHt;
        const uint32_t k_head_offset = cur_head * St * DHt;

        // Read K, V chunks
        const uint32_t k_chunk_offset = k_chunk_start * Sk_chunk_t_dynamic * DHt;
        uint32_t k_start_tile_id = k_batch_offset + k_head_offset + k_chunk_offset;

        read_kv_chunks<DHt, vDHt, barrier_threshold, cb_k_in, cb_v_in>(
            k_chunk_start,
            k_chunk_end,
            k_start_tile_id,
            Sk_chunk_t_dynamic,
            k_chunk_tiles,
            v_chunk_tiles,
            k_reader,
            k_tile_bytes,
            v_tile_bytes);
    }
}
