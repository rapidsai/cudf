/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/types.hpp>

#include <cuda/std/algorithm>

namespace CUDF_EXPORT cudf {
namespace transformation {
namespace jit {

/// @param total Pointer to global memory to accumulate the total. Must be initialized to zero.
/// @param thread_total The per-thread total to add to the global total.
__device__ void device_reduce_sum(cudf::size_type* total, cudf::size_type thread_total)
{
  static_assert(sizeof(cudf::size_type) <= sizeof(unsigned int));

  cudf::size_type warp_total = thread_total;

  auto participation_mask = __activemask();

  for (int num_warp_sums = 16; num_warp_sums > 0; num_warp_sums /= 2) {
    warp_total += __shfl_down_sync(participation_mask, warp_total, num_warp_sums);
  }

  if (threadIdx.x == 0) { atomicAdd(total, warp_total); }
}

/// @brief Compute the null bitmask of chunks (word-sized) from a boolean source array.
/// @param src The source boolean array
/// @param word_chunk_start The starting word chunk index to process
/// @param num_word_chunks The number of word chunks to process
/// @param dst_word The output bitmask word
/// @return The number of valid (set) bits written to the null mask
__device__ cudf::size_type bools_to_bits_chunk(bool const* __restrict__ src,
                                               cudf::size_type src_size,
                                               cudf::size_type word_chunk,
                                               cudf::bitmask_type* __restrict__ dst_word)
{
  static_assert(sizeof(cudf::bitmask_type) <= sizeof(unsigned int));

  static constexpr auto num_word_bits =
    static_cast<cudf::size_type>(sizeof(cudf::bitmask_type) * 8);

  auto bit_start = word_chunk * num_word_bits;
  auto bit_end   = cuda::std::min(bit_start + num_word_bits, src_size);

  cudf::bitmask_type out_word = 0;
  for (auto b = bit_start; b < bit_end; b++) {
    auto bit_pos = (b % num_word_bits);
    auto bits    = (src[b] ? cudf::bitmask_type{1} : cudf::bitmask_type{0}) << bit_pos;
    out_word |= bits;
  }

  *dst_word = out_word;

  return __popc(out_word);
}

__device__ void boolean_mask_to_nullmask_subkernel(bool const* __restrict__ src,
                                                   cudf::size_type src_size,
                                                   cudf::bitmask_type* __restrict__ dst,
                                                   cudf::size_type* __restrict__ valid_count)
{
  constexpr auto num_word_bits       = static_cast<cudf::size_type>(sizeof(cudf::bitmask_type) * 8);
  auto num_chunks                    = (src_size + (num_word_bits - 1)) / num_word_bits;
  cudf::size_type thread_valid_count = 0;

  auto i = static_cast<int64_t>(threadIdx.x) +
           static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x);
  auto stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);

  for (; i < num_chunks; i += stride) {
    thread_valid_count += bools_to_bits_chunk(src, src_size, i, &dst[i]);
  }

  device_reduce_sum(valid_count, thread_valid_count);
}

}  // namespace jit
}  // namespace transformation
}  // namespace CUDF_EXPORT cudf
