/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cudf/types.hpp>

#include <cuda/std/algorithm>

namespace cudf {
namespace detail {

struct byte_gather_params {
  cudf::size_type const* __restrict__ indices      = nullptr;
  void const* const __restrict__* __restrict__ src = nullptr;
  void* const __restrict__* __restrict__ dst       = nullptr;
  cudf::size_type size                             = 0;
  cudf::size_type num_b8                           = 0;
  cudf::size_type num_b16                          = 0;
  cudf::size_type num_b32                          = 0;
  cudf::size_type num_b64                          = 0;
  cudf::size_type num_b128                         = 0;
};

struct bit_gather_params {
  cudf::size_type const* __restrict__ indices                    = nullptr;
  cudf::bitmask_type const* __restrict__ const* __restrict__ src = nullptr;
  cudf::size_type const* __restrict__ offsets                    = nullptr;
  cudf::bitmask_type* __restrict__ const* __restrict__ dst       = nullptr;
  cudf::size_type* __restrict__ valid_counts                     = nullptr;
  cudf::size_type size                                           = 0;
  cudf::size_type num_srcs                                       = 0;
};

/// @brief a size-based type generic wide gather kernel for byte-addressable types
__global__ void wide_byte_gather_kernel_generic(byte_gather_params params)
{
  auto const tid = (std::int64_t)blockIdx.x * (std::int64_t)blockDim.x + (std::int64_t)threadIdx.x;
  auto const stride = (std::int64_t)blockDim.x * (std::int64_t)gridDim.x;

  for (cudf::size_type i = tid; i < params.size; i += stride) {
    auto gather_index = params.indices[i];
    auto dst_index    = i;

    cudf::size_type column = 0;

    for (cudf::size_type c = 0; c < params.num_b8; c++) {
      static_cast<std::uint8_t*>(params.dst[column + c])[dst_index] =
        static_cast<std::uint8_t const*>(params.src[column + c])[gather_index];
    }

    column += params.num_b8;

    for (cudf::size_type c = 0; c < params.num_b16; c++) {
      static_cast<std::uint16_t*>(params.dst[column + c])[dst_index] =
        static_cast<std::uint16_t const*>(params.src[column + c])[gather_index];
    }

    column += params.num_b16;

    for (cudf::size_type c = 0; c < params.num_b32; c++) {
      static_cast<std::uint32_t*>(params.dst[column + c])[dst_index] =
        static_cast<std::uint32_t const*>(params.src[column + c])[gather_index];
    }

    column += params.num_b32;

    for (cudf::size_type c = 0; c < params.num_b64; c++) {
      static_cast<std::uint64_t*>(params.dst[column + c])[dst_index] =
        static_cast<std::uint64_t const*>(params.src[column + c])[gather_index];
    }

    column += params.num_b64;

    for (cudf::size_type c = 0; c < params.num_b128; c++) {
      static_cast<unsigned __int128*>(params.dst[column + c])[dst_index] =
        static_cast<unsigned __int128 const*>(params.src[column + c])[gather_index];
    }

    column += params.num_b128;
  }
}

/// @brief a generic wide gather kernel for bitmask types
__global__ void wide_bit_gather_kernel_generic(bit_gather_params params)
{
  auto const tid = (std::int64_t)blockIdx.x * (std::int64_t)blockDim.x + (std::int64_t)threadIdx.x;
  auto const stride = (std::int64_t)blockDim.x * (std::int64_t)gridDim.x;

  constexpr cudf::size_type BITS_PER_WORD     = sizeof(cudf::bitmask_type) * 8;
  constexpr cudf::size_type COLUMN_BATCH_SIZE = 16;

  cudf::size_type const num_words = (params.size + BITS_PER_WORD - 1) / BITS_PER_WORD;

  cudf::size_type indices[BITS_PER_WORD];

  for (cudf::size_type columns_begin = 0; columns_begin < params.num_srcs;
       columns_begin += COLUMN_BATCH_SIZE) {
    cudf::size_type thread_valid_counts[COLUMN_BATCH_SIZE] = {};

    cudf::size_type const columns_end =
      cuda::std::min(columns_begin + COLUMN_BATCH_SIZE, params.num_srcs);
    cudf::size_type const num_batch_columns = columns_end - columns_begin;

    for (cudf::size_type i = tid; i < num_words; i += stride) {
      auto const begin_bit_index = i * BITS_PER_WORD;

      cudf::size_type num_indices = 0;
      for (num_indices = 0;
           num_indices < BITS_PER_WORD && (begin_bit_index + num_indices) < params.size;
           num_indices++) {
        indices[num_indices] = params.indices[begin_bit_index + num_indices];
      }

      for (cudf::size_type batch_column = 0; batch_column < num_batch_columns; batch_column++) {
        cudf::bitmask_type bitword  = 0;
        cudf::size_type valid_count = 0;

        for (cudf::size_type bit_index = 0; bit_index < BITS_PER_WORD && bit_index < num_indices;
             bit_index++) {
          auto gather_index = indices[bit_index] + params.offsets[columns_begin + batch_column];
          auto bit          = cudf::bitmask_type{1} &
                     (params.src[columns_begin + batch_column][gather_index / BITS_PER_WORD] >>
                      (gather_index % BITS_PER_WORD));
          bitword |= (bit << bit_index);
          valid_count += static_cast<cudf::size_type>(bit);
        }

        params.dst[columns_begin + batch_column][i] = bitword;

        thread_valid_counts[batch_column] += valid_count;
      }
    }

    for (cudf::size_type batch_column = 0; batch_column < num_batch_columns; batch_column++) {
      std::uint32_t mask = __activemask();

      cudf::size_type warp_total = thread_valid_counts[batch_column];

      // warp-reduction
      for (cudf::size_type num_warp_sums = warpSize / 2; num_warp_sums > 0; num_warp_sums /= 2) {
        warp_total += __shfl_down_sync(mask, warp_total, num_warp_sums);
      }

      // global-reduction
      auto lane = threadIdx.x % warpSize;
      if (lane == 0) { atomicAdd(params.valid_counts + columns_begin + batch_column, warp_total); }
    }
  }
}

}  // namespace detail
}  // namespace cudf
