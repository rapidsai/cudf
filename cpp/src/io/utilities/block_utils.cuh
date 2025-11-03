/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/detail/utilities/cuda.cuh>

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>

namespace cudf {
namespace io {

template <typename T>
inline __device__ T shuffle(T var, int lane = 0)
{
  return __shfl_sync(~0, var, lane);
}

template <typename T>
inline __device__ T shuffle_xor(T var, uint32_t delta)
{
  return __shfl_xor_sync(~0, var, delta);
}

inline __device__ uint32_t ballot(int pred) { return __ballot_sync(~0, pred); }

// Warp reduction helpers
template <cudf::size_type size, typename T>
inline __device__ T warp_reduce_or(T acc)
{
  static_assert(size >= 1 and size <= cudf::detail::warp_size and (size & (size - 1)) == 0,
                "Size must be a power of 2 and less than or equal to the warp size");
  if constexpr (size == 1) {
    return acc;
  } else {
    acc = warp_reduce_or<size / 2>(acc);
    return acc | shuffle_xor(acc, size / 2);
  }
}

template <typename T>
inline __device__ T WarpReducePos2(T pos, uint32_t t)
{
  T tmp = shuffle(pos, t & 0x1e);
  pos += (t & 1) ? tmp : 0;
  return pos;
}
template <typename T>
inline __device__ T WarpReducePos4(T pos, uint32_t t)
{
  T tmp;
  pos = WarpReducePos2(pos, t);
  tmp = shuffle(pos, (t & 0x1c) | 1);
  pos += (t & 2) ? tmp : 0;
  return pos;
}
template <typename T>
inline __device__ T WarpReducePos8(T pos, uint32_t t)
{
  T tmp;
  pos = WarpReducePos4(pos, t);
  tmp = shuffle(pos, (t & 0x18) | 3);
  pos += (t & 4) ? tmp : 0;
  return pos;
}
template <typename T>
inline __device__ T WarpReducePos16(T pos, uint32_t t)
{
  T tmp;
  pos = WarpReducePos8(pos, t);
  tmp = shuffle(pos, (t & 0x10) | 7);
  pos += (t & 8) ? tmp : 0;
  return pos;
}
template <typename T>
inline __device__ T WarpReducePos32(T pos, uint32_t t)
{
  T tmp;
  pos = WarpReducePos16(pos, t);
  tmp = shuffle(pos, 0xf);
  pos += (t & 16) ? tmp : 0;
  return pos;
}

inline __device__ double Int128ToDouble_rn(uint64_t lo, int64_t hi)
{
  double sign;
  if (hi < 0) {
    sign = -1.0;
    lo   = (~lo) + 1;
    hi   = (~hi) + (lo == 0);
  } else {
    sign = 1.0;
  }
  return sign * __fma_rn(__ll2double_rn(hi), 4294967296.0 * 4294967296.0, __ull2double_rn(lo));
}

template <typename T>
  requires(cuda::std::is_same_v<T, uint32_t> or cuda::std::is_same_v<T, uint64_t>)
inline __device__ T unaligned_load(uint8_t const* p)
{
  T value;
  cuda::std::memcpy(&value, p, sizeof(T));
  return value;
}

template <uint32_t nthreads, bool sync_before_store>
inline __device__ void memcpy_block(void* dstv,
                                    void const* srcv,
                                    uint32_t len,
                                    cooperative_groups::thread_block const& block)
{
  static_assert(
    nthreads >= sizeof(uint32_t),
    "The kernel block size (nthreads) must be greater than or equal to the size of uint32_t");
  auto const t    = block.thread_rank();
  auto* dst       = static_cast<uint8_t*>(dstv);
  auto const* src = static_cast<uint8_t const*>(srcv);
  // Align output to 32-bit
  auto const dst_align_bytes = static_cast<uint32_t>(0x3 & -reinterpret_cast<intptr_t>(dst));
  if (dst_align_bytes != 0) {
    auto const align_len = cuda::std::min<uint32_t>(dst_align_bytes, len);
    uint8_t byte;
    if (t < align_len) { byte = src[t]; }
    if constexpr (sync_before_store) { block.sync(); }
    if (t < align_len) { dst[t] = byte; }
    src += align_len;
    dst += align_len;
    len -= align_len;
  }
  // Copy 32-bit chunks
  while (len >= sizeof(uint32_t)) {
    auto const copy_cnt = cuda::std::min<uint32_t>(len / sizeof(uint32_t), nthreads);
    uint32_t value;
    if (t < copy_cnt) { value = unaligned_load<uint32_t>(src + (t * sizeof(uint32_t))); }
    if constexpr (sync_before_store) { block.sync(); }
    if (t < copy_cnt) { reinterpret_cast<uint32_t*>(dst)[t] = value; }
    src += copy_cnt * sizeof(uint32_t);
    dst += copy_cnt * sizeof(uint32_t);
    len -= copy_cnt * sizeof(uint32_t);
  }
  // Copy the remaining bytes
  if (len != 0) {
    uint8_t byte;
    if (t < len) { byte = src[t]; }
    if constexpr (sync_before_store) { block.sync(); }
    if (t < len) { dst[t] = byte; }
  }
}

}  // namespace io
}  // namespace cudf
