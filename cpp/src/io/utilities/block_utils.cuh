/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
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

inline __device__ void syncwarp() { __syncwarp(); }

inline __device__ uint32_t ballot(int pred) { return __ballot_sync(~0, pred); }

// Warp reduction helpers
template <typename T>
inline __device__ T WarpReduceOr2(T acc)
{
  return acc | shuffle_xor(acc, 1);
}
template <typename T>
inline __device__ T WarpReduceOr4(T acc)
{
  acc = WarpReduceOr2(acc);
  return acc | shuffle_xor(acc, 2);
}
template <typename T>
inline __device__ T WarpReduceOr8(T acc)
{
  acc = WarpReduceOr4(acc);
  return acc | shuffle_xor(acc, 4);
}
template <typename T>
inline __device__ T WarpReduceOr16(T acc)
{
  acc = WarpReduceOr8(acc);
  return acc | shuffle_xor(acc, 8);
}
template <typename T>
inline __device__ T WarpReduceOr32(T acc)
{
  acc = WarpReduceOr16(acc);
  return acc | shuffle_xor(acc, 16);
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

inline __device__ uint32_t unaligned_load32(uint8_t const* p)
{
  uint32_t value;
  cuda::std::memcpy(&value, p, sizeof(uint32_t));
  return value;
}

inline __device__ uint64_t unaligned_load64(uint8_t const* p)
{
  uint64_t value;
  cuda::std::memcpy(&value, p, sizeof(uint64_t));
  return value;
}

template <unsigned int nthreads, bool sync_before_store>
inline __device__ void memcpy_block(void* dstv, void const* srcv, uint32_t len, uint32_t t)
{
  auto* dst       = static_cast<uint8_t*>(dstv);
  auto const* src = static_cast<uint8_t const*>(srcv);
  uint32_t dst_align_bytes, src_align_bytes, src_align_bits;
  // Align output to 32-bit
  dst_align_bytes = 3 & -reinterpret_cast<intptr_t>(dst);
  if (dst_align_bytes != 0) {
    uint32_t align_len = min(dst_align_bytes, len);
    uint8_t b;
    if (t < align_len) { b = src[t]; }
    if constexpr (sync_before_store) { __syncthreads(); }
    if (t < align_len) { dst[t] = b; }
    src += align_len;
    dst += align_len;
    len -= align_len;
  }
  src_align_bytes = (uint32_t)(3 & reinterpret_cast<uintptr_t>(src));
  src_align_bits  = src_align_bytes * 8;
  while (len >= 4) {
    auto const* src32 = reinterpret_cast<uint32_t const*>(src - src_align_bytes);
    uint32_t copy_cnt = min(len >> 2, nthreads);
    uint32_t v;
    if (t < copy_cnt) {
      v = src32[t];
      if (src_align_bits != 0) { v = __funnelshift_r(v, src32[t + 1], src_align_bits); }
    }
    if constexpr (sync_before_store) { __syncthreads(); }
    if (t < copy_cnt) { reinterpret_cast<uint32_t*>(dst)[t] = v; }
    src += copy_cnt * 4;
    dst += copy_cnt * 4;
    len -= copy_cnt * 4;
  }
  if (len != 0) {
    uint8_t b;
    if (t < len) { b = src[t]; }
    if constexpr (sync_before_store) { __syncthreads(); }
    if (t < len) { dst[t] = b; }
  }
}

}  // namespace io
}  // namespace cudf
