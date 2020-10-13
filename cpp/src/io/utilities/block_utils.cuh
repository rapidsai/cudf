/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <stdint.h>

namespace cudf {
namespace io {
#if (__CUDACC_VER_MAJOR__ >= 9)
#define SHFL0(v) __shfl_sync(~0, v, 0)
#define SHFL(v, t) __shfl_sync(~0, v, t)
#define SHFL_XOR(v, m) __shfl_xor_sync(~0, v, m)
#define SYNCWARP() __syncwarp()
#define BALLOT(v) __ballot_sync(~0, v)
#else
#define SHFL0(v) __shfl(v, 0)
#define SHFL(v, t) __shfl(v, t)
#define SHFL_XOR(v, m) __shfl_xor(v, m)
#define SYNCWARP()
#define BALLOT(v) __ballot(v)
#endif

#if (__CUDA_ARCH__ >= 700)
#define NANOSLEEP(d) __nanosleep(d)
#else
#define NANOSLEEP(d) clock()
#endif

// Warp reduction helpers
template <typename T>
inline __device__ T WarpReduceSum2(T acc)
{
  return acc + SHFL_XOR(acc, 1);
}
template <typename T>
inline __device__ T WarpReduceSum4(T acc)
{
  acc = WarpReduceSum2(acc);
  return acc + SHFL_XOR(acc, 2);
}
template <typename T>
inline __device__ T WarpReduceSum8(T acc)
{
  acc = WarpReduceSum4(acc);
  return acc + SHFL_XOR(acc, 4);
}
template <typename T>
inline __device__ T WarpReduceSum16(T acc)
{
  acc = WarpReduceSum8(acc);
  return acc + SHFL_XOR(acc, 8);
}
template <typename T>
inline __device__ T WarpReduceSum32(T acc)
{
  acc = WarpReduceSum16(acc);
  return acc + SHFL_XOR(acc, 16);
}

template <typename T>
inline __device__ T WarpReduceOr2(T acc)
{
  return acc | SHFL_XOR(acc, 1);
}
template <typename T>
inline __device__ T WarpReduceOr4(T acc)
{
  acc = WarpReduceOr2(acc);
  return acc | SHFL_XOR(acc, 2);
}
template <typename T>
inline __device__ T WarpReduceOr8(T acc)
{
  acc = WarpReduceOr4(acc);
  return acc | SHFL_XOR(acc, 4);
}
template <typename T>
inline __device__ T WarpReduceOr16(T acc)
{
  acc = WarpReduceOr8(acc);
  return acc | SHFL_XOR(acc, 8);
}
template <typename T>
inline __device__ T WarpReduceOr32(T acc)
{
  acc = WarpReduceOr16(acc);
  return acc | SHFL_XOR(acc, 16);
}

template <typename T>
inline __device__ T WarpReducePos2(T pos, uint32_t t)
{
  T tmp = SHFL(pos, t & 0x1e);
  pos += (t & 1) ? tmp : 0;
  return pos;
}
template <typename T>
inline __device__ T WarpReducePos4(T pos, uint32_t t)
{
  T tmp;
  pos = WarpReducePos2(pos, t);
  tmp = SHFL(pos, (t & 0x1c) | 1);
  pos += (t & 2) ? tmp : 0;
  return pos;
}
template <typename T>
inline __device__ T WarpReducePos8(T pos, uint32_t t)
{
  T tmp;
  pos = WarpReducePos4(pos, t);
  tmp = SHFL(pos, (t & 0x18) | 3);
  pos += (t & 4) ? tmp : 0;
  return pos;
}
template <typename T>
inline __device__ T WarpReducePos16(T pos, uint32_t t)
{
  T tmp;
  pos = WarpReducePos8(pos, t);
  tmp = SHFL(pos, (t & 0x10) | 7);
  pos += (t & 8) ? tmp : 0;
  return pos;
}
template <typename T>
inline __device__ T WarpReducePos32(T pos, uint32_t t)
{
  T tmp;
  pos = WarpReducePos16(pos, t);
  tmp = SHFL(pos, 0xf);
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

inline __device__ uint32_t unaligned_load32(const uint8_t *p)
{
  uint32_t ofs        = 3 & reinterpret_cast<uintptr_t>(p);
  const uint32_t *p32 = reinterpret_cast<const uint32_t *>(p - ofs);
  uint32_t v          = p32[0];
  return (ofs) ? __funnelshift_r(v, p32[1], ofs * 8) : v;
}

inline __device__ uint64_t unaligned_load64(const uint8_t *p)
{
  uint32_t ofs        = 3 & reinterpret_cast<uintptr_t>(p);
  const uint32_t *p32 = reinterpret_cast<const uint32_t *>(p - ofs);
  uint32_t v0         = p32[0];
  uint32_t v1         = p32[1];
  if (ofs) {
    v0 = __funnelshift_r(v0, v1, ofs * 8);
    v1 = __funnelshift_r(v1, p32[2], ofs * 8);
  }
  return (((uint64_t)v1) << 32) | v0;
}

template <unsigned int nthreads, bool sync_before_store>
inline __device__ void memcpy_block(void *dstv, const void *srcv, uint32_t len, uint32_t t)
{
  uint8_t *dst       = static_cast<uint8_t *>(dstv);
  const uint8_t *src = static_cast<const uint8_t *>(srcv);
  uint32_t dst_align_bytes, src_align_bytes, src_align_bits;
  // Align output to 32-bit
  dst_align_bytes = 3 & -reinterpret_cast<intptr_t>(dst);
  if (dst_align_bytes != 0) {
    uint32_t align_len = min(dst_align_bytes, len);
    uint8_t b;
    if (t < align_len) { b = src[t]; }
    if (sync_before_store) { __syncthreads(); }
    if (t < align_len) { dst[t] = b; }
    src += align_len;
    dst += align_len;
    len -= align_len;
  }
  src_align_bytes = (uint32_t)(3 & reinterpret_cast<uintptr_t>(src));
  src_align_bits  = src_align_bytes * 8;
  while (len >= 4) {
    const uint32_t *src32 = reinterpret_cast<const uint32_t *>(src - src_align_bytes);
    uint32_t copy_cnt     = min(len >> 2, nthreads);
    uint32_t v;
    if (t < copy_cnt) {
      v = src32[t];
      if (src_align_bits != 0) { v = __funnelshift_r(v, src32[t + 1], src_align_bits); }
    }
    if (sync_before_store) { __syncthreads(); }
    if (t < copy_cnt) { reinterpret_cast<uint32_t *>(dst)[t] = v; }
    src += copy_cnt * 4;
    dst += copy_cnt * 4;
    len -= copy_cnt * 4;
  }
  if (len != 0) {
    uint8_t b;
    if (t < len) { b = src[t]; }
    if (sync_before_store) { __syncthreads(); }
    if (t < len) { dst[t] = b; }
  }
}

/**
 * @brief Compares two strings
 */
template <class T, const T lesser, const T greater, const T equal>
inline __device__ T nvstr_compare(const char *as, uint32_t alen, const char *bs, uint32_t blen)
{
  uint32_t len = min(alen, blen);
  uint32_t i   = 0;
  if (len >= 4) {
    uint32_t align_a     = 3 & reinterpret_cast<uintptr_t>(as);
    uint32_t align_b     = 3 & reinterpret_cast<uintptr_t>(bs);
    const uint32_t *as32 = reinterpret_cast<const uint32_t *>(as - align_a);
    const uint32_t *bs32 = reinterpret_cast<const uint32_t *>(bs - align_b);
    uint32_t ofsa        = align_a * 8;
    uint32_t ofsb        = align_b * 8;
    do {
      uint32_t a = *as32++;
      uint32_t b = *bs32++;
      if (ofsa) a = __funnelshift_r(a, *as32, ofsa);
      if (ofsb) b = __funnelshift_r(b, *bs32, ofsb);
      if (a != b) {
        return (lesser == greater || __byte_perm(a, 0, 0x0123) < __byte_perm(b, 0, 0x0123))
                 ? lesser
                 : greater;
      }
      i += 4;
    } while (i + 4 <= len);
  }
  while (i < len) {
    uint8_t a = as[i];
    uint8_t b = bs[i];
    if (a != b) { return (a < b) ? lesser : greater; }
    ++i;
  }
  return (alen == blen) ? equal : (alen < blen) ? lesser : greater;
}

inline __device__ bool nvstr_is_lesser(const char *as, uint32_t alen, const char *bs, uint32_t blen)
{
  return nvstr_compare<bool, true, false, false>(as, alen, bs, blen);
}

inline __device__ bool nvstr_is_greater(const char *as,
                                        uint32_t alen,
                                        const char *bs,
                                        uint32_t blen)
{
  return nvstr_compare<bool, false, true, false>(as, alen, bs, blen);
}

inline __device__ bool nvstr_is_equal(const char *as, uint32_t alen, const char *bs, uint32_t blen)
{
  return nvstr_compare<bool, false, false, true>(as, alen, bs, blen);
}

}  // namespace io
}  // namespace cudf
