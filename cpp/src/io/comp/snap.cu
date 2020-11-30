/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include "gpuinflate.h"

#include <io/utilities/block_utils.cuh>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace io {
constexpr int hash_bits = 12;

// TBD: Tentatively limits to 2-byte codes to prevent long copy search followed by long literal
// encoding

/**
 * @brief snappy compressor state
 **/
struct snap_state_s {
  const uint8_t *src;                 ///< Ptr to uncompressed data
  uint32_t src_len;                   ///< Uncompressed data length
  uint8_t *dst_base;                  ///< Base ptr to output compressed data
  uint8_t *dst;                       ///< Current ptr to uncompressed data
  uint8_t *end;                       ///< End of uncompressed data buffer
  volatile uint32_t literal_length;   ///< Number of literal bytes
  volatile uint32_t copy_length;      ///< Number of copy bytes
  volatile uint32_t copy_distance;    ///< Distance for copy bytes
  uint16_t hash_map[1 << hash_bits];  ///< Low 16-bit offset from hash
};

/**
 * @brief 12-bit hash from four consecutive bytes
 **/
static inline __device__ uint32_t snap_hash(uint32_t v)
{
  return (v * ((1 << 20) + (0x2a00) + (0x6a) + 1)) >> (32 - hash_bits);
}

/**
 * @brief Fetches four consecutive bytes
 **/
static inline __device__ uint32_t fetch4(const uint8_t *src)
{
  uint32_t src_align    = 3 & reinterpret_cast<uintptr_t>(src);
  const uint32_t *src32 = reinterpret_cast<const uint32_t *>(src - src_align);
  uint32_t v            = src32[0];
  return (src_align) ? __funnelshift_r(v, src32[1], src_align * 8) : v;
}

/**
 * @brief Outputs a snappy literal symbol
 *
 * @param dst Destination compressed byte stream
 * @param end End of compressed data buffer
 * @param src Pointer to literal bytes
 * @param len_minus1 Number of literal bytes minus 1
 * @param t Thread in warp
 *
 * @return Updated pointer to compressed byte stream
 **/
static __device__ uint8_t *StoreLiterals(
  uint8_t *dst, uint8_t *end, const uint8_t *src, uint32_t len_minus1, uint32_t t)
{
  if (len_minus1 < 60) {
    if (!t && dst < end) dst[0] = (len_minus1 << 2);
    dst += 1;
  } else if (len_minus1 <= 0xff) {
    if (!t && dst + 1 < end) {
      dst[0] = 60 << 2;
      dst[1] = len_minus1;
    }
    dst += 2;
  } else if (len_minus1 <= 0xffff) {
    if (!t && dst + 2 < end) {
      dst[0] = 61 << 2;
      dst[1] = len_minus1;
      dst[2] = len_minus1 >> 8;
    }
    dst += 3;
  } else if (len_minus1 <= 0xffffff) {
    if (!t && dst + 3 < end) {
      dst[0] = 62 << 2;
      dst[1] = len_minus1;
      dst[2] = len_minus1 >> 8;
      dst[3] = len_minus1 >> 16;
    }
    dst += 4;
  } else {
    if (!t && dst + 4 < end) {
      dst[0] = 63 << 2;
      dst[1] = len_minus1;
      dst[2] = len_minus1 >> 8;
      dst[3] = len_minus1 >> 16;
      dst[4] = len_minus1 >> 24;
    }
    dst += 5;
  }
  for (uint32_t i = t; i <= len_minus1; i += 32) {
    if (dst + i < end) dst[i] = src[i];
  }
  return dst + len_minus1 + 1;
}

/**
 * @brief Outputs a snappy copy symbol (assumed to be called by a single thread)
 *
 * @param dst Destination compressed byte stream
 * @param end End of compressed data buffer
 * @param copy_len Copy length
 * @param distance Copy distance
 *
 * @return Updated pointer to compressed byte stream
 **/
static __device__ uint8_t *StoreCopy(uint8_t *dst,
                                     uint8_t *end,
                                     uint32_t copy_len,
                                     uint32_t distance)
{
  if (copy_len < 12 && distance < 2048) {
    // xxxxxx01.oooooooo: copy with 3-bit length, 11-bit offset
    if (dst + 2 <= end) {
      dst[0] = ((distance & 0x700) >> 3) | ((copy_len - 4) << 2) | 0x01;
      dst[1] = distance;
    }
    return dst + 2;
  } else {
    // xxxxxx1x: copy with 6-bit length, 16-bit offset
    if (dst + 3 <= end) {
      dst[0] = ((copy_len - 1) << 2) | 0x2;
      dst[1] = distance;
      dst[2] = distance >> 8;
    }
    return dst + 3;
  }
}

/**
 * @brief Returns mask of any thread in the warp that has a hash value
 * equal to that of the calling thread
 **/
static inline __device__ uint32_t HashMatchAny(uint32_t v, uint32_t t)
{
#if (__CUDA_ARCH__ >= 700)
  return __match_any_sync(~0, v);
#else
  uint32_t err_map = 0;
  for (uint32_t i = 0; i < hash_bits; i++, v >>= 1) {
    uint32_t b       = v & 1;
    uint32_t match_b = ballot(b);
    err_map |= match_b ^ -(int32_t)b;
  }
  return ~err_map;
#endif
}

/**
 * @brief Finds the first occurrence of a consecutive 4-byte match in the input sequence,
 * or at most 256 bytes
 *
 * @param s Compressor state (copy_length set to 4 if a match is found, zero otherwise)
 * @param src Uncompressed buffer
 * @param pos0 Position in uncompressed buffer
 * @param t thread in warp
 *
 * @return Number of bytes before first match (literal length)
 **/
static __device__ uint32_t FindFourByteMatch(snap_state_s *s,
                                             const uint8_t *src,
                                             uint32_t pos0,
                                             uint32_t t)
{
  constexpr int max_literal_length = 256;
  // Matches encoder limit as described in snappy format description
  constexpr int max_copy_distance = 32768;
  uint32_t len                    = s->src_len;
  uint32_t pos                    = pos0;
  uint32_t maxpos                 = pos0 + max_literal_length - 31;
  uint32_t match_mask, literal_cnt;
  if (t == 0) { s->copy_length = 0; }
  do {
    bool valid4               = (pos + t + 4 <= len);
    uint32_t data32           = (valid4) ? fetch4(src + pos + t) : 0;
    uint32_t hash             = (valid4) ? snap_hash(data32) : 0;
    uint32_t local_match      = HashMatchAny(hash, t);
    uint32_t local_match_lane = 31 - __clz(local_match & ((1 << t) - 1));
    uint32_t local_match_data = shuffle(data32, min(local_match_lane, t));
    uint32_t offset, match;
    if (valid4) {
      if (local_match_lane < t && local_match_data == data32) {
        match  = 1;
        offset = pos + local_match_lane;
      } else {
        offset = (pos & ~0xffff) | s->hash_map[hash];
        if (offset >= pos) { offset = (offset >= 0x10000) ? offset - 0x10000 : pos; }
        match =
          (offset < pos && offset + max_copy_distance >= pos + t && fetch4(src + offset) == data32);
      }
    } else {
      match       = 0;
      local_match = 0;
      offset      = pos + t;
    }
    match_mask = ballot(match);
    if (match_mask != 0) {
      literal_cnt = __ffs(match_mask) - 1;
      if (t == literal_cnt) {
        s->copy_distance = pos + t - offset;
        s->copy_length   = 4;
      }
    } else {
      literal_cnt = 32;
    }
    // Update hash up to the first 4 bytes of the copy length
    local_match &= (0x2 << literal_cnt) - 1;
    if (t <= literal_cnt && t == 31 - __clz(local_match)) { s->hash_map[hash] = pos + t; }
    pos += literal_cnt;
  } while (literal_cnt == 32 && pos < maxpos);
  return min(pos, len) - pos0;
}

/// @brief Returns the number of matching bytes for two byte sequences up to 63 bytes
static __device__ uint32_t Match60(const uint8_t *src1,
                                   const uint8_t *src2,
                                   uint32_t len,
                                   uint32_t t)
{
  uint32_t mismatch = ballot(t >= len || src1[t] != src2[t]);
  if (mismatch == 0) {
    mismatch = ballot(32 + t >= len || src1[32 + t] != src2[32 + t]);
    return 31 + __ffs(mismatch);  // mismatch cannot be zero here if len <= 63
  } else {
    return __ffs(mismatch) - 1;
  }
}

/**
 * @brief Snappy compression kernel
 * See http://github.com/google/snappy/blob/master/format_description.txt
 *
 * blockDim {128,1,1}
 *
 * @param[in] inputs Source/Destination buffer information per block
 * @param[out] outputs Compression status per block
 * @param[in] count Number of blocks to compress
 **/
extern "C" __global__ void __launch_bounds__(128)
  snap_kernel(gpu_inflate_input_s *inputs, gpu_inflate_status_s *outputs, int count)
{
  __shared__ __align__(16) snap_state_s state_g;

  snap_state_s *const s = &state_g;
  uint32_t t            = threadIdx.x;
  uint32_t pos;
  const uint8_t *src;

  if (!t) {
    const uint8_t *src = static_cast<const uint8_t *>(inputs[blockIdx.x].srcDevice);
    uint32_t src_len   = static_cast<uint32_t>(inputs[blockIdx.x].srcSize);
    uint8_t *dst       = static_cast<uint8_t *>(inputs[blockIdx.x].dstDevice);
    uint32_t dst_len   = static_cast<uint32_t>(inputs[blockIdx.x].dstSize);
    uint8_t *end       = dst + dst_len;
    s->src             = src;
    s->src_len         = src_len;
    s->dst_base        = dst;
    s->end             = end;
    while (src_len > 0x7f) {
      if (dst < end) { dst[0] = src_len | 0x80; }
      dst++;
      src_len >>= 7;
    }
    if (dst < end) { dst[0] = src_len; }
    s->dst            = dst + 1;
    s->literal_length = 0;
    s->copy_length    = 0;
    s->copy_distance  = 0;
  }
  for (uint32_t i = t; i < sizeof(s->hash_map) / sizeof(uint32_t); i += 128) {
    *reinterpret_cast<volatile uint32_t *>(&s->hash_map[i * 2]) = 0;
  }
  __syncthreads();
  src = s->src;
  pos = 0;
  while (pos < s->src_len) {
    uint32_t literal_len = s->literal_length;
    uint32_t copy_len    = s->copy_length;
    uint32_t distance    = s->copy_distance;
    __syncthreads();
    if (t < 32) {
      // WARP0: Encode literals and copies
      uint8_t *dst = s->dst;
      uint8_t *end = s->end;
      if (literal_len > 0) {
        dst = StoreLiterals(dst, end, src + pos, literal_len - 1, t);
        pos += literal_len;
      }
      if (copy_len > 0) {
        if (t == 0) { dst = StoreCopy(dst, end, copy_len, distance); }
        pos += copy_len;
      }
      __syncwarp();
      if (t == 0) { s->dst = dst; }
    } else {
      pos += literal_len + copy_len;
      if (t < 32 * 2) {
        // WARP1: Find a match using 12-bit hashes of 4-byte blocks
        uint32_t t5 = t & 0x1f;
        literal_len = FindFourByteMatch(s, src, pos, t5);
        if (t5 == 0) { s->literal_length = literal_len; }
        copy_len = s->copy_length;
        if (copy_len != 0) {
          uint32_t match_pos = pos + literal_len + copy_len;  // NOTE: copy_len is always 4 here
          copy_len += Match60(src + match_pos,
                              src + match_pos - s->copy_distance,
                              min(s->src_len - match_pos, 64 - copy_len),
                              t5);
          if (t5 == 0) { s->copy_length = copy_len; }
        }
      }
    }
    __syncthreads();
  }
  __syncthreads();
  if (!t) {
    outputs[blockIdx.x].bytes_written = s->dst - s->dst_base;
    outputs[blockIdx.x].status        = (s->dst > s->end) ? 1 : 0;
    outputs[blockIdx.x].reserved      = 0;
  }
}

cudaError_t __host__ gpu_snap(gpu_inflate_input_s *inputs,
                              gpu_inflate_status_s *outputs,
                              int count,
                              rmm::cuda_stream_view stream)
{
  dim3 dim_block(128, 1);  // 4 warps per stream, 1 stream per block
  dim3 dim_grid(count, 1);
  if (count > 0) {
    snap_kernel<<<dim_grid, dim_block, 0, stream.value()>>>(inputs, outputs, count);
  }
  return cudaSuccess;
}

}  // namespace io
}  // namespace cudf
