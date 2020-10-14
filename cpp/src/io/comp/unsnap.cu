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

#include <cub/cub.cuh>
#include <io/utilities/block_utils.cuh>
#include "gpuinflate.h"

namespace cudf {
namespace io {
// Not supporting streams longer than this (not what snappy is intended for)
#define SNAPPY_MAX_STREAM_SIZE 0x7fffffff

#define LOG2_BATCH_SIZE 5
#define BATCH_SIZE (1 << LOG2_BATCH_SIZE)
#define LOG2_BATCH_COUNT 2
#define BATCH_COUNT (1 << LOG2_BATCH_COUNT)
#define LOG2_PREFETCH_SIZE 9
#define PREFETCH_SIZE (1 << LOG2_PREFETCH_SIZE)  // 512B, in 32B chunks

#define LOG_CYCLECOUNT 0

/**
 * @brief Describes a single LZ77 symbol (single entry in batch)
 **/
struct unsnap_batch_s {
  int32_t len;  // 1..64 = Number of bytes
  uint32_t
    offset;  // copy distance if greater than zero or negative of literal offset in byte stream
};

/**
 * @brief Queue structure used to exchange data between warps
 **/
struct unsnap_queue_s {
  uint32_t prefetch_wrpos;         ///< Prefetcher write position
  uint32_t prefetch_rdpos;         ///< Prefetch consumer read position
  int32_t prefetch_end;            ///< Prefetch enable flag (nonzero stops prefetcher)
  int32_t batch_len[BATCH_COUNT];  ///< Length of each batch - <0:end, 0:not ready, >0:symbol count
  unsnap_batch_s batch[BATCH_COUNT * BATCH_SIZE];  ///< LZ77 batch data
  uint8_t buf[PREFETCH_SIZE];                      ///< Prefetch buffer
};

/**
 * @brief snappy decompression state
 **/
struct unsnap_state_s {
  const uint8_t *base;         ///< base ptr of compressed stream
  const uint8_t *end;          ///< end of compressed stream
  uint32_t uncompressed_size;  ///< uncompressed stream size
  uint32_t bytes_left;         ///< bytes to uncompressed remaining
  int32_t error;               ///< current error status
  uint32_t tstart;             ///< start time for perf logging
  volatile unsnap_queue_s q;   ///< queue for cross-warp communication
  gpu_inflate_input_s in;      ///< input parameters for current block
};

/**
 * @brief prefetches data for the symbol decoding stage
 *
 * @param s decompression state
 * @param t warp lane id
 **/
__device__ void snappy_prefetch_bytestream(unsnap_state_s *s, int t)
{
  const uint8_t *base  = s->base;
  uint32_t end         = (uint32_t)(s->end - base);
  uint32_t align_bytes = (uint32_t)(0x20 - (0x1f & reinterpret_cast<uintptr_t>(base)));
  int32_t pos          = min(align_bytes, end);
  int32_t blen;
  // Start by prefetching up to the next a 32B-aligned location
  if (t < pos) { s->q.buf[t] = base[t]; }
  blen = 0;
  do {
    SYNCWARP();
    if (!t) {
      uint32_t minrdpos;
      s->q.prefetch_wrpos = pos;
      minrdpos            = pos - min(pos, PREFETCH_SIZE - 32u);
      blen                = (int)min(32u, end - pos);
      for (;;) {
        uint32_t rdpos = s->q.prefetch_rdpos;
        if (rdpos >= minrdpos) break;
        if (s->q.prefetch_end) {
          blen = 0;
          break;
        }
        NANOSLEEP(100);
      }
    }
    blen = SHFL0(blen);
    if (t < blen) { s->q.buf[(pos + t) & (PREFETCH_SIZE - 1)] = base[pos + t]; }
    pos += blen;
  } while (blen > 0);
}

/**
 * @brief Lookup table for get_len3_mask()
 *
 * Indexed by a 10-bit pattern, contains the corresponding 4-bit mask of
 * 3-byte code lengths in the lower 4 bits, along with the total number of
 * bytes used for coding the four lengths in the upper 4 bits.
 * The upper 4-bit value could also be obtained by 8+__popc(mask4)
 *
 *   for (uint32_t k = 0; k < 1024; k++)
 *   {
 *       for (uint32_t i = 0, v = 0, b = k, n = 0; i < 4; i++)
 *       {
 *           v |= (b & 1) << i;
 *           n += (b & 1) + 2;
 *           b >>= (b & 1) + 2;
 *       }
 *       k_len3lut[k] = v | (n << 4);
 *   }
 *
 **/
static const uint8_t __device__ __constant__ k_len3lut[1 << 10] = {
  0x80, 0x91, 0x80, 0x91, 0x92, 0x91, 0x92, 0x91, 0x80, 0xa3, 0x80, 0xa3, 0x92, 0xa3, 0x92, 0xa3,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xa3, 0x94, 0xa3, 0x92, 0xa3, 0x92, 0xa3,
  0x80, 0xa5, 0x80, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x80, 0xa3, 0x80, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x94, 0xa5, 0x94, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x94, 0xa3, 0x94, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x98, 0x91, 0x98, 0x91, 0x92, 0x91, 0x92, 0x91, 0x98, 0xb7, 0x98, 0xb7, 0x92, 0xb7, 0x92, 0xb7,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xb7, 0x94, 0xb7, 0x92, 0xb7, 0x92, 0xb7,
  0x98, 0xa5, 0x98, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x98, 0xb7, 0x98, 0xb7, 0xa6, 0xb7, 0xa6, 0xb7,
  0x94, 0xa5, 0x94, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x94, 0xb7, 0x94, 0xb7, 0xa6, 0xb7, 0xa6, 0xb7,
  0x80, 0xa9, 0x80, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x80, 0xa3, 0x80, 0xa3, 0xaa, 0xa3, 0xaa, 0xa3,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xa3, 0xac, 0xa3, 0xaa, 0xa3, 0xaa, 0xa3,
  0x80, 0xa5, 0x80, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x80, 0xa3, 0x80, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0xac, 0xa5, 0xac, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0xac, 0xa3, 0xac, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x98, 0xa9, 0x98, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x98, 0xb7, 0x98, 0xb7, 0xaa, 0xb7, 0xaa, 0xb7,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xb7, 0xac, 0xb7, 0xaa, 0xb7, 0xaa, 0xb7,
  0x98, 0xa5, 0x98, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x98, 0xb7, 0x98, 0xb7, 0xa6, 0xb7, 0xa6, 0xb7,
  0xac, 0xa5, 0xac, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0xac, 0xb7, 0xac, 0xb7, 0xa6, 0xb7, 0xa6, 0xb7,
  0x80, 0x91, 0x80, 0x91, 0x92, 0x91, 0x92, 0x91, 0x80, 0xbb, 0x80, 0xbb, 0x92, 0xbb, 0x92, 0xbb,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xbb, 0x94, 0xbb, 0x92, 0xbb, 0x92, 0xbb,
  0x80, 0xbd, 0x80, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x80, 0xbb, 0x80, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x94, 0xbd, 0x94, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x94, 0xbb, 0x94, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x98, 0x91, 0x98, 0x91, 0x92, 0x91, 0x92, 0x91, 0x98, 0xb7, 0x98, 0xb7, 0x92, 0xb7, 0x92, 0xb7,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xb7, 0x94, 0xb7, 0x92, 0xb7, 0x92, 0xb7,
  0x98, 0xbd, 0x98, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x98, 0xb7, 0x98, 0xb7, 0xbe, 0xb7, 0xbe, 0xb7,
  0x94, 0xbd, 0x94, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x94, 0xb7, 0x94, 0xb7, 0xbe, 0xb7, 0xbe, 0xb7,
  0x80, 0xa9, 0x80, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x80, 0xbb, 0x80, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xbb, 0xac, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb,
  0x80, 0xbd, 0x80, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x80, 0xbb, 0x80, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0xac, 0xbd, 0xac, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0xac, 0xbb, 0xac, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x98, 0xa9, 0x98, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x98, 0xb7, 0x98, 0xb7, 0xaa, 0xb7, 0xaa, 0xb7,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xb7, 0xac, 0xb7, 0xaa, 0xb7, 0xaa, 0xb7,
  0x98, 0xbd, 0x98, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x98, 0xb7, 0x98, 0xb7, 0xbe, 0xb7, 0xbe, 0xb7,
  0xac, 0xbd, 0xac, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0xac, 0xb7, 0xac, 0xb7, 0xbe, 0xb7, 0xbe, 0xb7,
  0x80, 0x91, 0x80, 0x91, 0x92, 0x91, 0x92, 0x91, 0x80, 0xa3, 0x80, 0xa3, 0x92, 0xa3, 0x92, 0xa3,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xa3, 0x94, 0xa3, 0x92, 0xa3, 0x92, 0xa3,
  0x80, 0xa5, 0x80, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x80, 0xa3, 0x80, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x94, 0xa5, 0x94, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x94, 0xa3, 0x94, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x98, 0x91, 0x98, 0x91, 0x92, 0x91, 0x92, 0x91, 0x98, 0xcf, 0x98, 0xcf, 0x92, 0xcf, 0x92, 0xcf,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xcf, 0x94, 0xcf, 0x92, 0xcf, 0x92, 0xcf,
  0x98, 0xa5, 0x98, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x98, 0xcf, 0x98, 0xcf, 0xa6, 0xcf, 0xa6, 0xcf,
  0x94, 0xa5, 0x94, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x94, 0xcf, 0x94, 0xcf, 0xa6, 0xcf, 0xa6, 0xcf,
  0x80, 0xa9, 0x80, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x80, 0xa3, 0x80, 0xa3, 0xaa, 0xa3, 0xaa, 0xa3,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xa3, 0xac, 0xa3, 0xaa, 0xa3, 0xaa, 0xa3,
  0x80, 0xa5, 0x80, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x80, 0xa3, 0x80, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0xac, 0xa5, 0xac, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0xac, 0xa3, 0xac, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x98, 0xa9, 0x98, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x98, 0xcf, 0x98, 0xcf, 0xaa, 0xcf, 0xaa, 0xcf,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xcf, 0xac, 0xcf, 0xaa, 0xcf, 0xaa, 0xcf,
  0x98, 0xa5, 0x98, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x98, 0xcf, 0x98, 0xcf, 0xa6, 0xcf, 0xa6, 0xcf,
  0xac, 0xa5, 0xac, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0xac, 0xcf, 0xac, 0xcf, 0xa6, 0xcf, 0xa6, 0xcf,
  0x80, 0x91, 0x80, 0x91, 0x92, 0x91, 0x92, 0x91, 0x80, 0xbb, 0x80, 0xbb, 0x92, 0xbb, 0x92, 0xbb,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xbb, 0x94, 0xbb, 0x92, 0xbb, 0x92, 0xbb,
  0x80, 0xbd, 0x80, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x80, 0xbb, 0x80, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x94, 0xbd, 0x94, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x94, 0xbb, 0x94, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x98, 0x91, 0x98, 0x91, 0x92, 0x91, 0x92, 0x91, 0x98, 0xcf, 0x98, 0xcf, 0x92, 0xcf, 0x92, 0xcf,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xcf, 0x94, 0xcf, 0x92, 0xcf, 0x92, 0xcf,
  0x98, 0xbd, 0x98, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x98, 0xcf, 0x98, 0xcf, 0xbe, 0xcf, 0xbe, 0xcf,
  0x94, 0xbd, 0x94, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x94, 0xcf, 0x94, 0xcf, 0xbe, 0xcf, 0xbe, 0xcf,
  0x80, 0xa9, 0x80, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x80, 0xbb, 0x80, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xbb, 0xac, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb,
  0x80, 0xbd, 0x80, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x80, 0xbb, 0x80, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0xac, 0xbd, 0xac, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0xac, 0xbb, 0xac, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x98, 0xa9, 0x98, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x98, 0xcf, 0x98, 0xcf, 0xaa, 0xcf, 0xaa, 0xcf,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xcf, 0xac, 0xcf, 0xaa, 0xcf, 0xaa, 0xcf,
  0x98, 0xbd, 0x98, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x98, 0xcf, 0x98, 0xcf, 0xbe, 0xcf, 0xbe, 0xcf,
  0xac, 0xbd, 0xac, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0xac, 0xcf, 0xac, 0xcf, 0xbe, 0xcf, 0xbe, 0xcf};

/**
 * @brief Returns a 32-bit mask where 1 means 3-byte code length and 0 means 2-byte
 * code length, given an input mask of up to 96 bits.
 *
 * Implemented by doing 8 consecutive lookups, building the result 4-bit at a time
 **/
inline __device__ uint32_t get_len3_mask(uint32_t v0, uint32_t v1, uint32_t v2)
{
  uint32_t m, v, m4, n;
  v  = v0;
  m4 = k_len3lut[v & 0x3ff];
  m  = m4 & 0xf;
  n  = m4 >> 4;  // 8..12
  v  = v0 >> n;
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 4;
  n += m4 >> 4;  // 16..24
  v  = __funnelshift_r(v0, v1, n);
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 8;
  n += m4 >> 4;  // 24..36
  v >>= (m4 >> 4);
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 12;
  n  = (n + (m4 >> 4)) & 0x1f;  // (32..48) % 32 = 0..16
  v1 = __funnelshift_r(v1, v2, n);
  v2 >>= n;
  v  = v1;
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 16;
  n  = m4 >> 4;  // 8..12
  v  = v1 >> n;
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 20;
  n += m4 >> 4;  // 16..24
  v  = __funnelshift_r(v1, v2, n);
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 24;
  n += m4 >> 4;  // 24..36
  v >>= (m4 >> 4);
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 28;
  return m;
}

/**
 * @brief Returns a 32-bit mask where each 2-bit pair contains the symbol length
 * minus 2, given two input masks each containing bit0 or bit1 of the corresponding
 * code length minus 2 for up to 32 bytes
 **/
inline __device__ uint32_t get_len5_mask(uint32_t v0, uint32_t v1)
{
  uint32_t m;
  m = (v1 & 1) * 2 + (v0 & 1);
  v0 >>= (m + 2);
  v1 >>= (m + 1);
  for (uint32_t i = 1; i < 16; i++) {
    uint32_t m2 = (v1 & 2) | (v0 & 1);
    uint32_t n  = m2 + 2;
    m |= m2 << (i * 2);
    v0 >>= n;
    v1 >>= n;
  }
  return m;
}

#define READ_BYTE(pos) s->q.buf[(pos) & (PREFETCH_SIZE - 1)]

/**
 * @brief decode symbols and output LZ77 batches (single-warp)
 *
 * @param s decompression state
 * @param t warp lane id
 **/
__device__ void snappy_decode_symbols(unsnap_state_s *s, uint32_t t)
{
  uint32_t cur        = 0;
  uint32_t end        = static_cast<uint32_t>(s->end - s->base);
  uint32_t bytes_left = s->uncompressed_size;
  uint32_t dst_pos    = 0;
  int32_t batch       = 0;

  for (;;) {
    int32_t batch_len;
    volatile unsnap_batch_s *b;

    // Wait for prefetcher
    if (t == 0) {
      s->q.prefetch_rdpos = cur;
#pragma unroll(1)  // We don't want unrolling here
      while (s->q.prefetch_wrpos < min(cur + 5 * BATCH_SIZE, end)) { NANOSLEEP(50); }
      b = &s->q.batch[batch * BATCH_SIZE];
    }
    // Process small symbols in parallel: for data that does not get good compression,
    // the stream will consist of a large number of short literals (1-byte or 2-byte)
    // followed by short repeat runs. This results in many 2-byte or 3-byte symbols
    // that can all be decoded in parallel once we know the symbol length.
    {
      uint32_t v0, v1, v2, len3_mask, cur_t, is_long_sym, short_sym_mask;
      uint32_t b0;
      cur            = SHFL0(cur);
      cur_t          = cur + t;
      b0             = READ_BYTE(cur_t);
      v0             = BALLOT((b0 == 4) || (b0 & 2));
      b0             = READ_BYTE(cur_t + 32);
      v1             = BALLOT((b0 == 4) || (b0 & 2));
      b0             = READ_BYTE(cur_t + 64);
      v2             = BALLOT((b0 == 4) || (b0 & 2));
      len3_mask      = SHFL0((t == 0) ? get_len3_mask(v0, v1, v2) : 0);
      cur_t          = cur + 2 * t + __popc(len3_mask & ((1 << t) - 1));
      b0             = READ_BYTE(cur_t);
      is_long_sym    = ((b0 & ~4) != 0) && (((b0 + 1) & 2) == 0);
      short_sym_mask = BALLOT(is_long_sym);
      batch_len      = 0;
      b = reinterpret_cast<volatile unsnap_batch_s *>(SHFL0(reinterpret_cast<uintptr_t>(b)));
      if (!(short_sym_mask & 1)) {
        batch_len = SHFL0((t == 0) ? (short_sym_mask) ? __ffs(short_sym_mask) - 1 : 32 : 0);
        if (batch_len != 0) {
          uint32_t blen = 0;
          int32_t ofs   = 0;
          if (t < batch_len) {
            blen = (b0 & 1) ? ((b0 >> 2) & 7) + 4 : ((b0 >> 2) + 1);
            ofs  = (b0 & 1) ? ((b0 & 0xe0) << 3) | READ_BYTE(cur_t + 1)
                           : (b0 & 2) ? READ_BYTE(cur_t + 1) | (READ_BYTE(cur_t + 2) << 8)
                                      : -(int32_t)(cur_t + 1);
            b[t].len    = blen;
            b[t].offset = ofs;
            ofs += blen;  // for correct out-of-range detection below
          }
          blen           = WarpReducePos32(blen, t);
          bytes_left     = SHFL0(bytes_left);
          dst_pos        = SHFL0(dst_pos);
          short_sym_mask = __ffs(BALLOT(blen > bytes_left || ofs > (int32_t)(dst_pos + blen)));
          if (short_sym_mask != 0) { batch_len = min(batch_len, short_sym_mask - 1); }
          if (batch_len != 0) {
            blen = SHFL(blen, batch_len - 1);
            cur  = SHFL(cur_t, batch_len - 1) + 2 + ((len3_mask >> (batch_len - 1)) & 1);
            if (t == 0) {
              dst_pos += blen;
              bytes_left -= blen;
            }
          }
        }
      }
      // Check if the batch was stopped by a 3-byte or 4-byte literal
      if (batch_len < BATCH_SIZE - 2 && SHFL(b0 & ~4, batch_len) == 8) {
        // If so, run a slower version of the above that can also handle 3/4-byte literal sequences
        uint32_t batch_add;
        do {
          uint32_t clen, mask_t;
          cur_t     = cur + t;
          b0        = READ_BYTE(cur_t);
          clen      = (b0 & 3) ? (b0 & 2) ? 1 : 0 : (b0 >> 2);  // symbol length minus 2
          v0        = BALLOT(clen & 1);
          v1        = BALLOT((clen >> 1) & 1);
          len3_mask = SHFL0((t == 0) ? get_len5_mask(v0, v1) : 0);
          mask_t    = (1 << (2 * t)) - 1;
          cur_t     = cur + 2 * t + 2 * __popc((len3_mask & 0xaaaaaaaa) & mask_t) +
                  __popc((len3_mask & 0x55555555) & mask_t);
          b0          = READ_BYTE(cur_t);
          is_long_sym = ((b0 & 3) ? ((b0 & 3) == 3) : (b0 > 3 * 4)) || (cur_t >= cur + 32) ||
                        (batch_len + t >= BATCH_SIZE);
          batch_add = __ffs(BALLOT(is_long_sym)) - 1;
          if (batch_add != 0) {
            uint32_t blen = 0;
            int32_t ofs   = 0;
            if (t < batch_add) {
              blen = (b0 & 1) ? ((b0 >> 2) & 7) + 4 : ((b0 >> 2) + 1);
              ofs  = (b0 & 1) ? ((b0 & 0xe0) << 3) | READ_BYTE(cur_t + 1)
                             : (b0 & 2) ? READ_BYTE(cur_t + 1) | (READ_BYTE(cur_t + 2) << 8)
                                        : -(int32_t)(cur_t + 1);
              b[batch_len + t].len    = blen;
              b[batch_len + t].offset = ofs;
              ofs += blen;  // for correct out-of-range detection below
            }
            blen           = WarpReducePos32(blen, t);
            bytes_left     = SHFL0(bytes_left);
            dst_pos        = SHFL0(dst_pos);
            short_sym_mask = __ffs(BALLOT(blen > bytes_left || ofs > (int32_t)(dst_pos + blen)));
            if (short_sym_mask != 0) { batch_add = min(batch_add, short_sym_mask - 1); }
            if (batch_add != 0) {
              blen = SHFL(blen, batch_add - 1);
              cur  = SHFL(cur_t, batch_add - 1) + 2 + ((len3_mask >> ((batch_add - 1) * 2)) & 3);
              if (t == 0) {
                dst_pos += blen;
                bytes_left -= blen;
              }
              batch_len += batch_add;
            }
          }
        } while (batch_add >= 6 && batch_len < BATCH_SIZE - 2);
      }
    }
    if (t == 0) {
      while (bytes_left > 0 && batch_len < BATCH_SIZE) {
        uint32_t blen, offset;
        uint8_t b0 = READ_BYTE(cur);
        if (b0 & 3) {
          uint8_t b1 = READ_BYTE(cur + 1);
          if (!(b0 & 2)) {
            // xxxxxx01.oooooooo: copy with 3-bit length, 11-bit offset
            offset = ((b0 & 0xe0) << 3) | b1;
            blen   = ((b0 >> 2) & 7) + 4;
            cur += 2;
          } else {
            // xxxxxx1x: copy with 6-bit length, 2-byte or 4-byte offset
            offset = b1 | (READ_BYTE(cur + 2) << 8);
            if (b0 & 1)  // 4-byte offset
            {
              offset |= (READ_BYTE(cur + 3) << 16) | (READ_BYTE(cur + 4) << 24);
              cur += 5;
            } else {
              cur += 3;
            }
            blen = (b0 >> 2) + 1;
          }
          dst_pos += blen;
          if (offset - 1u >= dst_pos || bytes_left < blen) break;
          bytes_left -= blen;
        } else if (b0 < 4 * 4) {
          // 0000xx00: short literal
          blen   = (b0 >> 2) + 1;
          offset = -(int32_t)(cur + 1);
          cur += 1 + blen;
          dst_pos += blen;
          if (bytes_left < blen) break;
          bytes_left -= blen;
        } else {
          // xxxxxx00: literal
          blen = b0 >> 2;
          if (blen >= 60) {
            uint32_t num_bytes = blen - 59;
            blen               = READ_BYTE(cur + 1);
            if (num_bytes > 1) {
              blen |= READ_BYTE(cur + 2) << 8;
              if (num_bytes > 2) {
                blen |= READ_BYTE(cur + 3) << 16;
                if (num_bytes > 3) { blen |= READ_BYTE(cur + 4) << 24; }
              }
            }
            cur += num_bytes;
          }
          cur += 1;
          blen += 1;
          offset = -(int32_t)cur;
          cur += blen;
          // Wait for prefetcher
          s->q.prefetch_rdpos = cur;
#pragma unroll(1)  // We don't want unrolling here
          while (s->q.prefetch_wrpos < min(cur + 5 * BATCH_SIZE, end)) { NANOSLEEP(50); }
          dst_pos += blen;
          if (bytes_left < blen) break;
          bytes_left -= blen;
        }
        b[batch_len].len    = blen;
        b[batch_len].offset = offset;
        batch_len++;
      }
      if (batch_len != 0) {
        s->q.batch_len[batch] = batch_len;
        batch                 = (batch + 1) & (BATCH_COUNT - 1);
      }
    }
    batch_len = SHFL0(batch_len);
    if (t == 0) {
      while (s->q.batch_len[batch] != 0) { NANOSLEEP(100); }
    }
    if (batch_len != BATCH_SIZE) { break; }
  }
  if (!t) {
    s->q.prefetch_end     = 1;
    s->q.batch_len[batch] = -1;
    s->bytes_left         = bytes_left;
    if (bytes_left != 0) { s->error = -2; }
  }
}

/**
 * @brief process LZ77 symbols and output uncompressed stream
 *
 * @param s decompression state
 * @param t thread id within participating group (lane id)
 *
 * NOTE: No error checks at this stage (WARP0 responsible for not sending offsets and lengths that
 *would result in out-of-bounds accesses)
 **/
template <typename Storage>
__device__ void snappy_process_symbols(unsnap_state_s *s, int t, Storage &temp_storage)
{
  const uint8_t *literal_base = s->base;
  uint8_t *out                = static_cast<uint8_t *>(s->in.dstDevice);
  int batch                   = 0;

  do {
    volatile unsnap_batch_s *b = &s->q.batch[batch * BATCH_SIZE];
    int32_t batch_len, blen_t, dist_t;

    if (t == 0) {
      while ((batch_len = s->q.batch_len[batch]) == 0) { NANOSLEEP(100); }
    } else {
      batch_len = 0;
    }
    batch_len = SHFL0(batch_len);
    if (batch_len <= 0) { break; }
    if (t < batch_len) {
      blen_t = b[t].len;
      dist_t = b[t].offset;
    } else {
      blen_t = dist_t = 0;
    }
    // Try to combine as many small entries as possible, but try to avoid doing that
    // if we see a small repeat distance 8 bytes or less
    if (SHFL0(min((uint32_t)dist_t, (uint32_t)SHFL_XOR(dist_t, 1))) > 8) {
      uint32_t n;
      do {
        uint32_t bofs      = WarpReducePos32(blen_t, t);
        uint32_t stop_mask = BALLOT((uint32_t)dist_t < bofs);
        uint32_t start_mask =
          cub::WarpReduce<uint32_t>(temp_storage).Sum((bofs < 32 && t < batch_len) ? 1 << bofs : 0);
        start_mask = SHFL0(start_mask);
        n          = min(min((uint32_t)__popc(start_mask), (uint32_t)(__ffs(stop_mask) - 1u)),
                (uint32_t)batch_len);
        if (n != 0) {
          uint32_t it  = __popc(start_mask & ((2 << t) - 1));
          uint32_t tr  = t - SHFL(bofs - blen_t, it);
          int32_t dist = SHFL(dist_t, it);
          if (it < n) {
            const uint8_t *src = (dist > 0) ? (out + t - dist) : (literal_base + tr - dist);
            out[t]             = *src;
          }
          out += SHFL(bofs, n - 1);
          blen_t = SHFL(blen_t, (n + t) & 0x1f);
          dist_t = SHFL(dist_t, (n + t) & 0x1f);
          batch_len -= n;
        }
      } while (n >= 4);
    }
    for (int i = 0; i < batch_len; i++) {
      int32_t blen  = SHFL(blen_t, i);
      int32_t dist  = SHFL(dist_t, i);
      int32_t blen2 = (i + 1 < batch_len) ? SHFL(blen_t, i + 1) : 32;
      // Try to combine consecutive small entries if they are independent
      if ((uint32_t)dist >= (uint32_t)blen && blen + blen2 <= 32) {
        int32_t dist2 = SHFL(dist_t, i + 1);
        if ((uint32_t)dist2 >= (uint32_t)(blen + blen2)) {
          int32_t d;
          if (t < blen) {
            d = dist;
          } else {
            dist = dist2;
            d    = (dist2 <= 0) ? dist2 + blen : dist2;
          }
          blen += blen2;
          if (t < blen) {
            const uint8_t *src = (dist > 0) ? (out - d) : (literal_base - d);
            out[t]             = src[t];
          }
          out += blen;
          i++;
          continue;
        }
      }
      if (dist > 0) {
        // Copy
        uint8_t b0, b1;
        if (t < blen) {
          uint32_t pos       = t;
          const uint8_t *src = out + ((pos >= dist) ? (pos % dist) : pos) - dist;
          b0                 = *src;
        }
        if (32 + t < blen) {
          uint32_t pos       = 32 + t;
          const uint8_t *src = out + ((pos >= dist) ? (pos % dist) : pos) - dist;
          b1                 = *src;
        }
        if (t < blen) { out[t] = b0; }
        if (32 + t < blen) { out[32 + t] = b1; }
      } else {
        // Literal
        uint8_t b0, b1;
        dist = -dist;
        while (blen >= 64) {
          b0          = literal_base[dist + t];
          b1          = literal_base[dist + 32 + t];
          out[t]      = b0;
          out[32 + t] = b1;
          dist += 64;
          out += 64;
          blen -= 64;
        }
        if (t < blen) { b0 = literal_base[dist + t]; }
        if (32 + t < blen) { b1 = literal_base[dist + 32 + t]; }
        if (t < blen) { out[t] = b0; }
        if (32 + t < blen) { out[32 + t] = b1; }
      }
      out += blen;
    }
    SYNCWARP();
    if (t == 0) { s->q.batch_len[batch] = 0; }
    batch = (batch + 1) & (BATCH_COUNT - 1);
  } while (1);
}

/**
 * @brief Snappy decompression kernel
 * See http://github.com/google/snappy/blob/master/format_description.txt
 *
 * blockDim {128,1,1}
 *
 * @param[in] inputs Source & destination information per block
 * @param[out] outputs Decompression status per block
 **/
template <int block_size>
__global__ void __launch_bounds__(block_size)
  unsnap_kernel(gpu_inflate_input_s *inputs, gpu_inflate_status_s *outputs)
{
  __shared__ __align__(16) unsnap_state_s state_g;
  __shared__ cub::WarpReduce<uint32_t>::TempStorage temp_storage;
  int t             = threadIdx.x;
  unsnap_state_s *s = &state_g;
  int strm_id       = blockIdx.x;

  if (t < sizeof(gpu_inflate_input_s) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&s->in)[t] =
      reinterpret_cast<const uint32_t *>(&inputs[strm_id])[t];
    __threadfence_block();
  }
  if (t < BATCH_COUNT) { s->q.batch_len[t] = 0; }
  __syncthreads();
  if (!t) {
    const uint8_t *cur = static_cast<const uint8_t *>(s->in.srcDevice);
    const uint8_t *end = cur + s->in.srcSize;
    s->error           = 0;
#if LOG_CYCLECOUNT
    s->tstart = clock();
#endif
    if (cur < end) {
      // Read uncompressed size (varint), limited to 32-bit
      uint32_t uncompressed_size = *cur++;
      if (uncompressed_size > 0x7f) {
        uint32_t c        = (cur < end) ? *cur++ : 0;
        uncompressed_size = (uncompressed_size & 0x7f) | (c << 7);
        if (uncompressed_size >= (0x80 << 7)) {
          c                 = (cur < end) ? *cur++ : 0;
          uncompressed_size = (uncompressed_size & ((0x7f << 7) | 0x7f)) | (c << 14);
          if (uncompressed_size >= (0x80 << 14)) {
            c = (cur < end) ? *cur++ : 0;
            uncompressed_size =
              (uncompressed_size & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | (c << 21);
            if (uncompressed_size >= (0x80 << 21)) {
              c = (cur < end) ? *cur++ : 0;
              if (c < 0x8)
                uncompressed_size =
                  (uncompressed_size & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) |
                  (c << 28);
              else
                s->error = -1;
            }
          }
        }
      }
      s->uncompressed_size = uncompressed_size;
      s->bytes_left        = uncompressed_size;
      s->base              = cur;
      s->end               = end;
      if ((cur >= end && uncompressed_size != 0) || (uncompressed_size > s->in.dstSize)) {
        s->error = -1;
      }
    } else {
      s->error = -1;
    }
    s->q.prefetch_end   = 0;
    s->q.prefetch_wrpos = 0;
    s->q.prefetch_rdpos = 0;
  }
  __syncthreads();
  if (!s->error) {
    if (t < 32) {
      // WARP0: decode lengths and offsets
      snappy_decode_symbols(s, t);
    } else if (t < 64) {
      // WARP1: prefetch byte stream for WARP0
      snappy_prefetch_bytestream(s, t & 0x1f);
    } else if (t < 96) {
      // WARP2: LZ77
      snappy_process_symbols(s, t & 0x1f, temp_storage);
    }
    __syncthreads();
  }
  if (!t) {
    outputs[strm_id].bytes_written = s->uncompressed_size - s->bytes_left;
    outputs[strm_id].status        = s->error;
#if LOG_CYCLECOUNT
    outputs[strm_id].reserved = clock() - s->tstart;
#else
    outputs[strm_id].reserved = 0;
#endif
  }
}

cudaError_t __host__ gpu_unsnap(gpu_inflate_input_s *inputs,
                                gpu_inflate_status_s *outputs,
                                int count,
                                cudaStream_t stream)
{
  uint32_t count32 = (count > 0) ? count : 0;
  dim3 dim_block(128, 1);     // 4 warps per stream, 1 stream per block
  dim3 dim_grid(count32, 1);  // TODO: Check max grid dimensions vs max expected count

  unsnap_kernel<128><<<dim_grid, dim_block, 0, stream>>>(inputs, outputs);

  return cudaSuccess;
}

}  // namespace io
}  // namespace cudf
