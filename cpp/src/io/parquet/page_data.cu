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

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>
#include <cudf/detail/utilities/release_assert.cuh>
#include <cudf/utilities/bit.hpp>
#include <io/utilities/block_utils.cuh>
#include <io/utilities/column_buffer.hpp>

#include <io/parquet/parquet_gpu.hpp>

#define LOG2_NTHREADS (5 + 2)
#define NTHREADS (1 << LOG2_NTHREADS)
#define NZ_BFRSZ (NTHREADS * 2)

inline __device__ uint32_t rotl32(uint32_t x, uint32_t r)
{
  return __funnelshift_l(x, x, r);  // (x << r) | (x >> (32 - r));
}

inline __device__ int rolling_index(int index) { return index & (NZ_BFRSZ - 1); }

namespace cudf {
namespace io {
namespace parquet {
namespace gpu {

struct page_state_s {
  const uint8_t *data_start;
  const uint8_t *data_end;
  const uint8_t *dict_base;    // ptr to dictionary page data
  int32_t dict_size;           // size of dictionary data
  int32_t first_row;           // First row in page to output
  int32_t num_rows;            // Rows in page to decode (including rows to be skipped)
  int32_t first_output_value;  // First value in page to output
  int32_t num_input_values;    // total # of input/level values in the page
  int32_t dtype_len;           // Output data type length
  int32_t dtype_len_in;        // Can be larger than dtype_len if truncating 32-bit into 8-bit
  int32_t dict_bits;           // # of bits to store dictionary indices
  uint32_t dict_run;
  int32_t dict_val;
  uint32_t initial_rle_run[NUM_LEVEL_TYPES];   // [def,rep]
  int32_t initial_rle_value[NUM_LEVEL_TYPES];  // [def,rep]
  int32_t error;
  PageInfo page;
  ColumnChunkDesc col;

  // (leaf) value decoding
  int32_t nz_count;  // number of valid entries in nz_idx (write position in circular buffer)
  int32_t dict_pos;  // write position of dictionary indices
  int32_t out_pos;   // read position of final output
  int32_t ts_scale;  // timestamp scale: <0: divide by -ts_scale, >0: multiply by ts_scale
  uint32_t nz_idx[NZ_BFRSZ];    // circular buffer of non-null value positions
  uint32_t dict_idx[NZ_BFRSZ];  // Dictionary index, boolean, or string offset values
  uint32_t str_len[NZ_BFRSZ];   // String length for plain encoding of strings

  // repetition/definition level decoding
  int32_t input_value_count;                  // how many values of the input we've processed
  int32_t input_row_count;                    // how many rows of the input we've processed
  int32_t input_leaf_count;                   // how many leaf values of the input we've processed
  uint32_t rep[NZ_BFRSZ];                     // circular buffer of repetition level values
  uint32_t def[NZ_BFRSZ];                     // circular buffer of definition level values
  const uint8_t *lvl_start[NUM_LEVEL_TYPES];  // [def,rep]
  int32_t lvl_count[NUM_LEVEL_TYPES];         // how many of each of the streams we've decoded
  int32_t row_index_lower_bound;              // lower bound of row indices we should process
};

/**
 * @brief Computes a 32-bit hash when given a byte stream and range.
 *
 * MurmurHash3_32 implementation from
 * https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 *
 * MurmurHash3 was written by Austin Appleby, and is placed in the public
 * domain. The author hereby disclaims copyright to this source code.
 *
 * @param[in] key The input data to hash
 * @param[in] len The length of the input data
 * @param[in] seed An initialization value
 *
 * @return The hash value
 */
__device__ uint32_t device_str2hash32(const char *key, size_t len, uint32_t seed = 33)
{
  const uint8_t *p  = reinterpret_cast<const uint8_t *>(key);
  uint32_t h1       = seed, k1;
  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;
  int l             = len;
  // body
  while (l >= 4) {
    k1 = p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
    k1 *= c1;
    k1 = rotl32(k1, 15);
    k1 *= c2;
    h1 ^= k1;
    h1 = rotl32(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;
    p += 4;
    l -= 4;
  }
  // tail
  k1 = 0;
  switch (l) {
    case 3: k1 ^= p[2] << 16;
    case 2: k1 ^= p[1] << 8;
    case 1:
      k1 ^= p[0];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
  }
  // finalization
  h1 ^= len;
  h1 ^= h1 >> 16;
  h1 *= 0x85ebca6b;
  h1 ^= h1 >> 13;
  h1 *= 0xc2b2ae35;
  h1 ^= h1 >> 16;
  return h1;
}

/**
 * @brief Read a 32-bit varint integer
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The 32-bit value read
 */
inline __device__ uint32_t get_vlq32(const uint8_t *&cur, const uint8_t *end)
{
  uint32_t v = *cur++;
  if (v >= 0x80 && cur < end) {
    v = (v & 0x7f) | ((*cur++) << 7);
    if (v >= (0x80 << 7) && cur < end) {
      v = (v & ((0x7f << 7) | 0x7f)) | ((*cur++) << 14);
      if (v >= (0x80 << 14) && cur < end) {
        v = (v & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | ((*cur++) << 21);
        if (v >= (0x80 << 21) && cur < end) {
          v = (v & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) | ((*cur++) << 28);
        }
      }
    }
  }
  return v;
}

/**
 * @brief Parse the beginning of the level section (definition or repetition),
 * initializes the initial RLE run & value, and returns the section length
 *
 * @param[in,out] s The page state
 * @param[in] cur The current data position
 * @param[in] end The end of the data
 * @param[in] level_bits The bits required
 *
 * @return The length of the section
 */
__device__ uint32_t InitLevelSection(page_state_s *s,
                                     const uint8_t *cur,
                                     const uint8_t *end,
                                     level_type lvl)
{
  int32_t len;
  int level_bits = s->col.level_bits[lvl];
  int encoding   = lvl == level_type::DEFINITION ? s->page.definition_level_encoding
                                               : s->page.repetition_level_encoding;

  if (level_bits == 0) {
    len                       = 0;
    s->initial_rle_run[lvl]   = s->page.num_input_values * 2;  // repeated value
    s->initial_rle_value[lvl] = 0;
    s->lvl_start[lvl]         = cur;
  } else if (encoding == RLE) {
    if (cur + 4 < end) {
      uint32_t run;
      len = 4 + (cur[0]) + (cur[1] << 8) + (cur[2] << 16) + (cur[3] << 24);
      cur += 4;
      run                     = get_vlq32(cur, end);
      s->initial_rle_run[lvl] = run;
      if (!(run & 1)) {
        int v = (cur < end) ? cur[0] : 0;
        cur++;
        if (level_bits > 8) {
          v |= ((cur < end) ? cur[0] : 0) << 8;
          cur++;
        }
        s->initial_rle_value[lvl] = v;
      }
      s->lvl_start[lvl] = cur;
      if (cur > end) { s->error = 2; }
    } else {
      len      = 0;
      s->error = 2;
    }
  } else if (encoding == BIT_PACKED) {
    len                       = (s->page.num_input_values * level_bits + 7) >> 3;
    s->initial_rle_run[lvl]   = ((s->page.num_input_values + 7) >> 3) * 2 + 1;  // literal run
    s->initial_rle_value[lvl] = 0;
    s->lvl_start[lvl]         = cur;
  } else {
    s->error = 3;
    len      = 0;
  }
  return (uint32_t)len;
}

/**
 * @brief Decode values out of a definition or repetition stream
 *
 * @param[in,out] s Page state input/output
 * @param[in] t target_count Target count of stream values on output
 * @param[in] t Warp0 thread ID (0..31)
 * @param[in] lvl The level type we are decoding - DEFINITION or REPETITION
 */
__device__ void gpuDecodeStream(
  uint32_t *output, page_state_s *s, int32_t target_count, int t, level_type lvl)
{
  const uint8_t *cur_def    = s->lvl_start[lvl];
  const uint8_t *end        = s->data_start;
  uint32_t level_run        = s->initial_rle_run[lvl];
  int32_t level_val         = s->initial_rle_value[lvl];
  int level_bits            = s->col.level_bits[lvl];
  int32_t num_input_values  = s->num_input_values;
  int32_t value_count       = s->lvl_count[lvl];
  int32_t batch_coded_count = 0;

  while (value_count < target_count && value_count < num_input_values) {
    int batch_len;
    if (level_run <= 1) {
      // Get a new run symbol from the byte stream
      int sym_len = 0;
      if (!t) {
        const uint8_t *cur = cur_def;
        if (cur < end) { level_run = get_vlq32(cur, end); }
        if (!(level_run & 1)) {
          if (cur < end) level_val = cur[0];
          cur++;
          if (level_bits > 8) {
            if (cur < end) level_val |= cur[0] << 8;
            cur++;
          }
        }
        if (cur > end || level_run <= 1) { s->error = 0x10; }
        sym_len = (int32_t)(cur - cur_def);
        __threadfence_block();
      }
      sym_len   = SHFL0(sym_len);
      level_val = SHFL0(level_val);
      level_run = SHFL0(level_run);
      cur_def += sym_len;
    }
    if (s->error) { break; }

    batch_len = min(num_input_values - value_count, 32);
    if (level_run & 1) {
      // Literal run
      int batch_len8;
      batch_len  = min(batch_len, (level_run >> 1) * 8);
      batch_len8 = (batch_len + 7) >> 3;
      if (t < batch_len) {
        int bitpos         = t * level_bits;
        const uint8_t *cur = cur_def + (bitpos >> 3);
        bitpos &= 7;
        if (cur < end) level_val = cur[0];
        cur++;
        if (level_bits > 8 - bitpos && cur < end) {
          level_val |= cur[0] << 8;
          cur++;
          if (level_bits > 16 - bitpos && cur < end) level_val |= cur[0] << 16;
        }
        level_val = (level_val >> bitpos) & ((1 << level_bits) - 1);
      }
      level_run -= batch_len8 * 2;
      cur_def += batch_len8 * level_bits;
    } else {
      // Repeated value
      batch_len = min(batch_len, level_run >> 1);
      level_run -= batch_len * 2;
    }
    if (t < batch_len) {
      int idx                      = value_count + t;
      output[idx & (NZ_BFRSZ - 1)] = level_val;
    }
    batch_coded_count += batch_len;
    value_count += batch_len;
  }

  // update the stream info
  if (!t) {
    s->lvl_start[lvl]         = cur_def;
    s->initial_rle_run[lvl]   = level_run;
    s->initial_rle_value[lvl] = level_val;
    s->lvl_count[lvl]         = value_count;
  }
}

/**
 * @brief Performs RLE decoding of dictionary indexes
 *
 * @param[in,out] s Page state input/output
 * @param[in] target_pos Target index position in dict_idx buffer (may exceed this value by up to
 * 31)
 * @param[in] t Warp1 thread ID (0..31)
 *
 * @return The new output position
 */
__device__ int gpuDecodeDictionaryIndices(volatile page_state_s *s, int target_pos, int t)
{
  const uint8_t *end = s->data_end;
  int dict_bits      = s->dict_bits;
  int pos            = s->dict_pos;

  while (pos < target_pos) {
    int is_literal, batch_len;
    if (!t) {
      uint32_t run       = s->dict_run;
      const uint8_t *cur = s->data_start;
      if (run <= 1) {
        run = (cur < end) ? get_vlq32(cur, end) : 0;
        if (!(run & 1)) {
          // Repeated value
          int bytecnt = (dict_bits + 7) >> 3;
          if (cur + bytecnt <= end) {
            int32_t run_val = cur[0];
            if (bytecnt > 1) {
              run_val |= cur[1] << 8;
              if (bytecnt > 2) {
                run_val |= cur[2] << 16;
                if (bytecnt > 3) { run_val |= cur[3] << 24; }
              }
            }
            s->dict_val = run_val & ((1 << dict_bits) - 1);
          }
          cur += bytecnt;
        }
      }
      if (run & 1) {
        // Literal batch: must output a multiple of 8, except for the last batch
        int batch_len_div8;
        batch_len      = max(min(32, (int)(run >> 1) * 8), 1);
        batch_len_div8 = (batch_len + 7) >> 3;
        run -= batch_len_div8 * 2;
        cur += batch_len_div8 * dict_bits;
      } else {
        batch_len = max(min(32, (int)(run >> 1)), 1);
        run -= batch_len * 2;
      }
      s->dict_run   = run;
      s->data_start = cur;
      is_literal    = run & 1;
      __threadfence_block();
    }
    SYNCWARP();
    is_literal = SHFL0(is_literal);
    batch_len  = SHFL0(batch_len);
    if (t < batch_len) {
      int dict_idx = s->dict_val;
      if (is_literal) {
        int32_t ofs      = (t - ((batch_len + 7) & ~7)) * dict_bits;
        const uint8_t *p = s->data_start + (ofs >> 3);
        ofs &= 7;
        if (p < end) {
          uint32_t c = 8 - ofs;
          dict_idx   = (*p++) >> ofs;
          if (c < dict_bits && p < end) {
            dict_idx |= (*p++) << c;
            c += 8;
            if (c < dict_bits && p < end) {
              dict_idx |= (*p++) << c;
              c += 8;
              if (c < dict_bits && p < end) { dict_idx |= (*p++) << c; }
            }
          }
          dict_idx &= (1 << dict_bits) - 1;
        }
      }
      s->dict_idx[(pos + t) & (NZ_BFRSZ - 1)] = dict_idx;
    }
    pos += batch_len;
  }
  return pos;
}

/**
 * @brief Performs RLE decoding of dictionary indexes, for when dict_size=1
 *
 * @param[in,out] s Page state input/output
 * @param[in] target_pos Target write position
 * @param[in] t Thread ID
 *
 * @return The new output position
 */
__device__ int gpuDecodeRleBooleans(volatile page_state_s *s, int target_pos, int t)
{
  const uint8_t *end = s->data_end;
  int pos            = s->dict_pos;

  while (pos < target_pos) {
    int is_literal, batch_len;
    if (!t) {
      uint32_t run       = s->dict_run;
      const uint8_t *cur = s->data_start;
      if (run <= 1) {
        run = (cur < end) ? get_vlq32(cur, end) : 0;
        if (!(run & 1)) {
          // Repeated value
          s->dict_val = (cur < end) ? cur[0] & 1 : 0;
          cur++;
        }
      }
      if (run & 1) {
        // Literal batch: must output a multiple of 8, except for the last batch
        int batch_len_div8;
        batch_len = max(min(32, (int)(run >> 1) * 8), 1);
        if (batch_len >= 8) { batch_len &= ~7; }
        batch_len_div8 = (batch_len + 7) >> 3;
        run -= batch_len_div8 * 2;
        cur += batch_len_div8;
      } else {
        batch_len = max(min(32, (int)(run >> 1)), 1);
        run -= batch_len * 2;
      }
      s->dict_run   = run;
      s->data_start = cur;
      is_literal    = run & 1;
      __threadfence_block();
    }
    SYNCWARP();
    is_literal = SHFL0(is_literal);
    batch_len  = SHFL0(batch_len);
    if (t < batch_len) {
      int dict_idx;
      if (is_literal) {
        int32_t ofs      = t - ((batch_len + 7) & ~7);
        const uint8_t *p = s->data_start + (ofs >> 3);
        dict_idx         = (p < end) ? (p[0] >> (ofs & 7u)) & 1 : 0;
      } else {
        dict_idx = s->dict_val;
      }
      s->dict_idx[(pos + t) & (NZ_BFRSZ - 1)] = dict_idx;
    }
    pos += batch_len;
  }
  return pos;
}

/**
 * @brief Parses the length and position of strings
 *
 * @param[in,out] s Page state input/output
 * @param[in] target_pos Target output position
 * @param[in] t Thread ID
 *
 * @return The new output position
 */
__device__ void gpuInitStringDescriptors(volatile page_state_s *s, int target_pos, int t)
{
  int pos = s->dict_pos;
  // This step is purely serial
  if (!t) {
    const uint8_t *cur = s->data_start;
    int dict_size      = s->dict_size;
    int k              = s->dict_val;

    while (pos < target_pos) {
      int len;
      if (k + 4 <= dict_size) {
        len = (cur[k]) | (cur[k + 1] << 8) | (cur[k + 2] << 16) | (cur[k + 3] << 24);
        k += 4;
        if (k + len > dict_size) { len = 0; }
      } else {
        len = 0;
      }
      s->dict_idx[pos & (NZ_BFRSZ - 1)] = k;
      s->str_len[pos & (NZ_BFRSZ - 1)]  = len;
      k += len;
      pos++;
    }
    s->dict_val = k;
    __threadfence_block();
  }
}

/**
 * @brief Output a string descriptor
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dstv Pointer to row output data (string descriptor or 32-bit hash)
 */
inline __device__ void gpuOutputString(volatile page_state_s *s, int src_pos, void *dstv)
{
  const char *ptr = NULL;
  size_t len      = 0;

  if (s->dict_base) {
    // String dictionary
    uint32_t dict_pos =
      (s->dict_bits > 0) ? s->dict_idx[src_pos & (NZ_BFRSZ - 1)] * sizeof(nvstrdesc_s) : 0;
    if (dict_pos < (uint32_t)s->dict_size) {
      const nvstrdesc_s *src = reinterpret_cast<const nvstrdesc_s *>(s->dict_base + dict_pos);
      ptr                    = src->ptr;
      len                    = src->count;
    }
  } else {
    // Plain encoding
    uint32_t dict_pos = s->dict_idx[src_pos & (NZ_BFRSZ - 1)];
    if (dict_pos <= (uint32_t)s->dict_size) {
      ptr = reinterpret_cast<const char *>(s->data_start + dict_pos);
      len = s->str_len[src_pos & (NZ_BFRSZ - 1)];
    }
  }
  if (s->dtype_len == 4) {
    // Output hash
    *static_cast<uint32_t *>(dstv) = device_str2hash32(ptr, len);
  } else {
    // Output string descriptor
    nvstrdesc_s *dst = static_cast<nvstrdesc_s *>(dstv);
    dst->ptr         = ptr;
    dst->count       = len;
  }
}

/**
 * @brief Output a boolean
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
inline __device__ void gpuOutputBoolean(volatile page_state_s *s, int src_pos, uint8_t *dst)
{
  *dst = s->dict_idx[src_pos & (NZ_BFRSZ - 1)];
}

/**
 * @brief Store a 32-bit data element
 *
 * @param[out] dst ptr to output
 * @param[in] src8 raw input bytes
 * @param[in] dict_pos byte position in dictionary
 * @param[in] dict_size size of dictionary
 */
inline __device__ void gpuStoreOutput(uint32_t *dst,
                                      const uint8_t *src8,
                                      uint32_t dict_pos,
                                      uint32_t dict_size)
{
  uint32_t bytebuf;
  unsigned int ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos < dict_size) {
    bytebuf = *reinterpret_cast<const uint32_t *>(src8 + dict_pos);
    if (ofs) {
      uint32_t bytebufnext = *reinterpret_cast<const uint32_t *>(src8 + dict_pos + 4);
      bytebuf              = __funnelshift_r(bytebuf, bytebufnext, ofs);
    }
  } else {
    bytebuf = 0;
  }
  *dst = bytebuf;
}

/**
 * @brief Store a 64-bit data element
 *
 * @param[out] dst ptr to output
 * @param[in] src8 raw input bytes
 * @param[in] dict_pos byte position in dictionary
 * @param[in] dict_size size of dictionary
 */
inline __device__ void gpuStoreOutput(uint2 *dst,
                                      const uint8_t *src8,
                                      uint32_t dict_pos,
                                      uint32_t dict_size)
{
  uint2 v;
  unsigned int ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos < dict_size) {
    v.x = *reinterpret_cast<const uint32_t *>(src8 + dict_pos + 0);
    v.y = *reinterpret_cast<const uint32_t *>(src8 + dict_pos + 4);
    if (ofs) {
      uint32_t next = *reinterpret_cast<const uint32_t *>(src8 + dict_pos + 8);
      v.x           = __funnelshift_r(v.x, v.y, ofs);
      v.y           = __funnelshift_r(v.y, next, ofs);
    }
  } else {
    v.x = v.y = 0;
  }
  *dst = v;
}

/**
 * @brief Convert an INT96 Spark timestamp to 64-bit timestamp
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
inline __device__ void gpuOutputInt96Timestamp(volatile page_state_s *s, int src_pos, int64_t *dst)
{
  const uint8_t *src8;
  uint32_t dict_pos, dict_size = s->dict_size, ofs;
  int64_t ts;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? s->dict_idx[src_pos & (NZ_BFRSZ - 1)] : 0;
    src8     = s->dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    src8     = s->data_start;
  }
  dict_pos *= (uint32_t)s->dtype_len_in;
  ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos + 4 < dict_size) {
    uint3 v;
    int64_t nanos, secs, days;
    v.x = *reinterpret_cast<const uint32_t *>(src8 + dict_pos + 0);
    v.y = *reinterpret_cast<const uint32_t *>(src8 + dict_pos + 4);
    v.z = *reinterpret_cast<const uint32_t *>(src8 + dict_pos + 8);
    if (ofs) {
      uint32_t next = *reinterpret_cast<const uint32_t *>(src8 + dict_pos + 12);
      v.x           = __funnelshift_r(v.x, v.y, ofs);
      v.y           = __funnelshift_r(v.y, v.z, ofs);
      v.z           = __funnelshift_r(v.z, next, ofs);
    }
    nanos = v.y;
    nanos <<= 32;
    nanos |= v.x;
    // Convert from Julian day at noon to UTC seconds
    days = static_cast<int32_t>(v.z);
    secs = (days - 2440588) *
           (24 * 60 * 60);  // TBD: Should be noon instead of midnight, but this matches pyarrow
    if (s->col.ts_clock_rate)
      ts = (secs * s->col.ts_clock_rate) +
           nanos / (1000000000 / s->col.ts_clock_rate);  // Output to desired clock rate
    else
      ts = (secs * 1000000000) + nanos;
  } else {
    ts = 0;
  }
  *dst = ts;
}

/**
 * @brief Output a 64-bit timestamp
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
inline __device__ void gpuOutputInt64Timestamp(volatile page_state_s *s, int src_pos, int64_t *dst)
{
  const uint8_t *src8;
  uint32_t dict_pos, dict_size = s->dict_size, ofs;
  int64_t ts;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? s->dict_idx[src_pos & (NZ_BFRSZ - 1)] : 0;
    src8     = s->dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    src8     = s->data_start;
  }
  dict_pos *= (uint32_t)s->dtype_len_in;
  ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos + 4 < dict_size) {
    uint2 v;
    int64_t val;
    int32_t ts_scale;
    v.x = *reinterpret_cast<const uint32_t *>(src8 + dict_pos + 0);
    v.y = *reinterpret_cast<const uint32_t *>(src8 + dict_pos + 4);
    if (ofs) {
      uint32_t next = *reinterpret_cast<const uint32_t *>(src8 + dict_pos + 8);
      v.x           = __funnelshift_r(v.x, v.y, ofs);
      v.y           = __funnelshift_r(v.y, next, ofs);
    }
    val = v.y;
    val <<= 32;
    val |= v.x;
    // Output to desired clock rate
    ts_scale = s->ts_scale;
    if (ts_scale < 0) {
      // round towards negative infinity
      int sign = (val < 0);
      ts       = ((val + sign) / -ts_scale) + sign;
    } else {
      ts = val * ts_scale;
    }
  } else {
    ts = 0;
  }
  *dst = ts;
}

/**
 * @brief Powers of 10
 */
static const __device__ __constant__ double kPow10[40] = {
  1.0,   1.e1,  1.e2,  1.e3,  1.e4,  1.e5,  1.e6,  1.e7,  1.e8,  1.e9,  1.e10, 1.e11, 1.e12, 1.e13,
  1.e14, 1.e15, 1.e16, 1.e17, 1.e18, 1.e19, 1.e20, 1.e21, 1.e22, 1.e23, 1.e24, 1.e25, 1.e26, 1.e27,
  1.e28, 1.e29, 1.e30, 1.e31, 1.e32, 1.e33, 1.e34, 1.e35, 1.e36, 1.e37, 1.e38, 1.e39,
};

/**
 * @brief Output a decimal type ([INT32..INT128] + scale) as a 64-bit float
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 * @param[in] dtype Stored data type
 */
inline __device__ void gpuOutputDecimal(volatile page_state_s *s,
                                        int src_pos,
                                        double *dst,
                                        int dtype)
{
  const uint8_t *dict;
  uint32_t dict_pos, dict_size = s->dict_size, dtype_len_in;
  int64_t i128_hi, i128_lo;
  int32_t scale;
  double d;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? s->dict_idx[src_pos & (NZ_BFRSZ - 1)] : 0;
    dict     = s->dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    dict     = s->data_start;
  }
  dtype_len_in = s->dtype_len_in;
  dict_pos *= dtype_len_in;
  // FIXME: Not very efficient (currently reading 1 byte at a time) -> need a variable-length
  // unaligned load utility function (both little-endian and big-endian versions)
  if (dtype == INT32) {
    int32_t lo32 = 0;
    for (unsigned int i = 0; i < dtype_len_in; i++) {
      uint32_t v = (dict_pos + i < dict_size) ? dict[dict_pos + i] : 0;
      lo32 |= v << (i * 8);
    }
    i128_lo = lo32;
    i128_hi = lo32 >> 31;
  } else if (dtype == INT64) {
    int64_t lo64 = 0;
    for (unsigned int i = 0; i < dtype_len_in; i++) {
      uint64_t v = (dict_pos + i < dict_size) ? dict[dict_pos + i] : 0;
      lo64 |= v << (i * 8);
    }
    i128_lo = lo64;
    i128_hi = lo64 >> 63;
  } else  // if (dtype == FIXED_LENGTH_BYTE_ARRAY)
  {
    i128_lo = 0;
    for (unsigned int i = dtype_len_in - min(dtype_len_in, 8); i < dtype_len_in; i++) {
      uint32_t v = (dict_pos + i < dict_size) ? dict[dict_pos + i] : 0;
      i128_lo    = (i128_lo << 8) | v;
    }
    if (dtype_len_in > 8) {
      i128_hi = 0;
      for (unsigned int i = dtype_len_in - min(dtype_len_in, 16); i < dtype_len_in - 8; i++) {
        uint32_t v = (dict_pos + i < dict_size) ? dict[dict_pos + i] : 0;
        i128_hi    = (i128_hi << 8) | v;
      }
      if (dtype_len_in < 16) {
        i128_hi <<= 64 - (dtype_len_in - 8) * 8;
        i128_hi >>= 64 - (dtype_len_in - 8) * 8;
      }
    } else {
      if (dtype_len_in < 8) {
        i128_lo <<= 64 - dtype_len_in * 8;
        i128_lo >>= 64 - dtype_len_in * 8;
      }
      i128_hi = i128_lo >> 63;
    }
  }
  scale = s->col.decimal_scale;
  d     = Int128ToDouble_rn(i128_lo, i128_hi);
  *dst  = (scale < 0) ? (d * kPow10[min(-scale, 39)]) : (d / kPow10[min(scale, 39)]);
}

/**
 * @brief Output a small fixed-length value
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename T>
inline __device__ void gpuOutputFast(volatile page_state_s *s, int src_pos, T *dst)
{
  const uint8_t *dict;
  uint32_t dict_pos, dict_size = s->dict_size;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? s->dict_idx[src_pos & (NZ_BFRSZ - 1)] : 0;
    dict     = s->dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    dict     = s->data_start;
  }
  dict_pos *= (uint32_t)s->dtype_len_in;
  gpuStoreOutput(dst, dict, dict_pos, dict_size);
}

/**
 * @brief Output a N-byte value
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst8 Pointer to row output data
 * @param[in] len Length of element
 */
static __device__ void gpuOutputGeneric(volatile page_state_s *s,
                                        int src_pos,
                                        uint8_t *dst8,
                                        int len)
{
  const uint8_t *dict;
  uint32_t dict_pos, dict_size = s->dict_size;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? s->dict_idx[src_pos & (NZ_BFRSZ - 1)] : 0;
    dict     = s->dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    dict     = s->data_start;
  }
  dict_pos *= (uint32_t)s->dtype_len_in;
  if (len & 3) {
    // Generic slow path
    for (unsigned int i = 0; i < len; i++) {
      dst8[i] = (dict_pos + i < dict_size) ? dict[dict_pos + i] : 0;
    }
  } else {
    // Copy 4 bytes at a time
    const uint8_t *src8 = dict;
    unsigned int ofs    = 3 & reinterpret_cast<size_t>(src8);
    src8 -= ofs;  // align to 32-bit boundary
    ofs <<= 3;    // bytes -> bits
    for (unsigned int i = 0; i < len; i += 4) {
      uint32_t bytebuf;
      if (dict_pos < dict_size) {
        bytebuf = *reinterpret_cast<const uint32_t *>(src8 + dict_pos);
        if (ofs) {
          uint32_t bytebufnext = *reinterpret_cast<const uint32_t *>(src8 + dict_pos + 4);
          bytebuf              = __funnelshift_r(bytebuf, bytebufnext, ofs);
        }
      } else {
        bytebuf = 0;
      }
      dict_pos += 4;
      *reinterpret_cast<uint32_t *>(dst8 + i) = bytebuf;
    }
  }
}

/**
 * @brief Sets up block-local page state information from the global pages.
 *
 * @param[in, out] s The local page state to be filled in
 * @param[in] p The global page to be copied from
 * @param[in] chunks The global list of chunks
 * @param[in] num_rows Maximum number of rows to read
 * @param[in] min_row crop all rows below min_row
 * @param[in] num_chunk Number of column chunks
 */
static __device__ bool setupLocalPageInfo(page_state_s *const s,
                                          PageInfo *p,
                                          ColumnChunkDesc const *chunks,
                                          size_t min_row,
                                          size_t num_rows,
                                          int32_t num_chunks)
{
  int t = threadIdx.x;
  int chunk_idx;

  // Fetch page info
  // NOTE: Assumes that sizeof(PageInfo) <= 256 (and is padded to 4 bytes)
  if (t < sizeof(PageInfo) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&s->page)[t] = reinterpret_cast<const uint32_t *>(p)[t];
  }
  __syncthreads();
  if (s->page.flags & PAGEINFO_FLAGS_DICTIONARY) { return false; }
  // Fetch column chunk info
  chunk_idx = s->page.chunk_idx;
  if ((uint32_t)chunk_idx < (uint32_t)num_chunks) {
    // NOTE: Assumes that sizeof(ColumnChunkDesc) <= 256 (and is padded to 4 bytes)
    if (t < sizeof(ColumnChunkDesc) / sizeof(uint32_t)) {
      reinterpret_cast<uint32_t *>(&s->col)[t] =
        reinterpret_cast<const uint32_t *>(&chunks[chunk_idx])[t];
    }
  }

  // zero nested value and valid counts
  int d = 0;
  while (d < s->page.num_nesting_levels) {
    if (d + t < s->page.num_nesting_levels) {
      s->page.nesting[d + t].valid_count = 0;
      s->page.nesting[d + t].value_count = 0;
    }
    d += blockDim.x;
  }
  __syncthreads();

  if (!t) {
    s->error = 0;

    // our starting row (absolute index) is
    // col.start_row == absolute row index
    // page.chunk-row == relative row index within the chunk
    size_t page_start_row = s->col.start_row + s->page.chunk_row;

    // IMPORTANT : nested schemas can have 0 rows in a page but still have
    // values. The case is:
    // - On page N-1, the last row starts, with 2/6 values encoded
    // - On page N, the remaining 4/6 values are encoded, but there are no new rows.
    // if (s->page.num_input_values > 0 && s->page.num_rows > 0) {
    if (s->page.num_input_values > 0) {
      uint8_t *cur = s->page.page_data;
      uint8_t *end = cur + s->page.uncompressed_page_size;

      uint32_t dtype_len_out = s->col.data_type >> 3;
      s->ts_scale            = 0;
      // Validate data type
      switch (s->col.data_type & 7) {
        case BOOLEAN:
          s->dtype_len = 1;  // Boolean are stored as 1 byte on the output
          break;
        case INT32:
        case FLOAT: s->dtype_len = 4; break;
        case INT64:
          if (s->col.ts_clock_rate) {
            int32_t units = 0;
            if (s->col.converted_type == TIME_MICROS || s->col.converted_type == TIMESTAMP_MICROS)
              units = 1000000;
            else if (s->col.converted_type == TIME_MILLIS ||
                     s->col.converted_type == TIMESTAMP_MILLIS)
              units = 1000;
            if (units && units != s->col.ts_clock_rate)
              s->ts_scale = (s->col.ts_clock_rate < units) ? -(units / s->col.ts_clock_rate)
                                                           : (s->col.ts_clock_rate / units);
          }
          // Fall through to DOUBLE
        case DOUBLE: s->dtype_len = 8; break;
        case INT96: s->dtype_len = 12; break;
        case BYTE_ARRAY: s->dtype_len = sizeof(nvstrdesc_s); break;
        default:  // FIXED_LEN_BYTE_ARRAY:
          s->dtype_len = dtype_len_out;
          s->error |= (s->dtype_len <= 0);
          break;
      }
      // Special check for downconversions
      s->dtype_len_in = s->dtype_len;
      if (s->col.converted_type == DECIMAL) {
        s->dtype_len = 8;  // Convert DECIMAL to 64-bit float
      } else if ((s->col.data_type & 7) == INT32) {
        if (dtype_len_out == 1) s->dtype_len = 1;  // INT8 output
        if (dtype_len_out == 2) s->dtype_len = 2;  // INT16 output
      } else if ((s->col.data_type & 7) == BYTE_ARRAY && dtype_len_out == 4) {
        s->dtype_len = 4;  // HASH32 output
      } else if ((s->col.data_type & 7) == INT96) {
        s->dtype_len = 8;  // Convert to 64-bit timestamp
      }

      // first row within the page to start reading
      if (page_start_row >= min_row) {
        s->first_row = 0;
      } else {
        s->first_row = (int32_t)min(min_row - page_start_row, (size_t)s->page.num_rows);
      }
      // # of rows within the page to read
      s->num_rows = s->page.num_rows;
      if ((page_start_row + s->first_row) + s->num_rows > min_row + num_rows) {
        s->num_rows =
          (int32_t)max((int64_t)(min_row + num_rows - (page_start_row + s->first_row)), INT64_C(0));
      }

      // during the decoding step we need to offset the global output buffers
      // for each level of nesting so that we write to the section this page
      // is responsible for.
      // - for flat schemas, we can do this directly by using row counts
      // - for nested schemas, these offsets are computed during the preprocess step
      if (s->col.column_data_base != nullptr) {
        int max_depth = s->col.max_nesting_depth;
        for (int idx = 0; idx < max_depth; idx++) {
          PageNestingInfo *pni = &s->page.nesting[idx];

          size_t output_offset;
          // schemas without lists
          if (s->col.max_level[level_type::REPETITION] == 0) {
            output_offset = page_start_row >= min_row ? page_start_row - min_row : 0;
          }
          // for schemas with lists, we've already got the exactly value precomputed
          else {
            output_offset = pni->page_start_value;
          }

          pni->data_out = static_cast<uint8_t *>(s->col.column_data_base[idx]);
          if (pni->data_out != nullptr) {
            // anything below max depth with a valid data pointer must be a list, so the
            // element size is the size of the offset type.
            uint32_t len = idx < max_depth - 1 ? sizeof(cudf::size_type) : s->dtype_len;
            pni->data_out += (output_offset * len);
          }
          pni->valid_map = s->col.valid_map_base[idx];
          if (pni->valid_map != nullptr) {
            pni->valid_map += output_offset >> 5;
            pni->valid_map_offset = (int32_t)(output_offset & 0x1f);
          }
        }
      }
      s->first_output_value = 0;

      // Find the compressed size of repetition levels
      cur += InitLevelSection(s, cur, end, level_type::REPETITION);
      // Find the compressed size of definition levels
      cur += InitLevelSection(s, cur, end, level_type::DEFINITION);

      s->dict_bits = 0;
      s->dict_base = 0;
      s->dict_size = 0;
      switch (s->page.encoding) {
        case PLAIN_DICTIONARY:
        case RLE_DICTIONARY:
          // RLE-packed dictionary indices, first byte indicates index length in bits
          if (((s->col.data_type & 7) == BYTE_ARRAY) && (s->col.str_dict_index)) {
            // String dictionary: use index
            s->dict_base = reinterpret_cast<const uint8_t *>(s->col.str_dict_index);
            s->dict_size = s->col.page_info[0].num_input_values * sizeof(nvstrdesc_s);
          } else {
            s->dict_base =
              s->col.page_info[0].page_data;  // dictionary is always stored in the first page
            s->dict_size = s->col.page_info[0].uncompressed_page_size;
          }
          s->dict_run  = 0;
          s->dict_val  = 0;
          s->dict_bits = (cur < end) ? *cur++ : 0;
          if (s->dict_bits > 32 || !s->dict_base) { s->error = (10 << 8) | s->dict_bits; }
          break;
        case PLAIN:
          s->dict_size = static_cast<int32_t>(end - cur);
          s->dict_val  = 0;
          if ((s->col.data_type & 7) == BOOLEAN) { s->dict_run = s->dict_size * 2 + 1; }
          break;
        case RLE: s->dict_run = 0; break;
        default:
          s->error = 1;  // Unsupported encoding
          break;
      }
      if (cur > end) { s->error = 1; }
      s->data_start = cur;
      s->data_end   = end;
    } else {
      s->error = 1;
    }

    s->lvl_count[level_type::REPETITION] = 0;
    s->lvl_count[level_type::DEFINITION] = 0;
    s->nz_count                          = 0;
    s->num_input_values                  = s->page.num_input_values;
    s->dict_pos                          = 0;
    s->out_pos                           = 0;

    // handle row bounds (skip_rows, min_rows)
    s->input_row_count = s->first_row;

    // return the lower bound to compare (page-relative) thread row index against. Explanation:
    // In the case of nested schemas, rows can span page boundaries.  That is to say,
    // we can encounter the first value for row X on page M, but the last value for page M
    // might not be the last value for row X. page M+1 (or further) may contain the last value.
    //
    // This means that the first values we encounter for a given page (M+1) may not belong to the
    // row indicated by chunk_row, but to the row before it that spanned page boundaries. If that
    // previous row is within the overall row bounds, include the values by allowing relative row
    // index -1
    int max_row = (min_row + num_rows) - 1;
    if (min_row < page_start_row && max_row >= page_start_row - 1) {
      s->row_index_lower_bound = -1;
    } else {
      s->row_index_lower_bound = s->first_row;
    }

    // if we're in the decoding step, jump directly to the first
    // value we care about
    if (s->col.column_data_base != nullptr) {
      // for flat hierarchies, we haven't computed skipped_values yet, but we can do so trivially
      // now
      if (s->col.max_level[level_type::REPETITION] == 0) {
        s->page.skipped_values      = s->first_row;
        s->page.skipped_leaf_values = s->first_row;
      }

      s->input_value_count = s->page.skipped_values;
    } else {
      s->input_value_count        = 0;
      s->input_leaf_count         = 0;
      s->page.skipped_values      = -1;
      s->page.skipped_leaf_values = -1;
    }

    __threadfence_block();
  }
  __syncthreads();

  return true;
}

/**
 * @brief Store a validity mask containing value_count bits into the output validity buffer of the
 * page.
 *
 * @param[in,out] pni The page/nesting information to store the mask in. The validity map offset is
 * also updated
 * @param[in] valid_mask The validity mask to be stored
 * @param[in] value_count # of bits in the validity mask
 */
static __device__ void store_validity(PageNestingInfo *pni,
                                      uint32_t valid_mask,
                                      int32_t value_count)
{
  int word_offset = pni->valid_map_offset / 32;
  int bit_offset  = pni->valid_map_offset % 32;
  // if we fit entirely in the output word
  if (bit_offset + value_count <= 32) {
    uint32_t relevant_mask = static_cast<uint32_t>((static_cast<uint64_t>(1) << value_count) - 1);

    if (relevant_mask == ~0) {
      pni->valid_map[word_offset] = valid_mask;
    } else {
      atomicAnd(pni->valid_map + word_offset, ~(relevant_mask << bit_offset));
      atomicOr(pni->valid_map + word_offset, (valid_mask & relevant_mask) << bit_offset);
    }
  }
  // we're going to spill over into the next word.
  // note : writing both values here is the lazy/slow way.  we could be writing just
  // the first word and rolling the remaining bits over into the next call.
  // however, some basic performance tests shows almost no difference between these two
  // methods. More detailed performance testing might be worthwhile here.
  else {
    uint32_t bits_left = 32 - bit_offset;

    // first word. strip bits_left bits off the beginning and store that
    uint32_t relevant_mask = ((1 << bits_left) - 1);
    uint32_t mask_word0    = valid_mask & relevant_mask;
    atomicAnd(pni->valid_map + word_offset, ~(relevant_mask << bit_offset));
    atomicOr(pni->valid_map + word_offset, mask_word0 << bit_offset);

    // second word. strip the remainder of the bits off the end and store that
    relevant_mask       = ((1 << (value_count - bits_left)) - 1);
    uint32_t mask_word1 = valid_mask & (relevant_mask << bits_left);
    atomicAnd(pni->valid_map + word_offset + 1, ~(relevant_mask));
    atomicOr(pni->valid_map + word_offset + 1, mask_word1 >> bits_left);
  }

  pni->valid_map_offset += value_count;
}

/**
 * @brief Compute the nesting bounds within the hierarchy to add values to, and the definition level
 * D to which we should considered them null or not.
 *
 * @param[out] start_depth The start nesting depth
 * @param[out] end_depth The end nesting depth (inclusive)
 * @param[out] d The definition level up to which added values are not-null. if t is out of bounds,
 * d will be -1
 * @param[in] s Local page information
 * @param[in] input_value_count The current count of input level values we have processed
 * @param[in] target_input_value_count The desired # of input level values we want to process
 * @param[in] t Thread index
 */
inline __device__ void get_nesting_bounds(int &start_depth,
                                          int &end_depth,
                                          int &d,
                                          page_state_s *s,
                                          int input_value_count,
                                          int32_t target_input_value_count,
                                          int t)
{
  start_depth = -1;
  end_depth   = -1;
  d           = -1;
  if (input_value_count + t < target_input_value_count) {
    int index = rolling_index(input_value_count + t);
    d         = s->def[index];
    // if we have repetition (there are list columns involved) we have to
    // bound what nesting levels we apply values to
    if (s->col.max_level[level_type::REPETITION] > 0) {
      int r       = s->rep[index];
      start_depth = s->page.nesting[r].start_depth;
      end_depth   = s->page.nesting[d].end_depth;
    }
    // for columns without repetition (even ones involving structs) we always
    // traverse the entire hierarchy.
    else {
      start_depth = 0;
      end_depth   = s->col.max_nesting_depth - 1;
    }
  }
}

/**
 * @brief Process a batch of incoming repetition/definition level values and generate
 *        validity, nested column offsets (where appropriate) and decoding indices.
 *
 * @param[in] target_input_value_count The # of repetition/definition levels to process up to
 * @param[in] s Local page information
 * @param[in] t Thread index
 */
static __device__ void gpuUpdateValidityOffsetsAndRowIndices(int32_t target_input_value_count,
                                                             page_state_s *s,
                                                             int t)
{
  // max nesting depth of the column
  int max_depth = s->col.max_nesting_depth;
  // how many (input) values we've processed in the page so far
  int input_value_count = s->input_value_count;
  // how many rows we've processed in the page so far
  int input_row_count = s->input_row_count;

  // process until we've reached the target
  while (input_value_count < target_input_value_count) {
    // determine the nesting bounds for this thread (the range of nesting depths we
    // will generate new value indices and validity bits for)
    int start_depth, end_depth, d;
    get_nesting_bounds(
      start_depth, end_depth, d, s, input_value_count, target_input_value_count, t);

    // 4 interesting things to track:
    // thread_value_count : # of output values from the view of this thread
    // warp_value_count   : # of output values for the whole warp
    //
    // thread_valid_count : # of valid values from the view of this thread
    // warp_valid_count   : # of valid values for the whole warp
    uint32_t thread_value_count, warp_value_count;
    uint32_t thread_valid_count, warp_valid_count;

    // track (page-relative) row index for the thread so we can compare against input bounds
    // keep track of overall # of rows we've read.
    int is_new_row               = start_depth == 0 ? 1 : 0;
    uint32_t warp_row_count_mask = BALLOT(is_new_row);
    int32_t thread_row_index =
      input_row_count + ((__popc(warp_row_count_mask & ((1 << t) - 1)) + is_new_row) - 1);
    input_row_count += __popc(warp_row_count_mask);
    // is this thread within row bounds?
    int in_row_bounds = thread_row_index >= s->row_index_lower_bound &&
                            thread_row_index < (s->first_row + s->num_rows)
                          ? 1
                          : 0;

    // compute warp and thread value counts
    uint32_t warp_count_mask =
      BALLOT((0 >= start_depth && 0 <= end_depth) && in_row_bounds ? 1 : 0);

    warp_value_count = __popc(warp_count_mask);
    // Note : ((1 << t) - 1) implies "for all threads before me"
    thread_value_count = __popc(warp_count_mask & ((1 << t) - 1));

    // walk from 0 to max_depth
    uint32_t next_thread_value_count, next_warp_value_count;
    for (int s_idx = 0; s_idx < max_depth; s_idx++) {
      PageNestingInfo *pni = &s->page.nesting[s_idx];

      // if we are within the range of nesting levels we should be adding value indices for
      int in_nesting_bounds =
        ((s_idx >= start_depth && s_idx <= end_depth) && in_row_bounds) ? 1 : 0;

      // everything up to the max_def_level is a non-null value
      uint32_t is_valid = 0;
      if (d >= pni->max_def_level && in_nesting_bounds) { is_valid = 1; }

      // compute warp and thread valid counts
      uint32_t warp_valid_mask;
      // for flat schemas, a simple ballot_sync gives us the correct count and bit positions because
      // every value in the input matches to a value in the output
      if (max_depth == 0) {
        warp_valid_mask = BALLOT(is_valid);
      }
      // for nested schemas, it's more complicated.  This warp will visit 32 incoming values,
      // however not all of them will necessarily represent a value at this nesting level. so the
      // validity bit for thread t might actually represent output value t-6. the correct position
      // for thread t's bit is cur_value_count. for cuda 11 we could use __reduce_or_sync(), but
      // until then we have to do a warp reduce.
      else {
        warp_valid_mask = WarpReduceOr32(is_valid << thread_value_count);
      }
      thread_valid_count = __popc(warp_valid_mask & ((1 << thread_value_count) - 1));
      warp_valid_count   = __popc(warp_valid_mask);

      // if this is the value column emit an index for value decoding
      if (is_valid && s_idx == max_depth - 1) {
        int idx                       = pni->valid_count + thread_valid_count;
        int ofs                       = pni->value_count + thread_value_count;
        s->nz_idx[rolling_index(idx)] = ofs;
      }

      // compute warp and thread value counts for the -next- nesting level. we need to
      // do this for nested schemas so that we can emit an offset for the -current- nesting
      // level. more concretely : the offset for the current nesting level == current length of the
      // next nesting level
      if (s_idx < max_depth - 1) {
        uint32_t next_warp_count_mask =
          BALLOT((s_idx + 1 >= start_depth && s_idx + 1 <= end_depth && in_row_bounds) ? 1 : 0);
        next_warp_value_count   = __popc(next_warp_count_mask);
        next_thread_value_count = __popc(next_warp_count_mask & ((1 << t) - 1));

        // if we're -not- at a leaf column and we're within nesting/row bounds
        // and we have a valid data_out pointer, it implies this is a list column, so
        // emit an offset.
        if (in_nesting_bounds && pni->data_out != nullptr) {
          int idx             = pni->value_count + thread_value_count;
          cudf::size_type ofs = s->page.nesting[s_idx + 1].value_count + next_thread_value_count +
                                s->page.nesting[s_idx + 1].page_start_value;
          (reinterpret_cast<cudf::size_type *>(pni->data_out))[idx] = ofs;
        }
      }

      // increment count of valid values, count of total values, and validity mask
      if (!t) {
        if (pni->valid_map != nullptr && in_row_bounds) {
          store_validity(pni, warp_valid_mask, warp_value_count);
        }
        pni->valid_count += warp_valid_count;
        pni->value_count += warp_value_count;
      }

      // propagate value counts for the next level
      warp_value_count   = next_warp_value_count;
      thread_value_count = next_thread_value_count;
    }

    input_value_count += min(32, (target_input_value_count - input_value_count));
    SYNCWARP();
  }

  // update
  if (!t) {
    // update valid value count for decoding and total # of values we've processed
    s->nz_count          = s->page.nesting[max_depth - 1].valid_count;
    s->input_value_count = input_value_count;
    s->input_row_count   = input_row_count;
  }
}

/**
 * @brief Process repetition and definition levels up to the target count of leaf values.
 *
 * In order to decode actual leaf values from the input stream, we need to generate the
 * list of non-null value positions (page_state_s::nz_idx). We do this by processing
 * the repetition and definition level streams.  This process also generates validity information,
 * and offset column values in the case of nested schemas. Because of the way the streams
 * are encoded, this function may generate slightly more than target_leaf_count.
 *
 * Only runs on 1 warp.
 *
 * @param[in] s The local page state
 * @param[in] target_leaf_count Target count of non-null leaf values to generate indices for
 * @param[in] t Thread index
 */
__device__ void gpuDecodeLevels(page_state_s *s, int32_t target_leaf_count, int t)
{
  bool has_repetition = s->col.max_level[level_type::REPETITION] > 0;

  constexpr int batch_size = 32;
  int cur_leaf_count       = target_leaf_count;
  while (!s->error && s->nz_count < target_leaf_count &&
         s->input_value_count < s->num_input_values) {
    if (has_repetition) { gpuDecodeStream(s->rep, s, cur_leaf_count, t, level_type::REPETITION); }
    gpuDecodeStream(s->def, s, cur_leaf_count, t, level_type::DEFINITION);
    SYNCWARP();

    // because the rep and def streams are encoded seperately, we cannot request an exact
    // # of values to be decoded at once. we can only process the lowest # of decoded rep/def
    // levels we get.
    int actual_leaf_count = has_repetition ? min(s->lvl_count[level_type::REPETITION],
                                                 s->lvl_count[level_type::DEFINITION])
                                           : s->lvl_count[level_type::DEFINITION];

    // process what we got back
    gpuUpdateValidityOffsetsAndRowIndices(actual_leaf_count, s, t);
    cur_leaf_count = actual_leaf_count + batch_size;
    SYNCWARP();
  }
}

/**
 * @brief Process a batch of incoming repetition/definition level values to generate
 *        per-nesting level output column size for this page.
 *
 * Each page represents one piece of the overall output column. The total output (cudf)
 * column sizes are the sum of the values in each individual page.
 *
 * @param[in] s The local page info
 * @param[in] target_input_value_count The # of repetition/definition levels to process up to
 * @param[in] t Thread index
 * @param[in] bounds_set Whether or not s->row_index_lower_bound, s->first_row and s->num_rows
 * have been computed for this page (they will only be set in the second/trim pass).
 */
static __device__ void gpuUpdatePageSizes(page_state_s *s,
                                          int32_t target_input_value_count,
                                          int t,
                                          bool bounds_set)
{
  // max nesting depth of the column
  int max_depth = s->col.max_nesting_depth;
  // bool has_repetition = s->col.max_level[level_type::REPETITION] > 0 ? true : false;
  // how many input level values we've processed in the page so far
  int input_value_count = s->input_value_count;
  // how many leaf values we've processed in the page so far
  int input_leaf_count = s->input_leaf_count;
  // how many rows we've processed in the page so far
  int input_row_count = s->input_row_count;

  while (input_value_count < target_input_value_count) {
    int start_depth, end_depth, d;
    get_nesting_bounds(
      start_depth, end_depth, d, s, input_value_count, target_input_value_count, t);

    // count rows and leaf values
    int is_new_row                = start_depth == 0 ? 1 : 0;
    uint32_t warp_row_count_mask  = BALLOT(is_new_row);
    int is_new_leaf               = (d >= s->page.nesting[max_depth - 1].max_def_level) ? 1 : 0;
    uint32_t warp_leaf_count_mask = BALLOT(is_new_leaf);

    // is this thread within row bounds? on the first pass we don't know the bounds, so we will be
    // computing the full size of the column.  on the second pass, we will know our actual row
    // bounds, so the computation will cap sizes properly.
    int in_row_bounds = 1;
    if (bounds_set) {
      // absolute row index
      int32_t thread_row_index =
        input_row_count + ((__popc(warp_row_count_mask & ((1 << t) - 1)) + is_new_row) - 1);
      in_row_bounds = thread_row_index >= s->row_index_lower_bound &&
                          thread_row_index < (s->first_row + s->num_rows)
                        ? 1
                        : 0;

      uint32_t row_bounds_mask  = BALLOT(in_row_bounds);
      int first_thread_in_range = __ffs(row_bounds_mask) - 1;

      // if we've found the beginning of the first row, mark down the position
      // in the def/repetition buffer (skipped_values) and the data buffer (skipped_leaf_values)
      if (!t && first_thread_in_range >= 0 && s->page.skipped_values < 0) {
        // how many values we've skipped in the rep/def levels
        s->page.skipped_values = input_value_count + first_thread_in_range;
        // how many values we've skipped in the actual data stream
        s->page.skipped_leaf_values =
          input_leaf_count + __popc(warp_leaf_count_mask & ((1 << first_thread_in_range) - 1));
      }
    }

    // increment counts across all nesting depths
    for (int s_idx = 0; s_idx < max_depth; s_idx++) {
      // if we are within the range of nesting levels we should be adding value indices for
      int in_nesting_bounds = (s_idx >= start_depth && s_idx <= end_depth && in_row_bounds) ? 1 : 0;

      uint32_t count_mask = BALLOT(in_nesting_bounds);
      if (!t) { s->page.nesting[s_idx].size += __popc(count_mask); }
    }

    input_value_count += min(32, (target_input_value_count - input_value_count));
    input_row_count += __popc(warp_row_count_mask);
    input_leaf_count += __popc(warp_leaf_count_mask);
  }

  // update final page value count
  if (!t) {
    s->input_value_count = target_input_value_count;
    s->input_leaf_count  = input_leaf_count;
    s->input_row_count   = input_row_count;
  }
}

/**
 * @brief Kernel for computing per-page column size information for all nesting levels.
 *
 * This function will write out the size field for each level of nesting.
 *
 * @param[in,out] pages List of pages
 * @param[in] chunks List of column chunks
 * @param[in] num_chunks Number of column chunks
 * @param[in] min_row Row index to start reading at
 * @param[in] num_rows Maximum number of rows to read
 * @param[in] num_chunks Number of column chunks
 * @param[in] trim_pass Whether or not this is the trim pass.  We first have to compute
 * the full size information of every page before we come through in a second (trim) pass
 * to determine what subset of rows in this page we should be reading.
 */
// blockDim {NTHREADS,1,1}
extern "C" __global__ void __launch_bounds__(NTHREADS)
  gpuComputePageSizes(PageInfo *pages,
                      ColumnChunkDesc const *chunks,
                      size_t min_row,
                      size_t num_rows,
                      int32_t num_chunks,
                      bool trim_pass)
{
  __shared__ __align__(16) page_state_s state_g;

  page_state_s *const s = &state_g;
  int page_idx          = blockIdx.x;
  int t                 = threadIdx.x;
  PageInfo *pp          = &pages[page_idx];

  if (!setupLocalPageInfo(
        s, pp, chunks, trim_pass ? min_row : 0, trim_pass ? num_rows : INT_MAX, num_chunks)) {
    return;
  }

  // zero sizes
  int d = 0;
  while (d < s->page.num_nesting_levels) {
    if (d + t < s->page.num_nesting_levels) { s->page.nesting[d + t].size = 0; }
    d += blockDim.x;
  }
  if (!t) {
    s->page.skipped_values      = -1;
    s->page.skipped_leaf_values = -1;
    s->input_row_count          = 0;
    s->input_value_count        = 0;
    // if this isn't the trim pass, make sure we visit absolutely everything
    if (!trim_pass) {
      s->first_row             = 0;
      s->num_rows              = INT_MAX;
      s->row_index_lower_bound = -1;
    }
  }
  __syncthreads();

  bool has_repetition = s->col.max_level[level_type::REPETITION] > 0;

  // optimization : it might be useful to have a version of gpuDecodeStream that could go
  // wider than 1 warp.  Currently it only only uses 1 warp so that it can overlap work
  // with the value decoding step when in the actual value decoding kernel.  however during
  // this preprocess step we have no such limits -  we could go as wide as NTHREADS
  if (t < 32) {
    constexpr int batch_size = 32;
    int target_input_count   = batch_size;
    while (!s->error && s->input_value_count < s->num_input_values) {
      // decode repetition and definition levels. these will attempt to decode at
      // least up to the target, but may decode a few more.
      if (has_repetition) {
        gpuDecodeStream(s->rep, s, target_input_count, t, level_type::REPETITION);
      }
      gpuDecodeStream(s->def, s, target_input_count, t, level_type::DEFINITION);
      SYNCWARP();

      // we may have decoded different amounts from each stream, so only process what we've been
      int actual_input_count = has_repetition ? min(s->lvl_count[level_type::REPETITION],
                                                    s->lvl_count[level_type::DEFINITION])
                                              : s->lvl_count[level_type::DEFINITION];

      // process what we got back
      gpuUpdatePageSizes(s, actual_input_count, t, trim_pass);
      target_input_count = actual_input_count + batch_size;
      SYNCWARP();
    }
  }
  // update # rows in the actual page
  if (!t) {
    pp->num_rows            = s->page.nesting[0].size;
    pp->skipped_values      = s->page.skipped_values;
    pp->skipped_leaf_values = s->page.skipped_leaf_values;
  }
}

/**
 * @brief Kernel for co the column data stored in the pages
 *
 * This function will write the page data and the page data's validity to the
 * output specified in the page's column chunk. If necessary, additional
 * conversion will be performed to translate from the Parquet datatype to
 * desired output datatype (ex. 32-bit to 16-bit, string to hash).
 *
 * @param[in] pages List of pages
 * @param[in,out] chunks List of column chunks
 * @param[in] min_row Row index to start reading at
 * @param[in] num_rows Maximum number of rows to read
 * @param[in] num_chunks Number of column chunks
 */
// blockDim {NTHREADS,1,1}
extern "C" __global__ void __launch_bounds__(NTHREADS)
  gpuDecodePageData(PageInfo *pages,
                    ColumnChunkDesc const *chunks,
                    size_t min_row,
                    size_t num_rows,
                    int32_t num_chunks)
{
  __shared__ __align__(16) page_state_s state_g;

  page_state_s *const s = &state_g;
  int page_idx          = blockIdx.x;
  int t                 = threadIdx.x;
  int out_thread0;

  if (!setupLocalPageInfo(s, &pages[page_idx], chunks, min_row, num_rows, num_chunks)) { return; }

  if (s->dict_base) {
    out_thread0 = (s->dict_bits > 0) ? 64 : 32;
  } else {
    out_thread0 =
      ((s->col.data_type & 7) == BOOLEAN || (s->col.data_type & 7) == BYTE_ARRAY) ? 64 : 32;
  }

  uint32_t skipped_leaf_values = s->page.skipped_leaf_values;
  while (!s->error && (s->input_value_count < s->num_input_values || s->out_pos < s->nz_count)) {
    int target_pos;
    int out_pos = s->out_pos;

    if (t < out_thread0) {
      target_pos =
        min(out_pos + 2 * (NTHREADS - out_thread0), s->nz_count + (NTHREADS - out_thread0));
    } else {
      target_pos = min(s->nz_count, out_pos + NTHREADS - out_thread0);
      if (out_thread0 > 32) { target_pos = min(target_pos, s->dict_pos); }
    }
    __syncthreads();
    if (t < 32) {
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels(s, target_pos, t);
    } else if (t < out_thread0) {
      uint32_t src_target_pos = target_pos + skipped_leaf_values;

      // WARP1: Decode dictionary indices, booleans or string positions
      if (s->dict_base) {
        src_target_pos = gpuDecodeDictionaryIndices(s, src_target_pos, t & 0x1f);
      } else if ((s->col.data_type & 7) == BOOLEAN) {
        src_target_pos = gpuDecodeRleBooleans(s, src_target_pos, t & 0x1f);
      } else if ((s->col.data_type & 7) == BYTE_ARRAY) {
        gpuInitStringDescriptors(s, src_target_pos, t & 0x1f);
      }
      if (t == 32) { *(volatile int32_t *)&s->dict_pos = src_target_pos; }
    } else {
      // WARP1..WARP3: Decode values
      int dtype = s->col.data_type & 7;
      out_pos += t - out_thread0;
      uint32_t src_pos = out_pos + skipped_leaf_values;

      int output_value_idx = s->nz_idx[rolling_index(out_pos)];

      if (out_pos < target_pos && output_value_idx >= 0 && output_value_idx < s->num_input_values) {
        // nesting level that is storing actual leaf values
        int leaf_level_index = s->col.max_nesting_depth - 1;

        uint32_t dtype_len = s->dtype_len;
        void *dst          = s->page.nesting[leaf_level_index].data_out +
                    static_cast<size_t>(output_value_idx) * dtype_len;
        if (dtype == BYTE_ARRAY)
          gpuOutputString(s, src_pos, dst);
        else if (dtype == BOOLEAN)
          gpuOutputBoolean(s, src_pos, static_cast<uint8_t *>(dst));
        else if (s->col.converted_type == DECIMAL)
          gpuOutputDecimal(s, src_pos, static_cast<double *>(dst), dtype);
        else if (dtype == INT96)
          gpuOutputInt96Timestamp(s, src_pos, static_cast<int64_t *>(dst));
        else if (dtype_len == 8) {
          if (s->ts_scale)
            gpuOutputInt64Timestamp(s, src_pos, static_cast<int64_t *>(dst));
          else
            gpuOutputFast(s, src_pos, static_cast<uint2 *>(dst));
        } else if (dtype_len == 4)
          gpuOutputFast(s, src_pos, static_cast<uint32_t *>(dst));
        else
          gpuOutputGeneric(s, src_pos, static_cast<uint8_t *>(dst), dtype_len);
      }

      if (t == out_thread0) { *(volatile int32_t *)&s->out_pos = target_pos; }
    }
    __syncthreads();
  }
}

struct chunk_row_output_iter {
  PageInfo *p;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type *;
  using reference         = size_type &;
  using iterator_category = thrust::output_device_iterator_tag;

  chunk_row_output_iter operator+ __host__ __device__(int i)
  {
    return chunk_row_output_iter{p + i};
  }

  void operator++ __host__ __device__() { p++; }

  reference operator[] __device__(int i) { return p[i].chunk_row; }
  reference operator*__device__() { return p->chunk_row; }
  void operator= __device__(value_type v) { p->chunk_row = v; }
};

struct start_offset_output_iterator {
  PageInfo *pages;
  int *page_indices;
  int cur_index;
  int src_col_schema;
  int nesting_depth;
  int empty               = 0;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type *;
  using reference         = size_type &;
  using iterator_category = thrust::output_device_iterator_tag;

  start_offset_output_iterator operator+ __host__ __device__(int i)
  {
    return start_offset_output_iterator{
      pages, page_indices, cur_index + i, src_col_schema, nesting_depth};
  }

  void operator++ __host__ __device__() { cur_index++; }

  reference operator[] __device__(int i) { return dereference(cur_index + i); }
  reference operator*__device__() { return dereference(cur_index); }

 private:
  reference __device__ dereference(int index)
  {
    PageInfo const &p = pages[page_indices[index]];
    if (p.src_col_schema != src_col_schema || p.flags & PAGEINFO_FLAGS_DICTIONARY) { return empty; }
    return p.nesting[nesting_depth].page_start_value;
  }
};

/**
 * @copydoc cudf::io::parquet::gpu::PreprocessColumnData
 */
cudaError_t PreprocessColumnData(hostdevice_vector<PageInfo> &pages,
                                 hostdevice_vector<ColumnChunkDesc> const &chunks,
                                 std::vector<input_column_info> &input_columns,
                                 std::vector<cudf::io::detail::column_buffer> &output_columns,
                                 size_t num_rows,
                                 size_t min_row,
                                 cudaStream_t stream,
                                 rmm::mr::device_memory_resource *mr)
{
  dim3 dim_block(NTHREADS, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  // computes:
  // PageNestingInfo::size for each level of nesting, for each page.
  // The output from this does not take row bounds (num_rows, min_row) into account
  gpuComputePageSizes<<<dim_grid, dim_block, 0, stream>>>(
    pages.device_ptr(), chunks.device_ptr(), min_row, num_rows, chunks.size(), false);
  CUDA_TRY(cudaStreamSynchronize(stream));

  // computes:
  // PageInfo::chunk_row for all pages
  auto key_input = thrust::make_transform_iterator(
    pages.device_ptr(), [] __device__(PageInfo const &page) { return page.chunk_idx; });
  auto page_input = thrust::make_transform_iterator(
    pages.device_ptr(), [] __device__(PageInfo const &page) { return page.num_rows; });
  thrust::exclusive_scan_by_key(rmm::exec_policy(stream)->on(stream),
                                key_input,
                                key_input + pages.size(),
                                page_input,
                                chunk_row_output_iter{pages.device_ptr()});

  // computes:
  // PageNestingInfo::size for each level of nesting, for each page, taking row bounds into account.
  // PageInfo::skipped_values, which tells us where to start decoding in the input
  gpuComputePageSizes<<<dim_grid, dim_block, 0, stream>>>(
    pages.device_ptr(), chunks.device_ptr(), min_row, num_rows, chunks.size(), true);

  // retrieve pages back (PageInfo::num_rows has been set. if we don't bring it
  // back, this value will get overwritten later on).
  pages.device_to_host(stream, true);

  // ordering of pages is by input column schema, repeated across row groups.  so
  // if we had 3 columns, each with 2 pages, and 1 row group, our schema values might look like
  //
  // 1, 1, 2, 2, 3, 3
  //
  // However, if we had more than one row group, the pattern would be
  //
  // 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3
  // ^ row group 0     |
  //                   ^ row group 1
  //
  // To use exclusive_scan_by_key, the ordering we actually want is
  //
  // 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
  //
  // We also need to preserve key-relative page ordering, so we need to use a stable sort.
  rmm::device_uvector<int> page_keys(pages.size(), stream);
  rmm::device_uvector<int> page_index(pages.size(), stream);
  {
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      pages.device_ptr(),
                      pages.device_ptr() + pages.size(),
                      page_keys.begin(),
                      [] __device__(PageInfo const &page) { return page.src_col_schema; });

    thrust::sequence(rmm::exec_policy(stream)->on(stream), page_index.begin(), page_index.end());
    thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream),
                               page_keys.begin(),
                               page_keys.end(),
                               page_index.begin(),
                               thrust::less<int>());
  }

  // compute output column sizes by examining the pages of the -input- columns
  for (size_t idx = 0; idx < input_columns.size(); idx++) {
    auto const &input_col = input_columns[idx];
    auto src_col_schema   = input_col.schema_idx;
    size_t max_depth      = input_col.nesting_depth();

    auto *cols = &output_columns;
    for (size_t l_idx = 0; l_idx < input_col.nesting_depth(); l_idx++) {
      auto &out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      // size iterator. indexes pages by sorted order
      auto size_input = thrust::make_transform_iterator(
        page_index.begin(),
        [src_col_schema, l_idx, pages = pages.device_ptr()] __device__(int index) {
          auto const &page = pages[index];
          if (page.src_col_schema != src_col_schema || page.flags & PAGEINFO_FLAGS_DICTIONARY) {
            return 0;
          }
          return page.nesting[l_idx].size;
        });

      // compute column size.
      // for struct columns, higher levels of the output columns are shared between input
      // columns. so don't compute any given level more than once.
      if (out_buf.size == 0) {
        int size = thrust::reduce(
          rmm::exec_policy(stream)->on(stream), size_input, size_input + pages.size());

        // if this is a list column add 1 for non-leaf levels for the terminating offset
        if (out_buf.type.id() == type_id::LIST && l_idx < max_depth) { size++; }

        // allocate
        out_buf.create(size, stream, mr);
      }

      // compute per-page start offset
      thrust::exclusive_scan_by_key(rmm::exec_policy(stream)->on(stream),
                                    page_keys.begin(),
                                    page_keys.end(),
                                    size_input,
                                    start_offset_output_iterator{pages.device_ptr(),
                                                                 page_index.begin(),
                                                                 0,
                                                                 static_cast<int>(src_col_schema),
                                                                 static_cast<int>(l_idx)});
    }
  }

  return cudaSuccess;
}

/**
 * @copydoc cudf::io::parquet::gpu::DecodePageData
 */
cudaError_t __host__ DecodePageData(hostdevice_vector<PageInfo> &pages,
                                    hostdevice_vector<ColumnChunkDesc> const &chunks,
                                    size_t num_rows,
                                    size_t min_row,
                                    cudaStream_t stream)
{
  dim3 dim_block(NTHREADS, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  gpuDecodePageData<<<dim_grid, dim_block, 0, stream>>>(
    pages.device_ptr(), chunks.device_ptr(), min_row, num_rows, chunks.size());

  return cudaSuccess;
}

}  // namespace gpu
}  // namespace parquet
}  // namespace io
}  // namespace cudf
