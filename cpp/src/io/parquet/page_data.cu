/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include "parquet_gpu.hpp"
#include <io/utilities/block_utils.cuh>
#include <io/utilities/column_buffer.hpp>

#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

constexpr int block_size           = 128;
constexpr int non_zero_buffer_size = block_size * 2;

inline __device__ uint32_t rotl32(uint32_t x, uint32_t r)
{
  return __funnelshift_l(x, x, r);  // (x << r) | (x >> (32 - r));
}

inline __device__ int rolling_index(int index) { return index & (non_zero_buffer_size - 1); }

namespace cudf {
namespace io {
namespace parquet {
namespace gpu {

struct page_state_s {
  const uint8_t* data_start;
  const uint8_t* data_end;
  const uint8_t* lvl_end;
  const uint8_t* dict_base;    // ptr to dictionary page data
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
  int32_t src_pos;   // input read position of final output value
  int32_t ts_scale;  // timestamp scale: <0: divide by -ts_scale, >0: multiply by ts_scale
  uint32_t nz_idx[non_zero_buffer_size];    // circular buffer of non-null value positions
  uint32_t dict_idx[non_zero_buffer_size];  // Dictionary index, boolean, or string offset values
  uint32_t str_len[non_zero_buffer_size];   // String length for plain encoding of strings

  // repetition/definition level decoding
  int32_t input_value_count;                  // how many values of the input we've processed
  int32_t input_row_count;                    // how many rows of the input we've processed
  int32_t input_leaf_count;                   // how many leaf values of the input we've processed
  uint32_t rep[non_zero_buffer_size];         // circular buffer of repetition level values
  uint32_t def[non_zero_buffer_size];         // circular buffer of definition level values
  const uint8_t* lvl_start[NUM_LEVEL_TYPES];  // [def,rep]
  int32_t lvl_count[NUM_LEVEL_TYPES];         // how many of each of the streams we've decoded
  int32_t row_index_lower_bound;              // lower bound of row indices we should process
};

/**
 * @brief Read a 32-bit varint integer
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The 32-bit value read
 */
inline __device__ uint32_t get_vlq32(const uint8_t*& cur, const uint8_t* end)
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
__device__ uint32_t InitLevelSection(page_state_s* s,
                                     const uint8_t* cur,
                                     const uint8_t* end,
                                     level_type lvl)
{
  int32_t len;
  int level_bits    = s->col.level_bits[lvl];
  Encoding encoding = lvl == level_type::DEFINITION ? s->page.definition_level_encoding
                                                    : s->page.repetition_level_encoding;

  if (level_bits == 0) {
    len                       = 0;
    s->initial_rle_run[lvl]   = s->page.num_input_values * 2;  // repeated value
    s->initial_rle_value[lvl] = 0;
    s->lvl_start[lvl]         = cur;
  } else if (encoding == Encoding::RLE) {
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
  } else if (encoding == Encoding::BIT_PACKED) {
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
  uint32_t* output, page_state_s* s, int32_t target_count, int t, level_type lvl)
{
  const uint8_t* cur_def    = s->lvl_start[lvl];
  const uint8_t* end        = s->lvl_end;
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
        const uint8_t* cur = cur_def;
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
      sym_len   = shuffle(sym_len);
      level_val = shuffle(level_val);
      level_run = shuffle(level_run);
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
        const uint8_t* cur = cur_def + (bitpos >> 3);
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
      int idx                                  = value_count + t;
      output[idx & (non_zero_buffer_size - 1)] = level_val;
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
__device__ int gpuDecodeDictionaryIndices(volatile page_state_s* s, int target_pos, int t)
{
  const uint8_t* end = s->data_end;
  int dict_bits      = s->dict_bits;
  int pos            = s->dict_pos;

  while (pos < target_pos) {
    int is_literal, batch_len;
    if (!t) {
      uint32_t run       = s->dict_run;
      const uint8_t* cur = s->data_start;
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
    __syncwarp();
    is_literal = shuffle(is_literal);
    batch_len  = shuffle(batch_len);
    if (t < batch_len) {
      int dict_idx = s->dict_val;
      if (is_literal) {
        int32_t ofs      = (t - ((batch_len + 7) & ~7)) * dict_bits;
        const uint8_t* p = s->data_start + (ofs >> 3);
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
      s->dict_idx[(pos + t) & (non_zero_buffer_size - 1)] = dict_idx;
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
__device__ int gpuDecodeRleBooleans(volatile page_state_s* s, int target_pos, int t)
{
  const uint8_t* end = s->data_end;
  int pos            = s->dict_pos;

  while (pos < target_pos) {
    int is_literal, batch_len;
    if (!t) {
      uint32_t run       = s->dict_run;
      const uint8_t* cur = s->data_start;
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
    __syncwarp();
    is_literal = shuffle(is_literal);
    batch_len  = shuffle(batch_len);
    if (t < batch_len) {
      int dict_idx;
      if (is_literal) {
        int32_t ofs      = t - ((batch_len + 7) & ~7);
        const uint8_t* p = s->data_start + (ofs >> 3);
        dict_idx         = (p < end) ? (p[0] >> (ofs & 7u)) & 1 : 0;
      } else {
        dict_idx = s->dict_val;
      }
      s->dict_idx[(pos + t) & (non_zero_buffer_size - 1)] = dict_idx;
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
__device__ void gpuInitStringDescriptors(volatile page_state_s* s, int target_pos, int t)
{
  int pos = s->dict_pos;
  // This step is purely serial
  if (!t) {
    const uint8_t* cur = s->data_start;
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
      s->dict_idx[pos & (non_zero_buffer_size - 1)] = k;
      s->str_len[pos & (non_zero_buffer_size - 1)]  = len;
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
inline __device__ void gpuOutputString(volatile page_state_s* s, int src_pos, void* dstv)
{
  const char* ptr = nullptr;
  size_t len      = 0;

  if (s->dict_base) {
    // String dictionary
    uint32_t dict_pos = (s->dict_bits > 0) ? s->dict_idx[src_pos & (non_zero_buffer_size - 1)] *
                                               sizeof(string_index_pair)
                                           : 0;
    if (dict_pos < (uint32_t)s->dict_size) {
      const auto* src = reinterpret_cast<const string_index_pair*>(s->dict_base + dict_pos);
      ptr             = src->first;
      len             = src->second;
    }
  } else {
    // Plain encoding
    uint32_t dict_pos = s->dict_idx[src_pos & (non_zero_buffer_size - 1)];
    if (dict_pos <= (uint32_t)s->dict_size) {
      ptr = reinterpret_cast<const char*>(s->data_start + dict_pos);
      len = s->str_len[src_pos & (non_zero_buffer_size - 1)];
    }
  }
  if (s->dtype_len == 4) {
    // Output hash. This hash value is used if the option to convert strings to
    // categoricals is enabled. The seed value is chosen arbitrarily.
    uint32_t constexpr hash_seed = 33;
    cudf::string_view const sv{ptr, static_cast<size_type>(len)};
    *static_cast<uint32_t*>(dstv) = cudf::detail::MurmurHash3_32<cudf::string_view>{hash_seed}(sv);
  } else {
    // Output string descriptor
    auto* dst   = static_cast<string_index_pair*>(dstv);
    dst->first  = ptr;
    dst->second = len;
  }
}

/**
 * @brief Output a boolean
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
inline __device__ void gpuOutputBoolean(volatile page_state_s* s, int src_pos, uint8_t* dst)
{
  *dst = s->dict_idx[src_pos & (non_zero_buffer_size - 1)];
}

/**
 * @brief Store a 32-bit data element
 *
 * @param[out] dst ptr to output
 * @param[in] src8 raw input bytes
 * @param[in] dict_pos byte position in dictionary
 * @param[in] dict_size size of dictionary
 */
inline __device__ void gpuStoreOutput(uint32_t* dst,
                                      const uint8_t* src8,
                                      uint32_t dict_pos,
                                      uint32_t dict_size)
{
  uint32_t bytebuf;
  unsigned int ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos < dict_size) {
    bytebuf = *reinterpret_cast<const uint32_t*>(src8 + dict_pos);
    if (ofs) {
      uint32_t bytebufnext = *reinterpret_cast<const uint32_t*>(src8 + dict_pos + 4);
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
inline __device__ void gpuStoreOutput(uint2* dst,
                                      const uint8_t* src8,
                                      uint32_t dict_pos,
                                      uint32_t dict_size)
{
  uint2 v;
  unsigned int ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos < dict_size) {
    v.x = *reinterpret_cast<const uint32_t*>(src8 + dict_pos + 0);
    v.y = *reinterpret_cast<const uint32_t*>(src8 + dict_pos + 4);
    if (ofs) {
      uint32_t next = *reinterpret_cast<const uint32_t*>(src8 + dict_pos + 8);
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
 * @param[out] dst Pointer to row output data
 */
inline __device__ void gpuOutputInt96Timestamp(volatile page_state_s* s, int src_pos, int64_t* dst)
{
  using cuda::std::chrono::duration_cast;

  const uint8_t* src8;
  uint32_t dict_pos, dict_size = s->dict_size, ofs;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? s->dict_idx[src_pos & (non_zero_buffer_size - 1)] : 0;
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

  if (dict_pos + 4 >= dict_size) {
    *dst = 0;
    return;
  }

  uint3 v;
  int64_t nanos, days;
  v.x = *reinterpret_cast<const uint32_t*>(src8 + dict_pos + 0);
  v.y = *reinterpret_cast<const uint32_t*>(src8 + dict_pos + 4);
  v.z = *reinterpret_cast<const uint32_t*>(src8 + dict_pos + 8);
  if (ofs) {
    uint32_t next = *reinterpret_cast<const uint32_t*>(src8 + dict_pos + 12);
    v.x           = __funnelshift_r(v.x, v.y, ofs);
    v.y           = __funnelshift_r(v.y, v.z, ofs);
    v.z           = __funnelshift_r(v.z, next, ofs);
  }
  nanos = v.y;
  nanos <<= 32;
  nanos |= v.x;
  // Convert from Julian day at noon to UTC seconds
  days = static_cast<int32_t>(v.z);
  cudf::duration_D d_d{
    days - 2440588};  // TBD: Should be noon instead of midnight, but this matches pyarrow

  *dst = [&]() {
    switch (s->col.ts_clock_rate) {
      case 1:  // seconds
        return duration_cast<duration_s>(d_d).count() +
               duration_cast<duration_s>(duration_ns{nanos}).count();
      case 1'000:  // milliseconds
        return duration_cast<duration_ms>(d_d).count() +
               duration_cast<duration_ms>(duration_ns{nanos}).count();
      case 1'000'000:  // microseconds
        return duration_cast<duration_us>(d_d).count() +
               duration_cast<duration_us>(duration_ns{nanos}).count();
      case 1'000'000'000:  // nanoseconds
      default: return duration_cast<cudf::duration_ns>(d_d).count() + nanos;
    }
  }();
}

/**
 * @brief Output a 64-bit timestamp
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
inline __device__ void gpuOutputInt64Timestamp(volatile page_state_s* s, int src_pos, int64_t* dst)
{
  const uint8_t* src8;
  uint32_t dict_pos, dict_size = s->dict_size, ofs;
  int64_t ts;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? s->dict_idx[src_pos & (non_zero_buffer_size - 1)] : 0;
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
    v.x = *reinterpret_cast<const uint32_t*>(src8 + dict_pos + 0);
    v.y = *reinterpret_cast<const uint32_t*>(src8 + dict_pos + 4);
    if (ofs) {
      uint32_t next = *reinterpret_cast<const uint32_t*>(src8 + dict_pos + 8);
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
 * @brief Output a fixed-length byte array as int.
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename T>
__device__ void gpuOutputFixedLenByteArrayAsInt(volatile page_state_s* s, int src_pos, T* dst)
{
  uint32_t const dtype_len_in = s->dtype_len_in;
  uint8_t const* data         = s->dict_base ? s->dict_base : s->data_start;
  uint32_t const pos =
    (s->dict_base ? ((s->dict_bits > 0) ? s->dict_idx[src_pos & (non_zero_buffer_size - 1)] : 0)
                  : src_pos) *
    dtype_len_in;
  uint32_t const dict_size = s->dict_size;

  T unscaled = 0;
  for (unsigned int i = 0; i < dtype_len_in; i++) {
    uint32_t v = (pos + i < dict_size) ? data[pos + i] : 0;
    unscaled   = (unscaled << 8) | v;
  }
  // Shift the unscaled value up and back down when it isn't all 8 bytes,
  // which sign extend the value for correctly representing negative numbers.
  if (dtype_len_in < sizeof(T)) {
    unscaled <<= (sizeof(T) - dtype_len_in) * 8;
    unscaled >>= (sizeof(T) - dtype_len_in) * 8;
  }
  *dst = unscaled;
}

/**
 * @brief Output a small fixed-length value
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename T>
inline __device__ void gpuOutputFast(volatile page_state_s* s, int src_pos, T* dst)
{
  const uint8_t* dict;
  uint32_t dict_pos, dict_size = s->dict_size;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? s->dict_idx[src_pos & (non_zero_buffer_size - 1)] : 0;
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
static __device__ void gpuOutputGeneric(volatile page_state_s* s,
                                        int src_pos,
                                        uint8_t* dst8,
                                        int len)
{
  const uint8_t* dict;
  uint32_t dict_pos, dict_size = s->dict_size;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? s->dict_idx[src_pos & (non_zero_buffer_size - 1)] : 0;
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
    const uint8_t* src8 = dict;
    unsigned int ofs    = 3 & reinterpret_cast<size_t>(src8);
    src8 -= ofs;  // align to 32-bit boundary
    ofs <<= 3;    // bytes -> bits
    for (unsigned int i = 0; i < len; i += 4) {
      uint32_t bytebuf;
      if (dict_pos < dict_size) {
        bytebuf = *reinterpret_cast<const uint32_t*>(src8 + dict_pos);
        if (ofs) {
          uint32_t bytebufnext = *reinterpret_cast<const uint32_t*>(src8 + dict_pos + 4);
          bytebuf              = __funnelshift_r(bytebuf, bytebufnext, ofs);
        }
      } else {
        bytebuf = 0;
      }
      dict_pos += 4;
      *reinterpret_cast<uint32_t*>(dst8 + i) = bytebuf;
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
 * @param[in] min_row Crop all rows below min_row
 */
static __device__ bool setupLocalPageInfo(page_state_s* const s,
                                          PageInfo const* p,
                                          device_span<ColumnChunkDesc const> chunks,
                                          size_t min_row,
                                          size_t num_rows)
{
  int t = threadIdx.x;
  int chunk_idx;

  // Fetch page info
  if (t == 0) s->page = *p;
  __syncthreads();

  if (s->page.flags & PAGEINFO_FLAGS_DICTIONARY) { return false; }
  // Fetch column chunk info
  chunk_idx = s->page.chunk_idx;
  if (t == 0) { s->col = chunks[chunk_idx]; }

  // zero nested value and valid counts
  int d = 0;
  while (d < s->page.num_nesting_levels) {
    if (d + t < s->page.num_nesting_levels) {
      s->page.nesting[d + t].valid_count = 0;
      s->page.nesting[d + t].value_count = 0;
      s->page.nesting[d + t].null_count  = 0;
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
      uint8_t* cur = s->page.page_data;
      uint8_t* end = cur + s->page.uncompressed_page_size;

      uint32_t dtype_len_out = s->col.data_type >> 3;
      s->ts_scale            = 0;
      // Validate data type
      auto const data_type = s->col.data_type & 7;
      switch (data_type) {
        case BOOLEAN:
          s->dtype_len = 1;  // Boolean are stored as 1 byte on the output
          break;
        case INT32:
        case FLOAT: s->dtype_len = 4; break;
        case INT64:
          if (s->col.ts_clock_rate) {
            int32_t units = 0;
            if (s->col.converted_type == TIME_MILLIS or s->col.converted_type == TIMESTAMP_MILLIS) {
              units = cudf::timestamp_ms::period::den;
            } else if (s->col.converted_type == TIME_MICROS or
                       s->col.converted_type == TIMESTAMP_MICROS) {
              units = cudf::timestamp_us::period::den;
            } else if (s->col.logical_type.TIMESTAMP.unit.isset.NANOS) {
              units = cudf::timestamp_ns::period::den;
            }
            if (units and units != s->col.ts_clock_rate) {
              s->ts_scale = (s->col.ts_clock_rate < units) ? -(units / s->col.ts_clock_rate)
                                                           : (s->col.ts_clock_rate / units);
            }
          }
          // Fall through to DOUBLE
        case DOUBLE: s->dtype_len = 8; break;
        case INT96: s->dtype_len = 12; break;
        case BYTE_ARRAY: s->dtype_len = sizeof(string_index_pair); break;
        default:  // FIXED_LEN_BYTE_ARRAY:
          s->dtype_len = dtype_len_out;
          s->error |= (s->dtype_len <= 0);
          break;
      }
      // Special check for downconversions
      s->dtype_len_in = s->dtype_len;
      if (s->col.converted_type == DECIMAL && data_type == FIXED_LEN_BYTE_ARRAY) {
        s->dtype_len = s->dtype_len <= sizeof(int32_t)   ? sizeof(int32_t)
                       : s->dtype_len <= sizeof(int64_t) ? sizeof(int64_t)
                                                         : sizeof(__int128_t);
      } else if (data_type == INT32) {
        if (dtype_len_out == 1) s->dtype_len = 1;  // INT8 output
        if (dtype_len_out == 2) s->dtype_len = 2;  // INT16 output
      } else if (data_type == BYTE_ARRAY && dtype_len_out == 4) {
        s->dtype_len = 4;  // HASH32 output
      } else if (data_type == INT96) {
        s->dtype_len = 8;  // Convert to 64-bit timestamp
      }

      // first row within the page to output
      if (page_start_row >= min_row) {
        s->first_row = 0;
      } else {
        s->first_row = (int32_t)min(min_row - page_start_row, (size_t)s->page.num_rows);
      }
      // # of rows within the page to output
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
          PageNestingInfo* pni = &s->page.nesting[idx];

          size_t output_offset;
          // schemas without lists
          if (s->col.max_level[level_type::REPETITION] == 0) {
            output_offset = page_start_row >= min_row ? page_start_row - min_row : 0;
          }
          // for schemas with lists, we've already got the exactly value precomputed
          else {
            output_offset = pni->page_start_value;
          }

          pni->data_out = static_cast<uint8_t*>(s->col.column_data_base[idx]);
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
      s->dict_base = nullptr;
      s->dict_size = 0;
      switch (s->page.encoding) {
        case Encoding::PLAIN_DICTIONARY:
        case Encoding::RLE_DICTIONARY:
          // RLE-packed dictionary indices, first byte indicates index length in bits
          if (((s->col.data_type & 7) == BYTE_ARRAY) && (s->col.str_dict_index)) {
            // String dictionary: use index
            s->dict_base = reinterpret_cast<const uint8_t*>(s->col.str_dict_index);
            s->dict_size = s->col.page_info[0].num_input_values * sizeof(string_index_pair);
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
        case Encoding::PLAIN:
          s->dict_size = static_cast<int32_t>(end - cur);
          s->dict_val  = 0;
          if ((s->col.data_type & 7) == BOOLEAN) { s->dict_run = s->dict_size * 2 + 1; }
          break;
        case Encoding::RLE: s->dict_run = 0; break;
        default:
          s->error = 1;  // Unsupported encoding
          break;
      }
      if (cur > end) { s->error = 1; }
      s->lvl_end    = cur;
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
    s->src_pos                           = 0;

    // for flat hierarchies, we can't know how many leaf values to skip unless we do a full
    // preprocess of the definition levels (since nulls will have no actual decodable value, there
    // is no direct correlation between # of rows and # of decodable values).  so we will start
    // processing at the beginning of the value stream and disregard any indices that start
    // before the first row.
    if (s->col.max_level[level_type::REPETITION] == 0) {
      s->page.skipped_values      = 0;
      s->page.skipped_leaf_values = 0;
      s->input_value_count        = 0;
      s->input_row_count          = 0;

      s->row_index_lower_bound = -1;
    }
    // for nested hierarchies, we have run a preprocess that lets us skip directly to the values
    // we need to start decoding at
    else {
      // input_row_count translates to "how many rows we have processed so far", so since we are
      // skipping directly to where we want to start decoding, set it to first_row
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
      int const max_row = (min_row + num_rows) - 1;
      if (min_row < page_start_row && max_row >= page_start_row - 1) {
        s->row_index_lower_bound = -1;
      } else {
        s->row_index_lower_bound = s->first_row;
      }

      // if we're in the decoding step, jump directly to the first
      // value we care about
      if (s->col.column_data_base != nullptr) {
        s->input_value_count = s->page.skipped_values > -1 ? s->page.skipped_values : 0;
      } else {
        s->input_value_count        = 0;
        s->input_leaf_count         = 0;
        s->page.skipped_values      = -1;
        s->page.skipped_leaf_values = -1;
      }
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
static __device__ void store_validity(PageNestingInfo* pni,
                                      uint32_t valid_mask,
                                      int32_t value_count)
{
  int word_offset = pni->valid_map_offset / 32;
  int bit_offset  = pni->valid_map_offset % 32;
  // if we fit entirely in the output word
  if (bit_offset + value_count <= 32) {
    auto relevant_mask = static_cast<uint32_t>((static_cast<uint64_t>(1) << value_count) - 1);

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
inline __device__ void get_nesting_bounds(int& start_depth,
                                          int& end_depth,
                                          int& d,
                                          page_state_s* s,
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
                                                             page_state_s* s,
                                                             int t)
{
  // max nesting depth of the column
  int const max_depth = s->col.max_nesting_depth;
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
    int const is_new_row               = start_depth == 0 ? 1 : 0;
    uint32_t const warp_row_count_mask = ballot(is_new_row);
    int32_t const thread_row_index =
      input_row_count + ((__popc(warp_row_count_mask & ((1 << t) - 1)) + is_new_row) - 1);
    input_row_count += __popc(warp_row_count_mask);
    // is this thread within read row bounds?
    int const in_row_bounds = thread_row_index >= s->row_index_lower_bound &&
                                  thread_row_index < (s->first_row + s->num_rows)
                                ? 1
                                : 0;

    // compute warp and thread value counts
    uint32_t const warp_count_mask =
      ballot((0 >= start_depth && 0 <= end_depth) && in_row_bounds ? 1 : 0);

    warp_value_count = __popc(warp_count_mask);
    // Note : ((1 << t) - 1) implies "for all threads before me"
    thread_value_count = __popc(warp_count_mask & ((1 << t) - 1));

    // walk from 0 to max_depth
    uint32_t next_thread_value_count, next_warp_value_count;
    for (int s_idx = 0; s_idx < max_depth; s_idx++) {
      PageNestingInfo* pni = &s->page.nesting[s_idx];

      // if we are within the range of nesting levels we should be adding value indices for
      int const in_nesting_bounds =
        ((s_idx >= start_depth && s_idx <= end_depth) && in_row_bounds) ? 1 : 0;

      // everything up to the max_def_level is a non-null value
      uint32_t const is_valid = d >= pni->max_def_level && in_nesting_bounds ? 1 : 0;

      // compute warp and thread valid counts
      uint32_t const warp_valid_mask =
        // for flat schemas, a simple ballot_sync gives us the correct count and bit positions
        // because every value in the input matches to a value in the output
        max_depth == 1
          ? ballot(is_valid)
          :
          // for nested schemas, it's more complicated.  This warp will visit 32 incoming values,
          // however not all of them will necessarily represent a value at this nesting level. so
          // the validity bit for thread t might actually represent output value t-6. the correct
          // position for thread t's bit is cur_value_count. for cuda 11 we could use
          // __reduce_or_sync(), but until then we have to do a warp reduce.
          WarpReduceOr32(is_valid << thread_value_count);

      thread_valid_count = __popc(warp_valid_mask & ((1 << thread_value_count) - 1));
      warp_valid_count   = __popc(warp_valid_mask);

      // if this is the value column emit an index for value decoding
      if (is_valid && s_idx == max_depth - 1) {
        int const src_pos = pni->valid_count + thread_valid_count;
        int const dst_pos = pni->value_count + thread_value_count;
        // nz_idx is a mapping of src buffer indices to destination buffer indices
        s->nz_idx[rolling_index(src_pos)] = dst_pos;
      }

      // compute warp and thread value counts for the -next- nesting level. we need to
      // do this for nested schemas so that we can emit an offset for the -current- nesting
      // level. more concretely : the offset for the current nesting level == current length of the
      // next nesting level
      if (s_idx < max_depth - 1) {
        uint32_t const next_warp_count_mask =
          ballot((s_idx + 1 >= start_depth && s_idx + 1 <= end_depth && in_row_bounds) ? 1 : 0);
        next_warp_value_count   = __popc(next_warp_count_mask);
        next_thread_value_count = __popc(next_warp_count_mask & ((1 << t) - 1));

        // if we're -not- at a leaf column and we're within nesting/row bounds
        // and we have a valid data_out pointer, it implies this is a list column, so
        // emit an offset.
        if (in_nesting_bounds && pni->data_out != nullptr) {
          int const idx             = pni->value_count + thread_value_count;
          cudf::size_type const ofs = s->page.nesting[s_idx + 1].value_count +
                                      next_thread_value_count +
                                      s->page.nesting[s_idx + 1].page_start_value;
          (reinterpret_cast<cudf::size_type*>(pni->data_out))[idx] = ofs;
        }
      }

      // nested schemas always read and write to the same bounds (that is, read and write positions
      // are already pre-bounded by first_row/num_rows). flat schemas will start reading at the
      // first value, even if that is before first_row, because we cannot trivially jump to
      // the correct position to start reading. since we are about to write the validity vector here
      // we need to adjust our computed mask to take into account the write row bounds.
      int const in_write_row_bounds =
        max_depth == 1
          ? thread_row_index >= s->first_row && thread_row_index < (s->first_row + s->num_rows)
          : in_row_bounds;
      int const first_thread_in_write_range =
        max_depth == 1 ? __ffs(ballot(in_write_row_bounds)) - 1 : 0;
      // # of bits to of the validity mask to write out
      int const warp_valid_mask_bit_count =
        first_thread_in_write_range < 0 ? 0 : warp_value_count - first_thread_in_write_range;

      // increment count of valid values, count of total values, and update validity mask
      if (!t) {
        if (pni->valid_map != nullptr && warp_valid_mask_bit_count > 0) {
          uint32_t const warp_output_valid_mask = warp_valid_mask >> first_thread_in_write_range;
          store_validity(pni, warp_output_valid_mask, warp_valid_mask_bit_count);

          pni->null_count += warp_valid_mask_bit_count - __popc(warp_output_valid_mask);
        }
        pni->valid_count += warp_valid_count;
        pni->value_count += warp_value_count;
      }

      // propagate value counts for the next level
      warp_value_count   = next_warp_value_count;
      thread_value_count = next_thread_value_count;
    }

    input_value_count += min(32, (target_input_value_count - input_value_count));
    __syncwarp();
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
__device__ void gpuDecodeLevels(page_state_s* s, int32_t target_leaf_count, int t)
{
  bool has_repetition = s->col.max_level[level_type::REPETITION] > 0;

  constexpr int batch_size = 32;
  int cur_leaf_count       = target_leaf_count;
  while (!s->error && s->nz_count < target_leaf_count &&
         s->input_value_count < s->num_input_values) {
    if (has_repetition) { gpuDecodeStream(s->rep, s, cur_leaf_count, t, level_type::REPETITION); }
    gpuDecodeStream(s->def, s, cur_leaf_count, t, level_type::DEFINITION);
    __syncwarp();

    // because the rep and def streams are encoded separately, we cannot request an exact
    // # of values to be decoded at once. we can only process the lowest # of decoded rep/def
    // levels we get.
    int actual_leaf_count = has_repetition ? min(s->lvl_count[level_type::REPETITION],
                                                 s->lvl_count[level_type::DEFINITION])
                                           : s->lvl_count[level_type::DEFINITION];

    // process what we got back
    gpuUpdateValidityOffsetsAndRowIndices(actual_leaf_count, s, t);
    cur_leaf_count = actual_leaf_count + batch_size;
    __syncwarp();
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
static __device__ void gpuUpdatePageSizes(page_state_s* s,
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
    uint32_t warp_row_count_mask  = ballot(is_new_row);
    int is_new_leaf               = (d >= s->page.nesting[max_depth - 1].max_def_level) ? 1 : 0;
    uint32_t warp_leaf_count_mask = ballot(is_new_leaf);

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

      uint32_t row_bounds_mask  = ballot(in_row_bounds);
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

      uint32_t count_mask = ballot(in_nesting_bounds);
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
 * @param pages List of pages
 * @param chunks List of column chunks
 * @param min_row Row index to start reading at
 * @param num_rows Maximum number of rows to read. Pass as INT_MAX to guarantee reading all rows.
 * @param trim_pass Whether or not this is the trim pass.  We first have to compute
 * the full size information of every page before we come through in a second (trim) pass
 * to determine what subset of rows in this page we should be reading.
 */
__global__ void __launch_bounds__(block_size)
  gpuComputePageSizes(PageInfo* pages,
                      device_span<ColumnChunkDesc const> chunks,
                      size_t min_row,
                      size_t num_rows,
                      bool trim_pass)
{
  __shared__ __align__(16) page_state_s state_g;

  page_state_s* const s = &state_g;
  int page_idx          = blockIdx.x;
  int t                 = threadIdx.x;
  PageInfo* pp          = &pages[page_idx];

  if (!setupLocalPageInfo(s, pp, chunks, trim_pass ? min_row : 0, trim_pass ? num_rows : INT_MAX)) {
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

  // optimization : it might be useful to have a version of gpuDecodeStream that could go wider than
  // 1 warp.  Currently it only uses 1 warp so that it can overlap work with the value decoding step
  // when in the actual value decoding kernel. However, during this preprocess step we have no such
  // limits -  we could go as wide as block_size
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
      __syncwarp();

      // we may have decoded different amounts from each stream, so only process what we've been
      int actual_input_count = has_repetition ? min(s->lvl_count[level_type::REPETITION],
                                                    s->lvl_count[level_type::DEFINITION])
                                              : s->lvl_count[level_type::DEFINITION];

      // process what we got back
      gpuUpdatePageSizes(s, actual_input_count, t, trim_pass);
      target_input_count = actual_input_count + batch_size;
      __syncwarp();
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
 * @param pages List of pages
 * @param chunks List of column chunks
 * @param min_row Row index to start reading at
 * @param num_rows Maximum number of rows to read
 */
__global__ void __launch_bounds__(block_size) gpuDecodePageData(
  PageInfo* pages, device_span<ColumnChunkDesc const> chunks, size_t min_row, size_t num_rows)
{
  __shared__ __align__(16) page_state_s state_g;

  page_state_s* const s = &state_g;
  int page_idx          = blockIdx.x;
  int t                 = threadIdx.x;
  int out_thread0;

  if (!setupLocalPageInfo(s, &pages[page_idx], chunks, min_row, num_rows)) { return; }

  if (s->dict_base) {
    out_thread0 = (s->dict_bits > 0) ? 64 : 32;
  } else {
    out_thread0 =
      ((s->col.data_type & 7) == BOOLEAN || (s->col.data_type & 7) == BYTE_ARRAY) ? 64 : 32;
  }

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t skipped_leaf_values = s->page.skipped_leaf_values;
  while (!s->error && (s->input_value_count < s->num_input_values || s->src_pos < s->nz_count)) {
    int target_pos;
    int src_pos = s->src_pos;

    if (t < out_thread0) {
      target_pos =
        min(src_pos + 2 * (block_size - out_thread0), s->nz_count + (block_size - out_thread0));
    } else {
      target_pos = min(s->nz_count, src_pos + block_size - out_thread0);
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
      // skipped_leaf_values will always be 0 for flat hierarchies.
      uint32_t src_target_pos = target_pos + skipped_leaf_values;

      // WARP1: Decode dictionary indices, booleans or string positions
      if (s->dict_base) {
        src_target_pos = gpuDecodeDictionaryIndices(s, src_target_pos, t & 0x1f);
      } else if ((s->col.data_type & 7) == BOOLEAN) {
        src_target_pos = gpuDecodeRleBooleans(s, src_target_pos, t & 0x1f);
      } else if ((s->col.data_type & 7) == BYTE_ARRAY) {
        gpuInitStringDescriptors(s, src_target_pos, t & 0x1f);
      }
      if (t == 32) { *(volatile int32_t*)&s->dict_pos = src_target_pos; }
    } else {
      // WARP1..WARP3: Decode values
      int dtype = s->col.data_type & 7;
      src_pos += t - out_thread0;

      // the position in the output column/buffer
      int dst_pos = s->nz_idx[rolling_index(src_pos)];

      // for the flat hierarchy case we will be reading from the beginning of the value stream,
      // regardless of the value of first_row. so adjust our destination offset accordingly.
      // example:
      // - user has passed skip_rows = 2, so our first_row to output is 2
      // - the row values we get from nz_idx will be
      //   0, 1, 2, 3, 4 ....
      // - by shifting these values by first_row, the sequence becomes
      //   -1, -2, 0, 1, 2 ...
      // - so we will end up ignoring the first two input rows, and input rows 2..n will
      //   get written to the output starting at position 0.
      //
      if (s->col.max_nesting_depth == 1) { dst_pos -= s->first_row; }

      // target_pos will always be properly bounded by num_rows, but dst_pos may be negative (values
      // before first_row) in the flat hierarchy case.
      if (src_pos < target_pos && dst_pos >= 0) {
        // src_pos represents the logical row position we want to read from. But in the case of
        // nested hierarchies, there is no 1:1 mapping of rows to values.  So our true read position
        // has to take into account the # of values we have to skip in the page to get to the
        // desired logical row.  For flat hierarchies, skipped_leaf_values will always be 0.
        uint32_t val_src_pos = src_pos + skipped_leaf_values;

        // nesting level that is storing actual leaf values
        int leaf_level_index = s->col.max_nesting_depth - 1;

        uint32_t dtype_len = s->dtype_len;
        void* dst =
          s->page.nesting[leaf_level_index].data_out + static_cast<size_t>(dst_pos) * dtype_len;
        if (dtype == BYTE_ARRAY) {
          gpuOutputString(s, val_src_pos, dst);
        } else if (dtype == BOOLEAN) {
          gpuOutputBoolean(s, val_src_pos, static_cast<uint8_t*>(dst));
        } else if (s->col.converted_type == DECIMAL) {
          switch (dtype) {
            case INT32: gpuOutputFast(s, val_src_pos, static_cast<uint32_t*>(dst)); break;
            case INT64: gpuOutputFast(s, val_src_pos, static_cast<uint2*>(dst)); break;
            default:
              if (s->dtype_len_in <= sizeof(int32_t)) {
                gpuOutputFixedLenByteArrayAsInt(s, val_src_pos, static_cast<int32_t*>(dst));
              } else if (s->dtype_len_in <= sizeof(int64_t)) {
                gpuOutputFixedLenByteArrayAsInt(s, val_src_pos, static_cast<int64_t*>(dst));
              } else {
                gpuOutputFixedLenByteArrayAsInt(s, val_src_pos, static_cast<__int128_t*>(dst));
              }
              break;
          }
        } else if (dtype == INT96) {
          gpuOutputInt96Timestamp(s, val_src_pos, static_cast<int64_t*>(dst));
        } else if (dtype_len == 8) {
          if (s->ts_scale) {
            gpuOutputInt64Timestamp(s, val_src_pos, static_cast<int64_t*>(dst));
          } else {
            gpuOutputFast(s, val_src_pos, static_cast<uint2*>(dst));
          }
        } else if (dtype_len == 4) {
          gpuOutputFast(s, val_src_pos, static_cast<uint32_t*>(dst));
        } else {
          gpuOutputGeneric(s, val_src_pos, static_cast<uint8_t*>(dst), dtype_len);
        }
      }

      if (t == out_thread0) { *(volatile int32_t*)&s->src_pos = target_pos; }
    }
    __syncthreads();
  }
}

struct chunk_row_output_iter {
  PageInfo* p;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type*;
  using reference         = size_type&;
  using iterator_category = thrust::output_device_iterator_tag;

  __host__ __device__ chunk_row_output_iter operator+(int i)
  {
    return chunk_row_output_iter{p + i};
  }

  __host__ __device__ void operator++() { p++; }

  __device__ reference operator[](int i) { return p[i].chunk_row; }
  __device__ reference operator*() { return p->chunk_row; }
  __device__ void operator=(value_type v) { p->chunk_row = v; }
};

struct start_offset_output_iterator {
  PageInfo* pages;
  int* page_indices;
  int cur_index;
  int src_col_schema;
  int nesting_depth;
  int empty               = 0;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type*;
  using reference         = size_type&;
  using iterator_category = thrust::output_device_iterator_tag;

  __host__ __device__ start_offset_output_iterator operator+(int i)
  {
    return start_offset_output_iterator{
      pages, page_indices, cur_index + i, src_col_schema, nesting_depth};
  }

  __host__ __device__ void operator++() { cur_index++; }

  __device__ reference operator[](int i) { return dereference(cur_index + i); }
  __device__ reference operator*() { return dereference(cur_index); }

 private:
  __device__ reference dereference(int index)
  {
    PageInfo const& p = pages[page_indices[index]];
    if (p.src_col_schema != src_col_schema || p.flags & PAGEINFO_FLAGS_DICTIONARY) { return empty; }
    return p.nesting[nesting_depth].page_start_value;
  }
};

/**
 * @copydoc cudf::io::parquet::gpu::PreprocessColumnData
 */
void PreprocessColumnData(hostdevice_vector<PageInfo>& pages,
                          hostdevice_vector<ColumnChunkDesc> const& chunks,
                          std::vector<input_column_info>& input_columns,
                          std::vector<cudf::io::detail::column_buffer>& output_columns,
                          size_t num_rows,
                          size_t min_row,
                          bool uses_custom_row_bounds,
                          rmm::cuda_stream_view stream,
                          rmm::mr::device_memory_resource* mr)
{
  dim3 dim_block(block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  // computes:
  // PageNestingInfo::size for each level of nesting, for each page.
  // This computes the size for the entire page, not taking row bounds into account.
  // If uses_custom_row_bounds is set to true, we have to do a second pass later that "trims"
  // the starting and ending read values to account for these bounds.
  gpuComputePageSizes<<<dim_grid, dim_block, 0, stream.value()>>>(
    pages.device_ptr(),
    chunks,
    // if uses_custom_row_bounds is false, include all possible rows.
    uses_custom_row_bounds ? min_row : 0,
    uses_custom_row_bounds ? num_rows : INT_MAX,
    !uses_custom_row_bounds);

  // computes:
  // PageInfo::chunk_row for all pages
  auto key_input = thrust::make_transform_iterator(
    pages.device_ptr(), [] __device__(PageInfo const& page) { return page.chunk_idx; });
  auto page_input = thrust::make_transform_iterator(
    pages.device_ptr(), [] __device__(PageInfo const& page) { return page.num_rows; });
  thrust::exclusive_scan_by_key(rmm::exec_policy(stream),
                                key_input,
                                key_input + pages.size(),
                                page_input,
                                chunk_row_output_iter{pages.device_ptr()});

  // computes:
  // PageNestingInfo::size for each level of nesting, for each page, taking row bounds into account.
  // PageInfo::skipped_values, which tells us where to start decoding in the input  .
  // It is only necessary to do this second pass if uses_custom_row_bounds is set (if the user has
  // specified artifical bounds).
  if (uses_custom_row_bounds) {
    gpuComputePageSizes<<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, true);
  }

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
    thrust::transform(rmm::exec_policy(stream),
                      pages.device_ptr(),
                      pages.device_ptr() + pages.size(),
                      page_keys.begin(),
                      [] __device__(PageInfo const& page) { return page.src_col_schema; });

    thrust::sequence(rmm::exec_policy(stream), page_index.begin(), page_index.end());
    thrust::stable_sort_by_key(rmm::exec_policy(stream),
                               page_keys.begin(),
                               page_keys.end(),
                               page_index.begin(),
                               thrust::less<int>());
  }

  // compute output column sizes by examining the pages of the -input- columns
  for (size_t idx = 0; idx < input_columns.size(); idx++) {
    auto const& input_col = input_columns[idx];
    auto src_col_schema   = input_col.schema_idx;
    size_t max_depth      = input_col.nesting_depth();

    auto* cols = &output_columns;
    for (size_t l_idx = 0; l_idx < input_col.nesting_depth(); l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      // size iterator. indexes pages by sorted order
      auto size_input = thrust::make_transform_iterator(
        page_index.begin(),
        [src_col_schema, l_idx, pages = pages.device_ptr()] __device__(int index) {
          auto const& page = pages[index];
          if (page.src_col_schema != src_col_schema || page.flags & PAGEINFO_FLAGS_DICTIONARY) {
            return 0;
          }
          return page.nesting[l_idx].size;
        });

      // compute column size.
      // for struct columns, higher levels of the output columns are shared between input
      // columns. so don't compute any given level more than once.
      if (out_buf.size == 0) {
        int size = thrust::reduce(rmm::exec_policy(stream), size_input, size_input + pages.size());

        // Handle a specific corner case.  It is possible to construct a parquet file such that
        // a column within a row group contains more rows than the row group itself. This may be
        // invalid, but we have seen instances of this in the wild, including how they were created
        // using the apache parquet tools.  Normally, the trim pass would handle this case quietly,
        // but if we are not running the trim pass (which is most of the time) we need to cap the
        // number of rows we will allocate/read from the file with the amount specified in the
        // associated row group. This only applies to columns that are not children of lists as
        // those may have an arbitrary number of rows in them.
        if (!uses_custom_row_bounds &&
            !(out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) &&
            size > static_cast<size_type>(num_rows)) {
          size = static_cast<size_type>(num_rows);
        }

        // if this is a list column add 1 for non-leaf levels for the terminating offset
        if (out_buf.type.id() == type_id::LIST && l_idx < max_depth) { size++; }

        // allocate
        out_buf.create(size, stream, mr);
      }

      // compute per-page start offset
      thrust::exclusive_scan_by_key(rmm::exec_policy(stream),
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

  // retrieve pages back
  pages.device_to_host(stream);
}

/**
 * @copydoc cudf::io::parquet::gpu::DecodePageData
 */
void __host__ DecodePageData(hostdevice_vector<PageInfo>& pages,
                             hostdevice_vector<ColumnChunkDesc> const& chunks,
                             size_t num_rows,
                             size_t min_row,
                             rmm::cuda_stream_view stream)
{
  dim3 dim_block(block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  gpuDecodePageData<<<dim_grid, dim_block, 0, stream.value()>>>(
    pages.device_ptr(), chunks, min_row, num_rows);
}

}  // namespace gpu
}  // namespace parquet
}  // namespace io
}  // namespace cudf
