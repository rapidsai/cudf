/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "parquet_gpu.hpp"
#include "rle_stream.cuh"
#include <io/utilities/block_utils.cuh>
#include <io/utilities/column_buffer.hpp>

#include <cuda/std/tuple>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
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

namespace cudf {
namespace io {
namespace parquet {
namespace gpu {

struct page_state_s {
  uint8_t const* data_start;
  uint8_t const* data_end;
  uint8_t const* lvl_end;
  uint8_t const* dict_base;    // ptr to dictionary page data
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

  // repetition/definition level decoding
  int32_t input_value_count;                  // how many values of the input we've processed
  int32_t input_row_count;                    // how many rows of the input we've processed
  int32_t input_leaf_count;                   // how many leaf values of the input we've processed
  uint8_t const* lvl_start[NUM_LEVEL_TYPES];  // [def,rep]
  uint8_t const* abs_lvl_start[NUM_LEVEL_TYPES];  // [def,rep]
  uint8_t const* abs_lvl_end[NUM_LEVEL_TYPES];    // [def,rep]
  int32_t lvl_count[NUM_LEVEL_TYPES];             // how many of each of the streams we've decoded
  int32_t row_index_lower_bound;                  // lower bound of row indices we should process

  // a shared-memory cache of frequently used data when decoding. The source of this data is
  // normally stored in global memory which can yield poor performance. So, when possible
  // we copy that info here prior to decoding
  PageNestingDecodeInfo nesting_decode_cache[max_cacheable_nesting_decode_info];
  // points to either nesting_decode_cache above when possible, or to the global source otherwise
  PageNestingDecodeInfo* nesting_info;
};

// buffers only used in the decode kernel.  separated from page_state_s to keep
// shared memory usage in other kernels (eg, gpuComputePageSizes) down.
template <int _nz_buf_size, int _dict_buf_size, int _str_buf_size>
struct page_state_buffers_s {
  static constexpr int nz_buf_size   = _nz_buf_size;
  static constexpr int dict_buf_size = _dict_buf_size;
  static constexpr int str_buf_size  = _str_buf_size;

  uint32_t nz_idx[nz_buf_size];      // circular buffer of non-null value positions
  uint32_t dict_idx[dict_buf_size];  // Dictionary index, boolean, or string offset values
  uint32_t str_len[str_buf_size];    // String length for plain encoding of strings
};

/**
 * @brief Returns whether or not a page spans either the beginning or the end of the
 * specified row bounds
 *
 * @param s The page to be checked
 * @param start_row The starting row index
 * @param num_rows The number of rows
 *
 * @return True if the page spans the beginning or the end of the row bounds
 */
inline __device__ bool is_bounds_page(page_state_s* const s, size_t start_row, size_t num_rows)
{
  size_t const page_begin = s->col.start_row + s->page.chunk_row;
  size_t const page_end   = page_begin + s->page.num_rows;
  size_t const begin      = start_row;
  size_t const end        = start_row + num_rows;

  return ((page_begin <= begin && page_end >= begin) || (page_begin <= end && page_end >= end));
}

/**
 * @brief Returns whether or not a page is completely contained within the specified
 * row bounds
 *
 * @param s The page to be checked
 * @param start_row The starting row index
 * @param num_rows The number of rows
 *
 * @return True if the page is completely contained within the row bounds
 */
inline __device__ bool is_page_contained(page_state_s* const s, size_t start_row, size_t num_rows)
{
  size_t const page_begin = s->col.start_row + s->page.chunk_row;
  size_t const page_end   = page_begin + s->page.num_rows;
  size_t const begin      = start_row;
  size_t const end        = start_row + num_rows;

  return page_begin >= begin && page_end <= end;
}

/**
 * @brief Parse the beginning of the level section (definition or repetition),
 * initializes the initial RLE run & value, and returns the section length
 *
 * @param[in,out] s The page state
 * @param[in] cur The current data position
 * @param[in] end The end of the data
 * @param[in] level_bits The bits required
 * @param[in] is_decode_step True if we are performing the decode step.
 * @param[in,out] decoders The repetition and definition level stream decoders
 *
 * @return The length of the section
 */
static __device__ uint32_t InitLevelSection(page_state_s* s,
                                            uint8_t const* cur,
                                            uint8_t const* end,
                                            level_type lvl)
{
  int32_t len;
  int level_bits    = s->col.level_bits[lvl];
  Encoding encoding = lvl == level_type::DEFINITION ? s->page.definition_level_encoding
                                                    : s->page.repetition_level_encoding;

  auto start = cur;
  if (level_bits == 0) {
    len                       = 0;
    s->initial_rle_run[lvl]   = s->page.num_input_values * 2;  // repeated value
    s->initial_rle_value[lvl] = 0;
    s->lvl_start[lvl]         = cur;
    s->abs_lvl_start[lvl]     = cur;
  } else if (encoding == Encoding::RLE) {
    // V2 only uses RLE encoding, so only perform check here
    if (s->page.def_lvl_bytes || s->page.rep_lvl_bytes) {
      len = lvl == level_type::DEFINITION ? s->page.def_lvl_bytes : s->page.rep_lvl_bytes;
    } else if (cur + 4 < end) {
      len = 4 + (cur[0]) + (cur[1] << 8) + (cur[2] << 16) + (cur[3] << 24);
      cur += 4;
    } else {
      len      = 0;
      s->error = 2;
    }
    s->abs_lvl_start[lvl] = cur;
    if (!s->error) {
      uint32_t run            = get_vlq32(cur, end);
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
    }

    if (cur > end) { s->error = 2; }
  } else if (encoding == Encoding::BIT_PACKED) {
    len                       = (s->page.num_input_values * level_bits + 7) >> 3;
    s->initial_rle_run[lvl]   = ((s->page.num_input_values + 7) >> 3) * 2 + 1;  // literal run
    s->initial_rle_value[lvl] = 0;
    s->lvl_start[lvl]         = cur;
    s->abs_lvl_start[lvl]     = cur;
  } else {
    s->error = 3;
    len      = 0;
  }

  s->abs_lvl_end[lvl] = start + len;

  return static_cast<uint32_t>(len);
}

// Copies null counts back to `nesting_decode` at the end of scope
struct null_count_back_copier {
  page_state_s* s;
  int t;
  __device__ ~null_count_back_copier()
  {
    if (s->nesting_info != nullptr and s->nesting_info == s->nesting_decode_cache) {
      int depth = 0;
      while (depth < s->page.num_output_nesting_levels) {
        int const thread_depth = depth + t;
        if (thread_depth < s->page.num_output_nesting_levels) {
          s->page.nesting_decode[thread_depth].null_count =
            s->nesting_decode_cache[thread_depth].null_count;
        }
        depth += blockDim.x;
      }
    }
  }
};

/**
 * @brief Sets up block-local page state information from the global pages.
 *
 * @param[in, out] s The local page state to be filled in
 * @param[in] p The global page to be copied from
 * @param[in] chunks The global list of chunks
 * @param[in] min_row Crop all rows below min_row
 * @param[in] num_rows Maximum number of rows to read
 * @param[in] is_decode_step If we are setting up for the decode step (instead of the preprocess)
 * @param[in] decoders rle_stream decoders which will be used for decoding levels. Optional.
 * Currently only used by gpuComputePageSizes step)
 */
static __device__ bool setupLocalPageInfo(page_state_s* const s,
                                          PageInfo const* p,
                                          device_span<ColumnChunkDesc const> chunks,
                                          size_t min_row,
                                          size_t num_rows,
                                          bool is_decode_step)
{
  int t = threadIdx.x;
  int chunk_idx;

  // Fetch page info
  if (!t) {
    s->page         = *p;
    s->nesting_info = nullptr;
  }
  __syncthreads();

  if (s->page.flags & PAGEINFO_FLAGS_DICTIONARY) { return false; }
  // Fetch column chunk info
  chunk_idx = s->page.chunk_idx;
  if (!t) { s->col = chunks[chunk_idx]; }

  // if we can use the nesting decode cache, set it up now
  auto const can_use_decode_cache = s->page.nesting_info_size <= max_cacheable_nesting_decode_info;
  if (can_use_decode_cache) {
    int depth = 0;
    while (depth < s->page.nesting_info_size) {
      int const thread_depth = depth + t;
      if (thread_depth < s->page.nesting_info_size) {
        // these values need to be copied over from global
        s->nesting_decode_cache[thread_depth].max_def_level =
          s->page.nesting_decode[thread_depth].max_def_level;
        s->nesting_decode_cache[thread_depth].page_start_value =
          s->page.nesting_decode[thread_depth].page_start_value;
        s->nesting_decode_cache[thread_depth].start_depth =
          s->page.nesting_decode[thread_depth].start_depth;
        s->nesting_decode_cache[thread_depth].end_depth =
          s->page.nesting_decode[thread_depth].end_depth;
      }
      depth += blockDim.x;
    }
  }
  if (!t) {
    s->nesting_info = can_use_decode_cache ? s->nesting_decode_cache : s->page.nesting_decode;
  }

  __syncthreads();

  // zero counts
  int depth = 0;
  while (depth < s->page.num_output_nesting_levels) {
    int const thread_depth = depth + t;
    if (thread_depth < s->page.num_output_nesting_levels) {
      s->nesting_info[thread_depth].valid_count = 0;
      s->nesting_info[thread_depth].value_count = 0;
      s->nesting_info[thread_depth].null_count  = 0;
    }
    depth += blockDim.x;
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
        case INT32: [[fallthrough]];
        case FLOAT: s->dtype_len = 4; break;
        case INT64:
          if (s->col.ts_clock_rate) {
            int32_t units = 0;
            // Duration types are not included because no scaling is done when reading
            if (s->col.converted_type == TIMESTAMP_MILLIS) {
              units = cudf::timestamp_ms::period::den;
            } else if (s->col.converted_type == TIMESTAMP_MICROS) {
              units = cudf::timestamp_us::period::den;
            } else if (s->col.logical_type.TIMESTAMP.unit.isset.NANOS) {
              units = cudf::timestamp_ns::period::den;
            }
            if (units and units != s->col.ts_clock_rate) {
              s->ts_scale = (s->col.ts_clock_rate < units) ? -(units / s->col.ts_clock_rate)
                                                           : (s->col.ts_clock_rate / units);
            }
          }
          [[fallthrough]];
        case DOUBLE: s->dtype_len = 8; break;
        case INT96: s->dtype_len = 12; break;
        case BYTE_ARRAY:
          if (s->col.converted_type == DECIMAL) {
            auto const decimal_precision = s->col.decimal_precision;
            s->dtype_len                 = [decimal_precision]() {
              if (decimal_precision <= MAX_DECIMAL32_PRECISION) {
                return sizeof(int32_t);
              } else if (decimal_precision <= MAX_DECIMAL64_PRECISION) {
                return sizeof(int64_t);
              } else {
                return sizeof(__int128_t);
              }
            }();
          } else {
            s->dtype_len = sizeof(string_index_pair);
          }
          break;
        default:  // FIXED_LEN_BYTE_ARRAY:
          s->dtype_len = dtype_len_out;
          s->error |= (s->dtype_len <= 0);
          break;
      }
      // Special check for downconversions
      s->dtype_len_in = s->dtype_len;
      if (s->col.converted_type == DECIMAL && data_type == FIXED_LEN_BYTE_ARRAY) {
        s->dtype_len = [dtype_len = s->dtype_len]() {
          if (dtype_len <= sizeof(int32_t)) {
            return sizeof(int32_t);
          } else if (dtype_len <= sizeof(int64_t)) {
            return sizeof(int64_t);
          } else {
            return sizeof(__int128_t);
          }
        }();
      } else if (data_type == INT32) {
        if (dtype_len_out == 1) {
          // INT8 output
          s->dtype_len = 1;
        } else if (dtype_len_out == 2) {
          // INT16 output
          s->dtype_len = 2;
        } else if (s->col.converted_type == TIME_MILLIS) {
          // INT64 output
          s->dtype_len = 8;
        }
      } else if (data_type == BYTE_ARRAY && dtype_len_out == 4) {
        s->dtype_len = 4;  // HASH32 output
      } else if (data_type == INT96) {
        s->dtype_len = 8;  // Convert to 64-bit timestamp
      }

      // NOTE: s->page.num_rows, s->col.chunk_row, s->first_row and s->num_rows will be
      // invalid/bogus during first pass of the preprocess step for nested types. this is ok
      // because we ignore these values in that stage.
      {
        auto const max_row = min_row + num_rows;

        // if we are totally outside the range of the input, do nothing
        if ((page_start_row > max_row) || (page_start_row + s->page.num_rows < min_row)) {
          s->first_row = 0;
          s->num_rows  = 0;
        }
        // otherwise
        else {
          s->first_row             = page_start_row >= min_row ? 0 : min_row - page_start_row;
          auto const max_page_rows = s->page.num_rows - s->first_row;
          s->num_rows              = (page_start_row + s->first_row) + max_page_rows <= max_row
                                       ? max_page_rows
                                       : max_row - (page_start_row + s->first_row);
        }
      }

      // during the decoding step we need to offset the global output buffers
      // for each level of nesting so that we write to the section this page
      // is responsible for.
      // - for flat schemas, we can do this directly by using row counts
      // - for nested schemas, these offsets are computed during the preprocess step
      //
      // NOTE: in a chunked read situation, s->col.column_data_base and s->col.valid_map_base
      // will be aliased to memory that has been freed when we get here in the non-decode step, so
      // we cannot check against nullptr.  we'll just check a flag directly.
      if (is_decode_step) {
        int max_depth = s->col.max_nesting_depth;
        for (int idx = 0; idx < max_depth; idx++) {
          PageNestingDecodeInfo* nesting_info = &s->nesting_info[idx];

          size_t output_offset;
          // schemas without lists
          if (s->col.max_level[level_type::REPETITION] == 0) {
            output_offset = page_start_row >= min_row ? page_start_row - min_row : 0;
          }
          // for schemas with lists, we've already got the exact value precomputed
          else {
            output_offset = nesting_info->page_start_value;
          }

          nesting_info->data_out = static_cast<uint8_t*>(s->col.column_data_base[idx]);

          if (nesting_info->data_out != nullptr) {
            // anything below max depth with a valid data pointer must be a list, so the
            // element size is the size of the offset type.
            uint32_t len = idx < max_depth - 1 ? sizeof(cudf::size_type) : s->dtype_len;
            nesting_info->data_out += (output_offset * len);
          }
          nesting_info->valid_map = s->col.valid_map_base[idx];
          if (nesting_info->valid_map != nullptr) {
            nesting_info->valid_map += output_offset >> 5;
            nesting_info->valid_map_offset = (int32_t)(output_offset & 0x1f);
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
      // NOTE:  if additional encodings are supported in the future, modifications must
      // be made to is_supported_encoding() in reader_impl_preprocess.cu
      switch (s->page.encoding) {
        case Encoding::PLAIN_DICTIONARY:
        case Encoding::RLE_DICTIONARY:
          // RLE-packed dictionary indices, first byte indicates index length in bits
          if (((s->col.data_type & 7) == BYTE_ARRAY) && (s->col.str_dict_index)) {
            // String dictionary: use index
            s->dict_base = reinterpret_cast<uint8_t const*>(s->col.str_dict_index);
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
      s->input_leaf_count         = 0;

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
      if (is_decode_step) {
        s->input_value_count = s->page.skipped_values > -1 ? s->page.skipped_values : 0;
      } else {
        s->input_value_count = 0;
        s->input_leaf_count  = 0;
        s->page.skipped_values =
          -1;  // magic number to indicate it hasn't been set for use inside UpdatePageSizes
        s->page.skipped_leaf_values = 0;
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
 * @param[in,out] nesting_info The page/nesting information to store the mask in. The validity map
 * offset is also updated
 * @param[in] valid_mask The validity mask to be stored
 * @param[in] value_count # of bits in the validity mask
 */
static __device__ void store_validity(int valid_map_offset,
                                      bitmask_type* valid_map,
                                      uint32_t valid_mask,
                                      int32_t value_count)
{
  int word_offset = valid_map_offset / 32;
  int bit_offset  = valid_map_offset % 32;
  // if we fit entirely in the output word
  if (bit_offset + value_count <= 32) {
    auto relevant_mask = static_cast<uint32_t>((static_cast<uint64_t>(1) << value_count) - 1);

    if (relevant_mask == ~0) {
      valid_map[word_offset] = valid_mask;
    } else {
      atomicAnd(valid_map + word_offset, ~(relevant_mask << bit_offset));
      atomicOr(valid_map + word_offset, (valid_mask & relevant_mask) << bit_offset);
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
    atomicAnd(valid_map + word_offset, ~(relevant_mask << bit_offset));
    atomicOr(valid_map + word_offset, mask_word0 << bit_offset);

    // second word. strip the remainder of the bits off the end and store that
    relevant_mask       = ((1 << (value_count - bits_left)) - 1);
    uint32_t mask_word1 = valid_mask & (relevant_mask << bits_left);
    atomicAnd(valid_map + word_offset + 1, ~(relevant_mask));
    atomicOr(valid_map + word_offset + 1, mask_word1 >> bits_left);
  }
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
 * @param[in] rep Repetition level buffer
 * @param[in] def Definition level buffer
 * @param[in] input_value_count The current count of input level values we have processed
 * @param[in] target_input_value_count The desired # of input level values we want to process
 * @param[in] t Thread index
 */
template <int rolling_buf_size, typename level_t>
inline __device__ void get_nesting_bounds(int& start_depth,
                                          int& end_depth,
                                          int& d,
                                          page_state_s* s,
                                          level_t const* const rep,
                                          level_t const* const def,
                                          int input_value_count,
                                          int32_t target_input_value_count,
                                          int t)
{
  start_depth = -1;
  end_depth   = -1;
  d           = -1;
  if (input_value_count + t < target_input_value_count) {
    int const index = rolling_index<rolling_buf_size>(input_value_count + t);
    d               = static_cast<int>(def[index]);
    // if we have repetition (there are list columns involved) we have to
    // bound what nesting levels we apply values to
    if (s->col.max_level[level_type::REPETITION] > 0) {
      level_t const r = rep[index];
      start_depth     = s->nesting_info[r].start_depth;
      end_depth       = s->nesting_info[d].end_depth;
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
 * @brief Parses the length and position of strings and returns total length of all strings
 * processed
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] target_pos Target output position
 * @param[in] t Thread ID
 *
 * @return Total length of strings processed
 */
template <bool sizes_only, typename state_buf>
__device__ size_type gpuInitStringDescriptors(volatile page_state_s* s,
                                              [[maybe_unused]] volatile state_buf* sb,
                                              int target_pos,
                                              int t)
{
  int pos       = s->dict_pos;
  int total_len = 0;

  // This step is purely serial
  if (!t) {
    uint8_t const* cur = s->data_start;
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
      if constexpr (!sizes_only) {
        sb->dict_idx[rolling_index<state_buf::dict_buf_size>(pos)] = k;
        sb->str_len[rolling_index<state_buf::str_buf_size>(pos)]   = len;
      }
      k += len;
      total_len += len;
      pos++;
    }
    s->dict_val = k;
    __threadfence_block();
  }

  return total_len;
}

// #if 0
/**
 * @brief Retrieves string information for a string at the specified source position
 *
 * @param[in] s Page state input
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 *
 * @return A pair containing a pointer to the string and its length
 */
template <typename state_buf>
inline __device__ cuda::std::pair<char const*, size_t> gpuGetStringData(volatile page_state_s* s,
                                                                        volatile state_buf* sb,
                                                                        int src_pos)
{
  char const* ptr = nullptr;
  size_t len      = 0;

  if (s->dict_base) {
    // String dictionary
    uint32_t dict_pos =
      (s->dict_bits > 0)
        ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)] * sizeof(string_index_pair)
        : 0;
    if (dict_pos < (uint32_t)s->dict_size) {
      auto const* src = reinterpret_cast<string_index_pair const*>(s->dict_base + dict_pos);
      ptr             = src->first;
      len             = src->second;
    }
  } else {
    // Plain encoding
    uint32_t dict_pos = sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)];
    if (dict_pos <= (uint32_t)s->dict_size) {
      ptr = reinterpret_cast<char const*>(s->data_start + dict_pos);
      len = sb->str_len[rolling_index<state_buf::str_buf_size>(src_pos)];
    }
  }

  return {ptr, len};
}

/**
 * @brief Output a string descriptor
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dstv Pointer to row output data (string descriptor or 32-bit hash)
 */
template <typename state_buf>
inline __device__ void gpuOutputString(volatile page_state_s* s,
                                       volatile state_buf* sb,
                                       int src_pos,
                                       void* dstv)
{
  auto [ptr, len] = gpuGetStringData(s, sb, src_pos);
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
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename state_buf>
inline __device__ void gpuOutputBoolean(volatile state_buf* sb, int src_pos, uint8_t* dst)
{
  *dst = sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)];
}

/**
 * @brief Output a boolean
 *
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
inline __device__ void gpuOutputBooleanFast(uint8_t const* src, int src_pos, uint8_t* dst)
{
  uint8_t const b = src[src_pos / 8];
  *dst            = b & (1 << (src_pos % 8)) ? 1 : 0;
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
                                      uint8_t const* src8,
                                      uint32_t dict_pos,
                                      uint32_t dict_size)
{
  uint32_t bytebuf;
  unsigned int ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos < dict_size) {
    bytebuf = *reinterpret_cast<uint32_t const*>(src8 + dict_pos);
    if (ofs) {
      uint32_t bytebufnext = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 4);
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
                                      uint8_t const* src8,
                                      uint32_t dict_pos,
                                      uint32_t dict_size)
{
  uint2 v;
  unsigned int ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos < dict_size) {
    v.x = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 0);
    v.y = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 4);
    if (ofs) {
      uint32_t next = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 8);
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
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[out] dst Pointer to row output data
 */
template <typename state_buf>
inline __device__ void gpuOutputInt96Timestamp(volatile page_state_s* s,
                                               volatile state_buf* sb,
                                               int src_pos,
                                               int64_t* dst)
{
  using cuda::std::chrono::duration_cast;

  uint8_t const* src8;
  uint32_t dict_pos, dict_size = s->dict_size, ofs;

  if (s->dict_base) {
    // Dictionary
    dict_pos =
      (s->dict_bits > 0) ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)] : 0;
    src8 = s->dict_base;
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
  v.x = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 0);
  v.y = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 4);
  v.z = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 8);
  if (ofs) {
    uint32_t next = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 12);
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
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename state_buf>
inline __device__ void gpuOutputInt64Timestamp(volatile page_state_s* s,
                                               volatile state_buf* sb,
                                               int src_pos,
                                               int64_t* dst)
{
  uint8_t const* src8;
  uint32_t dict_pos, dict_size = s->dict_size, ofs;
  int64_t ts;

  if (s->dict_base) {
    // Dictionary
    dict_pos =
      (s->dict_bits > 0) ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)] : 0;
    src8 = s->dict_base;
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
    v.x = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 0);
    v.y = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 4);
    if (ofs) {
      uint32_t next = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 8);
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
 * @brief Output a byte array as int.
 *
 * @param[in] ptr Pointer to the byte array
 * @param[in] len Byte array length
 * @param[out] dst Pointer to row output data
 */
template <typename T>
__device__ void gpuOutputByteArrayAsInt(char const* ptr, int32_t len, T* dst)
{
  T unscaled = 0;
  for (auto i = 0; i < len; i++) {
    uint8_t v = ptr[i];
    unscaled  = (unscaled << 8) | v;
  }
  // Shift the unscaled value up and back down when it isn't all 8 bytes,
  // which sign extend the value for correctly representing negative numbers.
  unscaled <<= (sizeof(T) - len) * 8;
  unscaled >>= (sizeof(T) - len) * 8;
  *dst = unscaled;
}

/**
 * @brief Output a fixed-length byte array as int.
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename T, typename state_buf>
__device__ void gpuOutputFixedLenByteArrayAsInt(volatile page_state_s* s,
                                                volatile state_buf* sb,
                                                int src_pos,
                                                T* dst)
{
  uint32_t const dtype_len_in = s->dtype_len_in;
  uint8_t const* data         = s->dict_base ? s->dict_base : s->data_start;
  uint32_t const pos =
    (s->dict_base
       ? ((s->dict_bits > 0) ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)] : 0)
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
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename T, typename state_buf>
inline __device__ void gpuOutputFast(volatile page_state_s* s,
                                     volatile state_buf* sb,
                                     int src_pos,
                                     T* dst)
{
  uint8_t const* dict;
  uint32_t dict_pos, dict_size = s->dict_size;

  if (s->dict_base) {
    // Dictionary
    dict_pos =
      (s->dict_bits > 0) ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)] : 0;
    dict = s->dict_base;
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
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst8 Pointer to row output data
 * @param[in] len Length of element
 */
template <typename state_buf>
static __device__ void gpuOutputGeneric(
  volatile page_state_s* s, volatile state_buf* sb, int src_pos, uint8_t* dst8, int len)
{
  uint8_t const* dict;
  uint32_t dict_pos, dict_size = s->dict_size;

  if (s->dict_base) {
    // Dictionary
    dict_pos =
      (s->dict_bits > 0) ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)] : 0;
    dict = s->dict_base;
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
    uint8_t const* src8 = dict;
    unsigned int ofs    = 3 & reinterpret_cast<size_t>(src8);
    src8 -= ofs;  // align to 32-bit boundary
    ofs <<= 3;    // bytes -> bits
    for (unsigned int i = 0; i < len; i += 4) {
      uint32_t bytebuf;
      if (dict_pos < dict_size) {
        bytebuf = *reinterpret_cast<uint32_t const*>(src8 + dict_pos);
        if (ofs) {
          uint32_t bytebufnext = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 4);
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
 * @brief Performs RLE decoding of dictionary indexes
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] target_pos Target index position in dict_idx buffer (may exceed this value by up to
 * 31)
 * @param[in] t Warp1 thread ID (0..31)
 *
 * @return A pair containing the new output position, and the total length of strings decoded (this
 * will only be valid on thread 0 and if sizes_only is true). In the event that this function
 * decodes strings beyond target_pos, the total length of strings returned will include these
 * additional values.
 */
template <bool sizes_only, typename state_buf>
__device__ cuda::std::pair<int, int> gpuDecodeDictionaryIndices(
  volatile page_state_s* s, [[maybe_unused]] volatile state_buf* sb, int target_pos, int t)
{
  uint8_t const* end = s->data_end;
  int dict_bits      = s->dict_bits;
  int pos            = s->dict_pos;
  int str_len        = 0;

  while (pos < target_pos) {
    int is_literal, batch_len;
    if (!t) {
      uint32_t run       = s->dict_run;
      uint8_t const* cur = s->data_start;
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

    // compute dictionary index.
    int dict_idx = 0;
    if (t < batch_len) {
      dict_idx = s->dict_val;
      if (is_literal) {
        int32_t ofs      = (t - ((batch_len + 7) & ~7)) * dict_bits;
        uint8_t const* p = s->data_start + (ofs >> 3);
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

      // if we're not computing sizes, store off the dictionary index
      if constexpr (!sizes_only) {
        sb->dict_idx[rolling_index<state_buf::dict_buf_size>(pos + t)] = dict_idx;
      }
    }

    // if we're computing sizes, add the length(s)
    if constexpr (sizes_only) {
      int const len = [&]() {
        if (t >= batch_len || (pos + t >= target_pos)) { return 0; }
        uint32_t const dict_pos = (s->dict_bits > 0) ? dict_idx * sizeof(string_index_pair) : 0;
        if (dict_pos < (uint32_t)s->dict_size) {
          const auto* src = reinterpret_cast<string_index_pair const*>(s->dict_base + dict_pos);
          return src->second;
        }
        return 0;
      }();

      using WarpReduce = cub::WarpReduce<size_type>;
      __shared__ typename WarpReduce::TempStorage temp_storage;
      // note: str_len will only be valid on thread 0.
      str_len += WarpReduce(temp_storage).Sum(len);
    }

    pos += batch_len;
  }
  return {pos, str_len};
}

}  // namespace gpu
}  // namespace parquet
}  // namespace io
}  // namespace cudf
