/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include "error.hpp"
#include "io/utilities/block_utils.cuh"
#include "parquet_gpu.hpp"
#include "rle_stream.cuh"

#include <cuda/atomic>
#include <cuda/std/tuple>

namespace cudf::io::parquet::detail {

struct page_state_s {
  constexpr page_state_s() noexcept {}
  uint8_t const* data_start{};
  uint8_t const* data_end{};
  uint8_t const* lvl_end{};
  uint8_t const* dict_base{};    // ptr to dictionary page data
  int32_t dict_size{};           // size of dictionary data
  int32_t first_row{};           // First row in page to output
  int32_t num_rows{};            // Rows in page to decode (including rows to be skipped)
  int32_t first_output_value{};  // First value in page to output
  int32_t num_input_values{};    // total # of input/level values in the page
  int32_t dtype_len{};           // Output data type length
  int32_t dtype_len_in{};        // Can be larger than dtype_len if truncating 32-bit into 8-bit
  int32_t dict_bits{};           // # of bits to store dictionary indices
  uint32_t dict_run{};
  int32_t dict_val{};
  uint32_t initial_rle_run[NUM_LEVEL_TYPES]{};   // [def,rep]
  int32_t initial_rle_value[NUM_LEVEL_TYPES]{};  // [def,rep]
  kernel_error::value_type error{};
  PageInfo page{};
  ColumnChunkDesc col{};

  // (leaf) value decoding
  int32_t nz_count{};  // number of valid entries in nz_idx (write position in circular buffer)
  int32_t dict_pos{};  // write position of dictionary indices
  int32_t src_pos{};   // input read position of final output value
  int32_t ts_scale{};  // timestamp scale: <0: divide by -ts_scale, >0: multiply by ts_scale

  // repetition/definition level decoding
  int32_t input_value_count{};                  // how many values of the input we've processed
  int32_t input_row_count{};                    // how many rows of the input we've processed
  int32_t input_leaf_count{};                   // how many leaf values of the input we've processed
  uint8_t const* lvl_start[NUM_LEVEL_TYPES]{};  // [def,rep]
  uint8_t const* abs_lvl_start[NUM_LEVEL_TYPES]{};  // [def,rep]
  uint8_t const* abs_lvl_end[NUM_LEVEL_TYPES]{};    // [def,rep]
  int32_t lvl_count[NUM_LEVEL_TYPES]{};             // how many of each of the streams we've decoded
  int32_t row_index_lower_bound{};                  // lower bound of row indices we should process

  // a shared-memory cache of frequently used data when decoding. The source of this data is
  // normally stored in global memory which can yield poor performance. So, when possible
  // we copy that info here prior to decoding
  PageNestingDecodeInfo nesting_decode_cache[max_cacheable_nesting_decode_info]{};
  // points to either nesting_decode_cache above when possible, or to the global source otherwise
  PageNestingDecodeInfo* nesting_info{};

  inline __device__ void set_error_code(decode_error err)
  {
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{error};
    ref.fetch_or(static_cast<kernel_error::value_type>(err), cuda::std::memory_order_relaxed);
  }

  inline __device__ void reset_error_code()
  {
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{error};
    ref.store(0, cuda::std::memory_order_release);
  }
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
 * @brief Test if the given page is in a string column
 */
constexpr bool is_string_col(PageInfo const& page, device_span<ColumnChunkDesc const> chunks)
{
  if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) { return false; }
  auto const& col = chunks[page.chunk_idx];
  return is_string_col(col);
}

/**
 * @brief Returns whether or not a page spans either the beginning or the end of the
 * specified row bounds
 *
 * @param s The page to be checked
 * @param start_row The starting row index
 * @param num_rows The number of rows
 * @param has_repetition True if the schema has nesting
 *
 * @return True if the page spans the beginning or the end of the row bounds
 */
inline __device__ bool is_bounds_page(page_state_s* const s,
                                      size_t start_row,
                                      size_t num_rows,
                                      bool has_repetition)
{
  size_t const page_begin = s->col.start_row + s->page.chunk_row;
  size_t const page_end   = page_begin + s->page.num_rows;
  size_t const begin      = start_row;
  size_t const end        = start_row + num_rows;

  // for non-nested schemas, rows cannot span pages, so use a more restrictive test
  return has_repetition
           ? ((page_begin <= begin && page_end >= begin) || (page_begin <= end && page_end >= end))
           : ((page_begin < begin && page_end > begin) || (page_begin < end && page_end > end));
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
 * @brief Retrieves string information for a string at the specified source position
 *
 * @param[in] s Page state input
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @tparam state_buf Typename of the `state_buf` (usually inferred)
 *
 * @return A pair containing a pointer to the string and its length
 */
template <typename state_buf>
inline __device__ cuda::std::pair<char const*, size_t> gpuGetStringData(page_state_s* s,
                                                                        state_buf* sb,
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
 * @brief Performs RLE decoding of dictionary indexes
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] target_pos Target index position in dict_idx buffer (may exceed this value by up to
 * 31)
 * @param[in] t Warp1 thread ID (0..31)
 * @tparam sizes_only True if only sizes are to be calculated
 * @tparam state_buf Typename of the `state_buf` (usually inferred)
 *
 * @return A pair containing the new output position, and the total length of strings decoded (this
 * will only be valid on thread 0 and if sizes_only is true). In the event that this function
 * decodes strings beyond target_pos, the total length of strings returned will include these
 * additional values.
 */
template <bool sizes_only, typename state_buf>
__device__ cuda::std::pair<int, int> gpuDecodeDictionaryIndices(page_state_s* s,
                                                                [[maybe_unused]] state_buf* sb,
                                                                int target_pos,
                                                                int t)
{
  uint8_t const* end = s->data_end;
  int dict_bits      = s->dict_bits;
  int pos            = s->dict_pos;
  int str_len        = 0;

  // NOTE: racecheck warns about a RAW involving s->dict_pos, which is likely a false positive
  // because the only path that does not include a sync will lead to s->dict_pos being overwritten
  // with the same value

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
          const auto* src = reinterpret_cast<const string_index_pair*>(s->dict_base + dict_pos);
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

/**
 * @brief Performs RLE decoding of dictionary indexes, for when dict_size=1
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] target_pos Target write position
 * @param[in] t Thread ID
 * @tparam state_buf Typename of the `state_buf` (usually inferred)
 *
 * @return The new output position
 */
template <typename state_buf>
inline __device__ int gpuDecodeRleBooleans(page_state_s* s, state_buf* sb, int target_pos, int t)
{
  uint8_t const* end = s->data_end;
  int64_t pos        = s->dict_pos;

  // NOTE: racecheck warns about a RAW involving s->dict_pos, which is likely a false positive
  // because the only path that does not include a sync will lead to s->dict_pos being overwritten
  // with the same value

  while (pos < target_pos) {
    int is_literal, batch_len;
    if (!t) {
      uint32_t run       = s->dict_run;
      uint8_t const* cur = s->data_start;
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
        uint8_t const* p = s->data_start + (ofs >> 3);
        dict_idx         = (p < end) ? (p[0] >> (ofs & 7u)) & 1 : 0;
      } else {
        dict_idx = s->dict_val;
      }
      sb->dict_idx[rolling_index<state_buf::dict_buf_size>(pos + t)] = dict_idx;
    }
    pos += batch_len;
  }
  return pos;
}

/**
 * @brief Parses the length and position of strings and returns total length of all strings
 * processed
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] target_pos Target output position
 * @param[in] t Thread ID
 * @tparam sizes_only True if only sizes are to be calculated
 * @tparam state_buf Typename of the `state_buf` (usually inferred)
 *
 * @return Total length of strings processed
 */
template <bool sizes_only, typename state_buf>
__device__ size_type
gpuInitStringDescriptors(page_state_s* s, [[maybe_unused]] state_buf* sb, int target_pos, int t)
{
  int pos       = s->dict_pos;
  int total_len = 0;

  // This step is purely serial
  if (!t) {
    uint8_t const* cur = s->data_start;
    int dict_size      = s->dict_size;
    int k              = s->dict_val;

    while (pos < target_pos) {
      int len = 0;
      if (s->col.physical_type == FIXED_LEN_BYTE_ARRAY) {
        if (k < dict_size) { len = s->dtype_len_in; }
      } else {
        if (k + 4 <= dict_size) {
          len = (cur[k]) | (cur[k + 1] << 8) | (cur[k + 2] << 16) | (cur[k + 3] << 24);
          k += 4;
          if (k + len > dict_size) { len = 0; }
        }
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

/**
 * @brief Decode values out of a definition or repetition stream
 *
 * @param[out] output Level buffer output
 * @param[in,out] s Page state input/output
 * @param[in] target_count Target count of stream values on output
 * @param[in] t Warp0 thread ID (0..31)
 * @param[in] lvl The level type we are decoding - DEFINITION or REPETITION
 * @tparam level_t Type used to store decoded repetition and definition levels
 * @tparam rolling_buf_size Size of the cyclic buffer used to store value data
 */
template <typename level_t, int rolling_buf_size>
__device__ void gpuDecodeStream(
  level_t* output, page_state_s* s, int32_t target_count, int t, level_type lvl)
{
  uint8_t const* cur_def    = s->lvl_start[lvl];
  uint8_t const* end        = s->lvl_end;
  uint32_t level_run        = s->initial_rle_run[lvl];
  int32_t level_val         = s->initial_rle_value[lvl];
  int level_bits            = s->col.level_bits[lvl];
  int32_t num_input_values  = s->num_input_values;
  int32_t value_count       = s->lvl_count[lvl];
  int32_t batch_coded_count = 0;

  while (s->error == 0 && value_count < target_count && value_count < num_input_values) {
    int batch_len;
    if (level_run <= 1) {
      // Get a new run symbol from the byte stream
      int sym_len = 0;
      if (!t) {
        uint8_t const* cur = cur_def;
        if (cur < end) { level_run = get_vlq32(cur, end); }
        if (is_repeated_run(level_run)) {
          if (cur < end) level_val = cur[0];
          cur++;
          if (level_bits > 8) {
            if (cur < end) level_val |= cur[0] << 8;
            cur++;
          }
        }
        // If there are errors, set the error code and continue. The loop will be exited below.
        if (cur > end) { s->set_error_code(decode_error::LEVEL_STREAM_OVERRUN); }
        if (level_run <= 1) { s->set_error_code(decode_error::INVALID_LEVEL_RUN); }
        sym_len = (int32_t)(cur - cur_def);
        __threadfence_block();
      }
      sym_len   = shuffle(sym_len);
      level_val = shuffle(level_val);
      level_run = shuffle(level_run);
      cur_def += sym_len;
    }
    if (s->error != 0) { break; }

    batch_len = min(num_input_values - value_count, 32);
    if (is_literal_run(level_run)) {
      // Literal run
      int batch_len8;
      batch_len  = min(batch_len, (level_run >> 1) * 8);
      batch_len8 = (batch_len + 7) >> 3;
      if (t < batch_len) {
        int bitpos         = t * level_bits;
        uint8_t const* cur = cur_def + (bitpos >> 3);
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
      int idx                                      = value_count + t;
      output[rolling_index<rolling_buf_size>(idx)] = level_val;
    }
    batch_coded_count += batch_len;
    value_count += batch_len;
  }
  // issue #14597
  // racecheck reported race between reads at the start of this function and the writes below
  __syncwarp();

  // update the stream info
  if (!t) {
    s->lvl_start[lvl]         = cur_def;
    s->initial_rle_run[lvl]   = level_run;
    s->initial_rle_value[lvl] = level_val;
    s->lvl_count[lvl]         = value_count;
  }
}

/**
 * @brief Store a validity mask containing value_count bits into the output validity buffer of the
 * page.
 *
 * @param[in,out] nesting_info The page/nesting information to store the mask in. The validity map
 * offset is also updated
 * @param[in,out] valid_map Pointer to bitmask to store validity information to
 * @param[in] valid_mask The validity mask to be stored
 * @param[in] value_count # of bits in the validity mask
 */
inline __device__ void store_validity(int valid_map_offset,
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
 * @tparam rolling_buf_size Size of the cyclic buffer used to store value data
 * @tparam level_t Type used to store decoded repetition and definition levels
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
      int r       = rep[index];
      start_depth = s->nesting_info[r].start_depth;
      end_depth   = s->nesting_info[d].end_depth;
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
 * @param[out] sb Page state buffer output
 * @param[in] rep Repetition level buffer
 * @param[in] def Definition level buffer
 * @param[in] t Thread index
 * @tparam level_t Type used to store decoded repetition and definition levels
 * @tparam state_buf Typename of the `state_buf` (usually inferred)
 * @tparam rolling_buf_size Size of the cyclic buffer used to store value data
 */
template <typename level_t, typename state_buf, int rolling_buf_size>
__device__ void gpuUpdateValidityOffsetsAndRowIndices(int32_t target_input_value_count,
                                                      page_state_s* s,
                                                      state_buf* sb,
                                                      level_t const* const rep,
                                                      level_t const* const def,
                                                      int t)
{
  // exit early if there's no work to do
  if (s->input_value_count >= target_input_value_count) { return; }

  // max nesting depth of the column
  int const max_depth       = s->col.max_nesting_depth;
  bool const has_repetition = s->col.max_level[level_type::REPETITION] > 0;
  // how many (input) values we've processed in the page so far
  int input_value_count = s->input_value_count;
  // how many rows we've processed in the page so far
  int input_row_count = s->input_row_count;

  PageNestingDecodeInfo* nesting_info_base = s->nesting_info;

  // process until we've reached the target
  while (input_value_count < target_input_value_count) {
    // determine the nesting bounds for this thread (the range of nesting depths we
    // will generate new value indices and validity bits for)
    int start_depth, end_depth, d;
    get_nesting_bounds<rolling_buf_size, level_t>(
      start_depth, end_depth, d, s, rep, def, input_value_count, target_input_value_count, t);

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
      PageNestingDecodeInfo* nesting_info = &nesting_info_base[s_idx];

      // if we are within the range of nesting levels we should be adding value indices for
      int const in_nesting_bounds =
        ((s_idx >= start_depth && s_idx <= end_depth) && in_row_bounds) ? 1 : 0;

      // everything up to the max_def_level is a non-null value
      uint32_t const is_valid = d >= nesting_info->max_def_level && in_nesting_bounds ? 1 : 0;

      // compute warp and thread valid counts
      uint32_t const warp_valid_mask =
        // for flat schemas, a simple ballot_sync gives us the correct count and bit positions
        // because every value in the input matches to a value in the output
        !has_repetition
          ? ballot(is_valid)
          :
          // for nested schemas, it's more complicated.  This warp will visit 32 incoming values,
          // however not all of them will necessarily represent a value at this nesting level. so
          // the validity bit for thread t might actually represent output value t-6. the correct
          // position for thread t's bit is thread_value_count. for cuda 11 we could use
          // __reduce_or_sync(), but until then we have to do a warp reduce.
          WarpReduceOr32(is_valid << thread_value_count);

      thread_valid_count = __popc(warp_valid_mask & ((1 << thread_value_count) - 1));
      warp_valid_count   = __popc(warp_valid_mask);

      // if this is the value column emit an index for value decoding
      if (is_valid && s_idx == max_depth - 1) {
        int const src_pos = nesting_info->valid_count + thread_valid_count;
        int const dst_pos = nesting_info->value_count + thread_value_count;
        // nz_idx is a mapping of src buffer indices to destination buffer indices
        sb->nz_idx[rolling_index<rolling_buf_size>(src_pos)] = dst_pos;
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
        if (in_nesting_bounds && nesting_info->data_out != nullptr) {
          int const idx             = nesting_info->value_count + thread_value_count;
          cudf::size_type const ofs = nesting_info_base[s_idx + 1].value_count +
                                      next_thread_value_count +
                                      nesting_info_base[s_idx + 1].page_start_value;
          (reinterpret_cast<cudf::size_type*>(nesting_info->data_out))[idx] = ofs;
        }
      }

      // nested schemas always read and write to the same bounds (that is, read and write positions
      // are already pre-bounded by first_row/num_rows). flat schemas will start reading at the
      // first value, even if that is before first_row, because we cannot trivially jump to
      // the correct position to start reading. since we are about to write the validity vector here
      // we need to adjust our computed mask to take into account the write row bounds.
      int const in_write_row_bounds =
        !has_repetition
          ? thread_row_index >= s->first_row && thread_row_index < (s->first_row + s->num_rows)
          : in_row_bounds;
      int const first_thread_in_write_range =
        !has_repetition ? __ffs(ballot(in_write_row_bounds)) - 1 : 0;

      // # of bits to of the validity mask to write out
      int const warp_valid_mask_bit_count =
        first_thread_in_write_range < 0 ? 0 : warp_value_count - first_thread_in_write_range;

      // increment count of valid values, count of total values, and update validity mask
      if (!t) {
        if (nesting_info->valid_map != nullptr && warp_valid_mask_bit_count > 0) {
          uint32_t const warp_output_valid_mask = warp_valid_mask >> first_thread_in_write_range;
          store_validity(nesting_info->valid_map_offset,
                         nesting_info->valid_map,
                         warp_output_valid_mask,
                         warp_valid_mask_bit_count);
          nesting_info->valid_map_offset += warp_valid_mask_bit_count;
          nesting_info->null_count += warp_valid_mask_bit_count - __popc(warp_output_valid_mask);
        }
        nesting_info->valid_count += warp_valid_count;
        nesting_info->value_count += warp_value_count;
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
    s->nz_count          = nesting_info_base[max_depth - 1].valid_count;
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
 * @param[out] sb Page state buffer output
 * @param[in] target_leaf_count Target count of non-null leaf values to generate indices for
 * @param[in] rep Repetition level buffer
 * @param[in] def Definition level buffer
 * @param[in] t Thread index
 * @tparam rolling_buf_size Size of the cyclic buffer used to store value data
 * @tparam level_t Type used to store decoded repetition and definition levels
 * @tparam state_buf Typename of the `state_buf` (usually inferred)
 */
template <int rolling_buf_size, typename level_t, typename state_buf>
__device__ void gpuDecodeLevels(page_state_s* s,
                                state_buf* sb,
                                int32_t target_leaf_count,
                                level_t* const rep,
                                level_t* const def,
                                int t)
{
  bool has_repetition = s->col.max_level[level_type::REPETITION] > 0;

  constexpr int batch_size = 32;
  int cur_leaf_count       = target_leaf_count;
  while (s->error == 0 && s->nz_count < target_leaf_count &&
         s->input_value_count < s->num_input_values) {
    if (has_repetition) {
      gpuDecodeStream<level_t, rolling_buf_size>(rep, s, cur_leaf_count, t, level_type::REPETITION);
    }
    gpuDecodeStream<level_t, rolling_buf_size>(def, s, cur_leaf_count, t, level_type::DEFINITION);
    __syncwarp();

    // because the rep and def streams are encoded separately, we cannot request an exact
    // # of values to be decoded at once. we can only process the lowest # of decoded rep/def
    // levels we get.
    int actual_leaf_count = has_repetition ? min(s->lvl_count[level_type::REPETITION],
                                                 s->lvl_count[level_type::DEFINITION])
                                           : s->lvl_count[level_type::DEFINITION];

    // process what we got back
    gpuUpdateValidityOffsetsAndRowIndices<level_t, state_buf, rolling_buf_size>(
      actual_leaf_count, s, sb, rep, def, t);
    cur_leaf_count = actual_leaf_count + batch_size;
    __syncwarp();
  }
}

/**
 * @brief Parse the beginning of the level section (definition or repetition),
 * initializes the initial RLE run & value, and returns the section length
 *
 * @param[in,out] s The page state
 * @param[in] cur The current data position
 * @param[in] end The end of the data
 * @param[in] lvl Enum indicating whether this is to initialize repetition or definition level data
 *
 * @return The length of the section
 */
inline __device__ uint32_t InitLevelSection(page_state_s* s,
                                            uint8_t const* cur,
                                            uint8_t const* end,
                                            level_type lvl)
{
  int32_t len;
  int const level_bits = s->col.level_bits[lvl];
  auto const encoding  = lvl == level_type::DEFINITION ? s->page.definition_level_encoding
                                                       : s->page.repetition_level_encoding;

  auto start = cur;

  auto init_rle = [s, lvl, level_bits](uint8_t const* cur, uint8_t const* end) {
    uint32_t const run      = get_vlq32(cur, end);
    s->initial_rle_run[lvl] = run;
    if (!(run & 1)) {
      if (cur < end) {
        int v = cur[0];
        cur++;
        if (level_bits > 8) {
          v |= ((cur < end) ? cur[0] : 0) << 8;
          cur++;
        }
        s->initial_rle_value[lvl] = v;
      } else {
        s->initial_rle_value[lvl] = 0;
      }
    }
    s->lvl_start[lvl] = cur;

    if (cur > end) { s->set_error_code(decode_error::LEVEL_STREAM_OVERRUN); }
  };

  // this is a little redundant. if level_bits == 0, then nothing should be encoded
  // for the level, but some V2 files in the wild violate this and encode the data anyway.
  // thus we will handle V2 headers separately.
  if ((s->page.flags & PAGEINFO_FLAGS_V2) != 0 && (len = s->page.lvl_bytes[lvl]) != 0) {
    // V2 only uses RLE encoding so no need to check encoding
    s->abs_lvl_start[lvl] = cur;
    init_rle(cur, cur + len);
  } else if (level_bits == 0) {
    len                       = 0;
    s->initial_rle_run[lvl]   = s->page.num_input_values * 2;  // repeated value
    s->initial_rle_value[lvl] = 0;
    s->lvl_start[lvl]         = cur;
    s->abs_lvl_start[lvl]     = cur;
  } else if (encoding == Encoding::RLE) {  // V1 header with RLE encoding
    if (cur + 4 < end) {
      len = (cur[0]) + (cur[1] << 8) + (cur[2] << 16) + (cur[3] << 24);
      cur += 4;
      s->abs_lvl_start[lvl] = cur;
      init_rle(cur, cur + len);
      // add back the 4 bytes for the length
      len += 4;
    } else {
      len = 0;
      s->set_error_code(decode_error::LEVEL_STREAM_OVERRUN);
    }
  } else if (encoding == Encoding::BIT_PACKED) {
    len                       = (s->page.num_input_values * level_bits + 7) >> 3;
    s->initial_rle_run[lvl]   = ((s->page.num_input_values + 7) >> 3) * 2 + 1;  // literal run
    s->initial_rle_value[lvl] = 0;
    s->lvl_start[lvl]         = cur;
    s->abs_lvl_start[lvl]     = cur;
  } else {
    len = 0;
    s->set_error_code(decode_error::UNSUPPORTED_ENCODING);
  }

  s->abs_lvl_end[lvl] = start + len;

  return static_cast<uint32_t>(len);
}

/**
 * @brief Functor for setupLocalPageInfo that always returns true.
 */
struct all_types_filter {
  __device__ inline bool operator()(PageInfo const& page) { return true; }
};

/**
 * @brief Functor for setupLocalPageInfo that takes a mask of allowed types.
 */
struct mask_filter {
  uint32_t mask;

  __device__ mask_filter(uint32_t m) : mask(m) {}
  __device__ mask_filter(decode_kernel_mask m) : mask(static_cast<uint32_t>(m)) {}

  __device__ inline bool operator()(PageInfo const& page)
  {
    return BitAnd(mask, page.kernel_mask) != 0;
  }
};

enum class page_processing_stage {
  PREPROCESS,
  STRING_BOUNDS,
  DECODE,
};

/**
 * @brief Sets up block-local page state information from the global pages.
 *
 * @param[in, out] s The local page state to be filled in
 * @param[in] p The global page to be copied from
 * @param[in] chunks The global list of chunks
 * @param[in] min_row Crop all rows below min_row
 * @param[in] num_rows Maximum number of rows to read
 * @param[in] filter Filtering function used to decide which pages to operate on
 * @param[in] stage What stage of the decoding process is this being called from
 * @tparam Filter Function that takes a PageInfo reference and returns true if the given page should
 * be operated on Currently only used by gpuComputePageSizes step)
 * @return True if this page should be processed further
 */
template <typename Filter>
inline __device__ bool setupLocalPageInfo(page_state_s* const s,
                                          PageInfo const* p,
                                          device_span<ColumnChunkDesc const> chunks,
                                          size_t min_row,
                                          size_t num_rows,
                                          Filter filter,
                                          page_processing_stage stage)
{
  int t = threadIdx.x;

  // Fetch page info
  if (!t) {
    s->page         = *p;
    s->nesting_info = nullptr;
    s->col          = chunks[s->page.chunk_idx];
  }
  __syncthreads();

  // return false if this is a dictionary page or it does not pass the filter condition
  if ((s->page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0 || !filter(s->page)) { return false; }

  // our starting row (absolute index) is
  // col.start_row == absolute row index
  // page.chunk-row == relative row index within the chunk
  size_t const page_start_row = s->col.start_row + s->page.chunk_row;

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

    // NOTE: s->page.num_rows, s->col.chunk_row, s->first_row and s->num_rows will be
    // invalid/bogus during first pass of the preprocess step for nested types. this is ok
    // because we ignore these values in that stage.
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

  // if we have no work to do (eg, in a skip_rows/num_rows case) in this page.
  //
  // corner case: in the case of lists, we can have pages that contain "0" rows if the current row
  // starts before this page and ends after this page:
  //       P0        P1        P2
  //  |---------|---------|----------|
  //        ^------------------^
  //      row start           row end
  // P1 will contain 0 rows
  //
  // NOTE: this check needs to be done after the null counts have been zeroed out
  bool const has_repetition = s->col.max_level[level_type::REPETITION] > 0;
  if ((stage == page_processing_stage::STRING_BOUNDS || stage == page_processing_stage::DECODE) &&
      s->num_rows == 0 &&
      !(has_repetition && (is_bounds_page(s, min_row, num_rows, has_repetition) ||
                           is_page_contained(s, min_row, num_rows)))) {
    return false;
  }

  if (!t) {
    s->reset_error_code();

    // IMPORTANT : nested schemas can have 0 rows in a page but still have
    // values. The case is:
    // - On page N-1, the last row starts, with 2/6 values encoded
    // - On page N, the remaining 4/6 values are encoded, but there are no new rows.
    // if (s->page.num_input_values > 0 && s->page.num_rows > 0) {
    if (s->page.num_input_values > 0) {
      uint8_t* cur = s->page.page_data;
      uint8_t* end = cur + s->page.uncompressed_page_size;
      s->ts_scale  = 0;
      // Validate data type
      auto const data_type = s->col.physical_type;
      auto const is_decimal =
        s->col.logical_type.has_value() and s->col.logical_type->type == LogicalType::DECIMAL;
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
            if (s->col.logical_type.has_value()) {
              auto const& lt = *s->col.logical_type;
              if (lt.is_timestamp_millis()) {
                units = cudf::timestamp_ms::period::den;
              } else if (lt.is_timestamp_micros()) {
                units = cudf::timestamp_us::period::den;
              } else if (lt.is_timestamp_nanos()) {
                units = cudf::timestamp_ns::period::den;
              }
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
          if (is_decimal) {
            auto const decimal_precision = s->col.logical_type->precision();
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
          s->dtype_len = s->col.type_length;
          if (s->dtype_len <= 0) { s->set_error_code(decode_error::INVALID_DATA_TYPE); }
          break;
      }
      // Special check for downconversions
      s->dtype_len_in = s->dtype_len;
      if (data_type == FIXED_LEN_BYTE_ARRAY) {
        if (is_decimal) {
          s->dtype_len = [dtype_len = s->dtype_len]() {
            if (dtype_len <= sizeof(int32_t)) {
              return sizeof(int32_t);
            } else if (dtype_len <= sizeof(int64_t)) {
              return sizeof(int64_t);
            } else {
              return sizeof(__int128_t);
            }
          }();
        } else {
          s->dtype_len = sizeof(string_index_pair);
        }
      } else if (data_type == INT32) {
        // check for smaller bitwidths
        if (s->col.logical_type.has_value()) {
          auto const& lt = *s->col.logical_type;
          if (lt.type == LogicalType::INTEGER) {
            s->dtype_len = lt.bit_width() / 8;
          } else if (lt.is_time_millis()) {
            // cudf outputs as INT64
            s->dtype_len = 8;
          }
        }
      } else if (data_type == BYTE_ARRAY && s->col.is_strings_to_cat) {
        s->dtype_len = 4;  // HASH32 output
      } else if (data_type == INT96) {
        s->dtype_len = 8;  // Convert to 64-bit timestamp
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
      if (stage == page_processing_stage::DECODE) {
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

          if (s->col.column_data_base != nullptr) {
            nesting_info->data_out = static_cast<uint8_t*>(s->col.column_data_base[idx]);
            if (s->col.column_string_base != nullptr) {
              nesting_info->string_out = static_cast<uint8_t*>(s->col.column_string_base[idx]);
            }

            nesting_info->data_out = static_cast<uint8_t*>(s->col.column_data_base[idx]);

            if (nesting_info->data_out != nullptr) {
              // anything below max depth with a valid data pointer must be a list, so the
              // element size is the size of the offset type.
              uint32_t len = idx < max_depth - 1 ? sizeof(cudf::size_type) : s->dtype_len;
              // if this is a string column, then dtype_len is a lie. data will be offsets rather
              // than (ptr,len) tuples.
              if (is_string_col(s->col)) { len = sizeof(cudf::size_type); }
              nesting_info->data_out += (output_offset * len);
            }
            if (nesting_info->string_out != nullptr) {
              nesting_info->string_out += s->page.str_offset;
            }
            nesting_info->valid_map = s->col.valid_map_base[idx];
            if (nesting_info->valid_map != nullptr) {
              nesting_info->valid_map += output_offset >> 5;
              nesting_info->valid_map_offset = (int32_t)(output_offset & 0x1f);
            }
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
      s->dict_val  = 0;
      // NOTE:  if additional encodings are supported in the future, modifications must
      // be made to is_supported_encoding() in reader_impl_preprocess.cu
      switch (s->page.encoding) {
        case Encoding::PLAIN_DICTIONARY:
        case Encoding::RLE_DICTIONARY: {
          // RLE-packed dictionary indices, first byte indicates index length in bits
          auto const is_decimal =
            s->col.logical_type.has_value() and s->col.logical_type->type == LogicalType::DECIMAL;
          if ((s->col.physical_type == BYTE_ARRAY or
               s->col.physical_type == FIXED_LEN_BYTE_ARRAY) and
              not is_decimal and s->col.str_dict_index != nullptr) {
            // String dictionary: use index
            s->dict_base = reinterpret_cast<uint8_t const*>(s->col.str_dict_index);
            s->dict_size = s->col.dict_page->num_input_values * sizeof(string_index_pair);
          } else {
            s->dict_base = s->col.dict_page->page_data;
            s->dict_size = s->col.dict_page->uncompressed_page_size;
          }
          s->dict_run  = 0;
          s->dict_val  = 0;
          s->dict_bits = (cur < end) ? *cur++ : 0;
          if (s->dict_bits > 32 || (!s->dict_base && s->col.dict_page->num_input_values > 0)) {
            s->set_error_code(decode_error::INVALID_DICT_WIDTH);
          }
        } break;
        case Encoding::PLAIN:
        case Encoding::BYTE_STREAM_SPLIT:
          s->dict_size = static_cast<int32_t>(end - cur);
          s->dict_val  = 0;
          if (s->col.physical_type == BOOLEAN) { s->dict_run = s->dict_size * 2 + 1; }
          break;
        case Encoding::RLE: {
          // first 4 bytes are length of RLE data
          int const len = (cur[0]) + (cur[1] << 8) + (cur[2] << 16) + (cur[3] << 24);
          cur += 4;
          if (cur + len > end) { s->set_error_code(decode_error::DATA_STREAM_OVERRUN); }
          s->dict_run = 0;
        } break;
        case Encoding::DELTA_BINARY_PACKED:
        case Encoding::DELTA_LENGTH_BYTE_ARRAY:
        case Encoding::DELTA_BYTE_ARRAY:
          // nothing to do, just don't error
          break;
        default: {
          s->set_error_code(decode_error::UNSUPPORTED_ENCODING);
          break;
        }
      }
      if (cur > end) { s->set_error_code(decode_error::DATA_STREAM_OVERRUN); }
      s->lvl_end    = cur;
      s->data_start = cur;
      s->data_end   = end;
    } else {
      s->set_error_code(decode_error::EMPTY_PAGE);
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
      if (stage == page_processing_stage::DECODE) {
        s->input_value_count = s->page.skipped_values > -1 ? s->page.skipped_values : 0;
      } else if (stage == page_processing_stage::PREPROCESS) {
        s->input_value_count = 0;
        s->input_leaf_count  = 0;
        // magic number to indicate it hasn't been set for use inside UpdatePageSizes
        s->page.skipped_values      = -1;
        s->page.skipped_leaf_values = 0;
      }
    }

    __threadfence_block();
  }
  __syncthreads();

  return true;
}

}  // namespace cudf::io::parquet::detail
