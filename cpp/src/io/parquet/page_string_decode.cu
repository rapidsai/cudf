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

#include "delta_binary.cuh"
#include "error.hpp"
#include "page_decode.cuh"
#include "page_string_utils.cuh"
#include "rle_stream.cuh"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/strings/detail/gather.cuh>

#include <thrust/logical.h>
#include <thrust/transform_scan.h>

#include <bitset>

namespace cudf::io::parquet::detail {

namespace {

constexpr int preprocess_block_size    = 512;
constexpr int decode_block_size        = 128;
constexpr int delta_preproc_block_size = 64;
constexpr int delta_length_block_size  = 32;
constexpr int rolling_buf_size         = decode_block_size * 2;
constexpr int preproc_buf_size         = LEVEL_DECODE_BUF_SIZE;

/**
 * @brief Compute the start and end page value bounds for this page
 *
 * This uses definition and repetition level info to determine the number of valid and null
 * values for the page, taking into account skip_rows/num_rows (if set).
 *
 * @param s The local page info
 * @param min_row Row index to start reading at
 * @param num_rows Maximum number of rows to read
 * @param is_bounds_pg True if this page is clipped
 * @param has_repetition True if the schema is nested
 * @param decoders Definition and repetition level decoders
 * @return pair containing start and end value indexes
 * @tparam level_t Type used to store decoded repetition and definition levels
 * @tparam rle_buf_size Size of the buffer used when decoding repetition and definition levels
 */
template <typename level_t, int rle_buf_size>
__device__ thrust::pair<int, int> page_bounds(
  page_state_s* const s,
  size_t min_row,
  size_t num_rows,
  bool is_bounds_pg,
  bool has_repetition,
  rle_stream<level_t, rle_buf_size, preproc_buf_size>* decoders)
{
  using block_reduce = cub::BlockReduce<int, preprocess_block_size>;
  using block_scan   = cub::BlockScan<int, preprocess_block_size>;
  __shared__ union {
    typename block_reduce::TempStorage reduce_storage;
    typename block_scan::TempStorage scan_storage;
  } temp_storage;

  auto const t = threadIdx.x;

  // decode batches of level stream data using rle_stream objects and use the results to
  // calculate start and end value positions in the encoded string data.
  int const max_depth = s->col.max_nesting_depth;
  int const max_def   = s->nesting_info[max_depth - 1].max_def_level;

  // can skip all this if we know there are no nulls
  if (max_def == 0 && !is_bounds_pg) {
    if (t == 0) {
      s->page.num_valids = s->num_input_values;
      s->page.num_nulls  = 0;
    }
    return {0, s->num_input_values};
  }

  int start_value = 0;
  int end_value   = s->page.num_input_values;
  auto const pp   = &s->page;
  auto const col  = &s->col;

  // initialize the stream decoders (requires values computed in setupLocalPageInfo)
  auto const def_decode = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  auto const rep_decode = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::REPETITION]);
  decoders[level_type::DEFINITION].init(s->col.level_bits[level_type::DEFINITION],
                                        s->abs_lvl_start[level_type::DEFINITION],
                                        s->abs_lvl_end[level_type::DEFINITION],
                                        def_decode,
                                        s->page.num_input_values);
  // only need repetition if this is a bounds page. otherwise all we need is def level info
  // to count the nulls.
  if (has_repetition && is_bounds_pg) {
    decoders[level_type::REPETITION].init(s->col.level_bits[level_type::REPETITION],
                                          s->abs_lvl_start[level_type::REPETITION],
                                          s->abs_lvl_end[level_type::REPETITION],
                                          rep_decode,
                                          s->page.num_input_values);
  }

  int processed = 0;

  // if this is a bounds page, we need to do extra work to find the start and/or end value index
  if (is_bounds_pg) {
    __shared__ int skipped_values;
    __shared__ int skipped_leaf_values;
    __shared__ int last_input_value;
    __shared__ int end_val_idx;

    // need these for skip_rows case
    auto const page_start_row = col->start_row + pp->chunk_row;
    auto const max_row        = min_row + num_rows;
    auto const begin_row      = page_start_row >= min_row ? 0 : min_row - page_start_row;
    auto const max_page_rows  = pp->num_rows - begin_row;
    auto const page_rows      = page_start_row + begin_row + max_page_rows <= max_row
                                  ? max_page_rows
                                  : max_row - (page_start_row + begin_row);
    auto end_row              = begin_row + page_rows;
    int row_fudge             = -1;

    // short circuit for no nulls
    if (max_def == 0 && !has_repetition) {
      if (t == 0) {
        pp->num_nulls  = 0;
        pp->num_valids = end_row - begin_row;
      }
      return {begin_row, end_row};
    }

    int row_count           = 0;
    int leaf_count          = 0;
    bool skipped_values_set = false;
    bool end_value_set      = false;

    // If page_start_row >= min_row, then skipped_values is 0 and we don't have to search for
    // start_value. If there's repetition then we've already calculated
    // skipped_values/skipped_leaf_values.
    // TODO(ets): If we hit this condition, and end_row > last row in page, then we can skip
    // more of the processing below.
    if (has_repetition or page_start_row >= min_row) {
      if (t == 0) {
        if (has_repetition) {
          skipped_values      = pp->skipped_values;
          skipped_leaf_values = pp->skipped_leaf_values;
        } else {
          skipped_values      = 0;
          skipped_leaf_values = 0;
        }
      }
      skipped_values_set = true;
      __syncthreads();
    }

    while (processed < s->page.num_input_values) {
      thread_index_type start_val = processed;

      if (has_repetition) {
        decoders[level_type::REPETITION].decode_next(t);
        __syncthreads();

        // special case where page does not begin at a row boundary
        if (processed == 0 && rep_decode[0] != 0) {
          end_row++;  // need to finish off the previous row
          row_fudge = 0;
        }
      }

      // the # of rep/def levels will always be the same size
      processed += decoders[level_type::DEFINITION].decode_next(t);
      __syncthreads();

      // do something with the level data
      while (start_val < processed) {
        auto const idx_t = start_val + t;
        auto const idx   = rolling_index<preproc_buf_size>(idx_t);

        // get absolute thread row index
        int is_new_row = idx_t < processed && (!has_repetition || rep_decode[idx] == 0);
        int thread_row_count, block_row_count;
        block_scan(temp_storage.scan_storage)
          .InclusiveSum(is_new_row, thread_row_count, block_row_count);
        __syncthreads();

        // get absolute thread leaf index
        int const is_new_leaf = idx_t < processed && (def_decode[idx] >= max_def);
        int thread_leaf_count, block_leaf_count;
        block_scan(temp_storage.scan_storage)
          .InclusiveSum(is_new_leaf, thread_leaf_count, block_leaf_count);
        __syncthreads();

        // if we have not set skipped values yet, see if we found the first in-bounds row
        if (!skipped_values_set && row_count + block_row_count > begin_row) {
          // if this thread is in row bounds
          int const row_index = thread_row_count + row_count - 1;
          int const in_row_bounds =
            idx_t < processed && (row_index >= begin_row) && (row_index < end_row);

          int local_count, global_count;
          block_scan(temp_storage.scan_storage)
            .InclusiveSum(in_row_bounds, local_count, global_count);
          __syncthreads();

          // we found it
          if (global_count > 0) {
            // this is the thread that represents the first row. need to test in_row_bounds for
            // the case where we only want one row and local_count == 1 for many threads.
            if (local_count == 1 && in_row_bounds) {
              skipped_values = idx_t;
              skipped_leaf_values =
                leaf_count + (is_new_leaf ? thread_leaf_count - 1 : thread_leaf_count);
            }
            skipped_values_set = true;
          }
        }

        // test if row_count will exceed end_row in this batch
        if (!end_value_set && row_count + block_row_count >= end_row) {
          // if this thread exceeds row bounds. row_fudge change depending on whether we've faked
          // the end row to account for starting a page in the middle of a row.
          int const row_index          = thread_row_count + row_count + row_fudge;
          int const exceeds_row_bounds = row_index >= end_row;

          int local_count, global_count;
          block_scan(temp_storage.scan_storage)
            .InclusiveSum(exceeds_row_bounds, local_count, global_count);
          __syncthreads();

          // we found it
          if (global_count > 0) {
            // this is the thread that represents the end row.
            if (local_count == 1) {
              last_input_value = idx_t;
              end_val_idx = leaf_count + (is_new_leaf ? thread_leaf_count - 1 : thread_leaf_count);
            }
            end_value_set = true;
            break;
          }
        }

        row_count += block_row_count;
        leaf_count += block_leaf_count;

        start_val += preprocess_block_size;
      }
      __syncthreads();
      if (end_value_set) { break; }
    }

    start_value = skipped_values_set ? skipped_leaf_values : 0;
    end_value   = end_value_set ? end_val_idx : leaf_count;

    if (t == 0) {
      int const v0                = skipped_values_set ? skipped_values : 0;
      int const vn                = end_value_set ? last_input_value : s->num_input_values;
      int const total_values      = vn - v0;
      int const total_leaf_values = end_value - start_value;
      int const num_nulls         = total_values - total_leaf_values;
      pp->num_nulls               = num_nulls;
      pp->num_valids              = total_leaf_values;
    }
  }
  // already filtered out unwanted pages, so need to count all non-null values in this page
  else {
    int num_nulls = 0;
    while (processed < s->page.num_input_values) {
      thread_index_type start_val = processed;
      processed += decoders[level_type::DEFINITION].decode_next(t);
      __syncthreads();

      while (start_val < processed) {
        auto const idx_t = start_val + t;
        if (idx_t < processed) {
          auto const idx = rolling_index<preproc_buf_size>(idx_t);
          if (def_decode[idx] < max_def) { num_nulls++; }
        }
        start_val += preprocess_block_size;
      }
      __syncthreads();
    }

    int const null_count = block_reduce(temp_storage.reduce_storage).Sum(num_nulls);

    if (t == 0) {
      pp->num_nulls  = null_count;
      pp->num_valids = pp->num_input_values - null_count;
    }

    end_value -= pp->num_nulls;
  }

  return {start_value, end_value};
}

/**
 * @brief Compute string size information for dictionary encoded strings.
 *
 * @param data Pointer to the start of the page data stream
 * @param dict_base Pointer to the start of the dictionary
 * @param dict_bits The number of bits used to in the dictionary bit packing
 * @param dict_size Size of the dictionary in bytes
 * @param data_size Size of the page data in bytes
 * @param start_value Do not count values that occur before this index
 * @param end_value Do not count values that occur after this index
 */
__device__ size_t totalDictEntriesSize(uint8_t const* data,
                                       uint8_t const* dict_base,
                                       int dict_bits,
                                       int dict_size,
                                       int data_size,
                                       int start_value,
                                       int end_value)
{
  int const t              = threadIdx.x;
  uint8_t const* ptr       = data;
  uint8_t const* const end = data + data_size;
  int const bytecnt        = (dict_bits + 7) >> 3;
  size_t l_str_len         = 0;  // partial sums across threads
  int pos                  = 0;  // current value index in the data stream
  int t0                   = 0;  // thread 0 for this batch

  int dict_run = 0;
  int dict_val = 0;

  while (pos < end_value && ptr <= end) {
    if (dict_run <= 1) {
      dict_run = (ptr < end) ? get_vlq32(ptr, end) : 0;
      if (!(dict_run & 1)) {
        // Repeated value
        if (ptr + bytecnt <= end) {
          int32_t run_val = ptr[0];
          if (bytecnt > 1) {
            run_val |= ptr[1] << 8;
            if (bytecnt > 2) {
              run_val |= ptr[2] << 16;
              if (bytecnt > 3) { run_val |= ptr[3] << 24; }
            }
          }
          dict_val = run_val & ((1 << dict_bits) - 1);
        }
        ptr += bytecnt;
      }
    }

    int batch_len;
    if (dict_run & 1) {
      // Literal batch: must output a multiple of 8, except for the last batch
      int batch_len_div8;
      batch_len      = max(min(preprocess_block_size, (int)(dict_run >> 1) * 8), 1);
      batch_len_div8 = (batch_len + 7) >> 3;
      dict_run -= batch_len_div8 * 2;
      ptr += batch_len_div8 * dict_bits;
    } else {
      batch_len = dict_run >> 1;
      dict_run  = 0;
    }

    int const is_literal = dict_run & 1;

    // calculate my thread id for this batch.  way to round-robin the work.
    int mytid = t - t0;
    if (mytid < 0) mytid += preprocess_block_size;

    // compute dictionary index.
    if (is_literal) {
      int dict_idx = 0;
      if (mytid < batch_len) {
        dict_idx         = dict_val;
        int32_t ofs      = (mytid - ((batch_len + 7) & ~7)) * dict_bits;
        const uint8_t* p = ptr + (ofs >> 3);
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

        if (pos + mytid < end_value) {
          uint32_t const dict_pos = (dict_bits > 0) ? dict_idx * sizeof(string_index_pair) : 0;
          if (pos + mytid >= start_value && dict_pos < (uint32_t)dict_size) {
            const auto* src = reinterpret_cast<const string_index_pair*>(dict_base + dict_pos);
            l_str_len += src->second;
          }
        }
      }

      t0 += batch_len;
    } else {
      int const start_off =
        (pos < start_value && pos + batch_len > start_value) ? start_value - pos : 0;
      batch_len = min(batch_len, end_value - pos);
      if (mytid == 0) {
        uint32_t const dict_pos = (dict_bits > 0) ? dict_val * sizeof(string_index_pair) : 0;
        if (pos + batch_len > start_value && dict_pos < (uint32_t)dict_size) {
          const auto* src = reinterpret_cast<const string_index_pair*>(dict_base + dict_pos);
          l_str_len += (batch_len - start_off) * src->second;
        }
      }

      t0 += 1;
    }

    t0 = t0 % preprocess_block_size;
    pos += batch_len;
  }
  __syncthreads();

  using block_reduce = cub::BlockReduce<size_t, preprocess_block_size>;
  __shared__ typename block_reduce::TempStorage reduce_storage;
  size_t sum_l = block_reduce(reduce_storage).Sum(l_str_len);

  return sum_l;
}

/**
 * @brief Compute string size information for plain encoded strings.
 *
 * @param data Pointer to the start of the page data stream
 * @param data_size Length of data
 * @param start_value Do not count values that occur before this index
 * @param end_value Do not count values that occur after this index
 */
__device__ size_t totalPlainEntriesSize(uint8_t const* data,
                                        int data_size,
                                        int start_value,
                                        int end_value)
{
  int const t      = threadIdx.x;
  int pos          = 0;
  size_t total_len = 0;

  // This step is purely serial
  if (!t) {
    const uint8_t* cur = data;
    int k              = 0;

    while (pos < end_value && k < data_size) {
      int len;
      if (k + 4 <= data_size) {
        len = (cur[k]) | (cur[k + 1] << 8) | (cur[k + 2] << 16) | (cur[k + 3] << 24);
        k += 4;
        if (k + len > data_size) { len = 0; }
      } else {
        len = 0;
      }

      k += len;
      if (pos >= start_value) { total_len += len; }
      pos++;
    }
  }

  return total_len;
}

/**
 * @brief Compute string size information for DELTA_BYTE_ARRAY encoded strings.
 *
 * This traverses the packed prefix and suffix lengths, summing them to obtain the total
 * number of bytes needed for the decoded string data. It also calculates an upper bound
 * for the largest string length to obtain an upper bound on temporary space needed if
 * rows will be skipped.
 *
 * Called with 64 threads.
 *
 * @param data Pointer to the start of the page data stream
 * @param end Pointer to the end of the page data stream
 * @param start_value Do not count values that occur before this index
 * @param end_value Do not count values that occur after this index
 * @return A pair of `size_t` values representing the total string size and temp buffer size
 * required for decoding
 */
__device__ thrust::pair<size_t, size_t> totalDeltaByteArraySize(uint8_t const* data,
                                                                uint8_t const* end,
                                                                int start_value,
                                                                int end_value)
{
  using cudf::detail::warp_size;
  using WarpReduce = cub::WarpReduce<uleb128_t>;
  __shared__ typename WarpReduce::TempStorage temp_storage[2];

  __shared__ __align__(16) delta_binary_decoder prefixes;
  __shared__ __align__(16) delta_binary_decoder suffixes;

  int const t       = threadIdx.x;
  int const lane_id = t % warp_size;
  int const warp_id = t / warp_size;

  if (t == 0) {
    auto const* suffix_start = prefixes.find_end_of_block(data, end);
    suffixes.init_binary_block(suffix_start, end);
  }
  __syncthreads();

  // two warps will traverse the prefixes and suffixes and sum them up
  auto const db = t < warp_size ? &prefixes : t < 2 * warp_size ? &suffixes : nullptr;

  size_t total_bytes = 0;
  uleb128_t max_len  = 0;

  if (db != nullptr) {
    // initialize with first value (which is stored in last_value)
    if (lane_id == 0 && start_value == 0) { total_bytes = db->last_value; }

    uleb128_t lane_sum = 0;
    uleb128_t lane_max = 0;
    while (db->current_value_idx < end_value &&
           db->current_value_idx < db->num_encoded_values(true)) {
      // calculate values for current mini-block
      db->calc_mini_block_values(lane_id);

      // get per lane sum for mini-block
      for (uint32_t i = 0; i < db->values_per_mb; i += 32) {
        uint32_t const idx = db->current_value_idx + i + lane_id;
        if (idx >= start_value && idx < end_value && idx < db->value_count) {
          lane_sum += db->value[rolling_index<delta_rolling_buf_size>(idx)];
        }
        // need lane_max over all values, not just in bounds
        if (idx < db->value_count) {
          lane_max = max(lane_max, db->value[rolling_index<delta_rolling_buf_size>(idx)]);
        }
      }

      if (lane_id == 0) { db->setup_next_mini_block(true); }
      __syncwarp();
    }

    // get sum for warp.
    // note: warp_sum will only be valid on lane 0.
    auto const warp_sum = WarpReduce(temp_storage[warp_id]).Sum(lane_sum);
    __syncwarp();
    auto const warp_max = WarpReduce(temp_storage[warp_id]).Reduce(lane_max, cub::Max());

    if (lane_id == 0) {
      total_bytes += warp_sum;
      max_len = warp_max;
    }
  }
  __syncthreads();

  // now sum up total_bytes from the two warps
  auto const final_bytes =
    cudf::detail::single_lane_block_sum_reduce<delta_preproc_block_size, 0>(total_bytes);

  // Sum up prefix and suffix max lengths to get a max possible string length. Multiply that
  // by the number of strings in a mini-block, plus one to save the last string.
  auto const temp_bytes =
    cudf::detail::single_lane_block_sum_reduce<delta_preproc_block_size, 0>(max_len) *
    (db->values_per_mb + 1);

  return {final_bytes, temp_bytes};
}

/**
 * @brief Kernel for computing string page bounds information.
 *
 * This kernel traverses the repetition and definition level data to determine start and end values
 * for pages with string-like data. Also calculates the number of null and valid values in the
 * page. Does nothing if the page mask is neither `STRING` nor `DELTA_BYTE_ARRAY`. On exit the
 * `num_nulls`, `num_valids`, `start_val` and `end_val` fields of the `PageInfo` struct will be
 * populated.
 *
 * @param pages All pages to be decoded
 * @param chunks All chunks to be decoded
 * @param min_rows crop all rows below min_row
 * @param num_rows Maximum number of rows to read
 * @tparam level_t Type used to store decoded repetition and definition levels
 */
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(preprocess_block_size) gpuComputeStringPageBounds(
  PageInfo* pages, device_span<ColumnChunkDesc const> chunks, size_t min_row, size_t num_rows)
{
  __shared__ __align__(16) page_state_s state_g;

  page_state_s* const s = &state_g;
  int const page_idx    = blockIdx.x;
  int const t           = threadIdx.x;
  PageInfo* const pp    = &pages[page_idx];

  if (t == 0) {
    // don't clobber these if they're already computed from the index
    if (!pp->has_page_index) {
      s->page.num_nulls  = 0;
      s->page.num_valids = 0;
    }
    // reset str_bytes to 0 in case it's already been calculated (esp needed for chunked reads).
    pp->str_bytes = 0;
  }

  // whether or not we have repetition levels (lists)
  bool const has_repetition = chunks[pp->chunk_idx].max_level[level_type::REPETITION] > 0;

  // the required number of runs in shared memory we will need to provide the
  // rle_stream object
  constexpr int rle_run_buffer_size = rle_stream_required_run_buffer_size<preprocess_block_size>();

  // the level stream decoders
  __shared__ rle_run<level_t> def_runs[rle_run_buffer_size];
  __shared__ rle_run<level_t> rep_runs[rle_run_buffer_size];
  rle_stream<level_t, preprocess_block_size, preproc_buf_size>
    decoders[level_type::NUM_LEVEL_TYPES] = {{def_runs}, {rep_runs}};

  // setup page info
  if (!setupLocalPageInfo(s,
                          pp,
                          chunks,
                          min_row,
                          num_rows,
                          mask_filter{STRINGS_MASK},
                          page_processing_stage::STRING_BOUNDS)) {
    return;
  }

  bool const is_bounds_pg = is_bounds_page(s, min_row, num_rows, has_repetition);

  // if we have size info, then we only need to do this for bounds pages
  if (pp->has_page_index && !is_bounds_pg) { return; }

  // find start/end value indices
  auto const [start_value, end_value] =
    page_bounds(s, min_row, num_rows, is_bounds_pg, has_repetition, decoders);

  // need to save num_nulls and num_valids calculated in page_bounds in this page
  if (t == 0) {
    pp->num_nulls  = s->page.num_nulls;
    pp->num_valids = s->page.num_valids;
    pp->start_val  = start_value;
    pp->end_val    = end_value;
  }
}

/**
 * @brief Kernel for computing string page output size information for delta_byte_array encoding.
 *
 * This call ignores columns that are not DELTA_BYTE_ARRAY encoded. On exit the `str_bytes` field
 * of the `PageInfo` struct will be populated. Also fills in the `temp_string_size` field if rows
 * are to be skipped.
 *
 * @param pages All pages to be decoded
 * @param chunks All chunks to be decoded
 * @param min_rows crop all rows below min_row
 * @param num_rows Maximum number of rows to read
 */
CUDF_KERNEL void __launch_bounds__(delta_preproc_block_size) gpuComputeDeltaPageStringSizes(
  PageInfo* pages, device_span<ColumnChunkDesc const> chunks, size_t min_row, size_t num_rows)
{
  __shared__ __align__(16) page_state_s state_g;

  page_state_s* const s = &state_g;
  int const page_idx    = blockIdx.x;
  int const t           = threadIdx.x;
  PageInfo* const pp    = &pages[page_idx];

  // whether or not we have repetition levels (lists)
  bool const has_repetition = chunks[pp->chunk_idx].max_level[level_type::REPETITION] > 0;

  // setup page info
  if (!setupLocalPageInfo(s,
                          pp,
                          chunks,
                          min_row,
                          num_rows,
                          mask_filter{decode_kernel_mask::DELTA_BYTE_ARRAY},
                          page_processing_stage::STRING_BOUNDS)) {
    return;
  }

  auto const start_value = pp->start_val;

  // if data size is known, can short circuit here
  if ((chunks[pp->chunk_idx].data_type & 7) == FIXED_LEN_BYTE_ARRAY) {
    if (t == 0) {
      pp->str_bytes = pp->num_valids * s->dtype_len_in;

      // only need temp space if we're skipping values
      if (start_value > 0) {
        // just need to parse the header of the first delta binary block to get values_per_mb
        delta_binary_decoder db;
        db.init_binary_block(s->data_start, s->data_end);
        // save enough for one mini-block plus some extra to save the last_string
        pp->temp_string_size = s->dtype_len_in * (db.values_per_mb + 1);
      }
    }
  } else {
    bool const is_bounds_pg = is_bounds_page(s, min_row, num_rows, has_repetition);

    // if we have size info, then we only need to do this for bounds pages
    if (pp->has_page_index && !is_bounds_pg) {
      // check if we need to store values from the index
      if (is_page_contained(s, min_row, num_rows)) { pp->str_bytes = pp->str_bytes_from_index; }
      return;
    }

    // now process string info in the range [start_value, end_value)
    // set up for decoding strings...can be either plain or dictionary
    uint8_t const* data      = s->data_start;
    uint8_t const* const end = s->data_end;
    auto const end_value     = pp->end_val;

    auto const [len, temp_bytes] = totalDeltaByteArraySize(data, end, start_value, end_value);

    if (t == 0) {
      pp->str_bytes = len;

      // only need temp space if we're skipping values
      if (start_value > 0) { pp->temp_string_size = temp_bytes; }
    }
  }
}

/**
 * @brief Kernel for computing string page output size information for DELTA_LENGTH_BYTE_ARRAY
 * encoding.
 *
 * This call ignores columns that are not DELTA_LENGTH_BYTE_ARRAY encoded. On exit the `str_bytes`
 * field of the `PageInfo` struct will be populated.
 *
 * Currently this function only supports being called by a single warp.
 *
 * @param pages All pages to be decoded
 * @param chunks All chunks to be decoded
 * @param min_rows crop all rows below min_row
 * @param num_rows Maximum number of rows to read
 */
CUDF_KERNEL void __launch_bounds__(delta_length_block_size) gpuComputeDeltaLengthPageStringSizes(
  PageInfo* pages, device_span<ColumnChunkDesc const> chunks, size_t min_row, size_t num_rows)
{
  using cudf::detail::warp_size;
  using WarpReduce = cub::WarpReduce<uleb128_t>;
  __shared__ typename WarpReduce::TempStorage temp_storage;
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16) delta_binary_decoder string_lengths;

  page_state_s* const s = &state_g;
  int const page_idx    = blockIdx.x;
  int const t           = threadIdx.x;
  PageInfo* const pp    = &pages[page_idx];

  // whether or not we have repetition levels (lists)
  bool const has_repetition = chunks[pp->chunk_idx].max_level[level_type::REPETITION] > 0;

  // setup page info
  if (!setupLocalPageInfo(s,
                          pp,
                          chunks,
                          min_row,
                          num_rows,
                          mask_filter{decode_kernel_mask::DELTA_LENGTH_BA},
                          page_processing_stage::STRING_BOUNDS)) {
    return;
  }

  bool const is_bounds_pg = is_bounds_page(s, min_row, num_rows, has_repetition);

  // if we have size info, then we only need to do this for bounds pages
  if (pp->has_page_index && !is_bounds_pg) {
    // check if we need to store values from the index
    if (is_page_contained(s, min_row, num_rows)) { pp->str_bytes = pp->str_bytes_from_index; }
    return;
  }

  // for DELTA_LENGTH_BYTE_ARRAY, string size is page_data_size - size_of_delta_binary_block.
  // so all we need to do is skip the encoded string size info and then do pointer arithmetic,
  // if this isn't a bounds page.
  if (not is_bounds_pg) {
    if (t == 0) {
      auto const* string_start = string_lengths.find_end_of_block(s->data_start, s->data_end);
      size_t len               = static_cast<size_t>(s->data_end - string_start);
      pp->str_bytes            = len;
    }
  } else {
    // now process string info in the range [start_value, end_value)
    // set up for decoding strings...can be either plain or dictionary
    auto const start_value = pp->start_val;
    auto const end_value   = pp->end_val;

    if (t == 0) { string_lengths.init_binary_block(s->data_start, s->data_end); }
    __syncwarp();

    size_t total_bytes = 0;

    // initialize with first value (unless there are no values)
    if (t == 0 && start_value == 0 && start_value < end_value) {
      total_bytes = string_lengths.value_at(0);
    }

    uleb128_t lane_sum = 0;
    while (string_lengths.current_value_idx < end_value &&
           string_lengths.current_value_idx < string_lengths.num_encoded_values(true)) {
      // calculate values for current mini-block
      string_lengths.calc_mini_block_values(t);

      // get per lane sum for mini-block
      for (uint32_t i = 0; i < string_lengths.values_per_mb; i += warp_size) {
        uint32_t const idx = string_lengths.current_value_idx + i + t;
        if (idx >= start_value && idx < end_value && idx < string_lengths.value_count) {
          lane_sum += string_lengths.value[rolling_index<delta_rolling_buf_size>(idx)];
        }
      }

      if (t == 0) { string_lengths.setup_next_mini_block(true); }
      __syncwarp();
    }

    // get sum for warp.
    // note: warp_sum will only be valid on lane 0.
    auto const warp_sum = WarpReduce(temp_storage).Sum(lane_sum);

    if (t == 0) {
      total_bytes += warp_sum;
      pp->str_bytes = total_bytes;
    }
  }
}

/**
 * @brief Kernel for computing string page output size information.
 *
 * This call ignores non-string columns. On exit the `str_bytes` field of the `PageInfo` struct will
 * be populated.
 *
 * @param pages All pages to be decoded
 * @param chunks All chunks to be decoded
 * @param min_rows crop all rows below min_row
 * @param num_rows Maximum number of rows to read
 */
CUDF_KERNEL void __launch_bounds__(preprocess_block_size) gpuComputePageStringSizes(
  PageInfo* pages, device_span<ColumnChunkDesc const> chunks, size_t min_row, size_t num_rows)
{
  __shared__ __align__(16) page_state_s state_g;

  page_state_s* const s = &state_g;
  int const page_idx    = blockIdx.x;
  int const t           = threadIdx.x;
  PageInfo* const pp    = &pages[page_idx];

  // whether or not we have repetition levels (lists)
  bool const has_repetition = chunks[pp->chunk_idx].max_level[level_type::REPETITION] > 0;

  // setup page info
  if (!setupLocalPageInfo(s,
                          pp,
                          chunks,
                          min_row,
                          num_rows,
                          mask_filter{decode_kernel_mask::STRING},
                          page_processing_stage::STRING_BOUNDS)) {
    return;
  }

  bool const is_bounds_pg = is_bounds_page(s, min_row, num_rows, has_repetition);

  // if we have size info, then we only need to do this for bounds pages
  if (pp->has_page_index && !is_bounds_pg) {
    // check if we need to store values from the index
    if (is_page_contained(s, min_row, num_rows)) { pp->str_bytes = pp->str_bytes_from_index; }
    return;
  }

  auto const& col  = s->col;
  size_t str_bytes = 0;
  // short circuit for FIXED_LEN_BYTE_ARRAY
  if ((col.data_type & 7) == FIXED_LEN_BYTE_ARRAY) {
    str_bytes = pp->num_valids * s->dtype_len_in;
  } else {
    // now process string info in the range [start_value, end_value)
    // set up for decoding strings...can be either plain or dictionary
    uint8_t const* data      = s->data_start;
    uint8_t const* const end = s->data_end;
    uint8_t const* dict_base = nullptr;
    int dict_size            = 0;
    auto const start_value   = pp->start_val;
    auto const end_value     = pp->end_val;

    switch (pp->encoding) {
      case Encoding::PLAIN_DICTIONARY:
      case Encoding::RLE_DICTIONARY:
        // RLE-packed dictionary indices, first byte indicates index length in bits
        if (col.str_dict_index) {
          // String dictionary: use index
          dict_base = reinterpret_cast<const uint8_t*>(col.str_dict_index);
          dict_size = col.dict_page->num_input_values * sizeof(string_index_pair);
        } else {
          dict_base = col.dict_page->page_data;
          dict_size = col.dict_page->uncompressed_page_size;
        }

        // FIXME: need to return an error condition...this won't actually do anything
        if (s->dict_bits > 32 || (!dict_base && col.dict_page->num_input_values > 0)) {
          CUDF_UNREACHABLE("invalid dictionary bit size");
        }

        str_bytes = totalDictEntriesSize(
          data, dict_base, s->dict_bits, dict_size, (end - data), start_value, end_value);
        break;
      case Encoding::PLAIN:
        dict_size = static_cast<int32_t>(end - data);
        str_bytes = is_bounds_pg ? totalPlainEntriesSize(data, dict_size, start_value, end_value)
                                 : dict_size - sizeof(int) * pp->num_valids;
        break;
    }
  }

  if (t == 0) {
    // TODO check for overflow
    pp->str_bytes = str_bytes;

    // only need temp space for delta
    pp->temp_string_size = 0;
  }
}

/**
 * @brief Kernel for computing the string column data stored in the pages
 *
 * This function will write the page data and the page data's validity to the
 * output specified in the page's column chunk.
 *
 * This version uses a single warp to do the string copies.
 *
 * @param pages List of pages
 * @param chunks List of column chunks
 * @param min_row Row index to start reading at
 * @param num_rows Maximum number of rows to read
 * @tparam level_t Type used to store decoded repetition and definition levels
 */
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_block_size)
  gpuDecodeStringPageData(PageInfo* pages,
                          device_span<ColumnChunkDesc const> chunks,
                          size_t min_row,
                          size_t num_rows,
                          kernel_error::pointer error_code)
{
  using cudf::detail::warp_size;
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(4) size_type last_offset;
  __shared__ __align__(16)
    page_state_buffers_s<rolling_buf_size, rolling_buf_size, rolling_buf_size>
      state_buffers;

  page_state_s* const s = &state_g;
  auto* const sb        = &state_buffers;
  int const page_idx    = blockIdx.x;
  int const t           = threadIdx.x;
  int const lane_id     = t % warp_size;
  [[maybe_unused]] null_count_back_copier _{s, t};

  if (!setupLocalPageInfo(s,
                          &pages[page_idx],
                          chunks,
                          min_row,
                          num_rows,
                          mask_filter{decode_kernel_mask::STRING},
                          page_processing_stage::DECODE)) {
    return;
  }

  bool const has_repetition = s->col.max_level[level_type::REPETITION] > 0;

  // offsets are local to the page
  if (t == 0) { last_offset = 0; }
  __syncthreads();

  int const out_thread0                          = s->dict_base && s->dict_bits == 0 ? 32 : 64;
  int const leaf_level_index                     = s->col.max_nesting_depth - 1;
  PageNestingDecodeInfo* const nesting_info_base = s->nesting_info;

  __shared__ level_t rep[rolling_buf_size];  // circular buffer of repetition level values
  __shared__ level_t def[rolling_buf_size];  // circular buffer of definition level values

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t skipped_leaf_values = s->page.skipped_leaf_values;
  while (s->error == 0 &&
         (s->input_value_count < s->num_input_values || s->src_pos < s->nz_count)) {
    int target_pos;
    int src_pos = s->src_pos;

    if (t < out_thread0) {
      target_pos = min(src_pos + 2 * (decode_block_size - out_thread0),
                       s->nz_count + (decode_block_size - out_thread0));
    } else {
      target_pos = min(s->nz_count, src_pos + decode_block_size - out_thread0);
      if (out_thread0 > 32) { target_pos = min(target_pos, s->dict_pos); }
    }
    // this needs to be here to prevent warp 1/2 modifying src_pos before all threads have read it
    __syncthreads();
    if (t < 32) {
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<rolling_buf_size, level_t>(s, sb, target_pos, rep, def, t);
    } else if (t < out_thread0) {
      // skipped_leaf_values will always be 0 for flat hierarchies.
      uint32_t src_target_pos = target_pos + skipped_leaf_values;

      // WARP1: Decode dictionary indices, booleans or string positions
      if (s->dict_base) {
        src_target_pos = gpuDecodeDictionaryIndices<false>(s, sb, src_target_pos, lane_id).first;
      } else {
        gpuInitStringDescriptors<false>(s, sb, src_target_pos, lane_id);
      }
      if (t == 32) { s->dict_pos = src_target_pos; }
    } else {
      int const me = t - out_thread0;

      // WARP1..WARP3: Decode values
      src_pos += t - out_thread0;

      // the position in the output column/buffer
      int dst_pos = sb->nz_idx[rolling_index<rolling_buf_size>(src_pos)];

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
      if (!has_repetition) { dst_pos -= s->first_row; }

      if (me < warp_size) {
        for (int i = 0; i < decode_block_size - out_thread0; i += warp_size) {
          dst_pos = sb->nz_idx[rolling_index<rolling_buf_size>(src_pos + i)];
          if (!has_repetition) { dst_pos -= s->first_row; }

          auto [ptr, len] = src_pos + i < target_pos && dst_pos >= 0
                              ? gpuGetStringData(s, sb, src_pos + skipped_leaf_values + i)
                              : cuda::std::pair<char const*, size_t>{nullptr, 0};

          __shared__ cub::WarpScan<size_type>::TempStorage temp_storage;
          size_type offset, warp_total;
          cub::WarpScan<size_type>(temp_storage).ExclusiveSum(len, offset, warp_total);
          offset += last_offset;

          // choose a character parallel string copy when the average string is longer than a warp
          auto const use_char_ll = warp_total / warp_size >= warp_size;

          if (use_char_ll) {
            __shared__ __align__(8) uint8_t const* pointers[warp_size];
            __shared__ __align__(4) size_type offsets[warp_size];
            __shared__ __align__(4) int dsts[warp_size];
            __shared__ __align__(4) int lengths[warp_size];

            offsets[me]  = offset;
            pointers[me] = reinterpret_cast<uint8_t const*>(ptr);
            dsts[me]     = dst_pos;
            lengths[me]  = len;
            __syncwarp();

            for (int ss = 0; ss < warp_size && ss + i + s->src_pos < target_pos; ss++) {
              if (dsts[ss] >= 0) {
                auto offptr =
                  reinterpret_cast<int32_t*>(nesting_info_base[leaf_level_index].data_out) +
                  dsts[ss];
                *offptr      = lengths[ss];
                auto str_ptr = nesting_info_base[leaf_level_index].string_out + offsets[ss];
                ll_strcpy(str_ptr, pointers[ss], lengths[ss], me);
              }
            }

          } else {
            if (src_pos + i < target_pos && dst_pos >= 0) {
              auto offptr =
                reinterpret_cast<int32_t*>(nesting_info_base[leaf_level_index].data_out) + dst_pos;
              *offptr      = len;
              auto str_ptr = nesting_info_base[leaf_level_index].string_out + offset;
              memcpy(str_ptr, ptr, len);
            }
            __syncwarp();
          }

          // last thread in warp updates last_offset
          if (me == warp_size - 1) { last_offset = offset + len; }
          __syncwarp();
        }
      }

      if (t == out_thread0) { s->src_pos = target_pos; }
    }
    __syncthreads();
  }

  // now turn array of lengths into offsets
  int value_count = nesting_info_base[leaf_level_index].value_count;

  // if no repetition we haven't calculated start/end bounds and instead just skipped
  // values until we reach first_row. account for that here.
  if (!has_repetition) { value_count -= s->first_row; }

  auto const offptr = reinterpret_cast<size_type*>(nesting_info_base[leaf_level_index].data_out);
  block_excl_sum<decode_block_size>(offptr, value_count, s->page.str_offset);

  if (t == 0 and s->error != 0) { set_error(s->error, error_code); }
}

// Functor used to set the `temp_string_buf` pointer for each page. `data` points to a buffer
// to be used when skipping rows in the delta_byte_array decoder. Given a page and an offset,
// set the page's `temp_string_buf` to be `data + offset`.
struct page_tform_functor {
  uint8_t* const data;

  __device__ PageInfo operator()(PageInfo& page, int64_t offset)
  {
    if (page.temp_string_size != 0) { page.temp_string_buf = data + offset; }
    return page;
  }
};

}  // anonymous namespace

/**
 * @copydoc cudf::io::parquet::detail::ComputePageStringSizes
 */
void ComputePageStringSizes(cudf::detail::hostdevice_span<PageInfo> pages,
                            cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                            rmm::device_uvector<uint8_t>& temp_string_buf,
                            size_t min_row,
                            size_t num_rows,
                            int level_type_size,
                            uint32_t kernel_mask,
                            rmm::cuda_stream_view stream)
{
  dim3 const dim_block(preprocess_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);  // 1 threadblock per page
  if (level_type_size == 1) {
    gpuComputeStringPageBounds<uint8_t>
      <<<dim_grid, dim_block, 0, stream.value()>>>(pages.device_ptr(), chunks, min_row, num_rows);
  } else {
    gpuComputeStringPageBounds<uint16_t>
      <<<dim_grid, dim_block, 0, stream.value()>>>(pages.device_ptr(), chunks, min_row, num_rows);
  }

  // kernel mask may contain other kernels we don't need to count
  int const count_mask = kernel_mask & STRINGS_MASK;
  int const nkernels   = std::bitset<32>(count_mask).count();
  auto const streams   = cudf::detail::fork_streams(stream, nkernels);

  int s_idx = 0;
  if (BitAnd(kernel_mask, decode_kernel_mask::DELTA_BYTE_ARRAY) != 0) {
    dim3 dim_delta(delta_preproc_block_size, 1);
    gpuComputeDeltaPageStringSizes<<<dim_grid, dim_delta, 0, streams[s_idx++].value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows);
  }
  if (BitAnd(kernel_mask, decode_kernel_mask::DELTA_LENGTH_BA) != 0) {
    dim3 dim_delta(delta_length_block_size, 1);
    gpuComputeDeltaLengthPageStringSizes<<<dim_grid, dim_delta, 0, streams[s_idx++].value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows);
  }
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING) != 0) {
    gpuComputePageStringSizes<<<dim_grid, dim_block, 0, streams[s_idx++].value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows);
  }

  // synchronize the streams
  cudf::detail::join_streams(streams, stream);

  // check for needed temp space for DELTA_BYTE_ARRAY
  auto const need_sizes = thrust::any_of(
    rmm::exec_policy(stream), pages.device_begin(), pages.device_end(), [] __device__(auto& page) {
      return page.temp_string_size != 0;
    });

  if (need_sizes) {
    // sum up all of the temp_string_sizes
    auto const page_sizes = [] __device__(PageInfo const& page) { return page.temp_string_size; };
    auto const total_size = thrust::transform_reduce(rmm::exec_policy(stream),
                                                     pages.device_begin(),
                                                     pages.device_end(),
                                                     page_sizes,
                                                     0L,
                                                     thrust::plus<int64_t>{});

    // now do an exclusive scan over the temp_string_sizes to get offsets for each
    // page's chunk of the temp buffer
    rmm::device_uvector<int64_t> page_string_offsets(pages.size(), stream);
    thrust::transform_exclusive_scan(rmm::exec_policy_nosync(stream),
                                     pages.device_begin(),
                                     pages.device_end(),
                                     page_string_offsets.begin(),
                                     page_sizes,
                                     0L,
                                     thrust::plus<int64_t>{});

    // allocate the temp space
    temp_string_buf.resize(total_size, stream);

    // now use the offsets array to set each page's temp_string_buf pointers
    thrust::transform(rmm::exec_policy_nosync(stream),
                      pages.device_begin(),
                      pages.device_end(),
                      page_string_offsets.begin(),
                      pages.device_begin(),
                      page_tform_functor{temp_string_buf.data()});
  }
}

/**
 * @copydoc cudf::io::parquet::detail::DecodeStringPageData
 */
void __host__ DecodeStringPageData(cudf::detail::hostdevice_span<PageInfo> pages,
                                   cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                                   size_t num_rows,
                                   size_t min_row,
                                   int level_type_size,
                                   kernel_error::pointer error_code,
                                   rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 dim_block(decode_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    gpuDecodeStringPageData<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, error_code);
  } else {
    gpuDecodeStringPageData<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, error_code);
  }
}

}  // namespace cudf::io::parquet::detail
