/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include "io/utilities/block_utils.cuh"
#include "page_string_utils.cuh"
#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/transform_scan.h>

namespace cudf::io::parquet::detail {

namespace {

constexpr int decode_block_size = 128;

// DELTA_BYTE_ARRAY encoding (incremental encoding or front compression), is used for BYTE_ARRAY
// columns. For each element in a sequence of strings, a prefix length from the preceding string
// and a suffix is stored. The prefix lengths are DELTA_BINARY_PACKED encoded. The suffixes are
// encoded with DELTA_LENGTH_BYTE_ARRAY encoding, which is a DELTA_BINARY_PACKED list of suffix
// lengths, followed by the concatenated suffix data.
struct delta_byte_array_decoder {
  uint8_t const* last_string;       // pointer to last decoded string...needed for its prefix
  uint8_t const* suffix_char_data;  // pointer to the start of character data

  uint8_t* temp_buf;         // buffer used when skipping values
  uint32_t start_val;        // decoded strings up to this index will be dumped to temp_buf
  uint32_t last_string_len;  // length of the last decoded string

  delta_binary_decoder prefixes;  // state of decoder for prefix lengths
  delta_binary_decoder suffixes;  // state of decoder for suffix lengths

  // initialize the prefixes and suffixes blocks
  __device__ void init(uint8_t const* start, uint8_t const* end, uint32_t start_idx, uint8_t* temp)
  {
    auto const* suffix_start = prefixes.find_end_of_block(start, end);
    suffix_char_data         = suffixes.find_end_of_block(suffix_start, end);
    last_string              = nullptr;
    temp_buf                 = temp;
    start_val                = start_idx;
  }

  // kind of like an inclusive scan for strings. takes prefix_len bytes from preceding
  // string and prepends to the suffix we've already copied into place. called from
  // within loop over values_in_mb, so this only needs to handle a single warp worth of data
  // at a time.
  __device__ void string_scan(uint8_t* strings_out,
                              uint8_t const* last_string,
                              uint32_t start_idx,
                              uint32_t end_idx,
                              uint32_t offset,
                              uint32_t lane_id)
  {
    using cudf::detail::warp_size;

    // let p(n) === length(prefix(string_n))
    //
    // if p(n-1) > p(n), then string_n can be completed when string_n-2 is completed. likewise if
    // p(m) > p(n), then string_n can be completed with string_m-1. however, if p(m) < p(n), then m
    // is a "blocker" for string_n; string_n can be completed only after string_m is.
    //
    // we will calculate the nearest blocking position for each lane, and then fill in string_0. we
    // then iterate, finding all lanes that have had their "blocker" filled in and completing them.
    // when all lanes are filled in, we return. this will still hit the worst case if p(n-1) < p(n)
    // for all n
    __shared__ __align__(8) int64_t prefix_lens[warp_size];
    __shared__ __align__(8) uint8_t const* offsets[warp_size];

    uint32_t const ln_idx   = start_idx + lane_id;
    uint64_t prefix_len     = ln_idx < end_idx ? prefixes.value_at(ln_idx) : 0;
    uint8_t* const lane_out = ln_idx < end_idx ? strings_out + offset : nullptr;

    prefix_lens[lane_id] = prefix_len;
    offsets[lane_id]     = lane_out;

    // if all prefix_len's are zero, then there's nothing to do
    if (__all_sync(0xffff'ffff, prefix_len == 0)) { return; }

    // find a neighbor to the left that has a prefix length less than this lane. once that
    // neighbor is complete, this lane can be completed.
    int blocker = lane_id - 1;
    while (blocker > 0 && prefix_lens[blocker] != 0 && prefix_len <= prefix_lens[blocker]) {
      blocker--;
    }

    // fill in lane 0 (if necessary)
    if (lane_id == 0 && prefix_len > 0) {
      memcpy(lane_out, last_string, prefix_len);
      prefix_lens[0] = prefix_len = 0;
    }
    __syncwarp();

    // now fill in blockers until done
    for (uint32_t i = 1; i < warp_size && i + start_idx < end_idx; i++) {
      if (prefix_len != 0 && prefix_lens[blocker] == 0 && lane_out != nullptr) {
        memcpy(lane_out, offsets[blocker], prefix_len);
        prefix_lens[lane_id] = prefix_len = 0;
      }

      // check for finished
      if (__all_sync(0xffff'ffff, prefix_len == 0)) { return; }
    }
  }

  // calculate a mini-batch of string values, writing the results to
  // `strings_out`. starting at global index `start_idx` and decoding
  // up to `num_values` strings.
  // called by all threads in a warp. used for strings <= 32 chars.
  // returns number of bytes written
  __device__ size_t calculate_string_values(uint8_t* strings_out,
                                            uint32_t start_idx,
                                            uint32_t num_values,
                                            uint32_t lane_id)
  {
    using cudf::detail::warp_size;
    using WarpScan = cub::WarpScan<uint64_t>;
    __shared__ WarpScan::TempStorage scan_temp;

    if (start_idx >= suffixes.value_count) { return 0; }
    auto end_idx = start_idx + min(suffixes.values_per_mb, num_values);
    end_idx      = min(end_idx, static_cast<uint32_t>(suffixes.value_count));

    auto p_strings_out = strings_out;
    auto p_temp_out    = temp_buf;

    auto copy_batch = [&](uint8_t* out, uint32_t idx, uint32_t end) {
      uint32_t const ln_idx = idx + lane_id;

      // calculate offsets into suffix data
      uint64_t const suffix_len = ln_idx < end ? suffixes.value_at(ln_idx) : 0;
      uint64_t suffix_off       = 0;
      WarpScan(scan_temp).ExclusiveSum(suffix_len, suffix_off);

      // calculate offsets into string data
      uint64_t const prefix_len = ln_idx < end ? prefixes.value_at(ln_idx) : 0;
      uint64_t const string_len = prefix_len + suffix_len;

      // get offset into output for each lane
      uint64_t string_off, warp_total;
      WarpScan(scan_temp).ExclusiveSum(string_len, string_off, warp_total);
      auto const so_ptr = out + string_off;

      // copy suffixes into string data
      if (ln_idx < end) { memcpy(so_ptr + prefix_len, suffix_char_data + suffix_off, suffix_len); }
      __syncwarp();

      // copy prefixes into string data.
      string_scan(out, last_string, idx, end, string_off, lane_id);

      // save the position of the last computed string. this will be used in
      // the next iteration to reconstruct the string in lane 0.
      if (ln_idx == end - 1 || (ln_idx < end && lane_id == 31)) {
        // set last_string to this lane's string
        last_string     = out + string_off;
        last_string_len = string_len;
        // and consume used suffix_char_data
        suffix_char_data += suffix_off + suffix_len;
      }

      return warp_total;
    };

    uint64_t string_total = 0;
    for (int idx = start_idx; idx < end_idx; idx += warp_size) {
      auto const n_in_batch = min(warp_size, end_idx - idx);
      // account for the case where start_val occurs in the middle of this batch
      if (idx < start_val && idx + n_in_batch > start_val) {
        // dump idx...start_val into temp_buf
        copy_batch(p_temp_out, idx, start_val);
        __syncwarp();

        // start_val...idx + n_in_batch into strings_out
        auto nbytes = copy_batch(p_strings_out, start_val, idx + n_in_batch);
        p_strings_out += nbytes;
        string_total = nbytes;
      } else {
        if (idx < start_val) {
          p_temp_out += copy_batch(p_temp_out, idx, end_idx);
        } else {
          auto nbytes = copy_batch(p_strings_out, idx, end_idx);
          p_strings_out += nbytes;
          string_total += nbytes;
        }
      }
      __syncwarp();
    }

    return string_total;
  }

  // character parallel version of CalculateStringValues(). This is faster for strings longer than
  // 32 chars.
  __device__ size_t calculate_string_values_cp(uint8_t* strings_out,
                                               uint32_t start_idx,
                                               uint32_t num_values,
                                               uint32_t lane_id)
  {
    using cudf::detail::warp_size;
    __shared__ __align__(8) uint8_t* so_ptr;

    if (start_idx >= suffixes.value_count) { return 0; }
    auto end_idx = start_idx + min(suffixes.values_per_mb, num_values);
    end_idx      = min(end_idx, static_cast<uint32_t>(suffixes.value_count));

    if (lane_id == 0) { so_ptr = start_idx < start_val ? temp_buf : strings_out; }
    __syncwarp();

    uint64_t string_total = 0;
    for (int idx = start_idx; idx < end_idx; idx++) {
      uint64_t const suffix_len = suffixes.value_at(idx);
      uint64_t const prefix_len = prefixes.value_at(idx);
      uint64_t const string_len = prefix_len + suffix_len;

      // copy prefix and suffix data into current strings_out position
      // for longer strings use a 4-byte version stolen from gather_chars_fn_string_parallel.
      if (string_len > 64) {
        if (prefix_len > 0) { wideStrcpy(so_ptr, last_string, prefix_len, lane_id); }
        if (suffix_len > 0) {
          wideStrcpy(so_ptr + prefix_len, suffix_char_data, suffix_len, lane_id);
        }
      } else {
        for (int i = lane_id; i < string_len; i += warp_size) {
          so_ptr[i] = i < prefix_len ? last_string[i] : suffix_char_data[i - prefix_len];
        }
      }
      __syncwarp();

      if (idx >= start_val) { string_total += string_len; }

      if (lane_id == 0) {
        last_string     = so_ptr;
        last_string_len = string_len;
        suffix_char_data += suffix_len;
        if (idx == start_val - 1) {
          so_ptr = strings_out;
        } else {
          so_ptr += string_len;
        }
      }
      __syncwarp();
    }

    return string_total;
  }

  // dump strings before start_val to temp buf
  __device__ void skip(bool use_char_ll)
  {
    using cudf::detail::warp_size;
    int const t       = threadIdx.x;
    int const lane_id = t % warp_size;

    // is this even necessary? return if asking to skip the whole block.
    if (start_val >= prefixes.num_encoded_values(true)) { return; }

    // prefixes and suffixes will have the same parameters (it's checked earlier)
    auto const batch_size = prefixes.values_per_mb;

    uint32_t skip_pos = 0;
    while (prefixes.current_value_idx < start_val) {
      // warp 0 gets prefixes and warp 1 gets suffixes
      auto* const db = t < 32 ? &prefixes : &suffixes;

      // this will potentially decode past start_val, but that's ok
      if (t < 64) { db->decode_batch(); }
      __syncthreads();

      // warp 0 decodes the batch.
      if (t < 32) {
        auto const num_to_decode = min(batch_size, start_val - skip_pos);
        auto const bytes_written =
          use_char_ll ? calculate_string_values_cp(temp_buf, skip_pos, num_to_decode, lane_id)
                      : calculate_string_values(temp_buf, skip_pos, num_to_decode, lane_id);
        // store last_string someplace safe in temp buffer
        if (t == 0) {
          memcpy(temp_buf + bytes_written, last_string, last_string_len);
          last_string = temp_buf + bytes_written;
        }
      }
      skip_pos += prefixes.values_per_mb;
      __syncthreads();
    }
  }
};

// Decode page data that is DELTA_BINARY_PACKED encoded. This encoding is
// only used for int32 and int64 physical types (and appears to only be used
// with V2 page headers; see https://www.mail-archive.com/dev@parquet.apache.org/msg11826.html).
// this kernel only needs 96 threads (3 warps)(for now).
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(96)
  gpuDecodeDeltaBinary(PageInfo* pages,
                       device_span<ColumnChunkDesc const> chunks,
                       size_t min_row,
                       size_t num_rows,
                       kernel_error::pointer error_code)
{
  using cudf::detail::warp_size;
  __shared__ __align__(16) delta_binary_decoder db_state;
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16) page_state_buffers_s<delta_rolling_buf_size, 0, 0> state_buffers;

  page_state_s* const s = &state_g;
  auto* const sb        = &state_buffers;
  int const page_idx    = blockIdx.x;
  int const t           = threadIdx.x;
  int const lane_id     = t % warp_size;
  auto* const db        = &db_state;
  [[maybe_unused]] null_count_back_copier _{s, t};

  if (!setupLocalPageInfo(s,
                          &pages[page_idx],
                          chunks,
                          min_row,
                          num_rows,
                          mask_filter{decode_kernel_mask::DELTA_BINARY},
                          page_processing_stage::DECODE)) {
    return;
  }

  bool const has_repetition = s->col.max_level[level_type::REPETITION] > 0;

  // copying logic from gpuDecodePageData.
  PageNestingDecodeInfo const* nesting_info_base = s->nesting_info;

  __shared__ level_t rep[delta_rolling_buf_size];  // circular buffer of repetition level values
  __shared__ level_t def[delta_rolling_buf_size];  // circular buffer of definition level values

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t const skipped_leaf_values = s->page.skipped_leaf_values;

  // initialize delta state
  if (t == 0) { db->init_binary_block(s->data_start, s->data_end); }
  __syncthreads();

  auto const batch_size = db->values_per_mb;
  if (batch_size > max_delta_mini_block_size) {
    set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
              error_code);
    return;
  }

  // if skipped_leaf_values is non-zero, then we need to decode up to the first mini-block
  // that has a value we need.
  if (skipped_leaf_values > 0) { db->skip_values(skipped_leaf_values); }

  while (s->error == 0 &&
         (s->input_value_count < s->num_input_values || s->src_pos < s->nz_count)) {
    uint32_t target_pos;
    uint32_t const src_pos = s->src_pos;

    if (t < 2 * warp_size) {  // warp0..1
      target_pos = min(src_pos + 2 * batch_size, s->nz_count + batch_size);
    } else {  // warp2
      target_pos = min(s->nz_count, src_pos + batch_size);
    }
    // this needs to be here to prevent warp 2 modifying src_pos before all threads have read it
    __syncthreads();

    // warp0 will decode the rep/def levels, warp1 will unpack a mini-batch of deltas.
    // warp2 waits one cycle for warps 0/1 to produce a batch, and then stuffs values
    // into the proper location in the output.
    if (t < warp_size) {
      // warp 0
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<delta_rolling_buf_size, level_t>(s, sb, target_pos, rep, def, t);
    } else if (t < 2 * warp_size) {
      // warp 1
      db->decode_batch();

    } else if (src_pos < target_pos) {
      // warp 2
      // nesting level that is storing actual leaf values
      int const leaf_level_index = s->col.max_nesting_depth - 1;

      // process the mini-block in batches of 32
      for (uint32_t sp = src_pos + lane_id; sp < src_pos + batch_size; sp += 32) {
        // the position in the output column/buffer
        int32_t dst_pos = sb->nz_idx[rolling_index<delta_rolling_buf_size>(sp)];

        // handle skip_rows here. flat hierarchies can just skip up to first_row.
        if (!has_repetition) { dst_pos -= s->first_row; }

        // place value for this thread
        if (dst_pos >= 0 && sp < target_pos) {
          void* const dst = nesting_info_base[leaf_level_index].data_out + dst_pos * s->dtype_len;
          auto const val  = db->value_at(sp + skipped_leaf_values);
          switch (s->dtype_len) {
            case 1: *static_cast<int8_t*>(dst) = val; break;
            case 2: *static_cast<int16_t*>(dst) = val; break;
            case 4: *static_cast<int32_t*>(dst) = val; break;
            case 8: *static_cast<int64_t*>(dst) = val; break;
          }
        }
      }

      if (lane_id == 0) { s->src_pos = src_pos + batch_size; }
    }
    __syncthreads();
  }

  if (t == 0 and s->error != 0) { set_error(s->error, error_code); }
}

// Decode page data that is DELTA_BYTE_ARRAY packed. This encoding consists of a DELTA_BINARY_PACKED
// array of prefix lengths, followed by a DELTA_BINARY_PACKED array of suffix lengths, followed by
// the suffixes (technically the suffixes are DELTA_LENGTH_BYTE_ARRAY encoded). The latter two can
// be used to create an offsets array for the suffix data, but then this needs to be combined with
// the prefix lengths to do the final decode for each value. Because the lengths of the prefixes and
// suffixes are not encoded in the header, we're going to have to first do a quick pass through them
// to find the start/end of each structure.
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_block_size)
  gpuDecodeDeltaByteArray(PageInfo* pages,
                          device_span<ColumnChunkDesc const> chunks,
                          size_t min_row,
                          size_t num_rows,
                          kernel_error::pointer error_code)
{
  using cudf::detail::warp_size;
  __shared__ __align__(16) delta_byte_array_decoder db_state;
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16) page_state_buffers_s<delta_rolling_buf_size, 0, 0> state_buffers;

  page_state_s* const s = &state_g;
  auto* const sb        = &state_buffers;
  int const page_idx    = blockIdx.x;
  int const t           = threadIdx.x;
  int const lane_id     = t % warp_size;
  auto* const prefix_db = &db_state.prefixes;
  auto* const suffix_db = &db_state.suffixes;
  auto* const dba       = &db_state;
  [[maybe_unused]] null_count_back_copier _{s, t};

  if (!setupLocalPageInfo(s,
                          &pages[page_idx],
                          chunks,
                          min_row,
                          num_rows,
                          mask_filter{decode_kernel_mask::DELTA_BYTE_ARRAY},
                          page_processing_stage::DECODE)) {
    return;
  }

  bool const has_repetition = s->col.max_level[level_type::REPETITION] > 0;

  // choose a character parallel string copy when the average string is longer than a warp
  auto const use_char_ll = (s->page.str_bytes / s->page.num_valids) > warp_size;

  // copying logic from gpuDecodePageData.
  PageNestingDecodeInfo const* nesting_info_base = s->nesting_info;

  __shared__ level_t rep[delta_rolling_buf_size];  // circular buffer of repetition level values
  __shared__ level_t def[delta_rolling_buf_size];  // circular buffer of definition level values

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t const skipped_leaf_values = s->page.skipped_leaf_values;

  if (t == 0) {
    // initialize the prefixes and suffixes blocks
    dba->init(s->data_start, s->data_end, s->page.start_val, s->page.temp_string_buf);
  }
  __syncthreads();

  // assert that prefix and suffix have same mini-block size
  if (prefix_db->values_per_mb != suffix_db->values_per_mb or
      prefix_db->block_size != suffix_db->block_size or
      prefix_db->value_count != suffix_db->value_count) {
    set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAM_MISMATCH),
              error_code);
    return;
  }

  // pointer to location to output final strings
  int const leaf_level_index = s->col.max_nesting_depth - 1;
  auto strings_data          = nesting_info_base[leaf_level_index].string_out;

  // sanity check to make sure we can process this page
  auto const batch_size = prefix_db->values_per_mb;
  if (batch_size > max_delta_mini_block_size) {
    set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
              error_code);
    return;
  }

  // if this is a bounds page and nested, then we need to skip up front. non-nested will work
  // its way through the page.
  int string_pos          = has_repetition ? s->page.start_val : 0;
  auto const is_bounds_pg = is_bounds_page(s, min_row, num_rows, has_repetition);
  if (is_bounds_pg && string_pos > 0) { dba->skip(use_char_ll); }

  while (!s->error && (s->input_value_count < s->num_input_values || s->src_pos < s->nz_count)) {
    uint32_t target_pos;
    uint32_t const src_pos = s->src_pos;

    if (t < 3 * warp_size) {  // warp 0..2
      target_pos = min(src_pos + 2 * batch_size, s->nz_count + s->first_row + batch_size);
    } else {  // warp 3
      target_pos = min(s->nz_count, src_pos + batch_size);
    }
    // this needs to be here to prevent warp 3 modifying src_pos before all threads have read it
    __syncthreads();

    // warp0 will decode the rep/def levels, warp1 will unpack a mini-batch of prefixes, warp 2 will
    // unpack a mini-batch of suffixes. warp3 waits one cycle for warps 0-2 to produce a batch, and
    // then stuffs values into the proper location in the output.
    if (t < warp_size) {
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<delta_rolling_buf_size, level_t>(s, sb, target_pos, rep, def, t);

    } else if (t < 2 * warp_size) {
      // warp 1
      prefix_db->decode_batch();

    } else if (t < 3 * warp_size) {
      // warp 2
      suffix_db->decode_batch();

    } else if (src_pos < target_pos) {
      // warp 3

      int const nproc = min(batch_size, s->page.end_val - string_pos);
      strings_data += use_char_ll
                        ? dba->calculate_string_values_cp(strings_data, string_pos, nproc, lane_id)
                        : dba->calculate_string_values(strings_data, string_pos, nproc, lane_id);
      string_pos += nproc;

      // process the mini-block in batches of 32
      for (uint32_t sp = src_pos + lane_id; sp < src_pos + batch_size; sp += 32) {
        // the position in the output column/buffer
        int dst_pos = sb->nz_idx[rolling_index<delta_rolling_buf_size>(sp)];

        // handle skip_rows here. flat hierarchies can just skip up to first_row.
        if (!has_repetition) { dst_pos -= s->first_row; }

        if (dst_pos >= 0 && sp < target_pos) {
          auto const offptr =
            reinterpret_cast<size_type*>(nesting_info_base[leaf_level_index].data_out) + dst_pos;
          auto const src_idx = sp + skipped_leaf_values;
          *offptr            = prefix_db->value_at(src_idx) + suffix_db->value_at(src_idx);
        }
        __syncwarp();
      }

      if (lane_id == 0) { s->src_pos = src_pos + batch_size; }
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

// Decode page data that is DELTA_LENGTH_BYTE_ARRAY packed. This encoding consists of a
// DELTA_BINARY_PACKED array of string lengths, followed by the string data.
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_block_size)
  gpuDecodeDeltaLengthByteArray(PageInfo* pages,
                                device_span<ColumnChunkDesc const> chunks,
                                size_t min_row,
                                size_t num_rows,
                                kernel_error::pointer error_code)
{
  using cudf::detail::warp_size;
  __shared__ __align__(16) delta_binary_decoder db_state;
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16) page_state_buffers_s<delta_rolling_buf_size, 0, 0> state_buffers;
  __shared__ __align__(8) uint8_t const* page_string_data;
  __shared__ size_t string_offset;

  page_state_s* const s = &state_g;
  auto* const sb        = &state_buffers;
  int const page_idx    = blockIdx.x;
  int const t           = threadIdx.x;
  int const lane_id     = t % warp_size;
  auto* const db        = &db_state;
  [[maybe_unused]] null_count_back_copier _{s, t};

  auto const mask = decode_kernel_mask::DELTA_LENGTH_BA;
  if (!setupLocalPageInfo(s,
                          &pages[page_idx],
                          chunks,
                          min_row,
                          num_rows,
                          mask_filter{mask},
                          page_processing_stage::DECODE)) {
    return;
  }

  bool const has_repetition = s->col.max_level[level_type::REPETITION] > 0;

  // copying logic from gpuDecodePageData.
  PageNestingDecodeInfo const* nesting_info_base = s->nesting_info;

  __shared__ level_t rep[delta_rolling_buf_size];  // circular buffer of repetition level values
  __shared__ level_t def[delta_rolling_buf_size];  // circular buffer of definition level values

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t const skipped_leaf_values = s->page.skipped_leaf_values;

  // initialize delta state
  if (t == 0) {
    string_offset    = 0;
    page_string_data = db->find_end_of_block(s->data_start, s->data_end);
  }
  __syncthreads();

  int const leaf_level_index = s->col.max_nesting_depth - 1;

  // sanity check to make sure we can process this page
  auto const batch_size = db->values_per_mb;
  if (batch_size > max_delta_mini_block_size) {
    set_error(static_cast<int32_t>(decode_error::DELTA_PARAMS_UNSUPPORTED), error_code);
    return;
  }

  // if this is a bounds page, then we need to decode up to the first mini-block
  // that has a value we need, and set string_offset to the position of the first value in the
  // string data block.
  auto const is_bounds_pg = is_bounds_page(s, min_row, num_rows, has_repetition);
  if (is_bounds_pg && s->page.start_val > 0) {
    if (t < warp_size) {
      // string_off is only valid on thread 0
      auto const string_off = db->skip_values_and_sum(s->page.start_val);
      if (t == 0) {
        string_offset = string_off;

        // if there is no repetition, then we need to work through the whole page, so reset the
        // delta decoder to the beginning of the page
        if (not has_repetition) { db->init_binary_block(s->data_start, s->data_end); }
      }
    }
    __syncthreads();
  }

  int string_pos = has_repetition ? s->page.start_val : 0;

  while (!s->error && (s->input_value_count < s->num_input_values || s->src_pos < s->nz_count)) {
    uint32_t target_pos;
    uint32_t const src_pos = s->src_pos;

    if (t < 2 * warp_size) {  // warp0..1
      target_pos = min(src_pos + 2 * batch_size, s->nz_count + batch_size);
    } else {  // warp2
      target_pos = min(s->nz_count, src_pos + batch_size);
    }
    // this needs to be here to prevent warp 2 modifying src_pos before all threads have read it
    __syncthreads();

    // warp0 will decode the rep/def levels, warp1 will unpack a mini-batch of deltas.
    // warp2 waits one cycle for warps 0/1 to produce a batch, and then stuffs string sizes
    // into the proper location in the output. warp 3 does nothing until it's time to copy
    // string data.
    if (t < warp_size) {
      // warp 0
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<delta_rolling_buf_size, level_t>(s, sb, target_pos, rep, def, t);
    } else if (t < 2 * warp_size) {
      // warp 1
      db->decode_batch();

    } else if (t < 3 * warp_size && src_pos < target_pos) {
      // warp 2
      int const nproc = min(batch_size, s->page.end_val - string_pos);
      string_pos += nproc;

      // process the mini-block in batches of 32
      for (uint32_t sp = src_pos + lane_id; sp < src_pos + batch_size; sp += 32) {
        // the position in the output column/buffer
        int dst_pos = sb->nz_idx[rolling_index<delta_rolling_buf_size>(sp)];

        // handle skip_rows here. flat hierarchies can just skip up to first_row.
        if (!has_repetition) { dst_pos -= s->first_row; }

        // fill in offsets array
        if (dst_pos >= 0 && sp < target_pos) {
          auto const offptr =
            reinterpret_cast<size_type*>(nesting_info_base[leaf_level_index].data_out) + dst_pos;
          *offptr = db->value_at(sp + skipped_leaf_values);
        }
        __syncwarp();
      }

      if (lane_id == 0) { s->src_pos = src_pos + batch_size; }
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

  // finally, copy the string data into place
  auto const dst = nesting_info_base[leaf_level_index].string_out;
  auto const src = page_string_data + string_offset;
  memcpy_block<decode_block_size, true>(dst, src, s->page.str_bytes, t);

  if (t == 0 and s->error != 0) { set_error(s->error, error_code); }
}

}  // anonymous namespace

/**
 * @copydoc cudf::io::parquet::detail::DecodeDeltaBinary
 */
void DecodeDeltaBinary(cudf::detail::hostdevice_vector<PageInfo>& pages,
                       cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
                       size_t num_rows,
                       size_t min_row,
                       int level_type_size,
                       kernel_error::pointer error_code,
                       rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 dim_block(96, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    gpuDecodeDeltaBinary<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, error_code);
  } else {
    gpuDecodeDeltaBinary<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, error_code);
  }
}

/**
 * @copydoc cudf::io::parquet::gpu::DecodeDeltaByteArray
 */
void DecodeDeltaByteArray(cudf::detail::hostdevice_vector<PageInfo>& pages,
                          cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
                          size_t num_rows,
                          size_t min_row,
                          int level_type_size,
                          kernel_error::pointer error_code,
                          rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 const dim_block(decode_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    gpuDecodeDeltaByteArray<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, error_code);
  } else {
    gpuDecodeDeltaByteArray<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, error_code);
  }
}

/**
 * @copydoc cudf::io::parquet::gpu::DecodeDeltaByteArray
 */
void DecodeDeltaLengthByteArray(cudf::detail::hostdevice_vector<PageInfo>& pages,
                                cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
                                size_t num_rows,
                                size_t min_row,
                                int level_type_size,
                                kernel_error::pointer error_code,
                                rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 const dim_block(decode_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    gpuDecodeDeltaLengthByteArray<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, error_code);
  } else {
    gpuDecodeDeltaLengthByteArray<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, error_code);
  }
}

}  // namespace cudf::io::parquet::detail
