/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include "page_data.cuh"
#include "page_decode.cuh"
#include "parquet_gpu.hpp"
#include "rle_stream.cuh"

#include <cudf/detail/utilities/cuda.cuh>

namespace cudf::io::parquet::detail {

namespace {

// Unlike cub's algorithm, this provides warp-wide and block-wide results simultaneously.
// Also, this provides the ability to compute warp_bits & lane_mask manually, which we need for
// lists.
struct block_scan_results {
  uint32_t warp_bits;
  int thread_count_within_warp;
  int warp_count;

  int thread_count_within_block;
  int block_count;
};

template <int decode_block_size>
using block_scan_temp_storage = int[decode_block_size / cudf::detail::warp_size];

// Similar to CUB, must __syncthreads() after calling if reusing temp_storage
template <int decode_block_size>
__device__ inline static void scan_block_exclusive_sum(
  int thread_bit,
  block_scan_results& results,
  block_scan_temp_storage<decode_block_size>& temp_storage)
{
  int const t              = threadIdx.x;
  int const warp_index     = t / cudf::detail::warp_size;
  int const warp_lane      = t % cudf::detail::warp_size;
  uint32_t const lane_mask = (uint32_t(1) << warp_lane) - 1;

  uint32_t warp_bits = ballot(thread_bit);
  scan_block_exclusive_sum<decode_block_size>(
    warp_bits, warp_lane, warp_index, lane_mask, results, temp_storage);
}

// Similar to CUB, must __syncthreads() after calling if reusing temp_storage
template <int decode_block_size>
__device__ static void scan_block_exclusive_sum(
  uint32_t warp_bits,
  int warp_lane,
  int warp_index,
  uint32_t lane_mask,
  block_scan_results& results,
  block_scan_temp_storage<decode_block_size>& temp_storage)
{
  // Compute # warps
  constexpr int num_warps = decode_block_size / cudf::detail::warp_size;

  // Compute the warp-wide results
  results.warp_bits                = warp_bits;
  results.warp_count               = __popc(results.warp_bits);
  results.thread_count_within_warp = __popc(results.warp_bits & lane_mask);

  // Share the warp counts amongst the block threads
  if (warp_lane == 0) { temp_storage[warp_index] = results.warp_count; }
  __syncthreads();  // Sync to share counts between threads/warps

  // Compute block-wide results
  results.block_count               = 0;
  results.thread_count_within_block = results.thread_count_within_warp;
  for (int warp_idx = 0; warp_idx < num_warps; ++warp_idx) {
    results.block_count += temp_storage[warp_idx];
    if (warp_idx < warp_index) { results.thread_count_within_block += temp_storage[warp_idx]; }
  }
}

template <int block_size, bool has_lists_t, typename state_buf>
__device__ void gpuDecodeFixedWidthValues(
  page_state_s* s, state_buf* const sb, int start, int end, int t)
{
  constexpr int num_warps      = block_size / cudf::detail::warp_size;
  constexpr int max_batch_size = num_warps * cudf::detail::warp_size;

  // nesting level that is storing actual leaf values
  int const leaf_level_index = s->col.max_nesting_depth - 1;
  auto const data_out        = s->nesting_info[leaf_level_index].data_out;

  int const dtype          = s->col.physical_type;
  uint32_t const dtype_len = s->dtype_len;

  int const skipped_leaf_values = s->page.skipped_leaf_values;

  // decode values
  int pos = start;
  while (pos < end) {
    int const batch_size = min(max_batch_size, end - pos);
    int const target_pos = pos + batch_size;
    int const thread_pos = pos + t;

    // Index from value buffer (doesn't include nulls) to final array (has gaps for nulls)
    int const dst_pos = [&]() {
      int dst_pos = sb->nz_idx[rolling_index<state_buf::nz_buf_size>(thread_pos)];
      if constexpr (!has_lists_t) { dst_pos -= s->first_row; }
      return dst_pos;
    }();

    // target_pos will always be properly bounded by num_rows, but dst_pos may be negative (values
    // before first_row) in the flat hierarchy case.
    if (thread_pos < target_pos && dst_pos >= 0) {
      // nesting level that is storing actual leaf values

      // src_pos represents the logical row position we want to read from. But in the case of
      // nested hierarchies (lists), there is no 1:1 mapping of rows to values. So src_pos
      // has to take into account the # of values we have to skip in the page to get to the
      // desired logical row.  For flat hierarchies, skipped_leaf_values will always be 0.
      int const src_pos = [&]() {
        if constexpr (has_lists_t) { return thread_pos + skipped_leaf_values; }
        return thread_pos;
      }();

      void* const dst = data_out + (static_cast<size_t>(dst_pos) * dtype_len);

      if (s->col.logical_type.has_value() && s->col.logical_type->type == LogicalType::DECIMAL) {
        switch (dtype) {
          case INT32: gpuOutputFast(s, sb, src_pos, static_cast<uint32_t*>(dst)); break;
          case INT64: gpuOutputFast(s, sb, src_pos, static_cast<uint2*>(dst)); break;
          default:
            if (s->dtype_len_in <= sizeof(int32_t)) {
              gpuOutputFixedLenByteArrayAsInt(s, sb, src_pos, static_cast<int32_t*>(dst));
            } else if (s->dtype_len_in <= sizeof(int64_t)) {
              gpuOutputFixedLenByteArrayAsInt(s, sb, src_pos, static_cast<int64_t*>(dst));
            } else {
              gpuOutputFixedLenByteArrayAsInt(s, sb, src_pos, static_cast<__int128_t*>(dst));
            }
            break;
        }
      } else if (dtype == BOOLEAN) {
        gpuOutputBoolean(sb, src_pos, static_cast<uint8_t*>(dst));
      } else if (dtype == INT96) {
        gpuOutputInt96Timestamp(s, sb, src_pos, static_cast<int64_t*>(dst));
      } else if (dtype_len == 8) {
        if (s->dtype_len_in == 4) {
          // Reading INT32 TIME_MILLIS into 64-bit DURATION_MILLISECONDS
          // TIME_MILLIS is the only duration type stored as int32:
          // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#deprecated-time-convertedtype
          gpuOutputFast(s, sb, src_pos, static_cast<uint32_t*>(dst));
        } else if (s->ts_scale) {
          gpuOutputInt64Timestamp(s, sb, src_pos, static_cast<int64_t*>(dst));
        } else {
          gpuOutputFast(s, sb, src_pos, static_cast<uint2*>(dst));
        }
      } else if (dtype_len == 4) {
        gpuOutputFast(s, sb, src_pos, static_cast<uint32_t*>(dst));
      } else {
        gpuOutputGeneric(s, sb, src_pos, static_cast<uint8_t*>(dst), dtype_len);
      }
    }

    pos += batch_size;
  }
}

template <int block_size, bool has_lists_t, typename state_buf>
struct decode_fixed_width_values_func {
  __device__ inline void operator()(page_state_s* s, state_buf* const sb, int start, int end, int t)
  {
    gpuDecodeFixedWidthValues<block_size, has_lists_t, state_buf>(s, sb, start, end, t);
  }
};

template <int block_size, bool has_lists_t, typename state_buf>
__device__ inline void gpuDecodeFixedWidthSplitValues(
  page_state_s* s, state_buf* const sb, int start, int end, int t)
{
  using cudf::detail::warp_size;
  constexpr int num_warps      = block_size / warp_size;
  constexpr int max_batch_size = num_warps * warp_size;

  // nesting level that is storing actual leaf values
  int const leaf_level_index = s->col.max_nesting_depth - 1;
  auto const data_out        = s->nesting_info[leaf_level_index].data_out;

  int const dtype       = s->col.physical_type;
  auto const data_len   = thrust::distance(s->data_start, s->data_end);
  auto const num_values = data_len / s->dtype_len_in;

  int const skipped_leaf_values = s->page.skipped_leaf_values;

  // decode values
  int pos = start;
  while (pos < end) {
    int const batch_size = min(max_batch_size, end - pos);

    int const target_pos = pos + batch_size;
    int const thread_pos = pos + t;

    // the position in the output column/buffer
    // Index from value buffer (doesn't include nulls) to final array (has gaps for nulls)
    int const dst_pos = [&]() {
      int dst_pos = sb->nz_idx[rolling_index<state_buf::nz_buf_size>(thread_pos)];
      if constexpr (!has_lists_t) { dst_pos -= s->first_row; }
      return dst_pos;
    }();

    // target_pos will always be properly bounded by num_rows, but dst_pos may be negative (values
    // before first_row) in the flat hierarchy case.
    if (thread_pos < target_pos && dst_pos >= 0) {
      // src_pos represents the logical row position we want to read from. But in the case of
      // nested hierarchies (lists), there is no 1:1 mapping of rows to values. So src_pos
      // has to take into account the # of values we have to skip in the page to get to the
      // desired logical row.  For flat hierarchies, skipped_leaf_values will always be 0.
      int const src_pos = [&]() {
        if constexpr (has_lists_t) {
          return thread_pos + skipped_leaf_values;
        } else {
          return thread_pos;
        }
      }();

      uint32_t const dtype_len = s->dtype_len;
      uint8_t const* const src = s->data_start + src_pos;
      uint8_t* const dst       = data_out + static_cast<size_t>(dst_pos) * dtype_len;
      auto const is_decimal =
        s->col.logical_type.has_value() and s->col.logical_type->type == LogicalType::DECIMAL;

      // Note: non-decimal FIXED_LEN_BYTE_ARRAY will be handled in the string reader
      if (is_decimal) {
        switch (dtype) {
          case INT32: gpuOutputByteStreamSplit<int32_t>(dst, src, num_values); break;
          case INT64: gpuOutputByteStreamSplit<int64_t>(dst, src, num_values); break;
          case FIXED_LEN_BYTE_ARRAY:
            if (s->dtype_len_in <= sizeof(int32_t)) {
              gpuOutputSplitFixedLenByteArrayAsInt(
                reinterpret_cast<int32_t*>(dst), src, num_values, s->dtype_len_in);
              break;
            } else if (s->dtype_len_in <= sizeof(int64_t)) {
              gpuOutputSplitFixedLenByteArrayAsInt(
                reinterpret_cast<int64_t*>(dst), src, num_values, s->dtype_len_in);
              break;
            } else if (s->dtype_len_in <= sizeof(__int128_t)) {
              gpuOutputSplitFixedLenByteArrayAsInt(
                reinterpret_cast<__int128_t*>(dst), src, num_values, s->dtype_len_in);
              break;
            }
            // unsupported decimal precision
            [[fallthrough]];

          default: s->set_error_code(decode_error::UNSUPPORTED_ENCODING);
        }
      } else if (dtype_len == 8) {
        if (s->dtype_len_in == 4) {
          // Reading INT32 TIME_MILLIS into 64-bit DURATION_MILLISECONDS
          // TIME_MILLIS is the only duration type stored as int32:
          // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#deprecated-time-convertedtype
          gpuOutputByteStreamSplit<int32_t>(dst, src, num_values);
          // zero out most significant bytes
          memset(dst + 4, 0, 4);
        } else if (s->ts_scale) {
          gpuOutputSplitInt64Timestamp(
            reinterpret_cast<int64_t*>(dst), src, num_values, s->ts_scale);
        } else {
          gpuOutputByteStreamSplit<int64_t>(dst, src, num_values);
        }
      } else if (dtype_len == 4) {
        gpuOutputByteStreamSplit<int32_t>(dst, src, num_values);
      } else {
        s->set_error_code(decode_error::UNSUPPORTED_ENCODING);
      }
    }

    pos += batch_size;
  }
}

template <int block_size, bool has_lists_t, typename state_buf>
struct decode_fixed_width_split_values_func {
  __device__ inline void operator()(page_state_s* s, state_buf* const sb, int start, int end, int t)
  {
    gpuDecodeFixedWidthSplitValues<block_size, has_lists_t, state_buf>(s, sb, start, end, t);
  }
};

template <int decode_block_size, typename level_t, typename state_buf>
static __device__ int gpuUpdateValidityAndRowIndicesNested(
  int32_t target_value_count, page_state_s* s, state_buf* sb, level_t const* const def, int t)
{
  constexpr int num_warps      = decode_block_size / cudf::detail::warp_size;
  constexpr int max_batch_size = num_warps * cudf::detail::warp_size;

  // how many (input) values we've processed in the page so far
  int value_count = s->input_value_count;

  // cap by last row so that we don't process any rows past what we want to output.
  int const first_row                 = s->first_row;
  int const last_row                  = first_row + s->num_rows;
  int const capped_target_value_count = min(target_value_count, last_row);

  int const row_index_lower_bound = s->row_index_lower_bound;

  int const max_depth       = s->col.max_nesting_depth - 1;
  auto& max_depth_ni        = s->nesting_info[max_depth];
  int max_depth_valid_count = max_depth_ni.valid_count;

  __syncthreads();

  while (value_count < capped_target_value_count) {
    int const batch_size = min(max_batch_size, capped_target_value_count - value_count);

    // definition level
    int const d = [&]() {
      if (t >= batch_size) {
        return -1;
      } else if (def) {
        return static_cast<int>(def[rolling_index<state_buf::nz_buf_size>(value_count + t)]);
      }
      return 1;
    }();

    int const thread_value_count = t;
    int const block_value_count  = batch_size;

    // compute our row index, whether we're in row bounds, and validity
    int const row_index           = thread_value_count + value_count;
    int const in_row_bounds       = (row_index >= row_index_lower_bound) && (row_index < last_row);
    int const in_write_row_bounds = ballot(row_index >= first_row && row_index < last_row);
    int const write_start = __ffs(in_write_row_bounds) - 1;  // first bit in the warp to store

    // iterate by depth
    for (int d_idx = 0; d_idx <= max_depth; d_idx++) {
      auto& ni = s->nesting_info[d_idx];

      int const is_valid = ((d >= ni.max_def_level) && in_row_bounds) ? 1 : 0;

      // thread and block validity count
      using block_scan = cub::BlockScan<int, decode_block_size>;
      __shared__ typename block_scan::TempStorage scan_storage;
      int thread_valid_count, block_valid_count;
      block_scan(scan_storage).ExclusiveSum(is_valid, thread_valid_count, block_valid_count);

      // validity is processed per-warp
      //
      // nested schemas always read and write to the same bounds (that is, read and write
      // positions are already pre-bounded by first_row/num_rows). flat schemas will start reading
      // at the first value, even if that is before first_row, because we cannot trivially jump to
      // the correct position to start reading. since we are about to write the validity vector
      // here we need to adjust our computed mask to take into account the write row bounds.
      int warp_null_count = 0;
      if (ni.valid_map != nullptr) {
        uint32_t const warp_validity_mask = ballot(is_valid);
        // lane 0 from each warp writes out validity
        if ((write_start >= 0) && ((t % cudf::detail::warp_size) == 0)) {
          int const valid_map_offset = ni.valid_map_offset;
          int const vindex     = value_count + thread_value_count;  // absolute input value index
          int const bit_offset = (valid_map_offset + vindex + write_start) -
                                 first_row;  // absolute bit offset into the output validity map
          int const write_end =
            cudf::detail::warp_size - __clz(in_write_row_bounds);  // last bit in the warp to store
          int const bit_count = write_end - write_start;
          warp_null_count     = bit_count - __popc(warp_validity_mask >> write_start);

          store_validity(bit_offset, ni.valid_map, warp_validity_mask >> write_start, bit_count);
        }
      }

      // sum null counts. we have to do it this way instead of just incrementing by (value_count -
      // valid_count) because valid_count also includes rows that potentially start before our row
      // bounds. if we could come up with a way to clean that up, we could remove this and just
      // compute it directly at the end of the kernel.
      size_type const block_null_count =
        cudf::detail::single_lane_block_sum_reduce<decode_block_size, 0>(warp_null_count);
      if (t == 0) { ni.null_count += block_null_count; }

      // if this is valid and we're at the leaf, output dst_pos
      if (d_idx == max_depth) {
        if (is_valid) {
          int const dst_pos = value_count + thread_value_count;
          int const src_pos = max_depth_valid_count + thread_valid_count;

          sb->nz_idx[rolling_index<state_buf::nz_buf_size>(src_pos)] = dst_pos;
        }
        // update stuff
        max_depth_valid_count += block_valid_count;
      }

    }  // end depth loop

    value_count += block_value_count;
  }  // end loop

  if (t == 0) {
    // update valid value count for decoding and total # of values we've processed
    max_depth_ni.valid_count = max_depth_valid_count;
    s->nz_count              = max_depth_valid_count;
    s->input_value_count     = value_count;
    s->input_row_count       = value_count;
  }

  return max_depth_valid_count;
}

template <int decode_block_size, typename level_t, typename state_buf>
static __device__ int gpuUpdateValidityAndRowIndicesFlat(
  int32_t target_value_count, page_state_s* s, state_buf* sb, level_t const* const def, int t)
{
  constexpr int num_warps      = decode_block_size / cudf::detail::warp_size;
  constexpr int max_batch_size = num_warps * cudf::detail::warp_size;

  auto& ni = s->nesting_info[0];

  // how many (input) values we've processed in the page so far
  int value_count = s->input_value_count;
  int valid_count = ni.valid_count;

  // cap by last row so that we don't process any rows past what we want to output.
  int const first_row                 = s->first_row;
  int const last_row                  = first_row + s->num_rows;
  int const capped_target_value_count = min(target_value_count, last_row);

  int const valid_map_offset      = ni.valid_map_offset;
  int const row_index_lower_bound = s->row_index_lower_bound;

  __syncthreads();

  while (value_count < capped_target_value_count) {
    int const batch_size = min(max_batch_size, capped_target_value_count - value_count);

    int const thread_value_count = t;
    int const block_value_count  = batch_size;

    // compute our row index, whether we're in row bounds, and validity
    int const row_index     = thread_value_count + value_count;
    int const in_row_bounds = (row_index >= row_index_lower_bound) && (row_index < last_row);

    // use definition level & row bounds to determine if is valid
    int const is_valid = [&]() {
      if (t >= batch_size) {
        return 0;
      } else if (def) {
        int const def_level =
          static_cast<int>(def[rolling_index<state_buf::nz_buf_size>(value_count + t)]);
        return ((def_level > 0) && in_row_bounds) ? 1 : 0;
      }
      return in_row_bounds;
    }();

    // thread and block validity count
    using block_scan = cub::BlockScan<int, decode_block_size>;
    __shared__ typename block_scan::TempStorage scan_storage;
    int thread_valid_count, block_valid_count;
    block_scan(scan_storage).ExclusiveSum(is_valid, thread_valid_count, block_valid_count);
    uint32_t const warp_validity_mask = ballot(is_valid);

    // validity is processed per-warp
    //
    // nested schemas always read and write to the same bounds (that is, read and write
    // positions are already pre-bounded by first_row/num_rows). flat schemas will start reading
    // at the first value, even if that is before first_row, because we cannot trivially jump to
    // the correct position to start reading. since we are about to write the validity vector
    // here we need to adjust our computed mask to take into account the write row bounds.
    int const in_write_row_bounds = ballot(row_index >= first_row && row_index < last_row);
    int const write_start = __ffs(in_write_row_bounds) - 1;  // first bit in the warp to store
    int warp_null_count   = 0;
    // lane 0 from each warp writes out validity
    if ((write_start >= 0) && ((t % cudf::detail::warp_size) == 0)) {
      int const vindex     = value_count + thread_value_count;  // absolute input value index
      int const bit_offset = (valid_map_offset + vindex + write_start) -
                             first_row;  // absolute bit offset into the output validity map
      int const write_end =
        cudf::detail::warp_size - __clz(in_write_row_bounds);  // last bit in the warp to store
      int const bit_count = write_end - write_start;
      warp_null_count     = bit_count - __popc(warp_validity_mask >> write_start);

      store_validity(bit_offset, ni.valid_map, warp_validity_mask >> write_start, bit_count);
    }

    // sum null counts. we have to do it this way instead of just incrementing by (value_count -
    // valid_count) because valid_count also includes rows that potentially start before our row
    // bounds. if we could come up with a way to clean that up, we could remove this and just
    // compute it directly at the end of the kernel.
    size_type const block_null_count =
      cudf::detail::single_lane_block_sum_reduce<decode_block_size, 0>(warp_null_count);
    if (t == 0) { ni.null_count += block_null_count; }

    // output offset
    if (is_valid) {
      int const dst_pos = value_count + thread_value_count;
      int const src_pos = valid_count + thread_valid_count;

      sb->nz_idx[rolling_index<state_buf::nz_buf_size>(src_pos)] = dst_pos;
    }

    // update stuff
    value_count += block_value_count;
    valid_count += block_valid_count;
  }

  if (t == 0) {
    // update valid value count for decoding and total # of values we've processed
    ni.valid_count       = valid_count;
    ni.value_count       = value_count;
    s->nz_count          = valid_count;
    s->input_value_count = value_count;
    s->input_row_count   = value_count;
  }

  return valid_count;
}

template <int decode_block_size, typename state_buf>
static __device__ int gpuUpdateValidityAndRowIndicesNonNullable(int32_t target_value_count,
                                                                page_state_s* s,
                                                                state_buf* sb,
                                                                int t)
{
  constexpr int num_warps      = decode_block_size / cudf::detail::warp_size;
  constexpr int max_batch_size = num_warps * cudf::detail::warp_size;

  // cap by last row so that we don't process any rows past what we want to output.
  int const first_row                 = s->first_row;
  int const last_row                  = first_row + s->num_rows;
  int const capped_target_value_count = min(target_value_count, last_row);
  int const row_index_lower_bound     = s->row_index_lower_bound;

  // how many (input) values we've processed in the page so far
  int value_count = s->input_value_count;

  int const max_depth = s->col.max_nesting_depth - 1;
  auto& ni            = s->nesting_info[max_depth];
  int valid_count     = ni.valid_count;

  __syncthreads();

  while (value_count < capped_target_value_count) {
    int const batch_size = min(max_batch_size, capped_target_value_count - value_count);

    int const thread_value_count = t;
    int const block_value_count  = batch_size;

    // compute our row index, whether we're in row bounds, and validity
    int const row_index     = thread_value_count + value_count;
    int const in_row_bounds = (row_index >= row_index_lower_bound) && (row_index < last_row);

    int const is_valid           = in_row_bounds;
    int const thread_valid_count = thread_value_count;
    int const block_valid_count  = block_value_count;

    // if this is valid and we're at the leaf, output dst_pos
    if (is_valid) {
      // for non-list types, the value count is always the same across
      int const dst_pos = value_count + thread_value_count;
      int const src_pos = valid_count + thread_valid_count;

      sb->nz_idx[rolling_index<state_buf::nz_buf_size>(src_pos)] = dst_pos;
    }

    // update stuff
    value_count += block_value_count;
    valid_count += block_valid_count;
  }  // end loop

  if (t == 0) {
    // update valid value count for decoding and total # of values we've processed
    ni.valid_count       = valid_count;
    ni.value_count       = value_count;
    s->nz_count          = valid_count;
    s->input_value_count = value_count;
    s->input_row_count   = value_count;
  }

  return valid_count;
}

template <int decode_block_size, bool nullable, typename level_t, typename state_buf>
static __device__ int gpuUpdateValidityAndRowIndicesLists(int32_t target_value_count,
                                                          page_state_s* s,
                                                          state_buf* sb,
                                                          level_t const* const def,
                                                          level_t const* const rep,
                                                          int t)
{
  constexpr int num_warps      = decode_block_size / cudf::detail::warp_size;
  constexpr int max_batch_size = num_warps * cudf::detail::warp_size;

  // how many (input) values we've processed in the page so far, prior to this loop iteration
  int value_count = s->input_value_count;

  // how many rows we've processed in the page so far
  int input_row_count = s->input_row_count;

  // cap by last row so that we don't process any rows past what we want to output.
  int const first_row = s->first_row;
  int const last_row  = first_row + s->num_rows;

  int const row_index_lower_bound = s->row_index_lower_bound;
  int const max_depth             = s->col.max_nesting_depth - 1;
  int max_depth_valid_count       = s->nesting_info[max_depth].valid_count;

  int const warp_index     = t / cudf::detail::warp_size;
  int const warp_lane      = t % cudf::detail::warp_size;
  bool const is_first_lane = (warp_lane == 0);

  __syncthreads();
  __shared__ block_scan_temp_storage<decode_block_size> temp_storage;

  while (value_count < target_value_count) {
    bool const within_batch = value_count + t < target_value_count;

    // get definition level, use repetition level to get start/end depth
    // different for each thread, as each thread has a different r/d
    auto const [def_level, start_depth, end_depth] = [&]() {
      if (!within_batch) { return cuda::std::make_tuple(-1, -1, -1); }

      int const level_index = rolling_index<state_buf::nz_buf_size>(value_count + t);
      int const rep_level   = static_cast<int>(rep[level_index]);
      int const start_depth = s->nesting_info[rep_level].start_depth;

      if constexpr (!nullable) {
        return cuda::std::make_tuple(-1, start_depth, max_depth);
      } else {
        if (def != nullptr) {
          int const def_level = static_cast<int>(def[level_index]);
          return cuda::std::make_tuple(
            def_level, start_depth, s->nesting_info[def_level].end_depth);
        } else {
          return cuda::std::make_tuple(1, start_depth, max_depth);
        }
      }
    }();

    // Determine value count & row index
    //  track (page-relative) row index for the thread so we can compare against input bounds
    //  keep track of overall # of rows we've read.
    int const is_new_row = start_depth == 0 ? 1 : 0;
    int num_prior_new_rows, total_num_new_rows;
    {
      block_scan_results new_row_scan_results;
      scan_block_exclusive_sum<decode_block_size>(is_new_row, new_row_scan_results, temp_storage);
      __syncthreads();
      num_prior_new_rows = new_row_scan_results.thread_count_within_block;
      total_num_new_rows = new_row_scan_results.block_count;
    }

    int const row_index = input_row_count + ((num_prior_new_rows + is_new_row) - 1);
    input_row_count += total_num_new_rows;
    int const in_row_bounds = (row_index >= row_index_lower_bound) && (row_index < last_row);

    // VALUE COUNT:
    // in_nesting_bounds: if at a nesting level where we need to add value indices
    // the bounds: from current rep to the rep AT the def depth
    int in_nesting_bounds = ((0 >= start_depth && 0 <= end_depth) && in_row_bounds) ? 1 : 0;
    int thread_value_count_within_warp, warp_value_count, thread_value_count, block_value_count;
    {
      block_scan_results value_count_scan_results;
      scan_block_exclusive_sum<decode_block_size>(
        in_nesting_bounds, value_count_scan_results, temp_storage);
      __syncthreads();

      thread_value_count_within_warp = value_count_scan_results.thread_count_within_warp;
      warp_value_count               = value_count_scan_results.warp_count;
      thread_value_count             = value_count_scan_results.thread_count_within_block;
      block_value_count              = value_count_scan_results.block_count;
    }

    // iterate by depth
    for (int d_idx = 0; d_idx <= max_depth; d_idx++) {
      auto& ni = s->nesting_info[d_idx];

      // everything up to the max_def_level is a non-null value
      int const is_valid = [&](int input_def_level) {
        if constexpr (nullable) {
          return ((input_def_level >= ni.max_def_level) && in_nesting_bounds) ? 1 : 0;
        } else {
          return in_nesting_bounds;
        }
      }(def_level);

      // VALID COUNT:
      // Not all values visited by this block will represent a value at this nesting level.
      // the validity bit for thread t might actually represent output value t-6.
      // the correct position for thread t's bit is thread_value_count.
      uint32_t const warp_valid_mask =
        WarpReduceOr32((uint32_t)is_valid << thread_value_count_within_warp);
      int thread_valid_count, block_valid_count;
      {
        auto thread_mask = (uint32_t(1) << thread_value_count_within_warp) - 1;

        block_scan_results valid_count_scan_results;
        scan_block_exclusive_sum<decode_block_size>(warp_valid_mask,
                                                    warp_lane,
                                                    warp_index,
                                                    thread_mask,
                                                    valid_count_scan_results,
                                                    temp_storage);
        __syncthreads();
        thread_valid_count = valid_count_scan_results.thread_count_within_block;
        block_valid_count  = valid_count_scan_results.block_count;
      }

      // compute warp and thread value counts for the -next- nesting level. we need to
      // do this for lists so that we can emit an offset for the -current- nesting level.
      // the offset for the current nesting level == current length of the next nesting level
      int next_thread_value_count_within_warp = 0, next_warp_value_count = 0;
      int next_thread_value_count = 0, next_block_value_count = 0;
      int next_in_nesting_bounds = 0;
      if (d_idx < max_depth) {
        // NEXT DEPTH VALUE COUNT:
        next_in_nesting_bounds =
          ((d_idx + 1 >= start_depth) && (d_idx + 1 <= end_depth) && in_row_bounds) ? 1 : 0;
        {
          block_scan_results next_value_count_scan_results;
          scan_block_exclusive_sum<decode_block_size>(
            next_in_nesting_bounds, next_value_count_scan_results, temp_storage);
          __syncthreads();

          next_thread_value_count_within_warp =
            next_value_count_scan_results.thread_count_within_warp;
          next_warp_value_count   = next_value_count_scan_results.warp_count;
          next_thread_value_count = next_value_count_scan_results.thread_count_within_block;
          next_block_value_count  = next_value_count_scan_results.block_count;
        }

        // STORE OFFSET TO THE LIST LOCATION
        // if we're -not- at a leaf column and we're within nesting/row bounds
        // and we have a valid data_out pointer, it implies this is a list column, so
        // emit an offset.
        if (in_nesting_bounds && ni.data_out != nullptr) {
          const auto& next_ni = s->nesting_info[d_idx + 1];
          int const idx       = ni.value_count + thread_value_count;
          cudf::size_type const ofs =
            next_ni.value_count + next_thread_value_count + next_ni.page_start_value;

          (reinterpret_cast<cudf::size_type*>(ni.data_out))[idx] = ofs;
        }
      }

      // validity is processed per-warp (on lane 0's)
      // thi is because when atomic writes are needed, they are 32-bit operations
      //
      // lists always read and write to the same bounds
      // (that is, read and write positions are already pre-bounded by first_row/num_rows).
      // since we are about to write the validity vector
      // here we need to adjust our computed mask to take into account the write row bounds.
      if constexpr (nullable) {
        if (is_first_lane && (ni.valid_map != nullptr) && (warp_value_count > 0)) {
          // absolute bit offset into the output validity map
          // is cumulative sum of warp_value_count at the given nesting depth
          // DON'T subtract by first_row: since it's lists it's not 1-row-per-value
          int const bit_offset = ni.valid_map_offset + thread_value_count;

          store_validity(bit_offset, ni.valid_map, warp_valid_mask, warp_value_count);
        }

        if (t == 0) { ni.null_count += block_value_count - block_valid_count; }
      }

      // if this is valid and we're at the leaf, output dst_pos
      // Read value_count before the sync, so that when thread 0 modifies it we've already read its
      // value
      int const current_value_count = ni.value_count;
      __syncthreads();  // guard against modification of ni.value_count below
      if (d_idx == max_depth) {
        if (is_valid) {
          int const dst_pos      = current_value_count + thread_value_count;
          int const src_pos      = max_depth_valid_count + thread_valid_count;
          int const output_index = rolling_index<state_buf::nz_buf_size>(src_pos);

          // Index from rolling buffer of values (which doesn't include nulls) to final array (which
          // includes gaps for nulls)
          sb->nz_idx[output_index] = dst_pos;
        }
        max_depth_valid_count += block_valid_count;
      }

      // update stuff
      if (t == 0) {
        ni.value_count += block_value_count;
        ni.valid_map_offset += block_value_count;
      }
      __syncthreads();  // sync modification of ni.value_count

      // propagate value counts for the next depth level
      block_value_count              = next_block_value_count;
      thread_value_count             = next_thread_value_count;
      in_nesting_bounds              = next_in_nesting_bounds;
      warp_value_count               = next_warp_value_count;
      thread_value_count_within_warp = next_thread_value_count_within_warp;
    }  // END OF DEPTH LOOP

    int const batch_size = min(max_batch_size, target_value_count - value_count);
    value_count += batch_size;
  }

  if (t == 0) {
    // update valid value count for decoding and total # of values we've processed
    s->nesting_info[max_depth].valid_count = max_depth_valid_count;
    s->nz_count                            = max_depth_valid_count;
    s->input_value_count                   = value_count;

    // If we have lists # rows != # values
    s->input_row_count = input_row_count;
  }

  return max_depth_valid_count;
}

// is the page marked nullable or not
__device__ inline bool is_nullable(page_state_s* s)
{
  auto const lvl           = level_type::DEFINITION;
  auto const max_def_level = s->col.max_level[lvl];
  return max_def_level > 0;
}

// for a nullable page, check to see if it could have nulls
__device__ inline bool maybe_has_nulls(page_state_s* s)
{
  auto const lvl      = level_type::DEFINITION;
  auto const init_run = s->initial_rle_run[lvl];
  // literal runs, lets assume they could hold nulls
  if (is_literal_run(init_run)) { return true; }

  // repeated run with number of items in the run not equal
  // to the rows in the page, assume that means we could have nulls
  if (s->page.num_input_values != (init_run >> 1)) { return true; }

  auto const lvl_bits = s->col.level_bits[lvl];
  auto const run_val  = lvl_bits == 0 ? 0 : s->initial_rle_value[lvl];

  // the encoded repeated value isn't valid, we have (all) nulls
  return run_val != s->col.max_level[lvl];
}

template <int decode_block_size_t, typename state_buf>
inline __device__ void bool_plain_decode(page_state_s* s, state_buf* sb, int t, int to_decode)
{
  int pos              = s->dict_pos;
  int const target_pos = pos + to_decode;
  __syncthreads();  // Make sure all threads have read dict_pos before it changes at the end.

  while (pos < target_pos) {
    int const batch_len = min(target_pos - pos, decode_block_size_t);

    if (t < batch_len) {
      int const bit_pos           = pos + t;
      int const byte_offset       = bit_pos >> 3;
      int const bit_in_byte_index = bit_pos & 7;

      uint8_t const* const read_from = s->data_start + byte_offset;
      bool const read_bit            = (*read_from) & (1 << bit_in_byte_index);

      int const write_to_index     = rolling_index<state_buf::dict_buf_size>(bit_pos);
      sb->dict_idx[write_to_index] = read_bit;
    }

    pos += batch_len;
  }

  if (t == 0) { s->dict_pos = pos; }
}

template <int rolling_buf_size, typename stream_type>
__device__ int skip_decode(stream_type& parquet_stream, int num_to_skip, int t)
{
  // it could be that (e.g.) we skip 5000 but starting at row 4000 we have a run of length 2000:
  // in that case skip_decode() only skips 4000, and we have to process the remaining 1000 up front
  // modulo 2 * block_size of course, since that's as many as we process at once
  int num_skipped = parquet_stream.skip_decode(t, num_to_skip);
  while (num_skipped < num_to_skip) {
    // TODO: Instead of decoding, skip within the run to the appropriate location
    auto const to_decode = min(rolling_buf_size, num_to_skip - num_skipped);
    num_skipped += parquet_stream.decode_next(t, to_decode);
    __syncthreads();
  }

  return num_skipped;
}

/**
 * @brief Kernel for computing fixed width non dictionary column data stored in the pages
 *
 * This function will write the page data and the page data's validity to the
 * output specified in the page's column chunk. If necessary, additional
 * conversion will be performed to translate from the Parquet datatype to
 * desired output datatype.
 *
 * @param pages List of pages
 * @param chunks List of column chunks
 * @param min_row Row index to start reading at
 * @param num_rows Maximum number of rows to read
 * @param error_code Error code to set if an error is encountered
 */
template <typename level_t, int decode_block_size_t, decode_kernel_mask kernel_mask_t>
CUDF_KERNEL void __launch_bounds__(decode_block_size_t, 8)
  gpuDecodePageDataGeneric(PageInfo* pages,
                           device_span<ColumnChunkDesc const> chunks,
                           size_t min_row,
                           size_t num_rows,
                           kernel_error::pointer error_code)
{
  constexpr bool has_dict_t = (kernel_mask_t == decode_kernel_mask::FIXED_WIDTH_DICT) ||
                              (kernel_mask_t == decode_kernel_mask::FIXED_WIDTH_DICT_NESTED) ||
                              (kernel_mask_t == decode_kernel_mask::FIXED_WIDTH_DICT_LIST);
  constexpr bool has_bools_t = (kernel_mask_t == decode_kernel_mask::BOOLEAN) ||
                               (kernel_mask_t == decode_kernel_mask::BOOLEAN_NESTED) ||
                               (kernel_mask_t == decode_kernel_mask::BOOLEAN_LIST);
  constexpr bool has_nesting_t =
    (kernel_mask_t == decode_kernel_mask::BOOLEAN_NESTED) ||
    (kernel_mask_t == decode_kernel_mask::FIXED_WIDTH_DICT_NESTED) ||
    (kernel_mask_t == decode_kernel_mask::FIXED_WIDTH_NO_DICT_NESTED) ||
    (kernel_mask_t == decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_NESTED);
  constexpr bool has_lists_t =
    (kernel_mask_t == decode_kernel_mask::BOOLEAN_LIST) ||
    (kernel_mask_t == decode_kernel_mask::FIXED_WIDTH_DICT_LIST) ||
    (kernel_mask_t == decode_kernel_mask::FIXED_WIDTH_NO_DICT_LIST) ||
    (kernel_mask_t == decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_LIST);
  constexpr bool split_decode_t =
    (kernel_mask_t == decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_FLAT) ||
    (kernel_mask_t == decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_NESTED) ||
    (kernel_mask_t == decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_LIST);

  constexpr int rolling_buf_size    = decode_block_size_t * 2;
  constexpr int rle_run_buffer_size = rle_stream_required_run_buffer_size<decode_block_size_t>();

  __shared__ __align__(16) page_state_s state_g;
  using state_buf_t = page_state_buffers_s<rolling_buf_size,  // size of nz_idx buffer
                                           has_dict_t || has_bools_t ? rolling_buf_size : 1,
                                           1>;
  __shared__ __align__(16) state_buf_t state_buffers;

  page_state_s* const s = &state_g;
  auto* const sb        = &state_buffers;
  int const page_idx    = blockIdx.x;
  int const t           = threadIdx.x;
  PageInfo* pp          = &pages[page_idx];

  if (!(BitAnd(pages[page_idx].kernel_mask, kernel_mask_t))) { return; }

  // must come after the kernel mask check
  [[maybe_unused]] null_count_back_copier _{s, t};

  if (!setupLocalPageInfo(s,
                          pp,
                          chunks,
                          min_row,
                          num_rows,
                          mask_filter{kernel_mask_t},
                          page_processing_stage::DECODE)) {
    return;
  }

  // if we have no work to do (eg, in a skip_rows/num_rows case) in this page.
  if (s->num_rows == 0) { return; }

  using value_decoder_type = std::conditional_t<
    split_decode_t,
    decode_fixed_width_split_values_func<decode_block_size_t, has_lists_t, state_buf_t>,
    decode_fixed_width_values_func<decode_block_size_t, has_lists_t, state_buf_t>>;
  value_decoder_type decode_values;

  bool const should_process_nulls = is_nullable(s) && maybe_has_nulls(s);

  // shared buffer. all shared memory is suballocated out of here
  constexpr int rle_run_buffer_bytes =
    cudf::util::round_up_unsafe(rle_run_buffer_size * sizeof(rle_run), size_t{16});
  constexpr int shared_buf_size =
    rle_run_buffer_bytes * (static_cast<int>(has_dict_t) + static_cast<int>(has_bools_t) +
                            static_cast<int>(has_lists_t) + 1);
  __shared__ __align__(16) uint8_t shared_buf[shared_buf_size];

  // setup all shared memory buffers
  int shared_offset = 0;
  auto rep_runs     = reinterpret_cast<rle_run*>(shared_buf + shared_offset);
  if constexpr (has_lists_t) { shared_offset += rle_run_buffer_bytes; }
  auto dict_runs = reinterpret_cast<rle_run*>(shared_buf + shared_offset);
  if constexpr (has_dict_t) { shared_offset += rle_run_buffer_bytes; }
  auto bool_runs = reinterpret_cast<rle_run*>(shared_buf + shared_offset);
  if constexpr (has_bools_t) { shared_offset += rle_run_buffer_bytes; }
  auto def_runs = reinterpret_cast<rle_run*>(shared_buf + shared_offset);

  // initialize the stream decoders (requires values computed in setupLocalPageInfo)
  rle_stream<level_t, decode_block_size_t, rolling_buf_size> def_decoder{def_runs};
  level_t* const def = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  if (should_process_nulls) {
    def_decoder.init(s->col.level_bits[level_type::DEFINITION],
                     s->abs_lvl_start[level_type::DEFINITION],
                     s->abs_lvl_end[level_type::DEFINITION],
                     def,
                     s->page.num_input_values);
  }

  rle_stream<level_t, decode_block_size_t, rolling_buf_size> rep_decoder{rep_runs};
  level_t* const rep = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::REPETITION]);
  if constexpr (has_lists_t) {
    rep_decoder.init(s->col.level_bits[level_type::REPETITION],
                     s->abs_lvl_start[level_type::REPETITION],
                     s->abs_lvl_end[level_type::REPETITION],
                     rep,
                     s->page.num_input_values);
  }

  rle_stream<uint32_t, decode_block_size_t, rolling_buf_size> dict_stream{dict_runs};
  if constexpr (has_dict_t) {
    dict_stream.init(
      s->dict_bits, s->data_start, s->data_end, sb->dict_idx, s->page.num_input_values);
  }

  // Use dictionary stream memory for bools
  rle_stream<uint32_t, decode_block_size_t, rolling_buf_size> bool_stream{bool_runs};
  bool bools_are_rle_stream = (s->dict_run == 0);
  if constexpr (has_bools_t) {
    if (bools_are_rle_stream) {
      bool_stream.init(1, s->data_start, s->data_end, sb->dict_idx, s->page.num_input_values);
    }
  }
  __syncthreads();

  // We use two counters in the loop below: processed_count and valid_count.
  // - processed_count: number of values out of num_input_values that we have decoded so far.
  //   the definition stream returns the number of total rows it has processed in each call
  //   to decode_next and we accumulate in process_count.
  // - valid_count: number of non-null values we have decoded so far. In each iteration of the
  //   loop below, we look at the number of valid items (which could be all for non-nullable),
  //   and valid_count is that running count.
  int processed_count = 0;
  int valid_count     = 0;

  // Skip ahead in the decoding so that we don't repeat work (skipped_leaf_values = 0 for non-lists)
  if constexpr (has_lists_t) {
    auto const skipped_leaf_values = s->page.skipped_leaf_values;
    if (skipped_leaf_values > 0) {
      if (should_process_nulls) {
        skip_decode<rolling_buf_size>(def_decoder, skipped_leaf_values, t);
      }
      processed_count = skip_decode<rolling_buf_size>(rep_decoder, skipped_leaf_values, t);
      if constexpr (has_dict_t) {
        skip_decode<rolling_buf_size>(dict_stream, skipped_leaf_values, t);
      }
    }
  }

  // the core loop. decode batches of level stream data using rle_stream objects
  // and pass the results to gpuDecodeValues
  // For chunked reads we may not process all of the rows on the page; if not stop early
  int const last_row = s->first_row + s->num_rows;
  while ((s->error == 0) && (processed_count < s->page.num_input_values) &&
         (s->input_row_count <= last_row)) {
    int next_valid_count;

    // only need to process definition levels if this is a nullable column
    if (should_process_nulls) {
      processed_count += def_decoder.decode_next(t);
      __syncthreads();

      if constexpr (has_lists_t) {
        rep_decoder.decode_next(t);
        __syncthreads();
        next_valid_count = gpuUpdateValidityAndRowIndicesLists<decode_block_size_t, true, level_t>(
          processed_count, s, sb, def, rep, t);
      } else if constexpr (has_nesting_t) {
        next_valid_count = gpuUpdateValidityAndRowIndicesNested<decode_block_size_t, level_t>(
          processed_count, s, sb, def, t);
      } else {
        next_valid_count = gpuUpdateValidityAndRowIndicesFlat<decode_block_size_t, level_t>(
          processed_count, s, sb, def, t);
      }
    }
    // if we wanted to split off the skip_rows/num_rows case into a separate kernel, we could skip
    // this function call entirely since all it will ever generate is a mapping of (i -> i) for
    // nz_idx.  gpuDecodeFixedWidthValues would be the only work that happens.
    else {
      if constexpr (has_lists_t) {
        processed_count += rep_decoder.decode_next(t);
        __syncthreads();
        next_valid_count = gpuUpdateValidityAndRowIndicesLists<decode_block_size_t, false, level_t>(
          processed_count, s, sb, nullptr, rep, t);
      } else {
        processed_count += min(rolling_buf_size, s->page.num_input_values - processed_count);
        next_valid_count =
          gpuUpdateValidityAndRowIndicesNonNullable<decode_block_size_t>(processed_count, s, sb, t);
      }
    }
    __syncthreads();

    // if we have dictionary or bool data
    // We want to limit the number of dictionary/bool items we decode, that correspond to
    // the rows we have processed in this iteration that are valid.
    // We know the number of valid rows to process with: next_valid_count - valid_count.
    if constexpr (has_dict_t) {
      dict_stream.decode_next(t, next_valid_count - valid_count);
      __syncthreads();
    } else if constexpr (has_bools_t) {
      if (bools_are_rle_stream) {
        bool_stream.decode_next(t, next_valid_count - valid_count);
      } else {
        bool_plain_decode<decode_block_size_t>(s, sb, t, next_valid_count - valid_count);
      }
      __syncthreads();
    }

    // decode the values themselves
    decode_values(s, sb, valid_count, next_valid_count, t);
    __syncthreads();

    valid_count = next_valid_count;
  }
  if (t == 0 and s->error != 0) { set_error(s->error, error_code); }
}

}  // anonymous namespace

template <decode_kernel_mask mask>
using kernel_tag_t = std::integral_constant<decode_kernel_mask, mask>;

template <int value>
using int_tag_t = std::integral_constant<int, value>;

void __host__ DecodePageData(cudf::detail::hostdevice_span<PageInfo> pages,
                             cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                             size_t num_rows,
                             size_t min_row,
                             int level_type_size,
                             decode_kernel_mask kernel_mask,
                             kernel_error::pointer error_code,
                             rmm::cuda_stream_view stream)
{
  // No template parameters on lambdas until C++20, so use type tags instead
  auto launch_kernel = [&](auto block_size_tag, auto kernel_mask_tag) {
    constexpr int decode_block_size   = decltype(block_size_tag)::value;
    constexpr decode_kernel_mask mask = decltype(kernel_mask_tag)::value;

    dim3 dim_block(decode_block_size, 1);
    dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

    if (level_type_size == 1) {
      gpuDecodePageDataGeneric<uint8_t, decode_block_size, mask>
        <<<dim_grid, dim_block, 0, stream.value()>>>(
          pages.device_ptr(), chunks, min_row, num_rows, error_code);
    } else {
      gpuDecodePageDataGeneric<uint16_t, decode_block_size, mask>
        <<<dim_grid, dim_block, 0, stream.value()>>>(
          pages.device_ptr(), chunks, min_row, num_rows, error_code);
    }
  };

  switch (kernel_mask) {
    case decode_kernel_mask::FIXED_WIDTH_NO_DICT:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::FIXED_WIDTH_NO_DICT>{});
      break;
    case decode_kernel_mask::FIXED_WIDTH_NO_DICT_NESTED:
      launch_kernel(int_tag_t<128>{},
                    kernel_tag_t<decode_kernel_mask::FIXED_WIDTH_NO_DICT_NESTED>{});
      break;
    case decode_kernel_mask::FIXED_WIDTH_NO_DICT_LIST:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::FIXED_WIDTH_NO_DICT_LIST>{});
      break;
    case decode_kernel_mask::FIXED_WIDTH_DICT:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::FIXED_WIDTH_DICT>{});
      break;
    case decode_kernel_mask::FIXED_WIDTH_DICT_NESTED:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::FIXED_WIDTH_DICT_NESTED>{});
      break;
    case decode_kernel_mask::FIXED_WIDTH_DICT_LIST:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::FIXED_WIDTH_DICT_LIST>{});
      break;
    case decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_FLAT:
      launch_kernel(int_tag_t<128>{},
                    kernel_tag_t<decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_FLAT>{});
      break;
    case decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_NESTED:
      launch_kernel(int_tag_t<128>{},
                    kernel_tag_t<decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_NESTED>{});
      break;
    case decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_LIST:
      launch_kernel(int_tag_t<128>{},
                    kernel_tag_t<decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_LIST>{});
      break;
    case decode_kernel_mask::BOOLEAN:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::BOOLEAN>{});
      break;
    case decode_kernel_mask::BOOLEAN_NESTED:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::BOOLEAN_NESTED>{});
      break;
    case decode_kernel_mask::BOOLEAN_LIST:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::BOOLEAN_LIST>{});
      break;
    default: CUDF_EXPECTS(false, "Kernel type not handled by this function"); break;
  }
}

}  // namespace cudf::io::parquet::detail
