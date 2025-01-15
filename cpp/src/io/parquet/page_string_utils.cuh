/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "page_decode.cuh"

#include <cudf/strings/detail/gather.cuh>

#include <cuda/atomic>

namespace cudf::io::parquet::detail {

// stole this from cudf/strings/detail/gather.cuh. modified to run on a single string on one warp.
// copies from src to dst in 16B chunks per thread.
inline __device__ void wideStrcpy(uint8_t* dst, uint8_t const* src, size_t len, uint32_t lane_id)
{
  using cudf::detail::warp_size;
  using cudf::strings::detail::load_uint4;

  constexpr size_t out_datatype_size = sizeof(uint4);
  constexpr size_t in_datatype_size  = sizeof(uint);

  auto const alignment_offset = reinterpret_cast<std::uintptr_t>(dst) % out_datatype_size;
  uint4* out_chars_aligned    = reinterpret_cast<uint4*>(dst - alignment_offset);
  auto const in_start         = src;

  // Both `out_start_aligned` and `out_end_aligned` are indices into `dst`.
  // `out_start_aligned` is the first 16B aligned memory location after `dst + 4`.
  // `out_end_aligned` is the last 16B aligned memory location before `len - 4`. Characters
  // between `[out_start_aligned, out_end_aligned)` will be copied using uint4.
  // `dst + 4` and `len - 4` are used instead of `dst` and `len` to avoid
  // `load_uint4` reading beyond string boundaries.
  // use signed int since out_end_aligned can be negative.
  int64_t const out_start_aligned = (in_datatype_size + alignment_offset + out_datatype_size - 1) /
                                      out_datatype_size * out_datatype_size -
                                    alignment_offset;
  int64_t const out_end_aligned =
    (len - in_datatype_size + alignment_offset) / out_datatype_size * out_datatype_size -
    alignment_offset;

  for (int64_t ichar = out_start_aligned + lane_id * out_datatype_size; ichar < out_end_aligned;
       ichar += warp_size * out_datatype_size) {
    *(out_chars_aligned + (ichar + alignment_offset) / out_datatype_size) =
      load_uint4((char const*)in_start + ichar);
  }

  // Tail logic: copy characters of the current string outside
  // `[out_start_aligned, out_end_aligned)`.
  if (out_end_aligned <= out_start_aligned) {
    // In this case, `[out_start_aligned, out_end_aligned)` is an empty set, and we copy the
    // entire string.
    for (int64_t ichar = lane_id; ichar < len; ichar += warp_size) {
      dst[ichar] = in_start[ichar];
    }
  } else {
    // Copy characters in range `[0, out_start_aligned)`.
    if (lane_id < out_start_aligned) { dst[lane_id] = in_start[lane_id]; }
    // Copy characters in range `[out_end_aligned, len)`.
    int64_t ichar = out_end_aligned + lane_id;
    if (ichar < len) { dst[ichar] = in_start[ichar]; }
  }
}

/**
 * @brief Perform exclusive scan on an array of any length using a single block of threads.
 */
template <int block_size>
__device__ inline void block_excl_sum(size_type* arr, size_type length, size_type initial_value)
{
  using block_scan = cub::BlockScan<size_type, block_size>;
  __shared__ typename block_scan::TempStorage scan_storage;
  int const t = threadIdx.x;

  // do a series of block sums, storing results in arr as we go
  for (int pos = 0; pos < length; pos += block_size) {
    int const tidx = pos + t;
    size_type tval = tidx < length ? arr[tidx] : 0;

    size_type block_sum;
    size_type new_tval;
    block_scan(scan_storage).ExclusiveSum(tval, new_tval, block_sum);
    __syncthreads();

    if (tidx < length) { arr[tidx] = new_tval + initial_value; }
    initial_value += block_sum;
  }
}

/**
 * @brief Converts string sizes to offsets if this is not a large string column. Otherwise,
 * atomically update the initial string offset to be used during large string column construction
 */
template <int block_size>
__device__ void convert_string_lengths_to_offsets(page_state_s const* const state,
                                                  size_t* initial_str_offsets,
                                                  int32_t chunk_idx,
                                                  bool has_lists)
{
  // If this is a large string column. In the
  // latter case, offsets will be computed during string column creation.
  auto& ni        = state->nesting_info[state->col.max_nesting_depth - 1];
  int value_count = ni.value_count;

  // if no repetition we haven't calculated start/end bounds and instead just skipped
  // values until we reach first_row. account for that here.
  if (not has_lists) { value_count -= state->first_row; }

  auto const initial_value = state->page.str_offset;

  if (value_count > 0) {
    // Convert the array of lengths into offsets if this not a large string column.
    if (not state->col.is_large_string_col) {
      auto const offptr = reinterpret_cast<size_type*>(ni.data_out);
      block_excl_sum<block_size>(offptr, value_count, initial_value);
    }  // Atomically update the initial string offset if this is a large string column. This initial
    // offset will be used to compute (64-bit) offsets during large string column construction.
    else if (!threadIdx.x) {
      cuda::atomic_ref<size_t, cuda::std::thread_scope_device> initial_str_offsets_ref{
        initial_str_offsets[chunk_idx]};
      initial_str_offsets_ref.fetch_min(initial_value, cuda::std::memory_order_relaxed);
    }
  }
}

template <int block_size>
__device__ inline int calc_threads_per_string_log2(int avg)
{
  // From testing, performance is best when copying an average of B = 4 bytes at once.
  // So #-threads-per-string M = avg_string_length / 4
  // Help the compiler make the code fast by keeping everything a power of 2
  // For avg length < 4/8/16/..., length power-of-2 = 2/3/4/.../. Divide by 4: 0/1/...
  // This is the target (log2) for M, but we need to clamp its range

  // Clamp M (#-threads-per-string):
  // For T threads: clamp #-strings-at-once N btw T/32 (1/warp) & 32 (cache miss if larger)
  // So, clamp #-threads-per-string M = T / N between 32 (all in warp) & T/32 (cache miss)
  // Writing an equation M(T) is slower than just handling each T case separately
  auto caster = [](int value) { return static_cast<int>(value != 0); };  // branchless

  if constexpr (block_size > 512) {
    return 5;  // max of 32 strings at a time, no matter what
  } else if constexpr (block_size > 256) {
    return (avg < 64) ? 4 : 5;
  } else if constexpr (block_size > 128) {
    //(avg < 32) ? 3 : ((avg < 64) ? 4 : 5);
    return 3 + caster(avg >> 5) + caster(avg >> 6);
  } else if constexpr (block_size > 64) {
    //(avg < 16) ? 2 : ((avg < 32) ? 3 : ((avg < 64) ? 4 : 5));
    return 2 + caster(avg >> 4) + caster(avg >> 5) + caster(avg >> 6);
  } else if constexpr (block_size > 32) {
    //(avg < 8) ? 1 : ((avg < 16) ? 2 : ((avg < 32) ? 3 : ((avg < 64) ? 4 : 5)));
    return 1 + caster(avg >> 3) + caster(avg >> 4) + caster(avg >> 5) + caster(avg >> 6);
  } else {  // One warp
    //(avg<4) ? 0 : ((avg<8) ? 1 : ((avg<16) ? 2 : ((avg<32) ? 3 : ((avg<64) ? 4 : 5))));
    return caster(avg >> 2) + caster(avg >> 3) + caster(avg >> 4) + caster(avg >> 5) +
           caster(avg >> 6);
  }
}

/**
 * @brief Function for copying strings from the parquet page into column memory.
 *
 * All of the threads in the block will help with memcpy's, but only on a max of
 * 32 strings at once due to memory caching issues.
 * The # strings copied at once (and how many threads per string) is determined
 * from the average string length, with a target memcpy of 4-bytes per thread.
 *
 * @param s Page state
 * @param sb Page state buffers
 * @param start The value index to start copying strings for
 * @param end One past the end value index to stop copying strings for
 * @param t The current thread's index
 * @param string_output_offset Starting offset into the output column data for writing
 */
template <int block_size, bool has_lists_t, bool split_decode_t, typename state_buf>
__device__ size_t gpuDecodeString(
  page_state_s* s, state_buf* const sb, int start, int end, int t, size_t string_output_offset)
{
  // nesting level that is storing actual leaf values
  int const leaf_level_index    = s->col.max_nesting_depth - 1;
  int const skipped_leaf_values = s->page.skipped_leaf_values;

  auto const& ni = s->nesting_info[leaf_level_index];

  // decode values
  int pos = start;
  while (pos < end) {
    int const batch_size = min(block_size, end - pos);

    int const target_pos = pos + batch_size;
    int const thread_pos = pos + t;

    // Index from value buffer (doesn't include nulls) to final array (has gaps for nulls)
    int const dst_pos = [&]() {
      int dst_pos = sb->nz_idx[rolling_index<state_buf::nz_buf_size>(thread_pos)];
      if constexpr (!has_lists_t) { dst_pos -= s->first_row; }
      return dst_pos;
    }();

    // src_pos represents the logical row position we want to read from. But in the case of
    // nested hierarchies (lists), there is no 1:1 mapping of rows to values. So src_pos
    // has to take into account the # of values we have to skip in the page to get to the
    // desired logical row.  For flat hierarchies, skipped_leaf_values will always be 0.
    int const src_pos = [&]() {
      if constexpr (has_lists_t) { return thread_pos + skipped_leaf_values; }
      return thread_pos;
    }();

    // lookup input string pointer & length. store length.
    bool const in_range                       = (thread_pos < target_pos) && (dst_pos >= 0);
    auto [thread_input_string, string_length] = [&]() {
      // target_pos will always be properly bounded by num_rows, but dst_pos may be negative (values
      // before first_row) in the flat hierarchy case.
      if (!in_range) { return string_index_pair{nullptr, 0}; }
      string_index_pair string_pair = gpuGetStringData(s, sb, src_pos);
      int32_t* str_len_ptr          = reinterpret_cast<int32_t*>(ni.data_out) + dst_pos;
      *str_len_ptr                  = string_pair.second;
      return string_pair;
    }();

    // compute string offsets
    size_t thread_string_offset, block_total_string_length;
    {
      using scanner = cub::BlockScan<size_t, block_size>;
      __shared__ typename scanner::TempStorage scan_storage;
      scanner(scan_storage)
        .ExclusiveSum(string_length, thread_string_offset, block_total_string_length);

      // Make sure all threads have finished using scan_storage before next loop overwrites.
      __syncthreads();
    }

    // adjust for prior offset, get output string pointer
    thread_string_offset += string_output_offset;
    string_output_offset += block_total_string_length;
    auto const thread_output_string = ni.string_out + thread_string_offset;

    if constexpr (split_decode_t) {
      if (in_range) {
        auto const split_string_length = s->dtype_len_in;
        auto const stream_length       = s->page.str_bytes / split_string_length;

        for (int ii = 0; ii < split_string_length; ii++) {
          thread_output_string[ii] = s->data_start[src_pos + ii * stream_length];
        }
      }
    } else {
      // With a dictionary, performance is MUCH better for cache hits (reuse dict entry)
      // On a cache hit, we load a 128-byte L1/L2 cache line
      // If each thread (e.g. 128) copies a string (128/line): 128*128 = 16kb PER BLOCK
      // This likely overflows the L2 cache and results in a lot of cache misses
      // Plus, perf is O(longest_length), so we're bottlenecked for non-uniform data

      // Instead, the (T = block_size) threads cooperatively copy the T strings
      // N strings are copied at a time, with M = T / N threads per string
      __shared__ uint8_t const* inputs[block_size];
      __shared__ uint8_t* outputs[block_size];
      __shared__ int lengths[block_size];

      // Save string pointers & lengths so threads can share the work on them
      outputs[t] = thread_output_string;
      inputs[t]  = reinterpret_cast<uint8_t const*>(thread_input_string);
      lengths[t] = string_length;

      // Choose M, N to be powers of 2 to divide T evenly and allow bit shifts.
      // For T threads: clamp N btw T/32 (1 string per warp) & 32 (cache miss if larger)
      // Per string, each thread copies B = Length / M bytes in one contiguous memcpy

      // Performance: O(N*Avg_Length/M), with floor of at least N memcpy's (of O(Avg_length/M))
      // Note if M is large (vs. Avg_length) then many threads may do nothing: Slow
      // Note if M is small then N is large, increasing the time floor of N memcpy's
      // Determine M and N for each batch of strings based on their average length
      int const avg_string_length = block_total_string_length / batch_size;
      int const threads_per_string_log2 =
        calc_threads_per_string_log2<block_size>(avg_string_length);
      int const threads_per_string = 1 << threads_per_string_log2;  // M

      // For block_size = T = 128:
      // For an avg string length of 16 bytes or less (because N clamped): M = 4, N = 32
      // For an avg length of 65+ bytes (rounded): M = 32, N = 4 (1 string / warp at once)
      int const strings_at_once = block_size >> threads_per_string_log2;  // N
      int const string_lane     = t & (threads_per_string - 1);
      int const start_str_idx   = t >> threads_per_string_log2;

      // Sync writing the string info to shared memory above, prior to using it below.
      __syncthreads();

      // loop over all strings in this batch
      // threads work on consecutive strings so that all bytes are close in memory
      for (int str_idx = start_str_idx; str_idx < batch_size; str_idx += strings_at_once) {
        auto const input_string = inputs[str_idx];
        if (input_string == nullptr) { continue; }

        auto output_string = outputs[str_idx];
        int const length   = lengths[str_idx];

        // One-shot N chars per thread
        int const chars_at_once    = (length + threads_per_string - 1) >> threads_per_string_log2;
        int const start_index      = string_lane * chars_at_once;
        int const substring_length = min(chars_at_once, length - start_index);
        if (substring_length > 0) {
          memcpy(&(output_string[start_index]), &(input_string[start_index]), substring_length);
        }
      }

      // Make sure all threads have finished using shared memory before next loop overwrites
      __syncthreads();
    }

    pos += batch_size;
  }

  return string_output_offset;
}

}  // namespace cudf::io::parquet::detail
