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

#pragma once

#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

namespace cudf::io::parquet::detail {

template <int num_threads>
constexpr int rle_stream_required_run_buffer_size()
{
  constexpr int num_rle_stream_decode_warps = (num_threads / cudf::detail::warp_size) - 1;
  return (num_rle_stream_decode_warps * 2);
}

/**
 * @brief Read a 32-bit varint integer
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The 32-bit value read
 */
inline __device__ uint32_t get_vlq32(uint8_t const*& cur, uint8_t const* end)
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
 * @brief RLE run decode function per warp.
 *
 * @param output output data buffer
 * @param level_run RLE run header
 * @param run_start beginning of data for RLE run
 * @param end pointer to the end of data for RLE run
 * @param run_output_pos absolute output position for this run
 * @param run_offset offset after run_output_pos this call to decode starts outputting at
 * @param size length that will be decoded in this decode call, truncated to fit output buffer
 * @param level_bits bits needed to encode max values in the run (definition, dictionary)
 * @param lane warp lane that is executing this decode call
 */
template <typename level_t, int max_output_values>
__device__ inline void decode(level_t* const output,
                              int const level_run,
                              uint8_t const* const run_start,
                              uint8_t const* const end,
                              int const run_output_pos,
                              int const run_offset,
                              int const size,
                              int level_bits,
                              int lane)
{
  // local output_pos for this `decode` call.
  int decode_output_pos = 0;
  int remain            = size;

  // for bitpacked/literal runs, total size is always a multiple of 8. so we need to take care if
  // we are not starting/ending exactly on a run boundary
  uint8_t const* cur;
  if (is_literal_run(level_run)) {
    int const effective_offset = cudf::util::round_down_safe(run_offset, 8);
    int const lead_values      = (run_offset - effective_offset);
    decode_output_pos -= lead_values;
    remain += lead_values;
    cur = run_start + ((effective_offset >> 3) * level_bits);
  }

  // if this is a repeated run, compute the repeated value
  int level_val;
  if (is_repeated_run(level_run)) {
    level_val = run_start[0];
    if constexpr (sizeof(level_t) > 1) {
      if (level_bits > 8) {
        level_val |= run_start[1] << 8;
        if constexpr (sizeof(level_t) > 2) {
          if (level_bits > 16) {
            level_val |= run_start[2] << 16;
            if (level_bits > 24) { level_val |= run_start[3] << 24; }
          }
        }
      }
    }
  }

  // process
  while (remain > 0) {
    int const batch_len = min(32, remain);

    // if this is a literal run. each thread computes its own level_val
    if (is_literal_run(level_run)) {
      int const batch_len8 = (batch_len + 7) >> 3;
      if (lane < batch_len) {
        int bitpos                = lane * level_bits;
        uint8_t const* cur_thread = cur + (bitpos >> 3);
        bitpos &= 7;
        level_val = 0;
        if (cur_thread < end) { level_val = cur_thread[0]; }
        cur_thread++;
        if (level_bits > 8 - bitpos && cur_thread < end) {
          level_val |= cur_thread[0] << 8;
          cur_thread++;
          if (level_bits > 16 - bitpos && cur_thread < end) {
            level_val |= cur_thread[0] << 16;
            cur_thread++;
            if (level_bits > 24 - bitpos && cur_thread < end) { level_val |= cur_thread[0] << 24; }
          }
        }
        level_val = (level_val >> bitpos) & ((1 << level_bits) - 1);
      }

      cur += batch_len8 * level_bits;
    }

    // store level_val
    if (lane < batch_len && (lane + decode_output_pos) >= 0) {
      auto const idx = lane + run_output_pos + run_offset + decode_output_pos;
      output[rolling_index<max_output_values>(idx)] = level_val;
    }
    remain -= batch_len;
    decode_output_pos += batch_len;
  }
}

// a single rle run. may be broken up into multiple rle_batches
struct rle_run {
  int size;        // total size of the run
  int output_pos;  // absolute position of this run w.r.t output
  uint8_t const* start;
  int level_run;  // level_run header value
  int remaining;  // number of output items remaining to be decoded
};

// a stream of rle_runs
template <typename level_t, int decode_threads, int max_output_values>
struct rle_stream {
  static constexpr int num_rle_stream_decode_threads = decode_threads;
  // the -1 here is for the look-ahead warp that fills in the list of runs to be decoded
  // in an overlapped manner. so if we had 16 total warps:
  // - warp 0 would be filling in batches of runs to be processed
  // - warps 1-15 would be decoding the previous batch of runs generated
  static constexpr int num_rle_stream_decode_warps =
    (num_rle_stream_decode_threads / cudf::detail::warp_size) - 1;

  static constexpr int run_buffer_size = rle_stream_required_run_buffer_size<decode_threads>();

  int level_bits;
  uint8_t const* cur;
  uint8_t const* end;

  int total_values;
  int cur_values;

  level_t* output;

  rle_run* runs;

  int output_pos;

  int fill_index;
  int decode_index;

  __device__ rle_stream(rle_run* _runs) : runs(_runs) {}

  __device__ inline bool is_last_decode_warp(int warp_id)
  {
    return warp_id == num_rle_stream_decode_warps;
  }

  __device__ void init(int _level_bits,
                       uint8_t const* _start,
                       uint8_t const* _end,
                       level_t* _output,
                       int _total_values)
  {
    level_bits = _level_bits;
    cur        = _start;
    end        = _end;

    output = _output;

    output_pos = 0;

    total_values = _total_values;
    cur_values   = 0;
    fill_index   = 0;
    decode_index = -1;  // signals the first iteration. Nothing to decode.
  }

  __device__ inline int get_rle_run_info(rle_run& run)
  {
    run.start     = cur;
    run.level_run = get_vlq32(run.start, end);

    // run_bytes includes the header size
    int run_bytes = run.start - cur;
    if (is_literal_run(run.level_run)) {
      // from the parquet spec: literal runs always come in multiples of 8 values.
      run.size = (run.level_run >> 1) * 8;
      run_bytes += util::div_rounding_up_unsafe(run.size * level_bits, 8);
    } else {
      // repeated value run
      run.size = (run.level_run >> 1);
      run_bytes += util::div_rounding_up_unsafe(level_bits, 8);
    }

    return run_bytes;
  }

  __device__ inline void fill_run_batch()
  {
    // decode_index == -1 means we are on the very first decode iteration for this stream.
    // In this first iteration we are filling up to half of the runs array to decode in the next
    // iteration. On subsequent iterations, decode_index >= 0 and we are going to fill as many run
    // slots available as we can, to fill up to the slot before decode_index. We are also always
    // bound by cur < end, making sure we stop decoding once we've reached the end of the stream.
    while (((decode_index == -1 && fill_index < num_rle_stream_decode_warps) ||
            fill_index < decode_index + run_buffer_size) &&
           cur < end) {
      // Encoding::RLE
      // Pass by reference to fill the runs shared memory with the run data
      auto& run           = runs[rolling_index<run_buffer_size>(fill_index)];
      int const run_bytes = get_rle_run_info(run);

      run.remaining  = run.size;
      run.output_pos = output_pos;

      cur += run_bytes;
      output_pos += run.size;
      fill_index++;
    }
  }

  __device__ inline int decode_next(int t, int count)
  {
    int const output_count = min(count, total_values - cur_values);
    // special case. if level_bits == 0, just return all zeros. this should tremendously speed up
    // a very common case: columns with no nulls, especially if they are non-nested
    if (level_bits == 0) {
      int written = 0;
      while (written < output_count) {
        int const batch_size = min(num_rle_stream_decode_threads, output_count - written);
        if (t < batch_size) { output[rolling_index<max_output_values>(written + t)] = 0; }
        written += batch_size;
      }
      cur_values += output_count;
      return output_count;
    }

    // otherwise, full decode.
    int const warp_id        = t / cudf::detail::warp_size;
    int const warp_decode_id = warp_id - 1;
    int const warp_lane      = t % cudf::detail::warp_size;

    __shared__ int values_processed_shared;
    __shared__ int decode_index_shared;
    __shared__ int fill_index_shared;
    if (t == 0) {
      values_processed_shared = 0;
      decode_index_shared     = decode_index;
      fill_index_shared       = fill_index;
    }

    __syncthreads();

    fill_index = fill_index_shared;

    do {
      // protect against threads advancing past the end of this loop
      // and updating shared variables.
      __syncthreads();

      // warp 0 reads ahead and fills `runs` array to be decoded by remaining warps.
      if (warp_id == 0) {
        // fill the next set of runs. fill_runs will generally be the bottleneck for any
        // kernel that uses an rle_stream.
        if (warp_lane == 0) {
          fill_run_batch();
          if (decode_index == -1) {
            // first time, set it to the beginning of the buffer (rolled)
            decode_index        = 0;
            decode_index_shared = decode_index;
          }
          fill_index_shared = fill_index;
        }
      }
      // remaining warps decode the runs, starting on the second iteration of this. the pipeline of
      // runs is also persistent across calls to decode_next, so on the second call to decode_next,
      // this branch will start doing work immediately.
      // do/while loop (decode_index == -1 means "first iteration", so we should skip decoding)
      else if (decode_index >= 0 && decode_index + warp_decode_id < fill_index) {
        int const run_index = decode_index + warp_decode_id;
        auto& run           = runs[rolling_index<run_buffer_size>(run_index)];
        // this is the total amount (absolute) we will write in this invocation
        // of `decode_next`.
        int const max_count = cur_values + output_count;
        // run.output_pos is absolute position, we start decoding
        // if it's supposed to fit in this call to `decode_next`.
        if (max_count > run.output_pos) {
          int remaining        = run.remaining;
          int const run_offset = run.size - remaining;
          // last_run_pos is the absolute position of the run, including
          // what was decoded last time.
          int const last_run_pos = run.output_pos + run_offset;

          // the amount we should process is the smallest of current remaining, or
          // space available in the output buffer (for that last run at the end of
          // a call to decode_next).
          int const batch_len = min(remaining, max_count - last_run_pos);
          decode<level_t, max_output_values>(output,
                                             run.level_run,
                                             run.start,
                                             end,
                                             run.output_pos,
                                             run_offset,
                                             batch_len,
                                             level_bits,
                                             warp_lane);

          __syncwarp();
          if (warp_lane == 0) {
            // after writing this batch, are we at the end of the output buffer?
            auto const at_end = ((last_run_pos + batch_len - cur_values) == output_count);

            // update remaining for my warp
            remaining -= batch_len;
            // this is the last batch we will process this iteration if:
            // - either this run still has remaining values
            // - or it is consumed fully and its last index corresponds to output_count
            if (remaining > 0 || at_end) { values_processed_shared = output_count; }
            if (remaining == 0 && (at_end || is_last_decode_warp(warp_id))) {
              decode_index_shared = run_index + 1;
            }
            run.remaining = remaining;
          }
        }
      }
      __syncthreads();
      decode_index = decode_index_shared;
      fill_index   = fill_index_shared;
    } while (values_processed_shared < output_count);

    cur_values += values_processed_shared;

    // valid for every thread
    return values_processed_shared;
  }

  __device__ inline int skip_runs(int target_count)
  {
    // we want to process all runs UP TO BUT NOT INCLUDING the run that overlaps with the skip
    // amount so threads spin like crazy on fill_run_batch(), skipping writing unnecessary run info.
    // then when it hits the one that matters, we don't process it at all and bail as if we never
    // started basically we're setting up the rle_stream vars necessary to start fill_run_batch for
    // the first time
    while (cur < end) {
      rle_run run;
      int run_bytes = get_rle_run_info(run);

      if ((output_pos + run.size) > target_count) {
        return output_pos;  // bail! we've reached the starting run
      }

      // skip this run
      output_pos += run.size;
      cur += run_bytes;
    }

    return output_pos;  // we skipped everything
  }

  __device__ inline int skip_decode(int t, int count)
  {
    int const output_count = min(count, total_values - cur_values);

    // if level_bits == 0, there's nothing to do
    // a very common case: columns with no nulls, especially if they are non-nested
    cur_values = (level_bits == 0) ? output_count : skip_runs(output_count);
    return cur_values;
  }

  __device__ inline int decode_next(int t) { return decode_next(t, max_output_values); }
};

}  // namespace cudf::io::parquet::detail
