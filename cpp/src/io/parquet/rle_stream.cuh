/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/std/algorithm>
#include <cuda/std/iterator>
#include <cuda/std/memory>
#include <cuda/std/span>

namespace cudf::io::parquet::detail {

template <int num_threads>
__device__ constexpr int rle_stream_required_run_buffer_size()
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

// Chunked-expand path: number of run headers parsed per chunk.
//
// SMEM cost is (2 * kGenRuns + 1) * 4 bytes. Increasing kGenRuns reduces the
// number of outer-loop iterations in decode_next_chunked (each with a serial
// header-parse phase and a __syncthreads()), but competes with occupancy in
// preprocess_levels_kernel.
//
// Tuning (nvbench PARQUET_READER_NVBENCH parquet_read_decode -a
// data_type=[LIST,STRUCT,STRING]):
//   V100 (sm_70): 512 chosen to fit preprocess_levels_kernel SMEM budget.
//   A100 (sm_80): 1024 is optimal (80-95% faster than 512 on chunked path).
//                 2048 saturates or regresses ~10% on LIST (occupancy).
//   H100 (sm_90): TODO, expected ~1024 based on saturation of the A100 curve.
//   Blackwell:    TODO, inherits sm_80+ tier until measured.
#if __CUDA_ARCH__ >= 800
static constexpr int kGenRuns = 1024;
#else
static constexpr int kGenRuns = 512;
#endif

// a stream of rle_runs
template <typename level_t,
          int decode_threads,
          int max_output_values,
          bool use_chunked_expand = false>
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
  uint8_t const* s_start;
  uint8_t const* cur;
  uint8_t const* end;

  int total_values;
  int cur_values;

  level_t* output;

  rle_run* runs;

  int output_pos;

  int fill_index;
  int decode_index;

  // Optional shared-memory staging of the encoded byte stream. When init() is
  // given a scratch buffer large enough to hold [start, end), the stream is
  // copied into it once (block-cooperatively) and cur/end are rebased into
  // shared memory. This turns the serial run-header parse that dominates
  // fill_run_batch() from a chain of dependent L2 loads into shared-memory
  // loads. It stages *raw encoded bytes*, so it is level_t- and
  // level_bits-agnostic: definition/repetition levels, dictionary indices, and
  // boolean streams all benefit with identical code. Streams that do not fit
  // the budget transparently fall back to parsing from global.
  static constexpr int smem_stage_size = 8 * 1024;

  // Chunked-expand cross-call partial-run state.
  // When a run straddles a decode_next_chunked boundary we record its meta,
  // total count, and how many values were already emitted so the next call
  // can resume from the correct payload offset.  partial_run_meta == -1
  // means no pending partial run.
  int partial_run_meta;   // meta word for the split run (-1 = none)
  int partial_run_total;  // full value count of that run
  int partial_run_done;   // values already emitted from it

  __device__ rle_stream(rle_run* _runs) : runs(_runs) {}

  __device__ inline bool is_last_decode_warp(int warp_id)
  {
    return warp_id == num_rle_stream_decode_warps;
  }

  template <typename Group>
  __device__ void init(Group const& group,
                       int _level_bits,
                       uint8_t const* _start,
                       uint8_t const* _end,
                       level_t* _output,
                       int _total_values,
                       uint8_t* _smem_stage                                   = nullptr,
                       cuda::barrier<cuda::thread_scope_block>* _copy_barrier = nullptr)
  {
    level_bits = _level_bits;
    // s_start is set below after any smem-staging rebase, so downstream code
    // that stores offsets relative to s_start (chunked-expand meta) works
    // uniformly whether cur points into global or shared memory.
    cur = _start;
    end = _end;

    output = _output;

    output_pos = 0;

    total_values = _total_values;
    cur_values   = 0;
    fill_index   = 0;
    decode_index = -1;  // signals the first iteration. Nothing to decode.

    // If smem staging is active, use cuda::memcpy_async for a
    // block-cooperative global-to-shared copy that automatically dispatches to
    // the best copy path (cp.async, cp.async.bulk, or TMA) depending on the
    // hardware. Callers must provide a copy_barrier when using smem staging,
    // and must issue copy_barrier->arrive_and_wait() after init() to complete
    // the async copy.
    if (_smem_stage != nullptr) {
      auto* const smem_stage =
        static_cast<uint8_t const*>(cuda::std::assume_aligned<16>(_smem_stage));
      auto const len = static_cast<int>(cuda::std::distance(_start, _end));
      if (len > 0 && len <= smem_stage_size) {
        cuda::memcpy_async(group, _smem_stage, _start, static_cast<size_t>(len), *_copy_barrier);
        // Rebase the parse cursor and end onto the shared copy. All downstream
        // reads (get_rle_run_info, decode, skip_runs) follow cur/end and now hit
        // shared memory with no other changes required.
        cur = smem_stage;
        end = smem_stage + len;
      }
    }
    // Anchor s_start to the (possibly rebased) cur so chunked-expand meta
    // offsets index into the same memory space that the parse cursor uses.
    s_start = cur;

    partial_run_meta  = -1;
    partial_run_total = 0;
    partial_run_done  = 0;
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

  __device__ inline int decode_next_ring(int t, int count)
  {
    int const output_count = min(count, total_values - cur_values);

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

  __device__ __forceinline__ static void warp_fill(
    level_t* __restrict__ out, int abs_lo, int abs_hi, level_t value, int lane)
  {
    if (abs_lo >= abs_hi) return;
    // abs_lo and abs_hi are absolute indices; apply rolling_index at each write.
    for (int q = abs_lo + lane; q < abs_hi; q += 32) {
      out[rolling_index<max_output_values>(q)] = value;
    }
  }

  // Alternate decode path used when `use_chunked_expand` is true.
  //
  // Instead of the ring-buffer producer/consumer model in decode_next_ring
  // (one warp parses run headers, other warps expand one run each), thread 0
  // parses up to `kGenRuns` headers up-front into shared-memory tables
  // (gen_out_off / gen_meta), and then *all* warps cooperatively expand a
  // slice of the concatenated output range using binary search. This keeps
  // every warp busy even when runs are highly non-uniform in size, at the
  // cost of an extra intra-block sync per chunk.
  //
  // A single RLE run may exceed the requested `count`. In that case we emit
  // as much as fits, stash `partial_run_{meta,total,done}` in class state,
  // and resume it as slot 0 of the next invocation without re-parsing its
  // header (cur has already advanced past the payload).
  __device__ inline int decode_next_chunked(int t, int count)
  {
    int const output_count = min(count, total_values - cur_values);

    // ------------------------------------------------------------------
    // Per-chunk shared-memory scratch. `gen_out_off[i]` is the exclusive
    // prefix-sum of run lengths within the current chunk, so run `i`
    // occupies output positions [gen_out_off[i], gen_out_off[i+1]).
    // `gen_meta[i]` encodes both the payload offset (into s_start) and,
    // in the top bit, whether the run is literal (1) or RLE (0).
    // ------------------------------------------------------------------
    __shared__ int gen_out_off[kGenRuns + 1];
    __shared__ int gen_meta[kGenRuns];
    cuda::std::span<int> const gen_out_off_v{gen_out_off, kGenRuns + 1};
    cuda::std::span<int> const gen_meta_v{gen_meta, kGenRuns};
    // Payload offset within run slot 0: non-zero only when continuing a run
    // that was split across two decode_next_chunked calls.
    __shared__ int s_run0_payload_offset;
    __shared__ int s_chunk_runs;   // number of runs parsed in this chunk (n)
    __shared__ int s_chunk_total;  // sum of run lengths in this chunk (co)
    __shared__ int s_base_out;     // absolute output pos where this chunk starts

    int const lane        = t & 31;
    int const warp        = t >> 5;
    int constexpr kWarps  = num_rle_stream_decode_threads / cudf::detail::warp_size;
    int const value_width = (level_bits + 7) >> 3;
    // Bit mask used to extract a single level from a bit-packed literal-run
    // payload word. Invariant across the whole call; hoisted out of the
    // phase-2 expand loop to keep it out of the hot register set.
    uint32_t const level_mask = (level_bits == 32) ? 0xffffffffu : ((1u << level_bits) - 1);
    int out_pos_total         = cur_values;
    int const out_end         = cur_values + output_count;

    // ------------------------------------------------------------------
    // Outer loop: process the requested output range in chunks of up to
    // `kGenRuns` runs at a time until we have emitted `output_count`
    // values or run out of encoded input.
    // ------------------------------------------------------------------
    while (out_pos_total < out_end) {
      // ----- Phase 1: single-thread run-header parse ------------------
      // Thread 0 walks the encoded stream, decoding VLQ run headers and
      // filling gen_out_off / gen_meta. The other threads wait at the
      // __syncthreads() below. This is cheap because it is bounded by
      // kGenRuns headers and header parsing is inherently serial.
      if (t == 0) {
        int co                = 0;
        int n                 = 0;
        int out_base          = out_pos_total;
        gen_out_off_v[0]      = 0;
        s_run0_payload_offset = 0;

        // Slot 0 special case: resume a run that was split by the previous
        // call. `cur` already points past this run's payload (fully
        // consumed last call), so we do NOT re-parse its header - we just
        // reuse the saved meta and continue emitting values.
        if (partial_run_meta != -1) {
          int const remaining   = partial_run_total - partial_run_done;
          int const room        = out_end - out_base;
          int const cnt         = min(remaining, room);
          gen_meta_v[0]         = partial_run_meta;
          gen_out_off_v[1]      = cnt;
          s_run0_payload_offset = partial_run_done;
          n                     = 1;
          co                    = cnt;
          if (cnt < remaining) {
            partial_run_done += cnt;
          } else {
            partial_run_meta = -1;
          }
        }

        // Parse up to kGenRuns headers, stopping early if the output range
        // fills up or the encoded stream is exhausted.
        while (n < kGenRuns && (out_base + co) < out_end && cur < end) {
          uint32_t const level_run = get_vlq32(cur, end);
          int cnt;
          int meta;

          // Parquet RLE header format: LSB selects the encoding.
          //   bit 0 = 1  -> literal (bit-packed) run of `groups*8` values
          //   bit 0 = 0  -> RLE run of `level_run >> 1` copies of one value
          // The high bit of `meta` distinguishes the two at expand time.
          //
          // Assumption: (cur - s_start) fits in 31 bits, i.e. the encoded
          // level stream for a single Parquet page is < 2 GiB. This is
          // guaranteed by the Parquet format in practice (page payloads are
          // orders of magnitude smaller than 2 GiB) and is independent of
          // kGenRuns.
          if (level_run & 1u) {
            int const groups = level_run >> 1;
            cnt              = groups * 8;
            meta             = static_cast<int>(cur - s_start) | (1u << 31);
            cur += groups * level_bits;
          } else {
            cnt  = level_run >> 1;
            meta = static_cast<int>(cur - s_start);
            cur += value_width;
          }

          // If this run overflows the requested output window, clamp it and
          // stash the remainder as a pending partial run for the next call.
          // Note `cur` has already been advanced past the *full* payload
          // above, which is what we want - the resumed call reads the
          // payload out of s_start via the saved meta offset.
          int const room = out_end - (out_base + co);
          if (cnt > room) {
            partial_run_meta  = meta;
            partial_run_total = cnt;
            partial_run_done  = room;
            cnt               = room;
          }
          co += cnt;
          gen_meta_v[n]      = meta;
          gen_out_off_v[++n] = co;
          if (partial_run_meta != -1) { break; }
        }
        s_chunk_runs  = n;
        s_chunk_total = co;
        s_base_out    = out_base;
      }
      __syncthreads();

      // ----- Phase 2: cooperative expand ------------------------------
      // All warps see the same gen_out_off / gen_meta tables. We split
      // the flat output range [0, chunk_total) into `kWarps` equal-ish
      // slices and each warp writes its slice.
      int const chunk_runs          = s_chunk_runs;
      int const chunk_total         = s_chunk_total;
      int const base_out            = s_base_out;
      int const run0_payload_offset = s_run0_payload_offset;

      if (chunk_runs == 0) { break; }

      int const per = (chunk_total + kWarps - 1) / kWarps;
      int const lo  = warp * per;
      int const hi  = min(lo + per, chunk_total);
      if (lo < hi) {
        // Binary-search gen_out_off to find the first run that intersects
        // this warp's slice, then iterate forward until we pass `hi`.
        // This is the load-balancing trick: warps whose slice happens to
        // fall inside one huge run just write into it in parallel, and
        // warps whose slice covers many small runs walk through them.
        int const a =
          static_cast<int>(cuda::std::upper_bound(
                             gen_out_off_v.begin(), gen_out_off_v.begin() + chunk_runs + 1, lo) -
                           gen_out_off_v.begin()) -
          1;
        for (int r = a; r < chunk_runs && gen_out_off_v[r] < hi; ++r) {
          int const r_lo   = gen_out_off_v[r];
          int const r_hi   = gen_out_off_v[r + 1];
          int const seg_lo = max(r_lo, lo);
          int const seg_hi = min(r_hi, hi);
          int const meta   = gen_meta_v[r];
          // For slot 0 of a resumed partial run add the already-emitted offset
          // so we read from the correct position in the payload.
          int const run_payload_off = (r == 0) ? run0_payload_offset : 0;

          // Two expand kernels, selected by the meta top-bit:
          //   literal (bit 31 set) -> each output position needs its own
          //     bit-field extract from the packed payload.
          //   RLE    (bit 31 clear) -> single value read once, broadcast
          //     across [seg_lo, seg_hi) with warp_fill.
          if (meta & (1u << 31)) {
            int const payload_off  = meta & 0x7fffffff;
            uint8_t const* payload = s_start + payload_off;
            for (int p = seg_lo + lane; p < seg_hi; p += 32) {
              int const local       = (p - r_lo) + run_payload_off;
              int bitpos            = local * level_bits;
              uint8_t const* source = payload + (bitpos >> 3);
              bitpos &= 7;
              uint32_t level_val = 0;
              if (source < end) { level_val = source[0]; }
              ++source;
              if (level_bits > 8 - bitpos && source < end) {
                level_val |= static_cast<uint32_t>(source[0]) << 8;
                ++source;
                if (level_bits > 16 - bitpos && source < end) {
                  level_val |= static_cast<uint32_t>(source[0]) << 16;
                  ++source;
                  if (level_bits > 24 - bitpos && source < end) {
                    level_val |= static_cast<uint32_t>(source[0]) << 24;
                  }
                }
              }
              level_val = (level_val >> bitpos) & level_mask;
              output[rolling_index<max_output_values>(base_out + p)] =
                static_cast<level_t>(level_val);
            }
          } else {
            // RLE run: read the single repeated value once from s_start,
            // assembling up to 4 payload bytes into `level_val` based on
            // level_bits, then warp_fill the whole segment.
            uint8_t const* vptr = s_start + (meta & 0x7fffffff);
            uint32_t level_val  = vptr[0];
            if constexpr (sizeof(level_t) > 1) {
              if (level_bits > 8) {
                level_val |= static_cast<uint32_t>(vptr[1]) << 8;
                if constexpr (sizeof(level_t) > 2) {
                  if (level_bits > 16) {
                    level_val |= static_cast<uint32_t>(vptr[2]) << 16;
                    if (level_bits > 24) { level_val |= static_cast<uint32_t>(vptr[3]) << 24; }
                  }
                }
              }
            }
            warp_fill(
              output, base_out + seg_lo, base_out + seg_hi, static_cast<level_t>(level_val), lane);
          }
        }
      }
      // Barrier before rewriting the shared tables on the next iteration.
      __syncthreads();

      out_pos_total = base_out + chunk_total;
    }

    int const decoded = out_pos_total - cur_values;
    cur_values        = out_pos_total;
    return decoded;
  }

  __device__ inline int decode_next(int t, int count)
  {
    // Fast path: level_bits == 0 means every level is implicitly 0, so no
    // headers or payloads need parsing. This is a very common case: columns
    // with no nulls (especially non-nested ones) have all-zero definition
    // levels. Handled here so both decode_next_ring and decode_next_chunked
    // stay focused on the general RLE path.
    int const output_count = min(count, total_values - cur_values);
    if (level_bits == 0) {
      int written = 0;
      while (written < output_count) {
        int const batch_size = min(num_rle_stream_decode_threads, output_count - written);
        if (t < batch_size) {
          output[rolling_index<max_output_values>(cur_values + written + t)] = 0;
        }
        written += batch_size;
      }
      cur_values += output_count;
      return output_count;
    }
    if constexpr (use_chunked_expand) {
      return decode_next_chunked(t, count);
    } else {
      return decode_next_ring(t, count);
    }
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

template <typename level_t, int decode_threads, int max_output_values>
using rle_stream_chunked = rle_stream<level_t, decode_threads, max_output_values, true>;

}  // namespace cudf::io::parquet::detail
