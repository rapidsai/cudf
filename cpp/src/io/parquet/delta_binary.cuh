/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "page_decode.cuh"

namespace cudf::io::parquet::detail {

// DELTA_XXX encoding support
//
// DELTA_BINARY_PACKED is used for INT32 and INT64 data types. Encoding begins with a header
// containing a block size, number of mini-blocks in each block, total value count, and first
// value. The first three are ULEB128 variable length ints, and the last is a zigzag ULEB128
// varint.
//   -- the block size is a multiple of 128
//   -- the mini-block count is chosen so that each mini-block will contain a multiple of 32 values
//   -- the value count includes the first value stored in the header
//
// It seems most Parquet encoders will stick with a block size of 128, and 4 mini-blocks of 32
// elements each. arrow-rs will use a block size of 256 for 64-bit ints.
//
// Following the header are the data blocks. Each block is further divided into mini-blocks, with
// each mini-block having its own encoding bitwidth. Each block begins with a header containing a
// zigzag ULEB128 encoded minimum delta value, followed by an array of uint8 bitwidths, one entry
// per mini-block. While encoding, the lowest delta value is subtracted from all the deltas in the
// block to ensure that all encoded values are positive. The deltas for each mini-block are bit
// packed using the same encoding as the RLE/Bit-Packing Hybrid encoder.

// The DELTA_BINARY_PACKED spec requires the number of values in a mini-block to be a multiple of
// 32. That this equals the warp size is a coincidence the decoders below depend on: they produce
// values in warp_size-wide passes, so warp_size must divide every spec-valid mini-block size.
constexpr int delta_mini_block_size_multiple = 32;
static_assert(delta_mini_block_size_multiple % cudf::detail::warp_size == 0);

// The largest mini-block size the whole-mini-block skip paths (skip_values, skip_values_and_sum,
// and delta_byte_array_decoder::skip) can handle, since they require a full mini-block to be
// resident in the rolling buffer. The main decode loops sub-batch mini-blocks into warp_size-wide
// passes (see decode_next_pass) and support any mini-block size, but produce at most this many
// values per iteration.
constexpr int max_delta_mini_block_size = 64;

// The first pass decodes `values_per_mb` values, and then the second pass does another
// batch of size `values_per_mb`. The largest value for values_per_miniblock among the
// major writers seems to be 64, so 2 * 64 should be good. We save the first value separately
// since it is not encoded in the first mini-block.
// The extra 1 is for the first value, from the block header. It's not stored in the buffer, but it
// still impacts buffer indexing and we need to account for it to avoid race conditions.
constexpr int delta_rolling_buf_size = (2 * max_delta_mini_block_size) + 1;

/**
 * @brief Read a ULEB128 varint integer
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The value read
 */
inline __device__ uleb128_t get_uleb128(uint8_t const*& cur, uint8_t const* end)
{
  uleb128_t v = 0, l = 0, c;
  while (cur < end) {
    c = *cur++;
    v |= (c & 0x7f) << l;
    l += 7;
    if ((c & 0x80) == 0) { return v; }
  }
  return v;
}

/**
 * @brief Read a ULEB128 zig-zag encoded varint integer
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The value read
 */
inline __device__ zigzag128_t get_zz128(uint8_t const*& cur, uint8_t const* end)
{
  uleb128_t u = get_uleb128(cur, end);
  return static_cast<zigzag128_t>((u >> 1u) ^ -static_cast<zigzag128_t>(u & 1));
}

struct delta_binary_decoder {
  uint8_t const* block_start;  // start of data, but updated as data is read
  uint8_t const* block_end;    // end of data
  uleb128_t block_size;        // usually 128, must be multiple of 128
  uleb128_t mini_block_count;  // usually 4, chosen such that block_size/mini_block_count is a
                               // multiple of 32
  uleb128_t value_count;       // total values encoded in the block
  zigzag128_t first_value;     // initial value, stored in the header
  zigzag128_t last_value;      // last value decoded

  uint32_t values_per_mb;      // block_size / mini_block_count, must be multiple of 32
  uint32_t current_value_idx;  // current value index, initialized to 0 at start of block
  uint32_t cur_pass;           // current warp_size-wide pass within the mini-block, used by
                               // decode_next_pass for pipelined single-pass decoding

  zigzag128_t cur_min_delta;     // min delta for the block
  uint32_t cur_mb;               // index of the current mini-block within the block
  uint8_t const* cur_mb_start;   // pointer to the start of the current mini-block data
  uint8_t const* cur_bitwidths;  // pointer to the bitwidth array in the block
  bool error;                    // flag to catch malformed headers

  zigzag128_t value[delta_rolling_buf_size];  // circular buffer of delta values

  // returns the value stored in the `value` array at index
  // `rolling_index<delta_rolling_buf_size>(idx)`. If `idx` is `0`, then return `first_value`.
  __device__ constexpr zigzag128_t value_at(size_type idx)
  {
    return idx == 0 ? first_value : value[rolling_index<delta_rolling_buf_size>(idx)];
  }

  // returns the number of values encoded in the block data. when all_values is true,
  // account for the first value in the header. otherwise just count the values encoded
  // in the mini-block data.
  __device__ constexpr uint32_t num_encoded_values(bool all_values)
  {
    return value_count == 0 ? 0 : all_values ? value_count : value_count - 1;
  }

  // read mini-block header into state object. should only be called from init_binary_block or
  // setup_next_mini_block. header format is:
  //
  // | min delta (int) | bit-width array (1 byte * mini_block_count) |
  //
  // on exit db->cur_mb is 0 and db->cur_mb_start points to the first mini-block of data, or
  // nullptr if out of data.
  // is_decode indicates whether this is being called from initialization code (false) or
  // the actual decoding (true)
  inline __device__ void init_mini_block(bool is_decode)
  {
    cur_mb       = 0;
    cur_mb_start = nullptr;

    if (current_value_idx < num_encoded_values(is_decode)) {
      auto d_start  = block_start;
      cur_min_delta = get_zz128(d_start, block_end);
      cur_bitwidths = d_start;

      d_start += mini_block_count;
      cur_mb_start = d_start;
    }
  }

  // read delta binary header into state object. should be called on thread 0. header format is:
  //
  // | block size (uint) | mini-block count (uint) | value count (uint) | first value (int) |
  //
  // also initializes the first mini-block before exit
  inline __device__ void init_binary_block(uint8_t const* d_start, uint8_t const* d_end)
  {
    block_end        = d_end;
    block_size       = get_uleb128(d_start, d_end);
    mini_block_count = get_uleb128(d_start, d_end);
    value_count      = get_uleb128(d_start, d_end);
    first_value      = get_zz128(d_start, d_end);
    last_value       = first_value;

    current_value_idx = 0;
    cur_pass          = 0;
    error             = false;

    // Validate the header against the DELTA_BINARY_PACKED spec: the mini-block count must evenly
    // divide the block size, and each mini-block must hold a multiple of 32 values. The decoders
    // rely on the latter to advance from one mini-block to the next.
    if (mini_block_count == 0 or block_size == 0 or (block_size % mini_block_count) != 0 or
        ((block_size / mini_block_count) % delta_mini_block_size_multiple) != 0) {
      error         = true;
      value_count   = 0;
      values_per_mb = 1;
      block_start   = d_end;
      cur_mb        = 0;
      cur_mb_start  = d_end;
      cur_bitwidths = d_end;
      return;
    }

    values_per_mb = block_size / mini_block_count;

    // init the first mini-block
    block_start = d_start;

    // only call init if there are actually encoded values
    if (value_count > 1) { init_mini_block(false); }
  }

  // skip to the start of the next mini-block. should only be called on thread 0.
  // calls init_binary_block if currently on the last mini-block in a block.
  // is_decode indicates whether this is being called from initialization code (false) or
  // the actual decoding (true)
  inline __device__ void setup_next_mini_block(bool is_decode)
  {
    if (current_value_idx >= num_encoded_values(is_decode)) { return; }

    current_value_idx += values_per_mb;

    // just set pointer to start of next mini_block
    if (cur_mb < mini_block_count - 1) {
      cur_mb_start += cur_bitwidths[cur_mb] * values_per_mb / 8;
      cur_mb++;
    }
    // out of mini-blocks, start a new block
    else {
      block_start = cur_mb_start + cur_bitwidths[cur_mb] * values_per_mb / 8;
      init_mini_block(is_decode);
    }
  }

  // given start/end pointers in the data, find the end of the binary encoded block. when done,
  // `this` will be initialized with the correct start and end positions. returns the end, which is
  // start of data/next block. should only be called from thread 0.
  inline __device__ uint8_t const* find_end_of_block(uint8_t const* start, uint8_t const* end)
  {
    // read block header
    init_binary_block(start, end);

    // test for no encoded values. a single value will be in the block header.
    if (value_count <= 1) { return block_start; }

    // read mini-block headers and skip over data
    while (current_value_idx < num_encoded_values(false)) {
      setup_next_mini_block(false);
    }
    // calculate the correct end of the block
    auto const* const new_end = cur_mb == 0 ? block_start : cur_mb_start;
    // re-init block with correct end
    init_binary_block(start, new_end);
    return new_end;
  }

  // account for the first value from the block header before the first mini-block is decoded.
  // the first value is not encoded in the mini-block data, but it still occupies index 0 of the
  // value stream. returns true if there are more values to decode after the header value.
  // called by all threads in a single warp.
  inline __device__ bool advance_past_first_value(int lane_id)
  {
    if (current_value_idx >= value_count) { return false; }

    if (current_value_idx == 0) {
      // make sure all threads access current_value_idx above before incrementing
      __syncwarp();
      if (lane_id == 0) { current_value_idx++; }
      __syncwarp();
      if (current_value_idx >= value_count) { return false; }
    }
    return true;
  }

  // decode a single warp_size-wide pass (indexed by `pass`) of the current mini-block and convert
  // the deltas to values. factored out of calc_mini_block_values so the pipelined decoders can
  // produce one pass at a time and keep the rolling buffer small (see decode_next_pass). called by
  // all threads in a single warp.
  inline __device__ void calc_mini_block_pass(uint32_t pass, int lane_id)
  {
    using cudf::detail::warp_size;

    uint32_t const mb_bits = cur_bitwidths[cur_mb];

    // position at the end of this pass's values since the following calculates negative indexes
    auto const d_start = cur_mb_start + (pass + 1) * (warp_size * mb_bits / 8);

    // unpack deltas. modified from version in decode_dictionary_indices(), but
    // that one only unpacks up to bitwidths of 24. simplified some since this
    // will always do batches of 32.
    // NOTE: because this needs to handle up to 64 bits, the branching used in the other
    // implementation has been replaced with a loop. While this uses more registers, the
    // looping version is just as fast and easier to read.
    zigzag128_t delta = 0;
    if (lane_id + current_value_idx < value_count) {
      int32_t ofs      = (lane_id - warp_size) * mb_bits;
      uint8_t const* p = d_start + (ofs >> 3);
      ofs &= 7;
      if (p < block_end) {
        uint32_t c = 8 - ofs;  // 0 - 7 bits
        delta      = (*p++) >> ofs;

        while (c < mb_bits && p < block_end) {
          delta |= static_cast<zigzag128_t>(*p++) << c;
          c += 8;
        }
        delta &= (static_cast<zigzag128_t>(1) << mb_bits) - 1;
      }
    }

    // add min delta to get true delta
    delta += cur_min_delta;

    // do inclusive scan to get value - first_value at each position
    // NOTE: this function-scope shared TempStorage is shared by all warps that call this method
    // concurrently (e.g. the prefix and suffix decoder warps of the DELTA_BYTE_ARRAY kernels).
    // that is safe today because a 32-lane WarpScan over int64_t is shuffle-based and never
    // touches its (empty) TempStorage, but a cub change or a wider scan type could turn this
    // into a race.
    __shared__ cub::WarpScan<int64_t>::TempStorage temp_storage;
    cub::WarpScan<int64_t>(temp_storage).InclusiveSum(delta, delta);

    // now add first value from header or last value from previous pass to get true value
    delta += last_value;
    int const value_idx =
      rolling_index<delta_rolling_buf_size>(current_value_idx + warp_size * pass + lane_id);
    value[value_idx] = delta;

    // save value from last lane in warp. this will become the 'first value' added to the
    // deltas calculated in the next pass (or invocation).
    if (lane_id == warp_size - 1) { last_value = delta; }
    __syncwarp();
  }

  // decode the current mini-batch of deltas, and convert to values.
  // called by all threads in a warp, currently only one warp supported.
  inline __device__ void calc_mini_block_values(int lane_id)
  {
    using cudf::detail::warp_size;

    if (not advance_past_first_value(lane_id)) { return; }

    // need to do in multiple passes if values_per_mb != 32
    uint32_t const num_pass = values_per_mb / warp_size;

    for (uint32_t i = 0; i < num_pass; i++) {
      calc_mini_block_pass(i, lane_id);
    }
  }

  // decodes and skips values until the block containing the value after `skip` is reached.
  // called by all threads in a thread block.
  inline __device__ void skip_values(int skip)
  {
    using cudf::detail::warp_size;
    int const t       = threadIdx.x;
    int const lane_id = t % warp_size;

    while (current_value_idx < skip && current_value_idx < num_encoded_values(true)) {
      // calc_mini_block_values only runs in warp 0, but writes to current_value_idx,
      // so everyone must sync before we diverge
      __syncthreads();
      if (t < warp_size) {
        calc_mini_block_values(lane_id);
        if (lane_id == 0) { setup_next_mini_block(true); }
      }
      __syncthreads();
    }
  }

  // Decodes and skips values until the block containing the value after `skip` is reached.
  // Keeps a running sum of the values and returns that upon exit. Called by all threads in a
  // warp 0. Result is only valid on thread 0.
  // This is intended for use only by the DELTA_LENGTH_BYTE_ARRAY decoder.
  inline __device__ size_t skip_values_and_sum(int skip)
  {
    using cudf::detail::warp_size;
    // DELTA_LENGTH_BYTE_ARRAY lengths are encoded as INT32 by convention (since the PLAIN encoding
    // uses 4-byte lengths).
    using delta_length_type = int32_t;
    using warp_reduce       = cub::WarpReduce<size_t>;
    __shared__ warp_reduce::TempStorage temp_storage;
    int const t = threadIdx.x;

    // initialize sum with first value, which is stored in the block header. cast to
    // `delta_length_type` to ensure the value is interpreted properly before promoting it
    // back to `size_t`.
    size_t sum = static_cast<delta_length_type>(value_at(0));

    // if only skipping one value, we're done already
    if (skip == 1) { return sum; }

    // need to do in multiple passes if values_per_mb != 32
    uint32_t const num_pass = values_per_mb / warp_size;

    while (current_value_idx < skip && current_value_idx < num_encoded_values(true)) {
      calc_mini_block_values(t);

      int const idx = current_value_idx + t;

      for (uint32_t p = 0; p < num_pass; p++) {
        auto const pidx     = idx + p * warp_size;
        size_t const val    = pidx < skip ? static_cast<delta_length_type>(value_at(pidx)) : 0;
        auto const warp_sum = warp_reduce(temp_storage).Sum(val);
        if (t == 0) { sum += warp_sum; }
      }
      if (t == 0) { setup_next_mini_block(true); }
      __syncwarp();
    }

    return sum;
  }

  // decodes the current mini block and stores the values obtained. should only be called by
  // a single warp.
  inline __device__ void decode_batch()
  {
    using cudf::detail::warp_size;
    int const t       = threadIdx.x;
    int const lane_id = t % warp_size;

    // unpack deltas and save in db->value
    calc_mini_block_values(lane_id);

    // set up for next mini-block
    if (lane_id == 0) { setup_next_mini_block(true); }
  }

  // decode the next warp_size-wide pass of the current mini-block into db->value, advancing to the
  // next mini-block once all of its passes have been decoded. Unlike decode_batch(), which decodes
  // an entire mini-block at once, this decodes a single pass so the rolling buffer only needs to
  // hold a fixed number of values (2 * warp_size + 1) regardless of the mini-block size. This lets
  // the reader decode mini-blocks larger than max_delta_mini_block_size. Should only be called by a
  // single warp. NOTE: lane 0's state updates are not synchronized on exit; the caller must
  // synchronize the warp (or block) before the next call so all lanes observe them.
  inline __device__ void decode_next_pass()
  {
    using cudf::detail::warp_size;
    int const t       = threadIdx.x;
    int const lane_id = t % warp_size;

    if (not advance_past_first_value(lane_id)) { return; }

    // unpack one pass of deltas and save in db->value
    calc_mini_block_pass(cur_pass, lane_id);

    // advance within the mini-block; move to the next mini-block once all passes are decoded
    if (lane_id == 0) {
      if (++cur_pass == values_per_mb / warp_size) {
        cur_pass = 0;
        setup_next_mini_block(true);
      }
    }
  }
};

}  // namespace cudf::io::parquet::detail
