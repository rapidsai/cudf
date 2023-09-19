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

#include "page_decode.cuh"

namespace cudf::io::parquet::gpu {

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
//
// DELTA_BYTE_ARRAY encoding (incremental encoding or front compression), is used for BYTE_ARRAY
// columns. For each element in a sequence of strings, a prefix length from the preceding string
// and a suffix is stored. The prefix lengths are DELTA_BINARY_PACKED encoded. The suffixes are
// encoded with DELTA_LENGTH_BYTE_ARRAY encoding, which is a DELTA_BINARY_PACKED list of suffix
// lengths, followed by the concatenated suffix data.

// TODO: The delta encodings use ULEB128 integers, but for now we're only
// using max 64 bits. Need to see what the performance impact is of using
// __int128_t rather than int64_t.
using uleb128_t   = uint64_t;
using zigzag128_t = int64_t;

// we decode one mini-block at a time. max mini-block size seen is 64.
constexpr int delta_rolling_buf_size = 128;

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
  zigzag128_t last_value;      // last value decoded, initialized to first_value from header

  uint32_t values_per_mb;      // block_size / mini_block_count, must be multiple of 32
  uint32_t current_value_idx;  // current value index, initialized to 0 at start of block

  zigzag128_t cur_min_delta;     // min delta for the block
  uint32_t cur_mb;               // index of the current mini-block within the block
  uint8_t const* cur_mb_start;   // pointer to the start of the current mini-block data
  uint8_t const* cur_bitwidths;  // pointer to the bitwidth array in the block

  uleb128_t value[delta_rolling_buf_size];  // circular buffer of delta values

  // returns the number of values encoded in the block data. when all_values is true,
  // account for the first value in the header. otherwise just count the values encoded
  // in the mini-block data.
  constexpr uint32_t num_encoded_values(bool all_values)
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
    last_value       = get_zz128(d_start, d_end);

    current_value_idx = 0;
    values_per_mb     = block_size / mini_block_count;

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

  // decode the current mini-batch of deltas, and convert to values.
  // called by all threads in a warp, currently only one warp supported.
  inline __device__ void calc_mini_block_values(int lane_id)
  {
    using cudf::detail::warp_size;
    if (current_value_idx >= value_count) { return; }

    // need to save first value from header on first pass
    if (current_value_idx == 0) {
      if (lane_id == 0) {
        current_value_idx++;
        value[0] = last_value;
      }
      __syncwarp();
      if (current_value_idx >= value_count) { return; }
    }

    uint32_t const mb_bits = cur_bitwidths[cur_mb];

    // need to do in multiple passes if values_per_mb != 32
    uint32_t const num_pass = values_per_mb / warp_size;

    auto d_start = cur_mb_start;

    for (int i = 0; i < num_pass; i++) {
      // position at end of the current mini-block since the following calculates
      // negative indexes
      d_start += (warp_size * mb_bits) / 8;

      // unpack deltas. modified from version in gpuDecodeDictionaryIndices(), but
      // that one only unpacks up to bitwidths of 24. simplified some since this
      // will always do batches of 32.
      // NOTE: because this needs to handle up to 64 bits, the branching used in the other
      // implementation has been replaced with a loop. While this uses more registers, the
      // looping version is just as fast and easier to read. Might need to revisit this when
      // DELTA_BYTE_ARRAY is implemented.
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
      __shared__ cub::WarpScan<int64_t>::TempStorage temp_storage;
      cub::WarpScan<int64_t>(temp_storage).InclusiveSum(delta, delta);

      // now add first value from header or last value from previous block to get true value
      delta += last_value;
      int const value_idx =
        rolling_index<delta_rolling_buf_size>(current_value_idx + warp_size * i + lane_id);
      value[value_idx] = delta;

      // save value from last lane in warp. this will become the 'first value' added to the
      // deltas calculated in the next iteration (or invocation).
      if (lane_id == warp_size - 1) { last_value = delta; }
      __syncwarp();
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
      if (t < warp_size) {
        calc_mini_block_values(lane_id);
        if (lane_id == 0) { setup_next_mini_block(true); }
      }
      __syncthreads();
    }
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
};

}  // namespace cudf::io::parquet::gpu
