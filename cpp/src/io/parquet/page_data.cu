/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include "page_decode.cuh"

#include <io/utilities/column_buffer.hpp>

#include <cudf/detail/utilities/hash_functions.cuh>

namespace cudf {
namespace io {
namespace parquet {
namespace gpu {

namespace {

/**
 * @brief Output a string descriptor
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dstv Pointer to row output data (string descriptor or 32-bit hash)
 */
inline __device__ void gpuOutputString(volatile page_state_s* s,
                                       volatile page_state_buffers_s* sb,
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
inline __device__ void gpuOutputBoolean(volatile page_state_buffers_s* sb,
                                        int src_pos,
                                        uint8_t* dst)
{
  *dst = sb->dict_idx[rolling_index(src_pos)];
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
inline __device__ void gpuOutputInt96Timestamp(volatile page_state_s* s,
                                               volatile page_state_buffers_s* sb,
                                               int src_pos,
                                               int64_t* dst)
{
  using cuda::std::chrono::duration_cast;

  uint8_t const* src8;
  uint32_t dict_pos, dict_size = s->dict_size, ofs;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? sb->dict_idx[rolling_index(src_pos)] : 0;
    src8     = s->dict_base;
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
inline __device__ void gpuOutputInt64Timestamp(volatile page_state_s* s,
                                               volatile page_state_buffers_s* sb,
                                               int src_pos,
                                               int64_t* dst)
{
  uint8_t const* src8;
  uint32_t dict_pos, dict_size = s->dict_size, ofs;
  int64_t ts;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? sb->dict_idx[rolling_index(src_pos)] : 0;
    src8     = s->dict_base;
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
template <typename T>
__device__ void gpuOutputFixedLenByteArrayAsInt(volatile page_state_s* s,
                                                volatile page_state_buffers_s* sb,
                                                int src_pos,
                                                T* dst)
{
  uint32_t const dtype_len_in = s->dtype_len_in;
  uint8_t const* data         = s->dict_base ? s->dict_base : s->data_start;
  uint32_t const pos =
    (s->dict_base ? ((s->dict_bits > 0) ? sb->dict_idx[rolling_index(src_pos)] : 0) : src_pos) *
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
template <typename T>
inline __device__ void gpuOutputFast(volatile page_state_s* s,
                                     volatile page_state_buffers_s* sb,
                                     int src_pos,
                                     T* dst)
{
  uint8_t const* dict;
  uint32_t dict_pos, dict_size = s->dict_size;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? sb->dict_idx[rolling_index(src_pos)] : 0;
    dict     = s->dict_base;
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
static __device__ void gpuOutputGeneric(
  volatile page_state_s* s, volatile page_state_buffers_s* sb, int src_pos, uint8_t* dst8, int len)
{
  uint8_t const* dict;
  uint32_t dict_pos, dict_size = s->dict_size;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? sb->dict_idx[rolling_index(src_pos)] : 0;
    dict     = s->dict_base;
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
 *
 * This function expects the dictionary position to be at 0 and will traverse
 * the entire thing.
 *
 * Operates on a single warp only. Expects t < 32
 *
 * @param s The local page info
 * @param t Thread index
 */
__device__ size_type gpuDecodeTotalPageStringSize(page_state_s* s, int t)
{
  size_type target_pos = s->num_input_values;
  size_type str_len    = 0;
  if (s->dict_base) {
    auto const [new_target_pos, len] = gpuDecodeDictionaryIndices<true>(s, nullptr, target_pos, t);
    target_pos                       = new_target_pos;
    str_len                          = len;
  } else if ((s->col.data_type & 7) == BYTE_ARRAY) {
    str_len = gpuInitStringDescriptors<true>(s, nullptr, target_pos, t);
  }
  if (!t) { *(volatile int32_t*)&s->dict_pos = target_pos; }
  return str_len;
}

/**
 * @brief Update output column sizes for every nesting level based on a batch
 * of incoming decoded definition and repetition level values.
 *
 * If bounds_set is true, computes skipped_values and skipped_leaf_values for the
 * page to indicate where we need to skip to based on min/max row.
 *
 * Operates at the block level.
 *
 * @param s The local page info
 * @param target_value_count The target value count to process up to
 * @param rep Repetition level buffer
 * @param def Definition level buffer
 * @param t Thread index
 * @param bounds_set A boolean indicating whether or not min/max row bounds have been set
 */
template <int lvl_buf_size, typename level_t>
static __device__ void gpuUpdatePageSizes(page_state_s* s,
                                          int target_value_count,
                                          level_t const* const rep,
                                          level_t const* const def,
                                          int t,
                                          bool bounds_set)
{
  // max nesting depth of the column
  int const max_depth = s->col.max_nesting_depth;

  constexpr int num_warps      = preprocess_block_size / 32;
  constexpr int max_batch_size = num_warps * 32;

  using block_reduce = cub::BlockReduce<int, preprocess_block_size>;
  using block_scan   = cub::BlockScan<int, preprocess_block_size>;
  __shared__ union {
    typename block_reduce::TempStorage reduce_storage;
    typename block_scan::TempStorage scan_storage;
  } temp_storage;

  // how many input level values we've processed in the page so far
  int value_count = s->input_value_count;
  // how many rows we've processed in the page so far
  int row_count = s->input_row_count;
  // how many leaf values we've processed in the page so far
  int leaf_count = s->input_leaf_count;
  // whether or not we need to continue checking for the first row
  bool skipped_values_set = s->page.skipped_values >= 0;

  while (value_count < target_value_count) {
    int const batch_size = min(max_batch_size, target_value_count - value_count);

    // start/end depth
    int start_depth, end_depth, d;
    get_nesting_bounds<lvl_buf_size, level_t>(
      start_depth, end_depth, d, s, rep, def, value_count, value_count + batch_size, t);

    // is this thread within row bounds? in the non skip_rows/num_rows case this will always
    // be true.
    int in_row_bounds = 1;

    // if we are in the skip_rows/num_rows case, we need to check against these limits
    if (bounds_set) {
      // get absolute thread row index
      int const is_new_row = start_depth == 0;
      int thread_row_count, block_row_count;
      block_scan(temp_storage.scan_storage)
        .InclusiveSum(is_new_row, thread_row_count, block_row_count);
      __syncthreads();

      // get absolute thread leaf index
      int const is_new_leaf = (d >= s->nesting_info[max_depth - 1].max_def_level);
      int thread_leaf_count, block_leaf_count;
      block_scan(temp_storage.scan_storage)
        .InclusiveSum(is_new_leaf, thread_leaf_count, block_leaf_count);
      __syncthreads();

      // if this thread is in row bounds
      int const row_index = (thread_row_count + row_count) - 1;
      in_row_bounds =
        (row_index >= s->row_index_lower_bound) && (row_index < (s->first_row + s->num_rows));

      // if we have not set skipped values yet, see if we found the first in-bounds row
      if (!skipped_values_set) {
        int local_count, global_count;
        block_scan(temp_storage.scan_storage)
          .InclusiveSum(in_row_bounds, local_count, global_count);
        __syncthreads();

        // we found it
        if (global_count > 0) {
          // this is the thread that represents the first row.
          if (local_count == 1 && in_row_bounds) {
            s->page.skipped_values = value_count + t;
            s->page.skipped_leaf_values =
              leaf_count + (is_new_leaf ? thread_leaf_count - 1 : thread_leaf_count);
          }
          skipped_values_set = true;
        }
      }

      row_count += block_row_count;
      leaf_count += block_leaf_count;
    }

    // increment value counts across all nesting depths
    for (int s_idx = 0; s_idx < max_depth; s_idx++) {
      int const in_nesting_bounds = (s_idx >= start_depth && s_idx <= end_depth && in_row_bounds);
      int const count = block_reduce(temp_storage.reduce_storage).Sum(in_nesting_bounds);
      __syncthreads();
      if (!t) {
        PageNestingInfo* pni = &s->page.nesting[s_idx];
        pni->batch_size += count;
      }
    }

    value_count += batch_size;
  }

  // update final outputs
  if (!t) {
    s->input_value_count = value_count;

    // only used in the skip_rows/num_rows case
    s->input_leaf_count = leaf_count;
    s->input_row_count  = row_count;
  }
}

/**
 * @brief Kernel for computing per-page column size information for all nesting levels.
 *
 * This function will write out the size field for each level of nesting.
 *
 * @param pages List of pages
 * @param chunks List of column chunks
 * @param min_row Row index to start reading at
 * @param num_rows Maximum number of rows to read. Pass as INT_MAX to guarantee reading all rows
 * @param is_base_pass Whether or not this is the base pass.  We first have to compute
 * the full size information of every page before we come through in a second (trim) pass
 * to determine what subset of rows in this page we should be reading
 * @param compute_string_sizes Whether or not we should be computing string sizes
 * (PageInfo::str_bytes) as part of the pass
 */
template <int lvl_buf_size, typename level_t>
__global__ void __launch_bounds__(preprocess_block_size)
  gpuComputePageSizes(PageInfo* pages,
                      device_span<ColumnChunkDesc const> chunks,
                      size_t min_row,
                      size_t num_rows,
                      bool is_base_pass,
                      bool compute_string_sizes)
{
  __shared__ __align__(16) page_state_s state_g;

  page_state_s* const s = &state_g;
  int page_idx          = blockIdx.x;
  int t                 = threadIdx.x;
  PageInfo* pp          = &pages[page_idx];

  // whether or not we have repetition levels (lists)
  bool has_repetition = chunks[pp->chunk_idx].max_level[level_type::REPETITION] > 0;

  // the level stream decoders
  __shared__ rle_run<level_t> def_runs[run_buffer_size];
  __shared__ rle_run<level_t> rep_runs[run_buffer_size];
  rle_stream<level_t> decoders[level_type::NUM_LEVEL_TYPES] = {{def_runs}, {rep_runs}};

  // setup page info
  if (!setupLocalPageInfo(s, pp, chunks, min_row, num_rows, all_types_filter{}, false)) { return; }

  // initialize the stream decoders (requires values computed in setupLocalPageInfo)
  int const max_batch_size = lvl_buf_size;
  level_t* rep             = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::REPETITION]);
  level_t* def             = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  decoders[level_type::DEFINITION].init(s->col.level_bits[level_type::DEFINITION],
                                        s->abs_lvl_start[level_type::DEFINITION],
                                        s->abs_lvl_end[level_type::DEFINITION],
                                        max_batch_size,
                                        def,
                                        s->page.num_input_values);
  if (has_repetition) {
    decoders[level_type::REPETITION].init(s->col.level_bits[level_type::REPETITION],
                                          s->abs_lvl_start[level_type::REPETITION],
                                          s->abs_lvl_end[level_type::REPETITION],
                                          max_batch_size,
                                          rep,
                                          s->page.num_input_values);
  }
  __syncthreads();

  if (!t) {
    s->page.skipped_values      = -1;
    s->page.skipped_leaf_values = 0;
    s->page.str_bytes           = 0;
    s->input_row_count          = 0;
    s->input_value_count        = 0;

    // in the base pass, we're computing the number of rows, make sure we visit absolutely
    // everything
    if (is_base_pass) {
      s->first_row             = 0;
      s->num_rows              = INT_MAX;
      s->row_index_lower_bound = -1;
    }
  }

  // we only need to preprocess hierarchies with repetition in them (ie, hierarchies
  // containing lists anywhere within).
  compute_string_sizes =
    compute_string_sizes && ((s->col.data_type & 7) == BYTE_ARRAY && s->dtype_len != 4);

  // early out optimizations:

  // - if this is a flat hierarchy (no lists) and is not a string column. in this case we don't need
  // to do the expensive work of traversing the level data to determine sizes.  we can just compute
  // it directly.
  if (!has_repetition && !compute_string_sizes) {
    int depth = 0;
    while (depth < s->page.num_output_nesting_levels) {
      auto const thread_depth = depth + t;
      if (thread_depth < s->page.num_output_nesting_levels) {
        if (is_base_pass) { pp->nesting[thread_depth].size = pp->num_input_values; }
        pp->nesting[thread_depth].batch_size = pp->num_input_values;
      }
      depth += blockDim.x;
    }
    return;
  }

  // in the trim pass, for anything with lists, we only need to fully process bounding pages (those
  // at the beginning or the end of the row bounds)
  if (!is_base_pass && !is_bounds_page(s, min_row, num_rows, has_repetition)) {
    int depth = 0;
    while (depth < s->page.num_output_nesting_levels) {
      auto const thread_depth = depth + t;
      if (thread_depth < s->page.num_output_nesting_levels) {
        // if we are not a bounding page (as checked above) then we are either
        // returning all rows/values from this page, or 0 of them
        pp->nesting[thread_depth].batch_size =
          (s->num_rows == 0 && !is_page_contained(s, min_row, num_rows))
            ? 0
            : pp->nesting[thread_depth].size;
      }
      depth += blockDim.x;
    }
    return;
  }

  // zero sizes
  int depth = 0;
  while (depth < s->page.num_output_nesting_levels) {
    auto const thread_depth = depth + t;
    if (thread_depth < s->page.num_output_nesting_levels) {
      s->page.nesting[thread_depth].batch_size = 0;
    }
    depth += blockDim.x;
  }
  __syncthreads();

  // the core loop. decode batches of level stream data using rle_stream objects
  // and pass the results to gpuUpdatePageSizes
  int processed = 0;
  while (processed < s->page.num_input_values) {
    // TODO:  it would not take much more work to make it so that we could run both of these
    // decodes concurrently. there are a couple of shared variables internally that would have to
    // get dealt with but that's about it.
    if (has_repetition) {
      decoders[level_type::REPETITION].decode_next(t);
      __syncthreads();
    }
    // the # of rep/def levels will always be the same size
    processed += decoders[level_type::DEFINITION].decode_next(t);
    __syncthreads();

    // update page sizes
    gpuUpdatePageSizes<lvl_buf_size>(s, processed, rep, def, t, !is_base_pass);
    __syncthreads();
  }

  // retrieve total string size.
  // TODO: make this block-based instead of just 1 warp
  if (compute_string_sizes) {
    if (t < 32) { s->page.str_bytes = gpuDecodeTotalPageStringSize(s, t); }
  }

  // update output results:
  // - real number of rows for the whole page
  // - nesting sizes for the whole page
  // - skipped value information for trimmed pages
  // - string bytes
  if (is_base_pass) {
    // nesting level 0 is the root column, so the size is also the # of rows
    if (!t) { pp->num_rows = s->page.nesting[0].batch_size; }

    // store off this batch size as the "full" size
    int depth = 0;
    while (depth < s->page.num_output_nesting_levels) {
      auto const thread_depth = depth + t;
      if (thread_depth < s->page.num_output_nesting_levels) {
        pp->nesting[thread_depth].size = pp->nesting[thread_depth].batch_size;
      }
      depth += blockDim.x;
    }
  }

  if (!t) {
    pp->skipped_values      = s->page.skipped_values;
    pp->skipped_leaf_values = s->page.skipped_leaf_values;
    pp->str_bytes           = s->page.str_bytes;
  }
}

/**
 * @brief Kernel for computing the column data stored in the pages
 *
 * This function will write the page data and the page data's validity to the
 * output specified in the page's column chunk. If necessary, additional
 * conversion will be performed to translate from the Parquet datatype to
 * desired output datatype (ex. 32-bit to 16-bit, string to hash).
 *
 * @param pages List of pages
 * @param chunks List of column chunks
 * @param min_row Row index to start reading at
 * @param num_rows Maximum number of rows to read
 */
template <int lvl_buf_size, typename level_t>
__global__ void __launch_bounds__(decode_block_size) gpuDecodePageData(
  PageInfo* pages, device_span<ColumnChunkDesc const> chunks, size_t min_row, size_t num_rows)
{
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16) page_state_buffers_s state_buffers;

  page_state_s* const s          = &state_g;
  page_state_buffers_s* const sb = &state_buffers;
  int page_idx                   = blockIdx.x;
  int t                          = threadIdx.x;
  int out_thread0;
  [[maybe_unused]] null_count_back_copier _{s, t};

  if (!setupLocalPageInfo(
        s, &pages[page_idx], chunks, min_row, num_rows, non_string_filter{chunks}, true)) {
    return;
  }

  bool const has_repetition = s->col.max_level[level_type::REPETITION] > 0;

  if (s->dict_base) {
    out_thread0 = (s->dict_bits > 0) ? 64 : 32;
  } else {
    out_thread0 =
      ((s->col.data_type & 7) == BOOLEAN || (s->col.data_type & 7) == BYTE_ARRAY) ? 64 : 32;
  }

  PageNestingDecodeInfo* nesting_info_base = s->nesting_info;

  __shared__ level_t rep[non_zero_buffer_size];  // circular buffer of repetition level values
  __shared__ level_t def[non_zero_buffer_size];  // circular buffer of definition level values

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t skipped_leaf_values = s->page.skipped_leaf_values;
  while (!s->error && (s->input_value_count < s->num_input_values || s->src_pos < s->nz_count)) {
    int target_pos;
    int src_pos = s->src_pos;

    if (t < out_thread0) {
      target_pos = min(src_pos + 2 * (decode_block_size - out_thread0),
                       s->nz_count + (decode_block_size - out_thread0));
    } else {
      target_pos = min(s->nz_count, src_pos + decode_block_size - out_thread0);
      if (out_thread0 > 32) { target_pos = min(target_pos, s->dict_pos); }
    }
    __syncthreads();
    if (t < 32) {
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<lvl_buf_size, level_t>(s, sb, target_pos, rep, def, t);
    } else if (t < out_thread0) {
      // skipped_leaf_values will always be 0 for flat hierarchies.
      uint32_t src_target_pos = target_pos + skipped_leaf_values;

      // WARP1: Decode dictionary indices, booleans or string positions
      if (s->dict_base) {
        src_target_pos = gpuDecodeDictionaryIndices<false>(s, sb, src_target_pos, t & 0x1f).first;
      } else if ((s->col.data_type & 7) == BOOLEAN) {
        src_target_pos = gpuDecodeRleBooleans(s, sb, src_target_pos, t & 0x1f);
      } else if ((s->col.data_type & 7) == BYTE_ARRAY) {
        gpuInitStringDescriptors<false>(s, sb, src_target_pos, t & 0x1f);
      }
      if (t == 32) { *(volatile int32_t*)&s->dict_pos = src_target_pos; }
    } else {
      // WARP1..WARP3: Decode values
      int const dtype = s->col.data_type & 7;
      src_pos += t - out_thread0;

      // the position in the output column/buffer
      int dst_pos = sb->nz_idx[rolling_index(src_pos)];

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

      // target_pos will always be properly bounded by num_rows, but dst_pos may be negative (values
      // before first_row) in the flat hierarchy case.
      if (src_pos < target_pos && dst_pos >= 0) {
        // src_pos represents the logical row position we want to read from. But in the case of
        // nested hierarchies, there is no 1:1 mapping of rows to values.  So our true read position
        // has to take into account the # of values we have to skip in the page to get to the
        // desired logical row.  For flat hierarchies, skipped_leaf_values will always be 0.
        uint32_t val_src_pos = src_pos + skipped_leaf_values;

        // nesting level that is storing actual leaf values
        int leaf_level_index = s->col.max_nesting_depth - 1;

        uint32_t dtype_len = s->dtype_len;
        void* dst =
          nesting_info_base[leaf_level_index].data_out + static_cast<size_t>(dst_pos) * dtype_len;
        if (dtype == BYTE_ARRAY) {
          if (s->col.converted_type == DECIMAL) {
            auto const [ptr, len]        = gpuGetStringData(s, sb, val_src_pos);
            auto const decimal_precision = s->col.decimal_precision;
            if (decimal_precision <= MAX_DECIMAL32_PRECISION) {
              gpuOutputByteArrayAsInt(ptr, len, static_cast<int32_t*>(dst));
            } else if (decimal_precision <= MAX_DECIMAL64_PRECISION) {
              gpuOutputByteArrayAsInt(ptr, len, static_cast<int64_t*>(dst));
            } else {
              gpuOutputByteArrayAsInt(ptr, len, static_cast<__int128_t*>(dst));
            }
          } else {
            gpuOutputString(s, sb, val_src_pos, dst);
          }
        } else if (dtype == BOOLEAN) {
          gpuOutputBoolean(sb, val_src_pos, static_cast<uint8_t*>(dst));
        } else if (s->col.converted_type == DECIMAL) {
          switch (dtype) {
            case INT32: gpuOutputFast(s, sb, val_src_pos, static_cast<uint32_t*>(dst)); break;
            case INT64: gpuOutputFast(s, sb, val_src_pos, static_cast<uint2*>(dst)); break;
            default:
              if (s->dtype_len_in <= sizeof(int32_t)) {
                gpuOutputFixedLenByteArrayAsInt(s, sb, val_src_pos, static_cast<int32_t*>(dst));
              } else if (s->dtype_len_in <= sizeof(int64_t)) {
                gpuOutputFixedLenByteArrayAsInt(s, sb, val_src_pos, static_cast<int64_t*>(dst));
              } else {
                gpuOutputFixedLenByteArrayAsInt(s, sb, val_src_pos, static_cast<__int128_t*>(dst));
              }
              break;
          }
        } else if (dtype == INT96) {
          gpuOutputInt96Timestamp(s, sb, val_src_pos, static_cast<int64_t*>(dst));
        } else if (dtype_len == 8) {
          if (s->dtype_len_in == 4) {
            // Reading INT32 TIME_MILLIS into 64-bit DURATION_MILLISECONDS
            // TIME_MILLIS is the only duration type stored as int32:
            // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#deprecated-time-convertedtype
            gpuOutputFast(s, sb, val_src_pos, static_cast<uint32_t*>(dst));
          } else if (s->ts_scale) {
            gpuOutputInt64Timestamp(s, sb, val_src_pos, static_cast<int64_t*>(dst));
          } else {
            gpuOutputFast(s, sb, val_src_pos, static_cast<uint2*>(dst));
          }
        } else if (dtype_len == 4) {
          gpuOutputFast(s, sb, val_src_pos, static_cast<uint32_t*>(dst));
        } else {
          gpuOutputGeneric(s, sb, val_src_pos, static_cast<uint8_t*>(dst), dtype_len);
        }
      }

      if (t == out_thread0) { *(volatile int32_t*)&s->src_pos = target_pos; }
    }
    __syncthreads();
  }
}

}  // anonymous namespace

/**
 * @copydoc cudf::io::parquet::gpu::ComputePageSizes
 */
void ComputePageSizes(cudf::detail::hostdevice_vector<PageInfo>& pages,
                      cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
                      size_t min_row,
                      size_t num_rows,
                      bool compute_num_rows,
                      bool compute_string_sizes,
                      int level_type_size,
                      rmm::cuda_stream_view stream)
{
  dim3 dim_block(preprocess_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  // computes:
  // PageNestingInfo::size for each level of nesting, for each page.
  // This computes the size for the entire page, not taking row bounds into account.
  // If uses_custom_row_bounds is set to true, we have to do a second pass later that "trims"
  // the starting and ending read values to account for these bounds.
  if (level_type_size == 1) {
    gpuComputePageSizes<LEVEL_DECODE_BUF_SIZE, uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, compute_num_rows, compute_string_sizes);
  } else {
    gpuComputePageSizes<LEVEL_DECODE_BUF_SIZE, uint16_t>
      <<<dim_grid, dim_block, 0, stream.value()>>>(
        pages.device_ptr(), chunks, min_row, num_rows, compute_num_rows, compute_string_sizes);
  }
}

/**
 * @copydoc cudf::io::parquet::gpu::DecodePageData
 */
void __host__ DecodePageData(cudf::detail::hostdevice_vector<PageInfo>& pages,
                             cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
                             size_t num_rows,
                             size_t min_row,
                             int level_type_size,
                             rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 dim_block(decode_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    gpuDecodePageData<non_zero_buffer_size, uint8_t>
      <<<dim_grid, dim_block, 0, stream.value()>>>(pages.device_ptr(), chunks, min_row, num_rows);
  } else {
    gpuDecodePageData<non_zero_buffer_size, uint16_t>
      <<<dim_grid, dim_block, 0, stream.value()>>>(pages.device_ptr(), chunks, min_row, num_rows);
  }
}

}  // namespace gpu
}  // namespace parquet
}  // namespace io
}  // namespace cudf
