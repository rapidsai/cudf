/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include "orc_common.hpp"
#include "orc_gpu.hpp"

#include <cudf/table/table_device_view.cuh>
#include <io/utilities/block_utils.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace cudf {
namespace io {
namespace orc {
namespace gpu {
constexpr int init_hash_bits = 12;

struct dictinit_state_s {
  uint32_t nnz;
  uint32_t total_dupes;
  DictionaryChunk chunk;
  volatile uint32_t scratch_red[32];
  uint32_t* dict;
  union {
    uint16_t u16[1 << (init_hash_bits)];
    uint32_t u32[1 << (init_hash_bits - 1)];
  } map;
};

/**
 * @brief Return a 12-bit hash from a string
 */
static inline __device__ uint32_t hash_string(const string_view val)
{
  if (val.empty()) {
    return 0;
  } else {
    char const* ptr = val.data();
    uint32_t len    = val.size_bytes();
    return (ptr[0] + (ptr[len - 1] << 5) + (len << 10)) & ((1 << init_hash_bits) - 1);
  }
}

/**
 * @brief Fill dictionary with the indices of non-null rows
 *
 * @param[in,out] s dictionary builder state
 * @param[in] t thread id
 * @param[in] temp_storage shared memory storage to scan non-null positions
 */
template <int block_size, typename Storage>
static __device__ void LoadNonNullIndices(volatile dictinit_state_s* s,
                                          int t,
                                          Storage& temp_storage)
{
  if (t == 0) { s->nnz = 0; }
  for (uint32_t i = 0; i < s->chunk.num_rows; i += block_size) {
    const uint32_t* valid_map = s->chunk.leaf_column->null_mask();
    auto column_offset        = s->chunk.leaf_column->offset();
    uint32_t is_valid, nz_pos;
    if (t < block_size / 32) {
      if (!valid_map) {
        s->scratch_red[t] = 0xffffffffu;
      } else {
        uint32_t const row   = s->chunk.start_row + i + t * 32;
        auto const chunk_end = s->chunk.start_row + s->chunk.num_rows;

        auto const valid_map_idx = (row + column_offset) / 32;
        uint32_t valid           = (row < chunk_end) ? valid_map[valid_map_idx] : 0;

        auto const rows_in_next_word = (row + column_offset) & 0x1f;
        if (rows_in_next_word != 0) {
          auto const rows_in_current_word = 32 - rows_in_next_word;
          // Read next word if any rows are within the chunk
          uint32_t const valid_next =
            (row + rows_in_current_word < chunk_end) ? valid_map[valid_map_idx + 1] : 0;
          valid = __funnelshift_r(valid, valid_next, rows_in_next_word);
        }
        s->scratch_red[t] = valid;
      }
    }
    __syncthreads();
    is_valid = (i + t < s->chunk.num_rows) ? (s->scratch_red[t >> 5] >> (t & 0x1f)) & 1 : 0;
    uint32_t tmp_nnz;
    cub::BlockScan<uint32_t, block_size, cub::BLOCK_SCAN_WARP_SCANS>(temp_storage)
      .ExclusiveSum(is_valid, nz_pos, tmp_nnz);
    nz_pos += s->nnz;
    __syncthreads();
    if (!t) { s->nnz += tmp_nnz; }
    if (is_valid) { s->dict[nz_pos] = i + t; }
    __syncthreads();
  }
}

/**
 * @brief Gather all non-NULL string rows and compute total character data size
 */
// blockDim {block_size,1,1}
template <int block_size>
__global__ void __launch_bounds__(block_size, 2)
  gpuInitDictionaryIndices(device_2dspan<DictionaryChunk> chunks,
                           device_span<orc_column_device_view const> orc_columns,
                           device_span<device_span<uint32_t>> dict_data,
                           device_span<device_span<uint32_t>> dict_index,
                           device_span<device_span<uint32_t>> tmp_indices,
                           device_2dspan<rowgroup_rows const> rowgroup_bounds,
                           device_span<uint32_t const> str_col_indexes)
{
  __shared__ __align__(16) dictinit_state_s state_g;

  using block_reduce = cub::BlockReduce<uint32_t, block_size>;
  using block_scan   = cub::BlockScan<uint32_t, block_size, cub::BLOCK_SCAN_WARP_SCANS>;

  __shared__ union {
    typename block_reduce::TempStorage reduce_storage;
    typename block_scan::TempStorage scan_storage;
  } temp_storage;

  dictinit_state_s* const s = &state_g;
  // Index of the column in the `str_col_indexes` array
  uint32_t const str_col_idx = blockIdx.x;
  // Index of the column in the `orc_columns` array
  auto const col_idx      = str_col_indexes[str_col_idx];
  uint32_t group_id       = blockIdx.y;
  auto const num_str_cols = str_col_indexes.size();
  uint32_t nnz, start_row, dict_char_count;
  int t = threadIdx.x;

  if (t == 0) {
    s->chunk             = chunks[group_id][str_col_idx];
    s->chunk.leaf_column = &orc_columns[col_idx];
    s->chunk.dict_data   = dict_data[str_col_idx].data() + rowgroup_bounds[group_id][col_idx].begin;
    s->chunk.dict_index  = dict_index[str_col_idx].data();
    s->chunk.start_row   = rowgroup_bounds[group_id][col_idx].begin;
    s->chunk.num_rows    = rowgroup_bounds[group_id][col_idx].size();
    s->dict              = tmp_indices[str_col_idx].data() + s->chunk.start_row;
  }
  for (uint32_t i = 0; i < sizeof(s->map) / sizeof(uint32_t); i += block_size) {
    if (i + t < sizeof(s->map) / sizeof(uint32_t)) s->map.u32[i + t] = 0;
  }
  __syncthreads();
  // First, take care of NULLs, and count how many strings we have (TODO: bypass this step when
  // there are no nulls)
  LoadNonNullIndices<block_size>(s, t, temp_storage.scan_storage);
  // Sum the lengths of all the strings
  if (t == 0) {
    s->chunk.string_char_count = 0;
    s->total_dupes             = 0;
  }
  nnz              = s->nnz;
  auto t_dict_data = s->chunk.dict_data;
  start_row        = s->chunk.start_row;
  for (uint32_t i = 0; i < nnz; i += block_size) {
    uint32_t ck_row = 0;
    uint32_t hash   = 0;
    uint32_t len    = 0;
    if (i + t < nnz) {
      ck_row                 = s->dict[i + t];
      string_view string_val = s->chunk.leaf_column->element<string_view>(ck_row + start_row);
      len                    = static_cast<uint32_t>(string_val.size_bytes());
      hash                   = hash_string(string_val);
    }
    len = block_reduce(temp_storage.reduce_storage).Sum(len);
    if (t == 0) s->chunk.string_char_count += len;
    if (i + t < nnz) {
      atomicAdd(&s->map.u32[hash >> 1], 1 << ((hash & 1) ? 16 : 0));
      t_dict_data[i + t] = start_row + ck_row;
    }
    __syncthreads();
  }
  // Reorder the 16-bit local indices according to the hash value of the strings
  static_assert((init_hash_bits == 12), "Hardcoded for init_hash_bits=12");
  {
    // Cumulative sum of hash map counts
    uint32_t count01 = s->map.u32[t * 4 + 0];
    uint32_t count23 = s->map.u32[t * 4 + 1];
    uint32_t count45 = s->map.u32[t * 4 + 2];
    uint32_t count67 = s->map.u32[t * 4 + 3];
    uint32_t sum01   = count01 + (count01 << 16);
    uint32_t sum23   = count23 + (count23 << 16);
    uint32_t sum45   = count45 + (count45 << 16);
    uint32_t sum67   = count67 + (count67 << 16);
    sum23 += (sum01 >> 16) * 0x10001;
    sum45 += (sum23 >> 16) * 0x10001;
    sum67 += (sum45 >> 16) * 0x10001;
    uint32_t sum_w = sum67 >> 16;
    block_scan(temp_storage.scan_storage).InclusiveSum(sum_w, sum_w);
    __syncthreads();
    sum_w                 = (sum_w - (sum67 >> 16)) * 0x10001;
    s->map.u32[t * 4 + 0] = sum_w + sum01 - count01;
    s->map.u32[t * 4 + 1] = sum_w + sum23 - count23;
    s->map.u32[t * 4 + 2] = sum_w + sum45 - count45;
    s->map.u32[t * 4 + 3] = sum_w + sum67 - count67;
    __syncthreads();
  }
  // Put the indices back in hash order
  for (uint32_t i = 0; i < nnz; i += block_size) {
    uint32_t ck_row  = 0;
    uint32_t hash    = 0;
    uint32_t pos     = 0;
    uint32_t pos_old = 0;
    uint32_t sh      = 0;
    if (i + t < nnz) {
      ck_row                 = t_dict_data[i + t] - start_row;
      string_view string_val = s->chunk.leaf_column->element<string_view>(ck_row + start_row);
      hash                   = hash_string(string_val);
      sh                     = (hash & 1) ? 16 : 0;
      pos_old                = s->map.u16[hash];
    }
    // The isolation of the atomicAdd, along with pos_old/pos_new is to guarantee deterministic
    // behavior for the first row in the hash map that will be used for early duplicate detection
    __syncthreads();
    if (i + t < nnz) {
      pos          = (atomicAdd(&s->map.u32[hash >> 1], 1 << sh) >> sh) & 0xffff;
      s->dict[pos] = ck_row;
    }
    __syncthreads();
    bool collision         = false;
    uint32_t colliding_row = 0;
    uint32_t pos_new       = 0;
    if (i + t < nnz) {
      pos_new   = s->map.u16[hash];
      collision = (pos != pos_old && pos_new > pos_old + 1);
      if (collision) { colliding_row = s->dict[pos_old]; }
    }
    __syncthreads();
    if (collision) { atomicMin(s->dict + pos_old, ck_row); }

    __syncthreads();
    // Resolve collision
    if (collision && ck_row == s->dict[pos_old]) { s->dict[pos] = colliding_row; }
  }
  __syncthreads();
  // Now that the strings are ordered by hash, compare every string with the first entry in the hash
  // map, the position of the first string can be inferred from the hash map counts
  dict_char_count = 0;
  for (uint32_t i = 0; i < nnz; i += block_size) {
    uint32_t ck_row = 0, ck_row_ref = 0, is_dupe = 0;
    if (i + t < nnz) {
      ck_row                   = s->dict[i + t];
      string_view string_value = s->chunk.leaf_column->element<string_view>(ck_row + start_row);
      auto const string_length = static_cast<uint32_t>(string_value.size_bytes());
      auto const hash          = hash_string(string_value);
      ck_row_ref               = s->dict[(hash > 0) ? s->map.u16[hash - 1] : 0];
      if (ck_row_ref != ck_row) {
        string_view reference_string =
          s->chunk.leaf_column->element<string_view>(ck_row_ref + start_row);
        is_dupe = (string_value == reference_string);
        dict_char_count += (is_dupe) ? 0 : string_length;
      }
    }
    uint32_t dupes_in_block;
    uint32_t dupes_before;
    block_scan(temp_storage.scan_storage).InclusiveSum(is_dupe, dupes_before, dupes_in_block);
    dupes_before += s->total_dupes;
    __syncthreads();
    if (!t) { s->total_dupes += dupes_in_block; }
    if (i + t < nnz) {
      if (!is_dupe) {
        t_dict_data[i + t - dupes_before] = ck_row + start_row;
      } else {
        s->chunk.dict_index[ck_row + start_row] = (ck_row_ref + start_row) | (1u << 31);
      }
    }
  }
  // temp_storage is being used twice, so make sure there is `__syncthreads()` between them
  // while making any future changes.
  dict_char_count = block_reduce(temp_storage.reduce_storage).Sum(dict_char_count);
  if (!t) {
    chunks[group_id][str_col_idx].num_strings       = nnz;
    chunks[group_id][str_col_idx].string_char_count = s->chunk.string_char_count;
    chunks[group_id][str_col_idx].num_dict_strings  = nnz - s->total_dupes;
    chunks[group_id][str_col_idx].dict_char_count   = dict_char_count;
    chunks[group_id][str_col_idx].leaf_column       = s->chunk.leaf_column;

    chunks[group_id][str_col_idx].dict_data  = s->chunk.dict_data;
    chunks[group_id][str_col_idx].dict_index = s->chunk.dict_index;
    chunks[group_id][str_col_idx].start_row  = s->chunk.start_row;
    chunks[group_id][str_col_idx].num_rows   = s->chunk.num_rows;
  }
}

/**
 * @brief In-place concatenate dictionary data for all chunks in each stripe
 *
 * @param[in] stripes StripeDictionary device array [stripe][column]
 * @param[in] chunks DictionaryChunk device array [rowgroup][column]
 * @param[in] num_columns Number of columns
 */
// blockDim {1024,1,1}
__global__ void __launch_bounds__(1024)
  gpuCompactChunkDictionaries(device_2dspan<StripeDictionary> stripes,
                              device_2dspan<DictionaryChunk const> chunks)
{
  __shared__ __align__(16) StripeDictionary stripe_g;
  __shared__ __align__(16) DictionaryChunk chunk_g;
  __shared__ const uint32_t* volatile ck_curptr_g;
  __shared__ uint32_t volatile ck_curlen_g;

  uint32_t col_id    = blockIdx.x;
  uint32_t stripe_id = blockIdx.y;
  uint32_t chunk_len;
  int t = threadIdx.x;
  const uint32_t* src;
  uint32_t* dst;

  if (t == 0) stripe_g = stripes[stripe_id][col_id];
  __syncthreads();
  if (!stripe_g.dict_data) { return; }
  if (t == 0) chunk_g = chunks[stripe_g.start_chunk][col_id];
  __syncthreads();
  dst = stripe_g.dict_data + chunk_g.num_dict_strings;
  for (uint32_t g = 1; g < stripe_g.num_chunks; g++) {
    if (!t) {
      src         = chunks[stripe_g.start_chunk + g][col_id].dict_data;
      chunk_len   = chunks[stripe_g.start_chunk + g][col_id].num_dict_strings;
      ck_curptr_g = src;
      ck_curlen_g = chunk_len;
    }
    __syncthreads();
    src       = ck_curptr_g;
    chunk_len = ck_curlen_g;
    if (src != dst) {
      for (uint32_t i = 0; i < chunk_len; i += 1024) {
        uint32_t idx = (i + t < chunk_len) ? src[i + t] : 0;
        __syncthreads();
        if (i + t < chunk_len) dst[i + t] = idx;
      }
    }
    dst += chunk_len;
    __syncthreads();
  }
}

struct build_state_s {
  uint32_t total_dupes;
  StripeDictionary stripe;
  volatile uint32_t scratch_red[32];
};

/**
 * @brief Eliminate duplicates in-place and generate column dictionary index
 *
 * @param[in] stripes StripeDictionary device array [stripe][column]
 * @param[in] num_columns Number of string columns
 */
// NOTE: Prone to poor utilization on small datasets due to 1 block per dictionary
// blockDim {1024,1,1}
template <int block_size>
__global__ void __launch_bounds__(block_size)
  gpuBuildStripeDictionaries(device_2dspan<StripeDictionary> stripes)
{
  __shared__ __align__(16) build_state_s state_g;
  using block_reduce = cub::BlockReduce<uint32_t, block_size>;
  using block_scan   = cub::BlockScan<uint32_t, block_size, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ union {
    typename block_reduce::TempStorage reduce_storage;
    typename block_scan::TempStorage scan_storage;
  } temp_storage;

  build_state_s* const s = &state_g;
  uint32_t col_id        = blockIdx.x;
  uint32_t stripe_id     = blockIdx.y;
  uint32_t num_strings;
  uint32_t *dict_data, *dict_index;
  uint32_t dict_char_count;
  int t = threadIdx.x;

  if (t == 0) s->stripe = stripes[stripe_id][col_id];
  if (t == 31 * 32) { s->total_dupes = 0; }
  __syncthreads();
  num_strings = s->stripe.num_strings;
  dict_data   = s->stripe.dict_data;
  if (!dict_data) return;
  dict_index                 = s->stripe.dict_index;
  string_view current_string = string_view::min();
  dict_char_count            = 0;
  for (uint32_t i = 0; i < num_strings; i += block_size) {
    uint32_t cur     = (i + t < num_strings) ? dict_data[i + t] : 0;
    uint32_t cur_len = 0;
    bool is_dupe     = false;
    if (i + t < num_strings) {
      current_string = s->stripe.leaf_column->element<string_view>(cur);
      cur_len        = current_string.size_bytes();
    }
    if (i + t != 0 && i + t < num_strings) {
      uint32_t prev = dict_data[i + t - 1];
      is_dupe       = (current_string == (s->stripe.leaf_column->element<string_view>(prev)));
    }
    dict_char_count += (is_dupe) ? 0 : cur_len;
    uint32_t dupes_in_block;
    uint32_t dupes_before;
    block_scan(temp_storage.scan_storage).InclusiveSum(is_dupe, dupes_before, dupes_in_block);
    dupes_before += s->total_dupes;
    __syncthreads();
    if (!t) { s->total_dupes += dupes_in_block; }
    if (i + t < num_strings) {
      dict_index[cur] = i + t - dupes_before;
      if (!is_dupe && dupes_before != 0) { dict_data[i + t - dupes_before] = cur; }
    }
    __syncthreads();
  }
  dict_char_count = block_reduce(temp_storage.reduce_storage).Sum(dict_char_count);
  if (t == 0) {
    stripes[stripe_id][col_id].num_strings     = num_strings - s->total_dupes;
    stripes[stripe_id][col_id].dict_char_count = dict_char_count;
  }
}

void InitDictionaryIndices(device_span<orc_column_device_view const> orc_columns,
                           device_2dspan<DictionaryChunk> chunks,
                           device_span<device_span<uint32_t>> dict_data,
                           device_span<device_span<uint32_t>> dict_index,
                           device_span<device_span<uint32_t>> tmp_indices,
                           device_2dspan<rowgroup_rows const> rowgroup_bounds,
                           device_span<uint32_t const> str_col_indexes,
                           rmm::cuda_stream_view stream)
{
  static constexpr int block_size = 512;
  dim3 dim_block(block_size, 1);
  dim3 dim_grid(str_col_indexes.size(), rowgroup_bounds.size().first);
  gpuInitDictionaryIndices<block_size><<<dim_grid, dim_block, 0, stream.value()>>>(
    chunks, orc_columns, dict_data, dict_index, tmp_indices, rowgroup_bounds, str_col_indexes);
}

/**
 * @copydoc cudf::io::orc::gpu::BuildStripeDictionaries
 */
void BuildStripeDictionaries(device_2dspan<StripeDictionary> d_stripes_dicts,
                             host_2dspan<StripeDictionary const> h_stripe_dicts,
                             device_2dspan<DictionaryChunk const> chunks,
                             rmm::cuda_stream_view stream)
{
  auto const num_stripes = h_stripe_dicts.size().first;
  auto const num_columns = h_stripe_dicts.size().second;

  dim3 dim_block(1024, 1);  // 1024 threads per chunk
  dim3 dim_grid_build(num_columns, num_stripes);
  gpuCompactChunkDictionaries<<<dim_grid_build, dim_block, 0, stream.value()>>>(d_stripes_dicts,
                                                                                chunks);
  for (uint32_t stripe_idx = 0; stripe_idx < num_stripes; ++stripe_idx) {
    for (auto const& stripe_dict : h_stripe_dicts[stripe_idx]) {
      if (stripe_dict.dict_data != nullptr) {
        auto const dict_data_ptr = thrust::device_pointer_cast(stripe_dict.dict_data);
        auto const string_column = stripe_dict.leaf_column;
        // NOTE: Requires the --expt-extended-lambda nvcc flag
        thrust::sort(rmm::exec_policy(stream),
                     dict_data_ptr,
                     dict_data_ptr + stripe_dict.num_strings,
                     [string_column] __device__(const uint32_t& lhs, const uint32_t& rhs) {
                       return string_column->element<string_view>(lhs) <
                              string_column->element<string_view>(rhs);
                     });
      }
    }
  }
  gpuBuildStripeDictionaries<1024>
    <<<dim_grid_build, dim_block, 0, stream.value()>>>(d_stripes_dicts);
}

}  // namespace gpu
}  // namespace orc
}  // namespace io
}  // namespace cudf
