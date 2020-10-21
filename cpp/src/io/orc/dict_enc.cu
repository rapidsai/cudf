/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <io/utilities/block_utils.cuh>
#include "orc_common.h"
#include "orc_gpu.h"

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace cudf {
namespace io {
namespace orc {
namespace gpu {
#define MAX_SHORT_DICT_ENTRIES (10 * 1024)
#define INIT_HASH_BITS 12

struct dictinit_state_s {
  uint32_t nnz;
  uint32_t total_dupes;
  DictionaryChunk chunk;
  volatile uint32_t scratch_red[32];
  uint16_t dict[MAX_SHORT_DICT_ENTRIES];
  union {
    uint16_t u16[1 << (INIT_HASH_BITS)];
    uint32_t u32[1 << (INIT_HASH_BITS - 1)];
  } map;
};

/**
 * @brief Return a 12-bit hash from a byte sequence
 */
static inline __device__ uint32_t nvstr_init_hash(const uint8_t *ptr, uint32_t len)
{
  if (len != 0) {
    return (ptr[0] + (ptr[len - 1] << 5) + (len << 10)) & ((1 << INIT_HASH_BITS) - 1);
  } else {
    return 0;
  }
}

/**
 * @brief Fill dictionary with the indices of non-null rows
 *
 * @param[in,out] s dictionary builder state
 * @param[in] t thread id
 *
 **/
static __device__ void LoadNonNullIndices(volatile dictinit_state_s *s, int t)
{
  if (t == 0) { s->nnz = 0; }
  for (uint32_t i = 0; i < s->chunk.num_rows; i += 512) {
    const uint32_t *valid_map = s->chunk.valid_map_base;
    uint32_t is_valid, nz_map, nz_pos;
    if (t < 16) {
      if (!valid_map) {
        s->scratch_red[t] = 0xffffffffu;
      } else {
        uint32_t row = s->chunk.start_row + i + t * 32;
        uint32_t v   = (row < s->chunk.start_row + s->chunk.num_rows) ? valid_map[row >> 5] : 0;
        if (row & 0x1f) {
          uint32_t v1 =
            (row + 32 < s->chunk.start_row + s->chunk.num_rows) ? valid_map[(row >> 5) + 1] : 0;
          v = __funnelshift_r(v, v1, row & 0x1f);
        }
        s->scratch_red[t] = v;
      }
    }
    __syncthreads();
    is_valid = (i + t < s->chunk.num_rows) ? (s->scratch_red[t >> 5] >> (t & 0x1f)) & 1 : 0;
    nz_map   = BALLOT(is_valid);
    nz_pos   = s->nnz + __popc(nz_map & (0x7fffffffu >> (0x1fu - ((uint32_t)t & 0x1f))));
    if (!(t & 0x1f)) { s->scratch_red[16 + (t >> 5)] = __popc(nz_map); }
    __syncthreads();
    if (t < 32) {
      uint32_t nnz     = s->scratch_red[16 + (t & 0xf)];
      uint32_t nnz_pos = WarpReducePos16(nnz, t);
      if (t == 0xf) { s->nnz += nnz_pos; }
      if (t <= 0xf) { s->scratch_red[t] = nnz_pos - nnz; }
    }
    __syncthreads();
    if (is_valid) { s->dict[nz_pos + s->scratch_red[t >> 5]] = i + t; }
    __syncthreads();
  }
}

/**
 * @brief Gather all non-NULL string rows and compute total character data size
 *
 * @param[in] chunks DictionaryChunk device array [rowgroup][column]
 * @param[in] num_columns Number of columns
 *
 **/
// blockDim {512,1,1}
template <int block_size>
__global__ void __launch_bounds__(block_size, 2)
  gpuInitDictionaryIndices(DictionaryChunk *chunks, uint32_t num_columns)
{
  __shared__ __align__(16) dictinit_state_s state_g;
  using warp_reduce      = cub::WarpReduce<uint32_t>;
  using half_warp_reduce = cub::WarpReduce<uint32_t, 16>;
  __shared__ union {
    typename warp_reduce::TempStorage full[block_size / 32];
    typename half_warp_reduce::TempStorage half[block_size / 32];
  } temp_storage;

  dictinit_state_s *const s = &state_g;
  uint32_t col_id           = blockIdx.x;
  uint32_t group_id         = blockIdx.y;
  const nvstrdesc_s *ck_data;
  uint32_t *dict_data;
  uint32_t nnz, start_row, dict_char_count;
  int t = threadIdx.x;

  if (t < sizeof(DictionaryChunk) / sizeof(uint32_t)) {
    ((volatile uint32_t *)&s->chunk)[t] =
      ((const uint32_t *)&chunks[group_id * num_columns + col_id])[t];
  }
  for (uint32_t i = 0; i < sizeof(s->map) / sizeof(uint32_t); i += block_size) {
    if (i + t < sizeof(s->map) / sizeof(uint32_t)) s->map.u32[i + t] = 0;
  }
  __syncthreads();
  // First, take care of NULLs, and count how many strings we have (TODO: bypass this step when
  // there are no nulls)
  LoadNonNullIndices(s, t);
  // Sum the lengths of all the strings
  if (t == 0) {
    s->chunk.string_char_count = 0;
    s->total_dupes             = 0;
  }
  nnz       = s->nnz;
  dict_data = s->chunk.dict_data;
  start_row = s->chunk.start_row;
  ck_data   = static_cast<const nvstrdesc_s *>(s->chunk.column_data_base) + start_row;
  for (uint32_t i = 0; i < nnz; i += block_size) {
    uint32_t ck_row = 0, len = 0, hash;
    const uint8_t *ptr = 0;
    if (i + t < nnz) {
      ck_row = s->dict[i + t];
      ptr    = reinterpret_cast<const uint8_t *>(ck_data[ck_row].ptr);
      len    = ck_data[ck_row].count;
      hash   = nvstr_init_hash(ptr, len);
    }
    len = half_warp_reduce(temp_storage.half[t / 32]).Sum(len);
    if (!(t & 0xf)) { s->scratch_red[t >> 4] = len; }
    __syncthreads();
    if (t < 32) {
      len = warp_reduce(temp_storage.full[t / 32]).Sum(s->scratch_red[t]);
      if (t == 0) s->chunk.string_char_count += len;
    }
    if (i + t < nnz) {
      atomicAdd(&s->map.u32[hash >> 1], 1 << ((hash & 1) ? 16 : 0));
      dict_data[i + t] = start_row + ck_row;
    }
    __syncthreads();
  }
  // Reorder the 16-bit local indices according to the hash value of the strings
#if (INIT_HASH_BITS != 12)
#error "Hardcoded for INIT_HASH_BITS=12"
#endif
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
    uint32_t sum_w, tmp;
    sum23 += (sum01 >> 16) * 0x10001;
    sum45 += (sum23 >> 16) * 0x10001;
    sum67 += (sum45 >> 16) * 0x10001;
    sum_w = sum67 >> 16;
    sum_w = WarpReducePos16(sum_w, t);
    if ((t & 0xf) == 0xf) { s->scratch_red[t >> 4] = sum_w; }
    __syncthreads();
    if (t < 32) {
      uint32_t sum_b    = WarpReducePos32(s->scratch_red[t], t);
      s->scratch_red[t] = sum_b;
    }
    __syncthreads();
    tmp                   = (t >= 16) ? s->scratch_red[(t >> 4) - 1] : 0;
    sum_w                 = (sum_w - (sum67 >> 16) + tmp) * 0x10001;
    s->map.u32[t * 4 + 0] = sum_w + sum01 - count01;
    s->map.u32[t * 4 + 1] = sum_w + sum23 - count23;
    s->map.u32[t * 4 + 2] = sum_w + sum45 - count45;
    s->map.u32[t * 4 + 3] = sum_w + sum67 - count67;
    __syncthreads();
  }
  // Put the indices back in hash order
  for (uint32_t i = 0; i < nnz; i += block_size) {
    uint32_t ck_row = 0, pos = 0, hash = 0, pos_old, pos_new, sh, colliding_row;
    bool collision;
    if (i + t < nnz) {
      const uint8_t *ptr;
      uint32_t len;
      ck_row  = dict_data[i + t] - start_row;
      ptr     = reinterpret_cast<const uint8_t *>(ck_data[ck_row].ptr);
      len     = (uint32_t)ck_data[ck_row].count;
      hash    = nvstr_init_hash(ptr, len);
      sh      = (hash & 1) ? 16 : 0;
      pos_old = s->map.u16[hash];
    }
    // The isolation of the atomicAdd, along with pos_old/pos_new is to guarantee deterministic
    // behavior for the first row in the hash map that will be used for early duplicate detection
    // The lack of 16-bit atomicMin makes this a bit messy...
    __syncthreads();
    if (i + t < nnz) {
      pos          = (atomicAdd(&s->map.u32[hash >> 1], 1 << sh) >> sh) & 0xffff;
      s->dict[pos] = ck_row;
    }
    __syncthreads();
    collision = false;
    if (i + t < nnz) {
      pos_new   = s->map.u16[hash];
      collision = (pos != pos_old && pos_new > pos_old + 1);
      if (collision) { colliding_row = s->dict[pos_old]; }
    }
    __syncthreads();
    // evens
    if (collision && !(pos_old & 1)) {
      uint32_t *dict32 = reinterpret_cast<uint32_t *>(&s->dict[pos_old]);
      atomicMin(dict32, (dict32[0] & 0xffff0000) | ck_row);
    }
    __syncthreads();
    // odds
    if (collision && (pos_old & 1)) {
      uint32_t *dict32 = reinterpret_cast<uint32_t *>(&s->dict[pos_old - 1]);
      atomicMin(dict32, (dict32[0] & 0x0000ffff) | (ck_row << 16));
    }
    __syncthreads();
    // Resolve collision
    if (collision && ck_row == s->dict[pos_old]) { s->dict[pos] = colliding_row; }
  }
  __syncthreads();
  // Now that the strings are ordered by hash, compare every string with the first entry in the hash
  // map, the position of the first string can be inferred from the hash map counts
  dict_char_count = 0;
  for (uint32_t i = 0; i < nnz; i += block_size) {
    uint32_t ck_row = 0, ck_row_ref = 0, is_dupe = 0, dupe_mask, dupes_before;
    if (i + t < nnz) {
      const char *str1, *str2;
      uint32_t len1, len2, hash;
      ck_row     = s->dict[i + t];
      str1       = ck_data[ck_row].ptr;
      len1       = (uint32_t)ck_data[ck_row].count;
      hash       = nvstr_init_hash(reinterpret_cast<const uint8_t *>(str1), len1);
      ck_row_ref = s->dict[(hash > 0) ? s->map.u16[hash - 1] : 0];
      if (ck_row_ref != ck_row) {
        str2    = ck_data[ck_row_ref].ptr;
        len2    = (uint32_t)ck_data[ck_row_ref].count;
        is_dupe = nvstr_is_equal(str1, len1, str2, len2);
        dict_char_count += (is_dupe) ? 0 : len1;
      }
    }
    dupe_mask    = BALLOT(is_dupe);
    dupes_before = s->total_dupes + __popc(dupe_mask & ((2 << (t & 0x1f)) - 1));
    if (!(t & 0x1f)) { s->scratch_red[t >> 5] = __popc(dupe_mask); }
    __syncthreads();
    if (t < 32) {
      uint32_t warp_dupes = (t < 16) ? s->scratch_red[t] : 0;
      uint32_t warp_pos   = WarpReducePos16(warp_dupes, t);
      if (t == 0xf) { s->total_dupes += warp_pos; }
      if (t < 16) { s->scratch_red[t] = warp_pos - warp_dupes; }
    }
    __syncthreads();
    if (i + t < nnz) {
      if (!is_dupe) {
        dupes_before += s->scratch_red[t >> 5];
        dict_data[i + t - dupes_before] = ck_row + start_row;
      } else {
        s->chunk.dict_index[ck_row + start_row] = (ck_row_ref + start_row) | (1u << 31);
      }
    }
  }
  dict_char_count = warp_reduce(temp_storage.full[t / 32]).Sum(dict_char_count);
  if (!(t & 0x1f)) { s->scratch_red[t >> 5] = dict_char_count; }
  __syncthreads();
  if (t < 32) {
    dict_char_count =
      half_warp_reduce(temp_storage.half[t / 32]).Sum((t < 16) ? s->scratch_red[t] : 0);
  }
  if (!t) {
    chunks[group_id * num_columns + col_id].num_strings       = nnz;
    chunks[group_id * num_columns + col_id].string_char_count = s->chunk.string_char_count;
    chunks[group_id * num_columns + col_id].num_dict_strings  = nnz - s->total_dupes;
    chunks[group_id * num_columns + col_id].dict_char_count   = dict_char_count;
  }
}

/**
 * @brief In-place concatenate dictionary data for all chunks in each stripe
 *
 * @param[in] stripes StripeDictionary device array [stripe][column]
 * @param[in] chunks DictionaryChunk device array [rowgroup][column]
 * @param[in] num_columns Number of columns
 *
 **/
// blockDim {1024,1,1}
extern "C" __global__ void __launch_bounds__(1024)
  gpuCompactChunkDictionaries(StripeDictionary *stripes,
                              DictionaryChunk const *chunks,
                              uint32_t num_columns)
{
  __shared__ __align__(16) StripeDictionary stripe_g;
  __shared__ __align__(16) DictionaryChunk chunk_g;
  __shared__ const uint32_t *volatile ck_curptr_g;
  __shared__ uint32_t volatile ck_curlen_g;

  uint32_t col_id    = blockIdx.x;
  uint32_t stripe_id = blockIdx.y;
  uint32_t chunk_len;
  int t = threadIdx.x;
  const uint32_t *src;
  uint32_t *dst;

  if (t < sizeof(StripeDictionary) / sizeof(uint32_t)) {
    ((volatile uint32_t *)&stripe_g)[t] =
      ((const uint32_t *)&stripes[stripe_id * num_columns + col_id])[t];
  }
  __syncthreads();
  if (!stripe_g.dict_data) { return; }
  if (t < sizeof(DictionaryChunk) / sizeof(uint32_t)) {
    ((volatile uint32_t *)&chunk_g)[t] =
      ((const uint32_t *)&chunks[stripe_g.start_chunk * num_columns + col_id])[t];
  }
  __syncthreads();
  dst = stripe_g.dict_data + chunk_g.num_dict_strings;
  for (uint32_t g = 1; g < stripe_g.num_chunks; g++) {
    if (!t) {
      src         = chunks[(stripe_g.start_chunk + g) * num_columns + col_id].dict_data;
      chunk_len   = chunks[(stripe_g.start_chunk + g) * num_columns + col_id].num_dict_strings;
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
 *
 **/
// NOTE: Prone to poor utilization on small datasets due to 1 block per dictionary
// blockDim {1024,1,1}
template <int block_size>
__global__ void __launch_bounds__(block_size)
  gpuBuildStripeDictionaries(StripeDictionary *stripes, uint32_t num_columns)
{
  __shared__ __align__(16) build_state_s state_g;
  using warp_reduce = cub::WarpReduce<uint32_t>;
  __shared__ typename warp_reduce::TempStorage temp_storage[block_size / 32];

  volatile build_state_s *const s = &state_g;
  uint32_t col_id                 = blockIdx.x;
  uint32_t stripe_id              = blockIdx.y;
  uint32_t num_strings;
  uint32_t *dict_data, *dict_index;
  uint32_t dict_char_count;
  const nvstrdesc_s *str_data;
  int t = threadIdx.x;

  if (t < sizeof(StripeDictionary) / sizeof(uint32_t)) {
    ((volatile uint32_t *)&s->stripe)[t] =
      ((const uint32_t *)&stripes[stripe_id * num_columns + col_id])[t];
  }
  if (t == 31 * 32) { s->total_dupes = 0; }
  __syncthreads();
  num_strings = s->stripe.num_strings;
  dict_data   = s->stripe.dict_data;
  if (!dict_data) return;
  dict_index      = s->stripe.dict_index;
  str_data        = static_cast<const nvstrdesc_s *>(s->stripe.column_data_base);
  dict_char_count = 0;
  for (uint32_t i = 0; i < num_strings; i += block_size) {
    uint32_t cur = (i + t < num_strings) ? dict_data[i + t] : 0;
    uint32_t dupe_mask, dupes_before, cur_len = 0;
    const char *cur_ptr;
    bool is_dupe = false;
    if (i + t < num_strings) {
      cur_ptr = str_data[cur].ptr;
      cur_len = str_data[cur].count;
    }
    if (i + t != 0 && i + t < num_strings) {
      uint32_t prev = dict_data[i + t - 1];
      is_dupe       = nvstr_is_equal(cur_ptr, cur_len, str_data[prev].ptr, str_data[prev].count);
    }
    dict_char_count += (is_dupe) ? 0 : cur_len;
    dupe_mask    = BALLOT(is_dupe);
    dupes_before = s->total_dupes + __popc(dupe_mask & ((2 << (t & 0x1f)) - 1));
    if (!(t & 0x1f)) { s->scratch_red[t >> 5] = __popc(dupe_mask); }
    __syncthreads();
    if (t < 32) {
      uint32_t warp_dupes = s->scratch_red[t];
      uint32_t warp_pos   = WarpReducePos32(warp_dupes, t);
      if (t == 0x1f) { s->total_dupes += warp_pos; }
      s->scratch_red[t] = warp_pos - warp_dupes;
    }
    __syncthreads();
    if (i + t < num_strings) {
      dupes_before += s->scratch_red[t >> 5];
      dict_index[cur] = i + t - dupes_before;
      if (!is_dupe && dupes_before != 0) { dict_data[i + t - dupes_before] = cur; }
    }
    __syncthreads();
  }
  dict_char_count = warp_reduce(temp_storage[t / 32]).Sum(dict_char_count);
  if (!(t & 0x1f)) { s->scratch_red[t >> 5] = dict_char_count; }
  __syncthreads();
  if (t < 32) { dict_char_count = warp_reduce(temp_storage[t / 32]).Sum(s->scratch_red[t]); }
  if (t == 0) {
    stripes[stripe_id * num_columns + col_id].num_strings     = num_strings - s->total_dupes;
    stripes[stripe_id * num_columns + col_id].dict_char_count = dict_char_count;
  }
}

/**
 * @brief Launches kernel for initializing dictionary chunks
 *
 * @param[in] chunks DictionaryChunk device array [rowgroup][column]
 * @param[in] num_columns Number of columns
 * @param[in] num_rowgroups Number of row groups
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t InitDictionaryIndices(DictionaryChunk *chunks,
                                  uint32_t num_columns,
                                  uint32_t num_rowgroups,
                                  cudaStream_t stream)
{
  dim3 dim_block(512, 1);  // 512 threads per chunk
  dim3 dim_grid(num_columns, num_rowgroups);
  gpuInitDictionaryIndices<512><<<dim_grid, dim_block, 0, stream>>>(chunks, num_columns);
  return cudaSuccess;
}

/**
 * @brief Launches kernel for building stripe dictionaries
 *
 * @param[in] stripes StripeDictionary device array [stripe][column]
 * @param[in] stripes_host StripeDictionary host array [stripe][column]
 * @param[in] chunks DictionaryChunk device array [rowgroup][column]
 * @param[in] num_stripes Number of stripes
 * @param[in] num_rowgroups Number of row groups
 * @param[in] num_columns Number of columns
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t BuildStripeDictionaries(StripeDictionary *stripes,
                                    StripeDictionary *stripes_host,
                                    DictionaryChunk const *chunks,
                                    uint32_t num_stripes,
                                    uint32_t num_rowgroups,
                                    uint32_t num_columns,
                                    cudaStream_t stream)
{
  dim3 dim_block(1024, 1);  // 1024 threads per chunk
  dim3 dim_grid_build(num_columns, num_stripes);
  gpuCompactChunkDictionaries<<<dim_grid_build, dim_block, 0, stream>>>(
    stripes, chunks, num_columns);
  for (uint32_t i = 0; i < num_stripes * num_columns; i++) {
    if (stripes_host[i].dict_data != nullptr) {
      thrust::device_ptr<uint32_t> p = thrust::device_pointer_cast(stripes_host[i].dict_data);
      const nvstrdesc_s *str_data =
        static_cast<const nvstrdesc_s *>(stripes_host[i].column_data_base);
      // NOTE: Requires the --expt-extended-lambda nvcc flag
      thrust::sort(rmm::exec_policy(stream)->on(stream),
                   p,
                   p + stripes_host[i].num_strings,
                   [str_data] __device__(const uint32_t &lhs, const uint32_t &rhs) {
                     return nvstr_is_lesser(str_data[lhs].ptr,
                                            (uint32_t)str_data[lhs].count,
                                            str_data[rhs].ptr,
                                            (uint32_t)str_data[rhs].count);
                   });
    }
  }
  gpuBuildStripeDictionaries<1024><<<dim_grid_build, dim_block, 0, stream>>>(stripes, num_columns);
  return cudaSuccess;
}

}  // namespace gpu
}  // namespace orc
}  // namespace io
}  // namespace cudf
