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
#include <cub/cub.cuh>
#include <cudf/utilities/error.hpp>
#include <io/parquet/parquet_gpu.hpp>
#include <io/utilities/block_utils.cuh>

namespace cudf {
namespace io {
namespace parquet {
namespace gpu {
struct dict_state_s {
  uint32_t row_cnt;
  PageFragment *cur_fragment;
  uint32_t *hashmap;
  uint32_t total_dict_entries;  //!< Total number of entries in dictionary
  uint32_t dictionary_size;     //!< Total dictionary size in bytes
  uint32_t num_dict_entries;    //!< Dictionary entries in current fragment to add
  uint32_t frag_dict_size;
  EncColumnChunk ck;
  EncColumnDesc col;
  PageFragment frag;
  volatile uint32_t scratch_red[32];
  uint16_t frag_dict[MAX_PAGE_FRAGMENT_SIZE];
};

/**
 * @brief Computes a 16-bit dictionary hash
 **/
inline __device__ uint32_t uint32_hash16(uint32_t v) { return (v + (v >> 16)) & 0xffff; }

inline __device__ uint32_t uint64_hash16(uint64_t v)
{
  return uint32_hash16((uint32_t)(v + (v >> 32)));
}

inline __device__ uint32_t nvstr_hash16(const uint8_t *p, uint32_t len)
{
  uint32_t hash = len;
  if (len > 0) {
    uint32_t align_p    = 3 & reinterpret_cast<uintptr_t>(p);
    const uint32_t *p32 = reinterpret_cast<const uint32_t *>(p - align_p);
    uint32_t ofs        = align_p * 8;
    uint32_t v;
    while (len > 4) {
      v = *p32++;
      if (ofs) { v = __funnelshift_r(v, *p32, ofs); }
      hash = __funnelshift_l(hash, hash, 5) + v;
      len -= 4;
    }
    v = *p32;
    if (ofs) { v = __funnelshift_r(v, (align_p + len > 4) ? p32[1] : 0, ofs); }
    v &= ((2 << (len * 8 - 1)) - 1);
    hash = __funnelshift_l(hash, hash, 5) + v;
  }
  return uint32_hash16(hash);
}

/**
 * @brief Fetch a page fragment and its dictionary entries in row-ascending order
 *
 * @param[in,out] s dictionary state
 * @param[in,out] dict_data fragment dictionary data for the current column (zeroed out after
 *fetching)
 * @param[in] frag_start_row row position of current fragment
 * @param[in] t thread id
 **/
__device__ void FetchDictionaryFragment(dict_state_s *s,
                                        uint32_t *dict_data,
                                        uint32_t frag_start_row,
                                        uint32_t t)
{
  if (t < sizeof(PageFragment) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&s->frag)[t] =
      reinterpret_cast<const uint32_t *>(s->cur_fragment)[t];
  }
  __syncthreads();
  // Store the row values in shared mem and set the corresponding dict_data to zero (end-of-list)
  // It's easiest to do this here since we're only dealing with values all within a 5K-row window
  for (uint32_t i = t; i < s->frag.num_dict_vals; i += 1024) {
    uint32_t r      = dict_data[frag_start_row + i] - frag_start_row;
    s->frag_dict[i] = r;
  }
  __syncthreads();
  for (uint32_t i = t; i < s->frag.num_dict_vals; i += 1024) {
    uint32_t r                    = s->frag_dict[i];
    dict_data[frag_start_row + r] = 0;
  }
  __syncthreads();
}

/// Generate dictionary indices in ascending row order
__device__ void GenerateDictionaryIndices(dict_state_s *s, uint32_t t)
{
  uint32_t *dict_index      = s->col.dict_index;
  uint32_t *dict_data       = s->col.dict_data + s->ck.start_row;
  const uint32_t *valid_map = s->col.valid_map_base;
  uint32_t num_dict_entries = 0;

  for (uint32_t i = 0; i < s->row_cnt; i += 1024) {
    uint32_t row      = s->ck.start_row + i + t;
    uint32_t is_valid = (i + t < s->row_cnt && row < s->col.num_rows)
                          ? (valid_map) ? (valid_map[row >> 5] >> (row & 0x1f)) & 1 : 1
                          : 0;
    uint32_t dict_idx = (is_valid) ? dict_index[row] : 0;
    uint32_t is_unique =
      (is_valid &&
       dict_idx ==
         row);  // Any value that doesn't have bit31 set should have dict_idx=row at this point
    uint32_t umask = BALLOT(is_unique);
    uint32_t pos   = num_dict_entries + __popc(umask & ((1 << (t & 0x1f)) - 1));
    if (!(t & 0x1f)) { s->scratch_red[t >> 5] = __popc(umask); }
    num_dict_entries += __syncthreads_count(is_unique);
    if (t < 32) { s->scratch_red[t] = WarpReducePos32(s->scratch_red[t], t); }
    __syncthreads();
    if (t >= 32) { pos += s->scratch_red[(t - 32) >> 5]; }
    if (is_valid && is_unique) {
      dict_data[pos]  = row;
      dict_index[row] = pos;
    }
    __syncthreads();
    if (is_valid && !is_unique) {
      // NOTE: Should have at most 3 iterations (once for early duplicate elimination, once for
      // final dictionary duplicate elimination and once for re-ordering) (If something went wrong
      // building the dictionary, it will likely hang or crash right here)
      do {
        dict_idx = dict_index[dict_idx & 0x7fffffff];
      } while (dict_idx > 0x7fffffff);
      dict_index[row] = dict_idx;
    }
  }
}

// blockDim(1024, 1, 1)
template <int block_size>
__global__ void __launch_bounds__(block_size, 1)
  gpuBuildChunkDictionaries(EncColumnChunk *chunks, uint32_t *dev_scratch)
{
  __shared__ __align__(8) dict_state_s state_g;
  using warp_reduce = cub::WarpReduce<uint32_t>;
  __shared__ typename warp_reduce::TempStorage temp_storage[block_size / 32];

  dict_state_s *const s = &state_g;
  uint32_t t            = threadIdx.x;
  uint32_t dtype, dtype_len, dtype_len_in;

  if (t < sizeof(EncColumnChunk) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&s->ck)[t] =
      reinterpret_cast<const uint32_t *>(&chunks[blockIdx.x])[t];
  }
  __syncthreads();
  if (!s->ck.has_dictionary) { return; }
  if (t < sizeof(EncColumnDesc) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&s->col)[t] =
      reinterpret_cast<const uint32_t *>(s->ck.col_desc)[t];
  }
  __syncthreads();
  if (!t) {
    s->hashmap               = dev_scratch + s->ck.dictionary_id * (size_t)(1 << kDictHashBits);
    s->row_cnt               = 0;
    s->cur_fragment          = s->ck.fragments;
    s->total_dict_entries    = 0;
    s->dictionary_size       = 0;
    s->ck.num_dict_fragments = 0;
  }
  dtype     = s->col.physical_type;
  dtype_len = (dtype == INT64 || dtype == DOUBLE) ? 8 : 4;
  if (dtype == INT32) {
    dtype_len_in = GetDtypeLogicalLen(s->col.converted_type);
  } else {
    dtype_len_in = (dtype == BYTE_ARRAY) ? sizeof(nvstrdesc_s) : dtype_len;
  }
  __syncthreads();
  while (s->row_cnt < s->ck.num_rows) {
    uint32_t frag_start_row = s->ck.start_row + s->row_cnt, num_dict_entries, frag_dict_size;
    FetchDictionaryFragment(s, s->col.dict_data, frag_start_row, t);
    __syncthreads();
    num_dict_entries = s->frag.num_dict_vals;
    if (!t) {
      s->num_dict_entries = 0;
      s->frag_dict_size   = 0;
    }
    for (uint32_t i = 0; i < num_dict_entries; i += 1024) {
      bool is_valid    = (i + t < num_dict_entries);
      uint32_t len     = 0;
      uint32_t is_dupe = 0;
      uint32_t row, hash, next, *next_addr;
      uint32_t new_dict_entries;

      if (is_valid) {
        row = frag_start_row + s->frag_dict[i + t];
        len = dtype_len;
        if (dtype == BYTE_ARRAY) {
          const char *ptr = static_cast<const nvstrdesc_s *>(s->col.column_data_base)[row].ptr;
          uint32_t count  = static_cast<uint32_t>(
            static_cast<const nvstrdesc_s *>(s->col.column_data_base)[row].count);
          len += count;
          hash = nvstr_hash16(reinterpret_cast<const uint8_t *>(ptr), count);
          // Walk the list of rows with the same hash
          next_addr = &s->hashmap[hash];
          while ((next = atomicCAS(next_addr, 0, row + 1)) != 0) {
            const char *ptr2 =
              static_cast<const nvstrdesc_s *>(s->col.column_data_base)[next - 1].ptr;
            uint32_t count2 =
              static_cast<const nvstrdesc_s *>(s->col.column_data_base)[next - 1].count;
            if (count2 == count && nvstr_is_equal(ptr, count, ptr2, count2)) {
              is_dupe = 1;
              break;
            }
            next_addr = &s->col.dict_data[next - 1];
          }
        } else {
          uint64_t val;

          if (dtype_len_in == 8) {
            val  = static_cast<const uint64_t *>(s->col.column_data_base)[row];
            hash = uint64_hash16(val);
          } else {
            val = (dtype_len_in == 4)
                    ? static_cast<const uint32_t *>(s->col.column_data_base)[row]
                    : (dtype_len_in == 2)
                        ? static_cast<const uint16_t *>(s->col.column_data_base)[row]
                        : static_cast<const uint8_t *>(s->col.column_data_base)[row];
            hash = uint32_hash16(val);
          }
          // Walk the list of rows with the same hash
          next_addr = &s->hashmap[hash];
          while ((next = atomicCAS(next_addr, 0, row + 1)) != 0) {
            uint64_t val2 =
              (dtype_len_in == 8)
                ? static_cast<const uint64_t *>(s->col.column_data_base)[next - 1]
                : (dtype_len_in == 4)
                    ? static_cast<const uint32_t *>(s->col.column_data_base)[next - 1]
                    : (dtype_len_in == 2)
                        ? static_cast<const uint16_t *>(s->col.column_data_base)[next - 1]
                        : static_cast<const uint8_t *>(s->col.column_data_base)[next - 1];
            if (val2 == val) {
              is_dupe = 1;
              break;
            }
            next_addr = &s->col.dict_data[next - 1];
          }
        }
      }
      // Count the non-duplicate entries
      frag_dict_size = warp_reduce(temp_storage[t / 32]).Sum((is_valid && !is_dupe) ? len : 0);
      if (!(t & 0x1f)) { s->scratch_red[t >> 5] = frag_dict_size; }
      new_dict_entries = __syncthreads_count(is_valid && !is_dupe);
      if (t < 32) {
        frag_dict_size = warp_reduce(temp_storage[t / 32]).Sum(s->scratch_red[t]);
        if (t == 0) {
          s->frag_dict_size += frag_dict_size;
          s->num_dict_entries += new_dict_entries;
        }
      }
      if (is_valid) {
        if (!is_dupe) {
          s->col.dict_index[row] = row;
        } else {
          s->col.dict_index[row] = (next - 1) | (1u << 31);
        }
      }
      __syncthreads();
      // At this point, the dictionary order is non-deterministic, and we want insertion order
      // Make sure that the non-duplicate entry corresponds to the lower row number
      // (The entry in dict_data (next-1) used for duplicate elimination does not need
      // to be the lowest row number)
      bool reorder_check = (is_valid && is_dupe && next - 1 > row);
      if (reorder_check) {
        next = s->col.dict_index[next - 1];
        while (next & (1u << 31)) { next = s->col.dict_index[next & 0x7fffffff]; }
      }
      if (__syncthreads_or(reorder_check)) {
        if (reorder_check) { atomicMin(&s->col.dict_index[next], row); }
        __syncthreads();
        if (reorder_check && s->col.dict_index[next] == row) {
          s->col.dict_index[next] = row | (1u << 31);
          s->col.dict_index[row]  = row;
        }
        __syncthreads();
      }
    }
    __syncthreads();
    num_dict_entries = s->num_dict_entries;
    frag_dict_size   = s->frag_dict_size;
    if (s->total_dict_entries + num_dict_entries > 65536 ||
        (s->dictionary_size != 0 && s->dictionary_size + frag_dict_size > 512 * 1024)) {
      break;
    }
    __syncthreads();
    if (!t) {
      if (num_dict_entries != s->frag.num_dict_vals) {
        s->cur_fragment->num_dict_vals = num_dict_entries;
      }
      if (frag_dict_size != s->frag.dict_data_size) { s->frag.dict_data_size = frag_dict_size; }
      s->total_dict_entries += num_dict_entries;
      s->dictionary_size += frag_dict_size;
      s->row_cnt += s->frag.num_rows;
      s->cur_fragment++;
      s->ck.num_dict_fragments++;
    }
    __syncthreads();
  }
  __syncthreads();
  GenerateDictionaryIndices(s, t);
  if (!t) {
    chunks[blockIdx.x].num_dict_fragments = s->ck.num_dict_fragments;
    chunks[blockIdx.x].dictionary_size    = s->dictionary_size;
    chunks[blockIdx.x].total_dict_entries = s->total_dict_entries;
  }
}

/**
 * @brief Launches kernel for building chunk dictionaries
 *
 * @param[in,out] chunks Column chunks
 * @param[in] dev_scratch Device scratch data (kDictScratchSize per dictionary)
 * @param[in] num_chunks Number of column chunks
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t BuildChunkDictionaries(EncColumnChunk *chunks,
                                   uint32_t *dev_scratch,
                                   size_t scratch_size,
                                   uint32_t num_chunks,
                                   cudaStream_t stream)
{
  if (num_chunks > 0 && scratch_size > 0) {  // zero scratch size implies no dictionaries
    CUDA_TRY(cudaMemsetAsync(dev_scratch, 0, scratch_size, stream));
    gpuBuildChunkDictionaries<1024><<<num_chunks, 1024, 0, stream>>>(chunks, dev_scratch);
  }
  return cudaSuccess;
}

}  // namespace gpu
}  // namespace parquet
}  // namespace io
}  // namespace cudf
