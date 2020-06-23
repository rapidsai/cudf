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
#include "parquet_gpu.h"

namespace cudf {
namespace io {
namespace parquet {
namespace gpu {
// Spark doesn't support RLE encoding for BOOLEANs
#define ENABLE_BOOL_RLE 0

#define INIT_HASH_BITS 12

struct frag_init_state_s {
  EncColumnDesc col;
  PageFragment frag;
  uint32_t total_dupes;
  volatile uint32_t scratch_red[32];
  uint32_t dict[MAX_PAGE_FRAGMENT_SIZE];
  union {
    uint16_t u16[1 << (INIT_HASH_BITS)];
    uint32_t u32[1 << (INIT_HASH_BITS - 1)];
  } map;
};

#define LOG2_RLE_BFRSZ 9
#define RLE_BFRSZ (1 << LOG2_RLE_BFRSZ)
#define RLE_MAX_LIT_RUN 0xfff8  // Maximum literal run for 2-byte run code

struct page_enc_state_s {
  uint8_t *cur;          //!< current output ptr
  uint8_t *rle_out;      //!< current RLE write ptr
  uint32_t rle_run;      //!< current RLE run
  uint32_t run_val;      //!< current RLE run value
  uint32_t rle_pos;      //!< RLE encoder positions
  uint32_t rle_numvals;  //!< RLE input value count
  uint32_t rle_lit_count;
  uint32_t rle_rpt_count;
  volatile uint32_t rpt_map[4];
  volatile uint32_t scratch_red[32];
  EncPage page;
  EncColumnChunk ck;
  EncColumnDesc col;
  gpu_inflate_input_s comp_in;
  gpu_inflate_status_s comp_out;
  uint16_t vals[RLE_BFRSZ];
};

/**
 * @brief Return a 12-bit hash from a byte sequence
 */
inline __device__ uint32_t nvstr_init_hash(const uint8_t *ptr, uint32_t len)
{
  if (len != 0) {
    return (ptr[0] + (ptr[len - 1] << 5) + (len << 10)) & ((1 << INIT_HASH_BITS) - 1);
  } else {
    return 0;
  }
}

inline __device__ uint32_t uint32_init_hash(uint32_t v)
{
  return (v + (v >> 11) + (v >> 22)) & ((1 << INIT_HASH_BITS) - 1);
}

inline __device__ uint32_t uint64_init_hash(uint64_t v)
{
  return uint32_init_hash(static_cast<uint32_t>(v + (v >> 32)));
}

/**
 * @brief Initializes encoder page fragments
 *
 * @param[in] frag Fragment array [fragment_id][column_id]
 * @param[in] col_desc Column description array [column_id]
 * @param[in] num_fragments Number of fragments per column
 * @param[in] num_columns Number of columns
 *
 **/
// blockDim {512,1,1}
__global__ void __launch_bounds__(512) gpuInitPageFragments(PageFragment *frag,
                                                            const EncColumnDesc *col_desc,
                                                            int32_t num_fragments,
                                                            int32_t num_columns,
                                                            uint32_t fragment_size,
                                                            uint32_t max_num_rows)
{
  __shared__ __align__(16) frag_init_state_s state_g;

  frag_init_state_s *const s = &state_g;
  uint32_t t                 = threadIdx.x;
  uint32_t start_row, nrows, dtype_len, dtype_len_in, dtype;

  if (t < sizeof(EncColumnDesc) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&s->col)[t] =
      reinterpret_cast<const uint32_t *>(&col_desc[blockIdx.x])[t];
  }
  for (uint32_t i = 0; i < sizeof(s->map) / sizeof(uint32_t); i += 512) {
    if (i + t < sizeof(s->map) / sizeof(uint32_t)) s->map.u32[i + t] = 0;
  }
  __syncthreads();
  start_row = blockIdx.y * fragment_size;
  if (!t) {
    s->col.num_rows            = min(s->col.num_rows, max_num_rows);
    s->frag.num_rows           = min(fragment_size, max_num_rows - min(start_row, max_num_rows));
    s->frag.non_nulls          = 0;
    s->frag.num_dict_vals      = 0;
    s->frag.fragment_data_size = 0;
    s->frag.dict_data_size     = 0;
    s->total_dupes             = 0;
  }
  dtype     = s->col.physical_type;
  dtype_len = (dtype == INT64 || dtype == DOUBLE) ? 8 : (dtype == BOOLEAN) ? 1 : 4;
  if (dtype == INT32) {
    dtype_len_in = GetDtypeLogicalLen(s->col.converted_type);
  } else {
    dtype_len_in = (dtype == BYTE_ARRAY) ? sizeof(nvstrdesc_s) : dtype_len;
  }
  __syncthreads();
  nrows = s->frag.num_rows;
  for (uint32_t i = 0; i < nrows; i += 512) {
    const uint32_t *valid = s->col.valid_map_base;
    uint32_t row          = start_row + i + t;
    uint32_t is_valid     = (i + t < nrows && row < s->col.num_rows)
                          ? (valid) ? (valid[row >> 5] >> (row & 0x1f)) & 1 : 1
                          : 0;
    uint32_t valid_warp = BALLOT(is_valid);
    uint32_t len, nz_pos, hash;
    if (is_valid) {
      len = dtype_len;
      if (dtype != BOOLEAN) {
        if (dtype == BYTE_ARRAY) {
          const char *ptr = reinterpret_cast<const nvstrdesc_s *>(s->col.column_data_base)[row].ptr;
          uint32_t count =
            (uint32_t) reinterpret_cast<const nvstrdesc_s *>(s->col.column_data_base)[row].count;
          len += count;
          hash = nvstr_init_hash(reinterpret_cast<const uint8_t *>(ptr), count);
        } else if (dtype_len_in == 8) {
          hash = uint64_init_hash(reinterpret_cast<const uint64_t *>(s->col.column_data_base)[row]);
        } else {
          hash = uint32_init_hash(
            (dtype_len_in == 4)
              ? reinterpret_cast<const uint32_t *>(s->col.column_data_base)[row]
              : (dtype_len_in == 2)
                  ? reinterpret_cast<const uint16_t *>(s->col.column_data_base)[row]
                  : reinterpret_cast<const uint8_t *>(s->col.column_data_base)[row]);
        }
      }
    } else {
      len = 0;
    }
    nz_pos =
      s->frag.non_nulls + __popc(valid_warp & (0x7fffffffu >> (0x1fu - ((uint32_t)t & 0x1f))));
    len = WarpReduceSum32(len);
    if (!(t & 0x1f)) {
      s->scratch_red[(t >> 5) + 0]  = __popc(valid_warp);
      s->scratch_red[(t >> 5) + 16] = len;
    }
    __syncthreads();
    if (t < 32) {
      uint32_t warp_pos  = WarpReducePos16((t < 16) ? s->scratch_red[t] : 0, t);
      uint32_t non_nulls = SHFL(warp_pos, 0xf);
      len                = WarpReduceSum16((t < 16) ? s->scratch_red[t + 16] : 0);
      if (t < 16) { s->scratch_red[t] = warp_pos; }
      if (!t) {
        s->frag.non_nulls = s->frag.non_nulls + non_nulls;
        s->frag.fragment_data_size += len;
      }
    }
    __syncthreads();
    if (is_valid && dtype != BOOLEAN) {
      uint32_t *dict_index = s->col.dict_index;
      if (t >= 32) { nz_pos += s->scratch_red[(t - 32) >> 5]; }
      if (dict_index) {
        atomicAdd(&s->map.u32[hash >> 1], (hash & 1) ? 1 << 16 : 1);
        dict_index[start_row + nz_pos] =
          ((i + t) << INIT_HASH_BITS) |
          hash;  // Store the hash along with the index, so we don't have to recompute it
      }
    }
  }
  __syncthreads();
  // Reorder the 16-bit local indices according to the hash values
  if (s->col.dict_index) {
#if (INIT_HASH_BITS != 12)
#error "Hardcoded for INIT_HASH_BITS=12"
#endif
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
  if (s->col.dict_index) {
    uint32_t *dict_index = s->col.dict_index + start_row;
    uint32_t nnz         = s->frag.non_nulls;
    for (uint32_t i = 0; i < nnz; i += 512) {
      uint32_t pos = 0, hash = 0, pos_old, pos_new, sh, colliding_row, val = 0;
      bool collision;
      if (i + t < nnz) {
        val     = dict_index[i + t];
        hash    = val & ((1 << INIT_HASH_BITS) - 1);
        sh      = (hash & 1) ? 16 : 0;
        pos_old = s->map.u16[hash];
      }
      // The isolation of the atomicAdd, along with pos_old/pos_new is to guarantee deterministic
      // behavior for the first row in the hash map that will be used for early duplicate detection
      __syncthreads();
      if (i + t < nnz) {
        pos          = (atomicAdd(&s->map.u32[hash >> 1], 1 << sh) >> sh) & 0xffff;
        s->dict[pos] = val;
      }
      __syncthreads();
      collision = false;
      if (i + t < nnz) {
        pos_new   = s->map.u16[hash];
        collision = (pos != pos_old && pos_new > pos_old + 1);
        if (collision) { colliding_row = s->dict[pos_old]; }
      }
      __syncthreads();
      if (collision) { atomicMin(&s->dict[pos_old], val); }
      __syncthreads();
      // Resolve collision
      if (collision && val == s->dict[pos_old]) { s->dict[pos] = colliding_row; }
    }
    __syncthreads();
    // Now that the values are ordered by hash, compare every entry with the first entry in the hash
    // map, the position of the first entry can be inferred from the hash map counts
    uint32_t dupe_data_size = 0;
    for (uint32_t i = 0; i < nnz; i += 512) {
      const void *col_data = s->col.column_data_base;
      uint32_t ck_row = 0, ck_row_ref = 0, is_dupe = 0, dupe_mask, dupes_before;
      if (i + t < nnz) {
        uint32_t dict_val = s->dict[i + t];
        uint32_t hash     = dict_val & ((1 << INIT_HASH_BITS) - 1);
        ck_row            = start_row + (dict_val >> INIT_HASH_BITS);
        ck_row_ref = start_row + (s->dict[(hash > 0) ? s->map.u16[hash - 1] : 0] >> INIT_HASH_BITS);
        if (ck_row_ref != ck_row) {
          if (dtype == BYTE_ARRAY) {
            const nvstrdesc_s *ck_data = reinterpret_cast<const nvstrdesc_s *>(col_data);
            const char *str1           = ck_data[ck_row].ptr;
            uint32_t len1              = (uint32_t)ck_data[ck_row].count;
            const char *str2           = ck_data[ck_row_ref].ptr;
            uint32_t len2              = (uint32_t)ck_data[ck_row_ref].count;
            is_dupe                    = nvstr_is_equal(str1, len1, str2, len2);
            dupe_data_size += (is_dupe) ? 4 + len1 : 0;
          } else {
            if (dtype_len_in == 8) {
              uint64_t v1 = reinterpret_cast<const uint64_t *>(col_data)[ck_row];
              uint64_t v2 = reinterpret_cast<const uint64_t *>(col_data)[ck_row_ref];
              is_dupe     = (v1 == v2);
              dupe_data_size += (is_dupe) ? 8 : 0;
            } else {
              uint32_t v1, v2;
              if (dtype_len_in == 4) {
                v1 = reinterpret_cast<const uint32_t *>(col_data)[ck_row];
                v2 = reinterpret_cast<const uint32_t *>(col_data)[ck_row_ref];
              } else if (dtype_len_in == 2) {
                v1 = reinterpret_cast<const uint16_t *>(col_data)[ck_row];
                v2 = reinterpret_cast<const uint16_t *>(col_data)[ck_row_ref];
              } else {
                v1 = reinterpret_cast<const uint8_t *>(col_data)[ck_row];
                v2 = reinterpret_cast<const uint8_t *>(col_data)[ck_row_ref];
              }
              is_dupe = (v1 == v2);
              dupe_data_size += (is_dupe) ? 4 : 0;
            }
          }
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
          s->col.dict_data[start_row + i + t - dupes_before] = ck_row;
        } else {
          s->col.dict_index[ck_row] = ck_row_ref | (1u << 31);
        }
      }
    }
    __syncthreads();
    dupe_data_size = WarpReduceSum32(dupe_data_size);
    if (!(t & 0x1f)) { s->scratch_red[t >> 5] = dupe_data_size; }
    __syncthreads();
    if (t < 32) {
      dupe_data_size = WarpReduceSum16((t < 16) ? s->scratch_red[t] : 0);
      if (!t) {
        s->frag.dict_data_size = s->frag.fragment_data_size - dupe_data_size;
        s->frag.num_dict_vals  = s->frag.non_nulls - s->total_dupes;
      }
    }
  }
  __syncthreads();
  if (t < sizeof(PageFragment) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&frag[blockIdx.x * num_fragments + blockIdx.y])[t] =
      reinterpret_cast<uint32_t *>(&s->frag)[t];
  }
}

// blockDim {128,1,1}
__global__ void __launch_bounds__(128) gpuInitFragmentStats(statistics_group *groups,
                                                            const PageFragment *fragments,
                                                            const EncColumnDesc *col_desc,
                                                            int32_t num_fragments,
                                                            int32_t num_columns,
                                                            uint32_t fragment_size)
{
  __shared__ __align__(8) statistics_group group_g[4];

  uint32_t t                = threadIdx.x & 0x1f;
  uint32_t frag_id          = blockIdx.y * 4 + (threadIdx.x >> 5);
  uint32_t column_id        = blockIdx.x;
  statistics_group *const g = &group_g[threadIdx.x >> 5];
  if (!t && frag_id < num_fragments) {
    g->col       = &col_desc[column_id];
    g->start_row = frag_id * fragment_size;
    g->num_rows  = fragments[column_id * num_fragments + frag_id].num_rows;
  }
  __syncthreads();
  if (t < sizeof(statistics_group) / sizeof(uint32_t) && frag_id < num_fragments) {
    reinterpret_cast<uint32_t *>(&groups[column_id * num_fragments + frag_id])[t] =
      reinterpret_cast<uint32_t *>(g)[t];
  }
}

// blockDim {128,1,1}
__global__ void __launch_bounds__(128) gpuInitPages(EncColumnChunk *chunks,
                                                    EncPage *pages,
                                                    const EncColumnDesc *col_desc,
                                                    statistics_merge_group *page_grstats,
                                                    statistics_merge_group *chunk_grstats,
                                                    int32_t num_rowgroups,
                                                    int32_t num_columns)
{
  __shared__ __align__(8) EncColumnDesc col_g;
  __shared__ __align__(8) EncColumnChunk ck_g;
  __shared__ __align__(8) PageFragment frag_g;
  __shared__ __align__(8) EncPage page_g;
  __shared__ __align__(8) statistics_merge_group pagestats_g;

  uint32_t t = threadIdx.x;

  if (t < sizeof(EncColumnDesc) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&col_g)[t] =
      reinterpret_cast<const uint32_t *>(&col_desc[blockIdx.x])[t];
  }
  if (t < sizeof(EncColumnChunk) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&ck_g)[t] =
      reinterpret_cast<const uint32_t *>(&chunks[blockIdx.y * num_columns + blockIdx.x])[t];
  }
  __syncthreads();
  if (t < 32) {
    uint32_t fragments_in_chunk = 0;
    uint32_t rows_in_page       = 0;
    uint32_t page_size          = 0;
    uint32_t num_pages          = 0;
    uint32_t num_rows           = 0;
    uint32_t page_start         = 0;
    uint32_t page_offset        = ck_g.ck_stat_size;
    uint32_t num_dict_entries   = 0;
    uint32_t comp_page_offset   = ck_g.ck_stat_size;
    uint32_t cur_row            = ck_g.start_row;
    uint32_t ck_max_stats_len   = 0;
    uint32_t max_stats_len      = 0;

    if (!t) {
      pagestats_g.col         = &col_desc[blockIdx.x];
      pagestats_g.start_chunk = ck_g.first_fragment;
      pagestats_g.num_chunks  = 0;
    }
    if (ck_g.has_dictionary) {
      if (!t) {
        page_g.page_data       = ck_g.uncompressed_bfr + page_offset;
        page_g.compressed_data = ck_g.compressed_bfr + comp_page_offset;
        page_g.num_fragments   = 0;
        page_g.page_type       = DICTIONARY_PAGE;
        page_g.dict_bits_plus1 = 0;
        page_g.chunk_id        = blockIdx.y * num_columns + blockIdx.x;
        page_g.hdr_size        = 0;
        page_g.max_hdr_size    = 32;
        page_g.max_data_size   = ck_g.dictionary_size;
        page_g.start_row       = cur_row;
        page_g.num_rows        = ck_g.total_dict_entries;
        page_offset += page_g.max_hdr_size + page_g.max_data_size;
        comp_page_offset += page_g.max_hdr_size + GetMaxCompressedBfrSize(page_g.max_data_size);
      }
      SYNCWARP();
      if (pages && t < sizeof(EncPage) / sizeof(uint32_t)) {
        reinterpret_cast<uint32_t *>(&pages[ck_g.first_page])[t] =
          reinterpret_cast<uint32_t *>(&page_g)[t];
      }
      if (page_grstats && t < sizeof(statistics_merge_group) / sizeof(uint32_t)) {
        reinterpret_cast<uint32_t *>(&page_grstats[ck_g.first_page])[t] =
          reinterpret_cast<uint32_t *>(&pagestats_g)[t];
      }
      num_pages = 1;
    }
    SYNCWARP();
    do {
      uint32_t fragment_data_size, max_page_size, minmax_len = 0;
      SYNCWARP();
      if (num_rows < ck_g.num_rows) {
        if (t < sizeof(PageFragment) / sizeof(uint32_t)) {
          reinterpret_cast<uint32_t *>(&frag_g)[t] =
            reinterpret_cast<const uint32_t *>(&ck_g.fragments[fragments_in_chunk])[t];
        }
        if (!t && ck_g.stats && col_g.stats_dtype == dtype_string) {
          minmax_len = max(ck_g.stats[fragments_in_chunk].min_value.str_val.length,
                           ck_g.stats[fragments_in_chunk].max_value.str_val.length);
        }
      } else if (!t) {
        frag_g.fragment_data_size = 0;
        frag_g.num_rows           = 0;
      }
      SYNCWARP();
      if (ck_g.has_dictionary && fragments_in_chunk < ck_g.num_dict_fragments) {
        fragment_data_size =
          frag_g.num_rows * 2;  // Assume worst-case of 2-bytes per dictionary index
      } else {
        fragment_data_size = frag_g.fragment_data_size;
      }
      max_page_size = (rows_in_page * 2 >= ck_g.num_rows)
                        ? 256 * 1024
                        : (rows_in_page * 3 >= ck_g.num_rows) ? 384 * 1024 : 512 * 1024;
      if (num_rows >= ck_g.num_rows ||
          (rows_in_page > 0 &&
           (page_size + fragment_data_size > max_page_size ||
            (ck_g.has_dictionary && fragments_in_chunk == ck_g.num_dict_fragments)))) {
        uint32_t dict_bits_plus1;

        if (ck_g.has_dictionary && page_start < ck_g.num_dict_fragments) {
          uint32_t dict_bits;
          if (num_dict_entries <= 2) {
            dict_bits = 1;
          } else if (num_dict_entries <= 4) {
            dict_bits = 2;
          } else if (num_dict_entries <= 16) {
            dict_bits = 4;
          } else if (num_dict_entries <= 256) {
            dict_bits = 8;
          } else if (num_dict_entries <= 4096) {
            dict_bits = 12;
          } else {
            dict_bits = 16;
          }
          page_size       = 1 + 5 + ((rows_in_page * dict_bits + 7) >> 3) + (rows_in_page >> 8);
          dict_bits_plus1 = dict_bits + 1;
        } else {
          dict_bits_plus1 = 0;
        }
        if (!t) {
          uint32_t def_level_bits = col_g.level_bits & 0xf;
          uint32_t def_level_size =
            (def_level_bits)
              ? 4 + 5 + ((def_level_bits * rows_in_page + 7) >> 3) + (rows_in_page >> 8)
              : 0;
          page_g.num_fragments   = fragments_in_chunk - page_start;
          page_g.chunk_id        = blockIdx.y * num_columns + blockIdx.x;
          page_g.page_type       = DATA_PAGE;
          page_g.dict_bits_plus1 = dict_bits_plus1;
          page_g.hdr_size        = 0;
          page_g.max_hdr_size    = 32;  // Max size excluding statistics
          if (ck_g.stats) {
            uint32_t stats_hdr_len = 16;
            if (col_g.stats_dtype == dtype_string) {
              stats_hdr_len += 5 * 3 + 2 * max_stats_len;
            } else {
              stats_hdr_len += ((col_g.stats_dtype >= dtype_int64) ? 10 : 5) * 3;
            }
            page_g.max_hdr_size += stats_hdr_len;
          }
          page_g.max_data_size    = page_size + def_level_size;
          page_g.page_data        = ck_g.uncompressed_bfr + page_offset;
          page_g.compressed_data  = ck_g.compressed_bfr + comp_page_offset;
          page_g.start_row        = cur_row;
          page_g.num_rows         = rows_in_page;
          pagestats_g.start_chunk = ck_g.first_fragment + page_start;
          pagestats_g.num_chunks  = page_g.num_fragments;
          page_offset += page_g.max_hdr_size + page_g.max_data_size;
          comp_page_offset += page_g.max_hdr_size + GetMaxCompressedBfrSize(page_g.max_data_size);
          cur_row += rows_in_page;
          ck_max_stats_len = max(ck_max_stats_len, max_stats_len);
        }
        SYNCWARP();
        if (pages && t < sizeof(EncPage) / sizeof(uint32_t)) {
          reinterpret_cast<uint32_t *>(&pages[ck_g.first_page + num_pages])[t] =
            reinterpret_cast<uint32_t *>(&page_g)[t];
        }
        if (page_grstats && t < sizeof(statistics_merge_group) / sizeof(uint32_t)) {
          reinterpret_cast<uint32_t *>(&page_grstats[ck_g.first_page + num_pages])[t] =
            reinterpret_cast<uint32_t *>(&pagestats_g)[t];
        }
        num_pages++;
        page_size     = 0;
        rows_in_page  = 0;
        page_start    = fragments_in_chunk;
        max_stats_len = 0;
      }
      max_stats_len = max(max_stats_len, minmax_len);
      num_dict_entries += frag_g.num_dict_vals;
      page_size += fragment_data_size;
      rows_in_page += frag_g.num_rows;
      num_rows += frag_g.num_rows;
      fragments_in_chunk++;
    } while (frag_g.num_rows != 0);
    SYNCWARP();
    if (!t) {
      if (ck_g.ck_stat_size == 0 && ck_g.stats) {
        uint32_t ck_stat_size = 48 + 2 * ck_max_stats_len;
        page_offset += ck_stat_size;
        comp_page_offset += ck_stat_size;
        ck_g.ck_stat_size = ck_stat_size;
      }
      ck_g.num_pages          = num_pages;
      ck_g.bfr_size           = page_offset;
      ck_g.compressed_size    = comp_page_offset;
      pagestats_g.start_chunk = ck_g.first_page + ck_g.has_dictionary;  // Exclude dictionary
      pagestats_g.num_chunks  = num_pages - ck_g.has_dictionary;
    }
  }
  __syncthreads();
  if (t < sizeof(EncColumnChunk) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&chunks[blockIdx.y * num_columns + blockIdx.x])[t] =
      reinterpret_cast<uint32_t *>(&ck_g)[t];
  }
  if (chunk_grstats && t < sizeof(statistics_merge_group) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&chunk_grstats[blockIdx.y * num_columns + blockIdx.x])[t] =
      reinterpret_cast<uint32_t *>(&pagestats_g)[t];
  }
}

/**
 * @brief Mask table representing how many consecutive repeats are needed to code a repeat run
 *[nbits-1]
 **/
static __device__ __constant__ uint32_t kRleRunMask[16] = {
  0x00ffffff, 0x0fff, 0x00ff, 0x3f, 0x0f, 0x0f, 0x7, 0x7, 0x3, 0x3, 0x3, 0x3, 0x1, 0x1, 0x1, 0x1};

/**
 * @brief Variable-length encode an integer
 **/
inline __device__ uint8_t *VlqEncode(uint8_t *p, uint32_t v)
{
  while (v > 0x7f) {
    *p++ = (v | 0x80);
    v >>= 7;
  }
  *p++ = v;
  return p;
}

/**
 * @brief Pack literal values in output bitstream (1,2,4,8,12 or 16 bits per value)
 **/
inline __device__ void PackLiterals(
  uint8_t *dst, uint32_t v, uint32_t count, uint32_t w, uint32_t t)
{
  if (t <= (count | 0x1f)) {
    if (w < 8) {
      uint32_t mask;
      if (w <= 1) {
        v |= SHFL_XOR(v, 1) << 1;
        v |= SHFL_XOR(v, 2) << 2;
        v |= SHFL_XOR(v, 4) << 4;
        mask = 0x7;
      } else if (w <= 2) {
        v |= SHFL_XOR(v, 1) << 2;
        v |= SHFL_XOR(v, 2) << 4;
        mask = 0x3;
      } else {  // w=4
        v |= SHFL_XOR(v, 1) << 4;
        mask = 0x1;
      }
      if (t < count && !(t & mask)) { dst[(t * w) >> 3] = v; }
    } else if (w < 12) {  // w=8
      if (t < count) { dst[t] = v; }
    } else if (w < 16) {  // w=12
      v |= SHFL_XOR(v, 1) << 12;
      if (t < count && !(t & 1)) {
        dst[(t >> 1) * 3 + 0] = v;
        dst[(t >> 1) * 3 + 1] = v >> 8;
        dst[(t >> 1) * 3 + 2] = v >> 16;
      }
    } else if (t < count) {  // w=16
      dst[t * 2 + 0] = v;
      dst[t * 2 + 1] = v >> 8;
    }
  }
}

/**
 * @brief RLE encoder
 *
 * @param[in,out] s Page encode state
 * @param[in] numvals Total count of input values
 * @param[in] nbits number of bits per symbol (1..16)
 * @param[in] flush nonzero if last batch in block
 * @param[in] t thread id (0..127)
 */
static __device__ void RleEncode(
  page_enc_state_s *s, uint32_t numvals, uint32_t nbits, uint32_t flush, uint32_t t)
{
  uint32_t rle_pos = s->rle_pos;
  uint32_t rle_run = s->rle_run;

  while (rle_pos < numvals) {
    uint32_t pos = rle_pos + t;
    if (rle_run > 0 && !(rle_run & 1)) {
      // Currently in a long repeat run
      uint32_t mask = BALLOT(pos < numvals && s->vals[pos & (RLE_BFRSZ - 1)] == s->run_val);
      uint32_t rle_rpt_count, max_rpt_count;
      if (!(t & 0x1f)) { s->rpt_map[t >> 5] = mask; }
      __syncthreads();
      if (t < 32) {
        uint32_t c32 = BALLOT(t >= 4 || s->rpt_map[t] != 0xffffffffu);
        if (!t) {
          uint32_t last_idx = __ffs(c32) - 1;
          s->rle_rpt_count =
            last_idx * 32 + ((last_idx < 4) ? __ffs(~s->rpt_map[last_idx]) - 1 : 0);
        }
      }
      __syncthreads();
      max_rpt_count = min(numvals - rle_pos, 128);
      rle_rpt_count = s->rle_rpt_count;
      rle_run += rle_rpt_count << 1;
      rle_pos += rle_rpt_count;
      if (rle_rpt_count < max_rpt_count || (flush && rle_pos == numvals)) {
        if (!t) {
          uint32_t run_val = s->run_val;
          uint8_t *dst     = VlqEncode(s->rle_out, rle_run);
          *dst++           = run_val;
          if (nbits > 8) { *dst++ = run_val >> 8; }
          s->rle_out = dst;
        }
        rle_run = 0;
      }
    } else {
      // New run or in a literal run
      uint32_t v0      = s->vals[pos & (RLE_BFRSZ - 1)];
      uint32_t v1      = s->vals[(pos + 1) & (RLE_BFRSZ - 1)];
      uint32_t mask    = BALLOT(pos + 1 < numvals && v0 == v1);
      uint32_t maxvals = min(numvals - rle_pos, 128);
      uint32_t rle_lit_count, rle_rpt_count;
      if (!(t & 0x1f)) { s->rpt_map[t >> 5] = mask; }
      __syncthreads();
      if (t < 32) {
        // Repeat run can only start on a multiple of 8 values
        uint32_t idx8        = (t * 8) >> 5;
        uint32_t pos8        = (t * 8) & 0x1f;
        uint32_t m0          = (idx8 < 4) ? s->rpt_map[idx8] : 0;
        uint32_t m1          = (idx8 < 3) ? s->rpt_map[idx8 + 1] : 0;
        uint32_t needed_mask = kRleRunMask[nbits - 1];
        mask                 = BALLOT((__funnelshift_r(m0, m1, pos8) & needed_mask) == needed_mask);
        if (!t) {
          uint32_t rle_run_start = (mask != 0) ? min((__ffs(mask) - 1) * 8, maxvals) : maxvals;
          uint32_t rpt_len       = 0;
          if (rle_run_start < maxvals) {
            uint32_t idx_cur = rle_run_start >> 5;
            uint32_t idx_ofs = rle_run_start & 0x1f;
            while (idx_cur < 4) {
              m0   = (idx_cur < 4) ? s->rpt_map[idx_cur] : 0;
              m1   = (idx_cur < 3) ? s->rpt_map[idx_cur + 1] : 0;
              mask = ~__funnelshift_r(m0, m1, idx_ofs);
              if (mask != 0) {
                rpt_len += __ffs(mask) - 1;
                break;
              }
              rpt_len += 32;
              idx_cur++;
            }
          }
          s->rle_lit_count = rle_run_start;
          s->rle_rpt_count = min(rpt_len, maxvals - rle_run_start);
        }
      }
      __syncthreads();
      rle_lit_count = s->rle_lit_count;
      rle_rpt_count = s->rle_rpt_count;
      if (rle_lit_count != 0 || (rle_run != 0 && rle_rpt_count != 0)) {
        uint32_t lit_div8;
        bool need_more_data = false;
        if (!flush && rle_pos + rle_lit_count == numvals) {
          // Wait for more data
          rle_lit_count -= min(rle_lit_count, 24);
          need_more_data = true;
        }
        if (rle_lit_count != 0) {
          lit_div8 = (rle_lit_count + ((flush && rle_pos + rle_lit_count == numvals) ? 7 : 0)) >> 3;
          if (rle_run + lit_div8 * 2 > 0x7f) {
            lit_div8      = 0x3f - (rle_run >> 1);  // Limit to fixed 1-byte header (504 literals)
            rle_rpt_count = 0;                      // Defer repeat run
          }
          if (lit_div8 != 0) {
            uint8_t *dst = s->rle_out + 1 + (rle_run >> 1) * nbits;
            PackLiterals(dst, (rle_pos + t < numvals) ? v0 : 0, lit_div8 * 8, nbits, t);
            rle_run = (rle_run + lit_div8 * 2) | 1;
            rle_pos = min(rle_pos + lit_div8 * 8, numvals);
          }
        }
        if (rle_run >= ((rle_rpt_count != 0 || (flush && rle_pos == numvals)) ? 0x03 : 0x7f)) {
          __syncthreads();
          // Complete literal run
          if (!t) {
            uint8_t *dst = s->rle_out;
            dst[0]       = rle_run;  // At most 0x7f
            dst += 1 + nbits * (rle_run >> 1);
            s->rle_out = dst;
          }
          rle_run = 0;
        }
        if (need_more_data) { break; }
      }
      // Start a repeat run
      if (rle_rpt_count != 0) {
        if (t == s->rle_lit_count) { s->run_val = v0; }
        rle_run = rle_rpt_count * 2;
        rle_pos += rle_rpt_count;
        if (rle_pos + 1 == numvals && !flush) { break; }
      }
    }
    __syncthreads();
  }
  __syncthreads();
  if (!t) {
    s->rle_run     = rle_run;
    s->rle_pos     = rle_pos;
    s->rle_numvals = numvals;
  }
}

/**
 * @brief PLAIN bool encoder
 *
 * @param[in,out] s Page encode state
 * @param[in] numvals Total count of input values
 * @param[in] flush nonzero if last batch in block
 * @param[in] t thread id (0..127)
 */
static __device__ void PlainBoolEncode(page_enc_state_s *s,
                                       uint32_t numvals,
                                       uint32_t flush,
                                       uint32_t t)
{
  uint32_t rle_pos = s->rle_pos;
  uint8_t *dst     = s->rle_out;

  while (rle_pos < numvals) {
    uint32_t pos    = rle_pos + t;
    uint32_t v      = (pos < numvals) ? s->vals[pos & (RLE_BFRSZ - 1)] : 0;
    uint32_t n      = min(numvals - rle_pos, 128);
    uint32_t nbytes = (n + ((flush) ? 7 : 0)) >> 3;
    if (!nbytes) { break; }
    v |= SHFL_XOR(v, 1) << 1;
    v |= SHFL_XOR(v, 2) << 2;
    v |= SHFL_XOR(v, 4) << 4;
    if (t < n && !(t & 7)) { dst[t >> 3] = v; }
    rle_pos = min(rle_pos + nbytes * 8, numvals);
    dst += nbytes;
  }
  __syncthreads();
  if (!t) {
    s->rle_pos     = rle_pos;
    s->rle_numvals = numvals;
    s->rle_out     = dst;
  }
}

// blockDim(128, 1, 1)
__global__ void __launch_bounds__(128, 8) gpuEncodePages(EncPage *pages,
                                                         const EncColumnChunk *chunks,
                                                         gpu_inflate_input_s *comp_in,
                                                         gpu_inflate_status_s *comp_out,
                                                         uint32_t start_page)
{
  __shared__ __align__(8) page_enc_state_s state_g;

  page_enc_state_s *const s = &state_g;
  uint32_t t                = threadIdx.x;
  uint32_t dtype, dtype_len_in, dtype_len_out;
  int32_t dict_bits;

  if (t < sizeof(EncPage) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&s->page)[t] =
      reinterpret_cast<uint32_t *>(&pages[start_page + blockIdx.x])[t];
  }
  __syncthreads();
  if (t < sizeof(EncColumnChunk) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&s->ck)[t] =
      reinterpret_cast<const uint32_t *>(&chunks[s->page.chunk_id])[t];
  }
  __syncthreads();
  if (t < sizeof(EncColumnDesc) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&s->col)[t] =
      reinterpret_cast<const uint32_t *>(s->ck.col_desc)[t];
  }
  __syncthreads();
  if (!t) { s->cur = s->page.page_data + s->page.max_hdr_size; }
  __syncthreads();
  // Encode NULLs
  if (s->page.page_type != DICTIONARY_PAGE && s->col.level_bits != 0) {
    const uint32_t *valid = s->col.valid_map_base;
    uint32_t def_lvl_bits = s->col.level_bits & 0xf;
    if (def_lvl_bits != 0) {
      if (!t) {
        s->rle_run     = 0;
        s->rle_pos     = 0;
        s->rle_numvals = 0;
        s->rle_out     = s->cur + 4;
      }
      __syncthreads();
      while (s->rle_numvals < s->page.num_rows) {
        uint32_t rle_numvals = s->rle_numvals;
        uint32_t nrows       = min(s->page.num_rows - rle_numvals, 128);
        uint32_t row         = s->page.start_row + rle_numvals + t;
        uint32_t def_lvl     = (rle_numvals + t < s->page.num_rows && row < s->col.num_rows)
                             ? (valid) ? (valid[row >> 5] >> (row & 0x1f)) & 1 : 1
                             : 0;
        s->vals[(rle_numvals + t) & (RLE_BFRSZ - 1)] = def_lvl;
        __syncthreads();
        rle_numvals += nrows;
        RleEncode(s, rle_numvals, def_lvl_bits, (rle_numvals == s->page.num_rows), t);
        __syncthreads();
      }
      if (t < 32) {
        uint8_t *cur     = s->cur;
        uint8_t *rle_out = s->rle_out;
        if (t < 4) {
          uint32_t rle_bytes = (uint32_t)(rle_out - cur) - 4;
          cur[t]             = rle_bytes >> (t * 8);
        }
        SYNCWARP();
        if (t == 0) { s->cur = rle_out; }
      }
    }
  }
  // Encode data values
  __syncthreads();
  dtype         = s->col.physical_type;
  dtype_len_out = (dtype == INT64 || dtype == DOUBLE) ? 8 : (dtype == BOOLEAN) ? 1 : 4;
  if (dtype == INT32) {
    dtype_len_in = GetDtypeLogicalLen(s->col.converted_type);
  } else {
    dtype_len_in = (dtype == BYTE_ARRAY) ? sizeof(nvstrdesc_s) : dtype_len_out;
  }
  dict_bits = (dtype == BOOLEAN) ? 1 : (s->page.dict_bits_plus1 - 1);
  if (t == 0) {
    uint8_t *dst   = s->cur;
    s->rle_run     = 0;
    s->rle_pos     = 0;
    s->rle_numvals = 0;
    s->rle_out     = dst;
    if (dict_bits >= 0 && dtype != BOOLEAN) {
      dst[0]     = dict_bits;
      s->rle_out = dst + 1;
    }
  }
  __syncthreads();
  for (uint32_t cur_row = 0; cur_row < s->page.num_rows;) {
    uint32_t nrows = min(s->page.num_rows - cur_row, 128);
    uint32_t row   = s->page.start_row + cur_row + t;
    uint32_t is_valid, warp_valids, len, pos;

    if (s->page.page_type == DICTIONARY_PAGE) {
      is_valid = (cur_row + t < s->page.num_rows);
      row      = (is_valid) ? s->col.dict_data[row] : row;
    } else {
      const uint32_t *valid = s->col.valid_map_base;
      is_valid              = (row < s->col.num_rows && cur_row + t < s->page.num_rows)
                   ? (valid) ? (valid[row >> 5] >> (row & 0x1f)) & 1 : 1
                   : 0;
    }
    warp_valids = BALLOT(is_valid);
    cur_row += nrows;
    if (dict_bits >= 0) {
      // Dictionary encoding
      if (dict_bits > 0) {
        uint32_t rle_numvals;

        pos = __popc(warp_valids & ((1 << (t & 0x1f)) - 1));
        if (!(t & 0x1f)) { s->scratch_red[t >> 5] = __popc(warp_valids); }
        __syncthreads();
        if (t < 32) { s->scratch_red[t] = WarpReducePos4((t < 4) ? s->scratch_red[t] : 0, t); }
        __syncthreads();
        pos         = pos + ((t >= 32) ? s->scratch_red[(t - 32) >> 5] : 0);
        rle_numvals = s->rle_numvals;
        if (is_valid) {
          uint32_t v;
          if (dtype == BOOLEAN) {
            v = reinterpret_cast<const uint8_t *>(s->col.column_data_base)[row];
          } else {
            v = s->col.dict_index[row];
          }
          s->vals[(rle_numvals + pos) & (RLE_BFRSZ - 1)] = v;
        }
        rle_numvals += s->scratch_red[3];
        __syncthreads();
#if !ENABLE_BOOL_RLE
        if (dtype == BOOLEAN) {
          PlainBoolEncode(s, rle_numvals, (cur_row == s->page.num_rows), t);
        } else
#endif
          RleEncode(s, rle_numvals, dict_bits, (cur_row == s->page.num_rows), t);
        __syncthreads();
      }
      if (t == 0) { s->cur = s->rle_out; }
      __syncthreads();
    } else {
      // Non-dictionary encoding
      uint8_t *dst = s->cur;

      if (is_valid) {
        len = dtype_len_out;
        if (dtype == BYTE_ARRAY) {
          len +=
            (uint32_t) reinterpret_cast<const nvstrdesc_s *>(s->col.column_data_base)[row].count;
        }
      } else {
        len = 0;
      }
      pos = WarpReducePos32(len, t);
      if ((t & 0x1f) == 0x1f) { s->scratch_red[t >> 5] = pos; }
      __syncthreads();
      if (t < 32) { s->scratch_red[t] = WarpReducePos4((t < 4) ? s->scratch_red[t] : 0, t); }
      __syncthreads();
      if (t == 0) { s->cur = dst + s->scratch_red[3]; }
      pos = pos + ((t >= 32) ? s->scratch_red[(t - 32) >> 5] : 0) - len;
      if (is_valid) {
        const uint8_t *src8 =
          reinterpret_cast<const uint8_t *>(s->col.column_data_base) + row * (size_t)dtype_len_in;
        switch (dtype) {
          case INT32:
          case FLOAT: {
            int32_t v;
            if (dtype_len_in == 4)
              v = *reinterpret_cast<const int32_t *>(src8);
            else if (dtype_len_in == 2)
              v = *reinterpret_cast<const int16_t *>(src8);
            else
              v = *reinterpret_cast<const int8_t *>(src8);
            dst[pos + 0] = v;
            dst[pos + 1] = v >> 8;
            dst[pos + 2] = v >> 16;
            dst[pos + 3] = v >> 24;
          } break;
          case INT64: {
            int64_t v        = *reinterpret_cast<const int64_t *>(src8);
            int32_t ts_scale = s->col.ts_scale;
            if (ts_scale != 0) {
              if (ts_scale < 0) {
                v /= -ts_scale;
              } else {
                v *= ts_scale;
              }
            }
            dst[pos + 0] = v;
            dst[pos + 1] = v >> 8;
            dst[pos + 2] = v >> 16;
            dst[pos + 3] = v >> 24;
            dst[pos + 4] = v >> 32;
            dst[pos + 5] = v >> 40;
            dst[pos + 6] = v >> 48;
            dst[pos + 7] = v >> 56;
          } break;
          case DOUBLE: memcpy(dst + pos, src8, 8); break;
          case BYTE_ARRAY: {
            const char *str_data = reinterpret_cast<const nvstrdesc_s *>(src8)->ptr;
            uint32_t v           = len - 4;  // string length
            dst[pos + 0]         = v;
            dst[pos + 1]         = v >> 8;
            dst[pos + 2]         = v >> 16;
            dst[pos + 3]         = v >> 24;
            if (v != 0) memcpy(dst + pos + 4, str_data, v);
          } break;
        }
      }
      __syncthreads();
    }
  }
  if (t == 0) {
    uint8_t *base                = s->page.page_data + s->page.max_hdr_size;
    uint32_t actual_data_size    = static_cast<uint32_t>(s->cur - base);
    uint32_t compressed_bfr_size = GetMaxCompressedBfrSize(actual_data_size);
    s->page.max_data_size        = actual_data_size;
    s->comp_in.srcDevice         = base;
    s->comp_in.srcSize           = actual_data_size;
    s->comp_in.dstDevice         = s->page.compressed_data + s->page.max_hdr_size;
    s->comp_in.dstSize           = compressed_bfr_size;
    s->comp_out.bytes_written    = 0;
    s->comp_out.status           = ~0;
    s->comp_out.reserved         = 0;
  }
  __syncthreads();
  if (t < sizeof(EncPage) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&pages[start_page + blockIdx.x])[t] =
      reinterpret_cast<uint32_t *>(&s->page)[t];
  }
  if (comp_in && t < sizeof(gpu_inflate_input_s) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&comp_in[blockIdx.x])[t] =
      reinterpret_cast<uint32_t *>(&s->comp_in)[t];
  }
  if (comp_out && t < sizeof(gpu_inflate_status_s) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&comp_out[blockIdx.x])[t] =
      reinterpret_cast<uint32_t *>(&s->comp_out)[t];
  }
}

// blockDim(128, 1, 1)
__global__ void __launch_bounds__(128) gpuDecideCompression(EncColumnChunk *chunks,
                                                            const EncPage *pages,
                                                            const gpu_inflate_status_s *comp_out,
                                                            uint32_t start_page)
{
  __shared__ __align__(8) EncColumnChunk ck_g;
  __shared__ __align__(4) unsigned int error_count;

  uint32_t t                      = threadIdx.x;
  uint32_t uncompressed_data_size = 0;
  uint32_t compressed_data_size   = 0;
  uint32_t first_page, num_pages;

  if (t < sizeof(EncColumnChunk) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&ck_g)[t] =
      reinterpret_cast<const uint32_t *>(&chunks[blockIdx.x])[t];
  }
  if (t == 0) { atomicAnd(&error_count, 0); }
  __syncthreads();
  if (t < 32) {
    first_page = ck_g.first_page;
    num_pages  = ck_g.num_pages;
    for (uint32_t page = t; page < num_pages; page += 32) {
      uint32_t page_data_size = pages[first_page + page].max_data_size;
      uint32_t comp_idx       = first_page + page - start_page;
      uncompressed_data_size += page_data_size;
      if (comp_out) {
        compressed_data_size += (uint32_t)comp_out[comp_idx].bytes_written;
        if (comp_out[comp_idx].status != 0) { atomicAdd(&error_count, 1); }
      }
    }
    uncompressed_data_size = WarpReduceSum32(uncompressed_data_size);
    compressed_data_size   = WarpReduceSum32(compressed_data_size);
  }
  __syncthreads();
  if (t == 0) {
    bool is_compressed;
    if (comp_out) {
      uint32_t compression_error = atomicAdd(&error_count, 0);
      is_compressed = (!compression_error && compressed_data_size < uncompressed_data_size);
    } else {
      is_compressed = false;
    }
    chunks[blockIdx.x].is_compressed = is_compressed;
    chunks[blockIdx.x].bfr_size      = uncompressed_data_size;
    chunks[blockIdx.x].compressed_size =
      (is_compressed) ? compressed_data_size : uncompressed_data_size;
  }
}

/**
 * Minimal thrift compact protocol support
 **/
inline __device__ uint8_t *cpw_put_uint32(uint8_t *p, uint32_t v)
{
  while (v > 0x7f) {
    *p++ = v | 0x80;
    v >>= 7;
  }
  *p++ = v;
  return p;
}

inline __device__ uint8_t *cpw_put_uint64(uint8_t *p, uint64_t v)
{
  while (v > 0x7f) {
    *p++ = v | 0x80;
    v >>= 7;
  }
  *p++ = v;
  return p;
}

inline __device__ uint8_t *cpw_put_int32(uint8_t *p, int32_t v)
{
  int32_t s = (v < 0);
  return cpw_put_uint32(p, (v ^ -s) * 2 + s);
}

inline __device__ uint8_t *cpw_put_int64(uint8_t *p, int64_t v)
{
  int64_t s = (v < 0);
  return cpw_put_uint64(p, (v ^ -s) * 2 + s);
}

inline __device__ uint8_t *cpw_put_fldh(uint8_t *p, int f, int cur, int t)
{
  if (f > cur && f <= cur + 15) {
    *p++ = ((f - cur) << 4) | t;
    return p;
  } else {
    *p++ = t;
    return cpw_put_int32(p, f);
  }
}

#define CPW_BEGIN_STRUCT(hdr_start) \
  {                                 \
    uint8_t *p  = hdr_start;        \
    int cur_fld = 0;

#define CPW_FLD_STRUCT_BEGIN(f)                         \
  p       = cpw_put_fldh(p, f, cur_fld, ST_FLD_STRUCT); \
  cur_fld = 0;

#define CPW_FLD_STRUCT_END(f) \
  *p++    = 0;                \
  cur_fld = f;

#define CPW_FLD_INT32(f, v)                          \
  p       = cpw_put_fldh(p, f, cur_fld, ST_FLD_I32); \
  p       = cpw_put_int32(p, v);                     \
  cur_fld = f;

#define CPW_FLD_INT64(f, v)                          \
  p       = cpw_put_fldh(p, f, cur_fld, ST_FLD_I64); \
  p       = cpw_put_int64(p, v);                     \
  cur_fld = f;

#define CPW_FLD_BINARY(f, v, l)                   \
  p = cpw_put_fldh(p, f, cur_fld, ST_FLD_BINARY); \
  p = cpw_put_uint32(p, l);                       \
  memcpy(p, v, l);                                \
  p += l;                                         \
  cur_fld = f;

#define CPW_END_STRUCT(hdr_end) \
  *p++    = 0;                  \
  hdr_end = p;                  \
  }

#define CPW_END_STRUCT_NOTERMINATION(hdr_end) \
  hdr_end = p;                                \
  }

__device__ uint8_t *EncodeStatistics(uint8_t *start,
                                     const statistics_chunk *s,
                                     const EncColumnDesc *col,
                                     float *fp_scratch)
{
  uint8_t *end, dtype, dtype_len;
  dtype = col->stats_dtype;
  switch (dtype) {
    case dtype_bool: dtype_len = 1; break;
    case dtype_int8:
    case dtype_int16:
    case dtype_int32:
    case dtype_date32:
    case dtype_float32: dtype_len = 4; break;
    case dtype_int64:
    case dtype_timestamp64:
    case dtype_float64:
    case dtype_decimal64: dtype_len = 8; break;
    case dtype_decimal128: dtype_len = 16; break;
    case dtype_string:
    default: dtype_len = 0; break;
  }
  CPW_BEGIN_STRUCT(start)
  CPW_FLD_INT64(3, s->null_count)
  if (s->has_minmax) {
    const void *vmin, *vmax;
    uint32_t lmin, lmax;

    if (dtype == dtype_string) {
      lmin = s->min_value.str_val.length;
      vmin = s->min_value.str_val.ptr;
      lmax = s->max_value.str_val.length;
      vmax = s->max_value.str_val.ptr;
    } else {
      lmin = lmax = dtype_len;
      if (dtype == dtype_float32) {  // Convert from double to float32
        fp_scratch[0] = s->min_value.fp_val;
        fp_scratch[1] = s->max_value.fp_val;
        vmin          = &fp_scratch[0];
        vmax          = &fp_scratch[1];
      } else {
        vmin = &s->min_value;
        vmax = &s->max_value;
      }
    }
    CPW_FLD_BINARY(5, vmax, lmax);
    CPW_FLD_BINARY(6, vmin, lmin);
  }
  CPW_END_STRUCT_NOTERMINATION(end);
  return end;
}

// blockDim(128, 1, 1)
__global__ void __launch_bounds__(128) gpuEncodePageHeaders(EncPage *pages,
                                                            EncColumnChunk *chunks,
                                                            const gpu_inflate_status_s *comp_out,
                                                            const statistics_chunk *page_stats,
                                                            const statistics_chunk *chunk_stats,
                                                            uint32_t start_page)
{
  __shared__ __align__(8) EncColumnDesc col_g;
  __shared__ __align__(8) EncColumnChunk ck_g;
  __shared__ __align__(8) EncPage page_g;
  __shared__ __align__(8) float fp_scratch[2];

  uint32_t t = threadIdx.x;

  if (t < sizeof(EncPage) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&page_g)[t] =
      reinterpret_cast<uint32_t *>(&pages[start_page + blockIdx.x])[t];
  }
  __syncthreads();
  if (t < sizeof(EncColumnChunk) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&ck_g)[t] =
      reinterpret_cast<const uint32_t *>(&chunks[page_g.chunk_id])[t];
  }
  __syncthreads();
  if (t < sizeof(EncColumnDesc) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&col_g)[t] = reinterpret_cast<const uint32_t *>(ck_g.col_desc)[t];
  }
  __syncthreads();
  if (!t) {
    uint8_t *hdr_start, *hdr_end;
    uint32_t compressed_page_size, uncompressed_page_size;

    if (chunk_stats && start_page + blockIdx.x == ck_g.first_page) {
      hdr_start = (ck_g.is_compressed) ? ck_g.compressed_bfr : ck_g.uncompressed_bfr;
      hdr_end   = EncodeStatistics(hdr_start, &chunk_stats[page_g.chunk_id], &col_g, fp_scratch);
      chunks[page_g.chunk_id].ck_stat_size = static_cast<uint32_t>(hdr_end - hdr_start);
    }
    uncompressed_page_size = page_g.max_data_size;
    if (ck_g.is_compressed) {
      hdr_start            = page_g.compressed_data;
      compressed_page_size = (uint32_t)comp_out[blockIdx.x].bytes_written;
      page_g.max_data_size = compressed_page_size;
    } else {
      hdr_start            = page_g.page_data;
      compressed_page_size = uncompressed_page_size;
    }
    CPW_BEGIN_STRUCT(hdr_start)
    int page_type = page_g.page_type;
    // NOTE: For dictionary encoding, parquet v2 recommends using PLAIN in dictionary page and
    // RLE_DICTIONARY in data page, but parquet v1 uses PLAIN_DICTIONARY in both dictionary and data
    // pages (actual encoding is identical).
#if ENABLE_BOOL_RLE
    int encoding =
      (col_g.physical_type != BOOLEAN)
        ? (page_type == DICTIONARY_PAGE || page_g.dict_bits_plus1 != 0) ? PLAIN_DICTIONARY : PLAIN
        : RLE;
#else
    int encoding =
      (page_type == DICTIONARY_PAGE || page_g.dict_bits_plus1 != 0) ? PLAIN_DICTIONARY : PLAIN;
#endif
    CPW_FLD_INT32(1, page_type)
    CPW_FLD_INT32(2, uncompressed_page_size)
    CPW_FLD_INT32(3, compressed_page_size)
    if (page_type == DATA_PAGE) {
      // DataPageHeader
      CPW_FLD_STRUCT_BEGIN(5)
      CPW_FLD_INT32(1, page_g.num_rows)  // NOTE: num_values != num_rows for list types
      CPW_FLD_INT32(2, encoding)         // encoding
      CPW_FLD_INT32(3, RLE)              // definition_level_encoding
      CPW_FLD_INT32(4, RLE)              // repetition_level_encoding
      // Optionally encode page-level statistics
      if (page_stats) {
        CPW_FLD_STRUCT_BEGIN(5)
        p = EncodeStatistics(p, &page_stats[start_page + blockIdx.x], &col_g, fp_scratch);
        CPW_FLD_STRUCT_END(5)
      }
      CPW_FLD_STRUCT_END(5)
    } else {
      // DictionaryPageHeader
      CPW_FLD_STRUCT_BEGIN(7)
      CPW_FLD_INT32(1, ck_g.total_dict_entries)  // number of values in dictionary
      CPW_FLD_INT32(2, encoding)
      CPW_FLD_STRUCT_END(7)
    }
    CPW_END_STRUCT(hdr_end)
    page_g.hdr_size = (uint32_t)(hdr_end - hdr_start);
  }
  __syncthreads();
  if (t < sizeof(EncPage) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&pages[start_page + blockIdx.x])[t] =
      reinterpret_cast<uint32_t *>(&page_g)[t];
  }
}

// blockDim(1024, 1, 1)
__global__ void __launch_bounds__(1024) gpuGatherPages(EncColumnChunk *chunks, const EncPage *pages)
{
  __shared__ __align__(8) EncColumnChunk ck_g;
  __shared__ __align__(8) EncPage page_g;

  uint32_t t = threadIdx.x;
  uint8_t *dst, *dst_base;
  const EncPage *first_page;
  uint32_t num_pages, uncompressed_size;

  if (t < sizeof(EncColumnChunk) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&ck_g)[t] =
      reinterpret_cast<const uint32_t *>(&chunks[blockIdx.x])[t];
  }
  __syncthreads();
  first_page = &pages[ck_g.first_page];
  num_pages  = ck_g.num_pages;
  dst        = (ck_g.is_compressed) ? ck_g.compressed_bfr : ck_g.uncompressed_bfr;
  dst += ck_g.ck_stat_size;  // Skip over chunk statistics
  dst_base          = dst;
  uncompressed_size = ck_g.bfr_size;
  for (uint32_t page = 0; page < num_pages; page++) {
    const uint8_t *src;
    uint32_t hdr_len, data_len;

    if (t < sizeof(EncPage) / sizeof(uint32_t)) {
      reinterpret_cast<uint32_t *>(&page_g)[t] =
        reinterpret_cast<const uint32_t *>(&first_page[page])[t];
    }
    __syncthreads();

    src = (ck_g.is_compressed) ? page_g.compressed_data : page_g.page_data;
    // Copy page header
    hdr_len = page_g.hdr_size;
    memcpy_block<1024, true>(dst, src, hdr_len, t);
    src += page_g.max_hdr_size;
    dst += hdr_len;
    // Copy page data
    uncompressed_size += hdr_len;
    data_len = page_g.max_data_size;
    memcpy_block<1024, true>(dst, src, data_len, t);
    dst += data_len;
    __syncthreads();
    if (!t && page == 0 && ck_g.has_dictionary) { ck_g.dictionary_size = hdr_len + data_len; }
  }
  if (t == 0) {
    chunks[blockIdx.x].bfr_size        = uncompressed_size;
    chunks[blockIdx.x].compressed_size = (dst - dst_base);
    if (ck_g.has_dictionary) { chunks[blockIdx.x].dictionary_size = ck_g.dictionary_size; }
  }
}

/**
 * @brief Launches kernel for initializing encoder page fragments
 *
 * @param[in] frag Fragment array [column_id][fragment_id]
 * @param[in] col_desc Column description array [column_id]
 * @param[in] num_fragments Number of fragments per column
 * @param[in] num_columns Number of columns
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t InitPageFragments(PageFragment *frag,
                              const EncColumnDesc *col_desc,
                              int32_t num_fragments,
                              int32_t num_columns,
                              uint32_t fragment_size,
                              uint32_t num_rows,
                              cudaStream_t stream)
{
  dim3 dim_grid(num_columns, num_fragments);  // 1 threadblock per fragment
  gpuInitPageFragments<<<dim_grid, 512, 0, stream>>>(
    frag, col_desc, num_fragments, num_columns, fragment_size, num_rows);
  return cudaSuccess;
}

/**
 * @brief Launches kernel for initializing fragment statistics groups
 *
 * @param[out] groups Statistics groups [num_columns x num_fragments]
 * @param[in] fragments Page fragments [num_columns x num_fragments]
 * @param[in] col_desc Column description [num_columns]
 * @param[in] num_fragments Number of fragments
 * @param[in] num_columns Number of columns
 * @param[in] fragment_size Max size of each fragment in rows
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t InitFragmentStatistics(statistics_group *groups,
                                   const PageFragment *fragments,
                                   const EncColumnDesc *col_desc,
                                   int32_t num_fragments,
                                   int32_t num_columns,
                                   uint32_t fragment_size,
                                   cudaStream_t stream)
{
  dim3 dim_grid(num_columns, (num_fragments + 3) >> 2);  // 1 warp per fragment
  gpuInitFragmentStats<<<dim_grid, 128, 0, stream>>>(
    groups, fragments, col_desc, num_fragments, num_columns, fragment_size);
  return cudaSuccess;
}

/**
 * @brief Launches kernel for initializing encoder data pages
 *
 * @param[in,out] chunks Column chunks [rowgroup][column]
 * @param[out] pages Encode page array (null if just counting pages)
 * @param[in] col_desc Column description array [column_id]
 * @param[in] num_rowgroups Number of fragments per column
 * @param[in] num_columns Number of columns
 * @param[in] page_grstats Setup for page-level stats
 * @param[in] chunk_grstats Setup for chunk-level stats
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t InitEncoderPages(EncColumnChunk *chunks,
                             EncPage *pages,
                             const EncColumnDesc *col_desc,
                             int32_t num_rowgroups,
                             int32_t num_columns,
                             statistics_merge_group *page_grstats,
                             statistics_merge_group *chunk_grstats,
                             cudaStream_t stream)
{
  dim3 dim_grid(num_columns, num_rowgroups);  // 1 threadblock per rowgroup
  gpuInitPages<<<dim_grid, 128, 0, stream>>>(
    chunks, pages, col_desc, page_grstats, chunk_grstats, num_rowgroups, num_columns);
  return cudaSuccess;
}

/**
 * @brief Launches kernel for packing column data into parquet pages
 *
 * @param[in,out] pages Device array of EncPages (unordered)
 * @param[in] chunks Column chunks
 * @param[in] num_pages Number of pages
 * @param[in] start_page First page to encode in page array
 * @param[out] comp_in Optionally initializes compressor input params
 * @param[out] comp_out Optionally initializes compressor output params
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t EncodePages(EncPage *pages,
                        const EncColumnChunk *chunks,
                        uint32_t num_pages,
                        uint32_t start_page,
                        gpu_inflate_input_s *comp_in,
                        gpu_inflate_status_s *comp_out,
                        cudaStream_t stream)
{
  gpuEncodePages<<<num_pages, 128, 0, stream>>>(pages, chunks, comp_in, comp_out, start_page);
  return cudaSuccess;
}

/**
 * @brief Launches kernel to make the compressed vs uncompressed chunk-level decision
 *
 * @param[in,out] chunks Column chunks
 * @param[in] pages Device array of EncPages (unordered)
 * @param[in] num_chunks Number of column chunks
 * @param[in] start_page First page to encode in page array
 * @param[in] comp_out Compressor status
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t DecideCompression(EncColumnChunk *chunks,
                              const EncPage *pages,
                              uint32_t num_chunks,
                              uint32_t start_page,
                              const gpu_inflate_status_s *comp_out,
                              cudaStream_t stream)
{
  gpuDecideCompression<<<num_chunks, 128, 0, stream>>>(chunks, pages, comp_out, start_page);
  return cudaSuccess;
}

/**
 * @brief Launches kernel to encode page headers
 *
 * @param[in,out] pages Device array of EncPages
 * @param[in,out] chunks Column chunks
 * @param[in] num_pages Number of pages
 * @param[in] start_page First page to encode in page array
 * @param[in] comp_out Compressor status or nullptr if no compression
 * @param[in] page_stats Optional page-level statistics to be included in page header
 * @param[in] chunk_stats Optional chunk-level statistics to be encoded
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t EncodePageHeaders(EncPage *pages,
                              EncColumnChunk *chunks,
                              uint32_t num_pages,
                              uint32_t start_page,
                              const gpu_inflate_status_s *comp_out,
                              const statistics_chunk *page_stats,
                              const statistics_chunk *chunk_stats,
                              cudaStream_t stream)
{
  gpuEncodePageHeaders<<<num_pages, 128, 0, stream>>>(
    pages, chunks, comp_out, page_stats, chunk_stats, start_page);
  return cudaSuccess;
}

/**
 * @brief Launches kernel to gather pages to a single contiguous block per chunk
 *
 * @param[in,out] chunks Column chunks
 * @param[in] pages Device array of EncPages
 * @param[in] num_chunks Number of column chunks
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t GatherPages(EncColumnChunk *chunks,
                        const EncPage *pages,
                        uint32_t num_chunks,
                        cudaStream_t stream)
{
  gpuGatherPages<<<num_chunks, 1024, 0, stream>>>(chunks, pages);
  return cudaSuccess;
}

}  // namespace gpu
}  // namespace parquet
}  // namespace io
}  // namespace cudf
