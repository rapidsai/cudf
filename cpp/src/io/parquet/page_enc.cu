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

#include "parquet_gpu.cuh"

#include <io/utilities/block_utils.cuh>

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>

#include <cuda/std/chrono>

#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>

namespace cudf {
namespace io {
namespace parquet {
namespace gpu {
// Spark doesn't support RLE encoding for BOOLEANs
#ifdef ENABLE_BOOL_RLE
constexpr bool enable_bool_rle = true;
#else
constexpr bool enable_bool_rle = false;
#endif

using ::cudf::detail::device_2dspan;

constexpr uint32_t rle_buffer_size = (1 << 9);

// do not truncate statistics
constexpr int32_t NO_TRUNC_STATS = 0;

// minimum scratch space required for encoding statistics
constexpr size_t MIN_STATS_SCRATCH_SIZE = sizeof(__int128_t);

// mask to determine lane id
constexpr uint32_t WARP_MASK = cudf::detail::warp_size - 1;

// currently 64k - 1
constexpr uint32_t MAX_GRID_Y_SIZE = (1 << 16) - 1;

struct frag_init_state_s {
  parquet_column_device_view col;
  PageFragment frag;
};

struct page_enc_state_s {
  uint8_t* cur;          //!< current output ptr
  uint8_t* rle_out;      //!< current RLE write ptr
  uint32_t rle_run;      //!< current RLE run
  uint32_t run_val;      //!< current RLE run value
  uint32_t rle_pos;      //!< RLE encoder positions
  uint32_t rle_numvals;  //!< RLE input value count
  uint32_t rle_lit_count;
  uint32_t rle_rpt_count;
  uint32_t page_start_val;
  uint32_t chunk_start_val;
  volatile uint32_t rpt_map[4];
  volatile uint32_t scratch_red[32];
  EncPage page;
  EncColumnChunk ck;
  parquet_column_device_view col;
  uint32_t vals[rle_buffer_size];
};

/**
 * @brief Returns the size of the type in the Parquet file.
 */
uint32_t __device__ physical_type_len(Type physical_type, type_id id)
{
  if (physical_type == FIXED_LEN_BYTE_ARRAY and id == type_id::DECIMAL128) {
    return sizeof(__int128_t);
  }
  switch (physical_type) {
    case INT96: return 12u;
    case INT64:
    case DOUBLE: return sizeof(int64_t);
    case BOOLEAN: return 1u;
    default: return sizeof(int32_t);
  }
}

// blockDim {512,1,1}
template <int block_size>
__global__ void __launch_bounds__(block_size)
  gpuInitPageFragments(device_2dspan<PageFragment> frag,
                       device_span<parquet_column_device_view const> col_desc,
                       device_span<partition_info const> partitions,
                       device_span<int const> part_frag_offset,
                       uint32_t fragment_size)
{
  __shared__ __align__(16) frag_init_state_s state_g;

  using block_reduce = cub::BlockReduce<uint32_t, block_size>;
  __shared__ typename block_reduce::TempStorage reduce_storage;

  frag_init_state_s* const s              = &state_g;
  uint32_t const t                        = threadIdx.x;
  auto const physical_type                = col_desc[blockIdx.x].physical_type;
  uint32_t const num_fragments_per_column = frag.size().second;

  if (t == 0) { s->col = col_desc[blockIdx.x]; }
  __syncthreads();

  auto const leaf_type = s->col.leaf_column->type().id();
  auto const dtype_len = physical_type_len(physical_type, leaf_type);

  for (uint32_t frag_y = blockIdx.y; frag_y < num_fragments_per_column; frag_y += gridDim.y) {
    if (t == 0) {
      // Find which partition this fragment came from
      auto it =
        thrust::upper_bound(thrust::seq, part_frag_offset.begin(), part_frag_offset.end(), frag_y);
      int p             = it - part_frag_offset.begin() - 1;
      int part_end_row  = partitions[p].start_row + partitions[p].num_rows;
      s->frag.start_row = (frag_y - part_frag_offset[p]) * fragment_size + partitions[p].start_row;

      // frag.num_rows = fragment_size except for the last fragment in partition which can be
      // smaller. num_rows is fixed but fragment size could be larger if the data is strings or
      // nested.
      s->frag.num_rows           = min(fragment_size, part_end_row - s->frag.start_row);
      s->frag.num_dict_vals      = 0;
      s->frag.fragment_data_size = 0;
      s->frag.dict_data_size     = 0;

      s->frag.start_value_idx = row_to_value_idx(s->frag.start_row, s->col);
      size_type end_value_idx = row_to_value_idx(s->frag.start_row + s->frag.num_rows, s->col);
      s->frag.num_leaf_values = end_value_idx - s->frag.start_value_idx;

      if (s->col.level_offsets != nullptr) {
        // For nested schemas, the number of values in a fragment is not directly related to the
        // number of encoded data elements or the number of rows.  It is simply the number of
        // repetition/definition values which together encode validity and nesting information.
        size_type first_level_val_idx = s->col.level_offsets[s->frag.start_row];
        size_type last_level_val_idx  = s->col.level_offsets[s->frag.start_row + s->frag.num_rows];
        s->frag.num_values            = last_level_val_idx - first_level_val_idx;
      } else {
        s->frag.num_values = s->frag.num_rows;
      }
    }
    __syncthreads();

    size_type nvals           = s->frag.num_leaf_values;
    size_type start_value_idx = s->frag.start_value_idx;

    for (uint32_t i = 0; i < nvals; i += block_size) {
      uint32_t val_idx  = start_value_idx + i + t;
      uint32_t is_valid = (i + t < nvals && val_idx < s->col.leaf_column->size())
                            ? s->col.leaf_column->is_valid(val_idx)
                            : 0;
      uint32_t len;
      if (is_valid) {
        len = dtype_len;
        if (physical_type == BYTE_ARRAY) {
          switch (leaf_type) {
            case type_id::STRING: {
              auto str = s->col.leaf_column->element<string_view>(val_idx);
              len += str.size_bytes();
            } break;
            case type_id::LIST: {
              auto list_element =
                get_element<statistics::byte_array_view>(*s->col.leaf_column, val_idx);
              len += list_element.size_bytes();
            } break;
            default: CUDF_UNREACHABLE("Unsupported data type for leaf column");
          }
        }
      } else {
        len = 0;
      }

      len = block_reduce(reduce_storage).Sum(len);
      if (t == 0) { s->frag.fragment_data_size += len; }
      __syncthreads();
    }
    __syncthreads();
    if (t == 0) { frag[blockIdx.x][frag_y] = s->frag; }
  }
}

// blockDim {128,1,1}
__global__ void __launch_bounds__(128)
  gpuInitFragmentStats(device_2dspan<statistics_group> groups,
                       device_2dspan<PageFragment const> fragments,
                       device_span<parquet_column_device_view const> col_desc)
{
  uint32_t const lane_id                  = threadIdx.x & WARP_MASK;
  uint32_t const column_id                = blockIdx.x;
  uint32_t const num_fragments_per_column = fragments.size().second;

  uint32_t frag_id = blockIdx.y * 4 + (threadIdx.x / cudf::detail::warp_size);
  while (frag_id < num_fragments_per_column) {
    if (lane_id == 0) {
      statistics_group g;
      g.col                      = &col_desc[column_id];
      g.start_row                = fragments[column_id][frag_id].start_value_idx;
      g.num_rows                 = fragments[column_id][frag_id].num_leaf_values;
      groups[column_id][frag_id] = g;
    }
    frag_id += gridDim.y * 4;
  }
}

constexpr uint32_t max_RLE_page_size(uint8_t value_bit_width, uint32_t num_values)
{
  if (value_bit_width == 0) return 0;

  // Run length = 4, max(rle/bitpack header) = 5, add one byte per 256 values for overhead
  return 4 + 5 + util::div_rounding_up_unsafe(num_values * value_bit_width, 8) + (num_values / 256);
}

// blockDim {128,1,1}
__global__ void __launch_bounds__(128)
  gpuInitPages(device_2dspan<EncColumnChunk> chunks,
               device_span<gpu::EncPage> pages,
               device_span<size_type> page_sizes,
               device_span<size_type> comp_page_sizes,
               device_span<parquet_column_device_view const> col_desc,
               statistics_merge_group* page_grstats,
               statistics_merge_group* chunk_grstats,
               int32_t num_columns,
               size_t max_page_size_bytes,
               size_type max_page_size_rows,
               uint32_t page_align)
{
  // TODO: All writing seems to be done by thread 0. Could be replaced by thrust foreach
  __shared__ __align__(8) parquet_column_device_view col_g;
  __shared__ __align__(8) EncColumnChunk ck_g;
  __shared__ __align__(8) PageFragment frag_g;
  __shared__ __align__(8) EncPage page_g;
  __shared__ __align__(8) statistics_merge_group pagestats_g;

  uint32_t t = threadIdx.x;

  if (t == 0) {
    col_g  = col_desc[blockIdx.x];
    ck_g   = chunks[blockIdx.y][blockIdx.x];
    page_g = {};
  }
  __syncthreads();
  if (t < 32) {
    uint32_t fragments_in_chunk  = 0;
    uint32_t rows_in_page        = 0;
    uint32_t values_in_page      = 0;
    uint32_t leaf_values_in_page = 0;
    uint32_t page_size           = 0;
    uint32_t num_pages           = 0;
    uint32_t num_rows            = 0;
    uint32_t page_start          = 0;
    uint32_t page_offset         = ck_g.ck_stat_size;
    uint32_t num_dict_entries    = 0;
    uint32_t comp_page_offset    = ck_g.ck_stat_size;
    uint32_t page_headers_size   = 0;
    uint32_t max_page_data_size  = 0;
    uint32_t cur_row             = ck_g.start_row;
    uint32_t ck_max_stats_len    = 0;
    uint32_t max_stats_len       = 0;

    if (!t) {
      pagestats_g.col_dtype   = col_g.leaf_column->type();
      pagestats_g.stats_dtype = col_g.stats_dtype;
      pagestats_g.start_chunk = ck_g.first_fragment;
      pagestats_g.num_chunks  = 0;
    }
    if (ck_g.use_dictionary) {
      if (!t) {
        page_g.page_data       = ck_g.uncompressed_bfr + page_offset;
        page_g.compressed_data = ck_g.compressed_bfr + comp_page_offset;
        page_g.num_fragments   = 0;
        page_g.page_type       = PageType::DICTIONARY_PAGE;
        page_g.chunk           = &chunks[blockIdx.y][blockIdx.x];
        page_g.chunk_id        = blockIdx.y * num_columns + blockIdx.x;
        page_g.hdr_size        = 0;
        page_g.max_hdr_size    = 32;
        page_g.max_data_size   = ck_g.uniq_data_size;
        page_g.start_row       = cur_row;
        page_g.num_rows        = ck_g.num_dict_entries;
        page_g.num_leaf_values = ck_g.num_dict_entries;
        page_g.num_values      = ck_g.num_dict_entries;  // TODO: shouldn't matter for dict page
        page_offset +=
          util::round_up_unsafe(page_g.max_hdr_size + page_g.max_data_size, page_align);
        if (not comp_page_sizes.empty()) {
          comp_page_offset += page_g.max_hdr_size + comp_page_sizes[ck_g.first_page];
        }
        page_headers_size += page_g.max_hdr_size;
        max_page_data_size = max(max_page_data_size, page_g.max_data_size);
      }
      __syncwarp();
      if (t == 0) {
        if (not pages.empty()) pages[ck_g.first_page] = page_g;
        if (not page_sizes.empty()) page_sizes[ck_g.first_page] = page_g.max_data_size;
        if (page_grstats) page_grstats[ck_g.first_page] = pagestats_g;
      }
      num_pages = 1;
    }
    __syncwarp();
    // This loop goes over one page fragment at a time and adds it to page.
    // When page size crosses a particular limit, then it moves on to the next page and then next
    // page fragment gets added to that one.

    // This doesn't actually deal with data. It's agnostic. It only cares about number of rows and
    // page size.
    do {
      uint32_t minmax_len = 0;
      __syncwarp();
      if (num_rows < ck_g.num_rows) {
        if (t == 0) { frag_g = ck_g.fragments[fragments_in_chunk]; }
        if (!t && ck_g.stats) {
          if (col_g.stats_dtype == dtype_string) {
            minmax_len = max(ck_g.stats[fragments_in_chunk].min_value.str_val.length,
                             ck_g.stats[fragments_in_chunk].max_value.str_val.length);
          } else if (col_g.stats_dtype == dtype_byte_array) {
            minmax_len = max(ck_g.stats[fragments_in_chunk].min_value.byte_val.length,
                             ck_g.stats[fragments_in_chunk].max_value.byte_val.length);
          }
        }
      } else if (!t) {
        frag_g.fragment_data_size = 0;
        frag_g.num_rows           = 0;
      }
      __syncwarp();
      uint32_t fragment_data_size =
        (ck_g.use_dictionary)
          ? frag_g.num_leaf_values * util::div_rounding_up_unsafe(ck_g.dict_rle_bits, 8)
          : frag_g.fragment_data_size;
      // TODO (dm): this convoluted logic to limit page size needs refactoring
      size_t this_max_page_size = (values_in_page * 2 >= ck_g.num_values)   ? 256 * 1024
                                  : (values_in_page * 3 >= ck_g.num_values) ? 384 * 1024
                                                                            : 512 * 1024;

      // override this_max_page_size if the requested size is smaller
      this_max_page_size = min(this_max_page_size, max_page_size_bytes);

      if (num_rows >= ck_g.num_rows ||
          (values_in_page > 0 && (page_size + fragment_data_size > this_max_page_size)) ||
          rows_in_page >= max_page_size_rows) {
        if (ck_g.use_dictionary) {
          // Additional byte to store entry bit width
          page_size = 1 + max_RLE_page_size(ck_g.dict_rle_bits, values_in_page);
        }
        if (!t) {
          page_g.num_fragments = fragments_in_chunk - page_start;
          page_g.chunk         = &chunks[blockIdx.y][blockIdx.x];
          page_g.chunk_id      = blockIdx.y * num_columns + blockIdx.x;
          page_g.page_type     = PageType::DATA_PAGE;
          page_g.hdr_size      = 0;
          page_g.max_hdr_size  = 32;  // Max size excluding statistics
          if (ck_g.stats) {
            uint32_t stats_hdr_len = 16;
            if (col_g.stats_dtype == dtype_string || col_g.stats_dtype == dtype_byte_array) {
              stats_hdr_len += 5 * 3 + 2 * max_stats_len;
            } else {
              stats_hdr_len += ((col_g.stats_dtype >= dtype_int64) ? 10 : 5) * 3;
            }
            page_g.max_hdr_size += stats_hdr_len;
          }
          page_g.max_hdr_size = util::round_up_unsafe(page_g.max_hdr_size, page_align);
          page_g.page_data    = ck_g.uncompressed_bfr + page_offset;
          if (not comp_page_sizes.empty()) {
            page_g.compressed_data = ck_g.compressed_bfr + comp_page_offset;
          }
          page_g.start_row          = cur_row;
          page_g.num_rows           = rows_in_page;
          page_g.num_leaf_values    = leaf_values_in_page;
          page_g.num_values         = values_in_page;
          auto const def_level_size = max_RLE_page_size(col_g.num_def_level_bits(), values_in_page);
          auto const rep_level_size = max_RLE_page_size(col_g.num_rep_level_bits(), values_in_page);
          page_g.max_data_size      = page_size + def_level_size + rep_level_size;

          pagestats_g.start_chunk = ck_g.first_fragment + page_start;
          pagestats_g.num_chunks  = page_g.num_fragments;
          page_offset +=
            util::round_up_unsafe(page_g.max_hdr_size + page_g.max_data_size, page_align);
          if (not comp_page_sizes.empty()) {
            comp_page_offset += page_g.max_hdr_size + comp_page_sizes[ck_g.first_page + num_pages];
          }
          page_headers_size += page_g.max_hdr_size;
          max_page_data_size = max(max_page_data_size, page_g.max_data_size);
          cur_row += rows_in_page;
          ck_max_stats_len = max(ck_max_stats_len, max_stats_len);
        }
        __syncwarp();
        if (t == 0) {
          if (not pages.empty()) { pages[ck_g.first_page + num_pages] = page_g; }
          if (not page_sizes.empty()) {
            page_sizes[ck_g.first_page + num_pages] = page_g.max_data_size;
          }
          if (page_grstats) { page_grstats[ck_g.first_page + num_pages] = pagestats_g; }
        }

        num_pages++;
        page_size           = 0;
        rows_in_page        = 0;
        values_in_page      = 0;
        leaf_values_in_page = 0;
        page_start          = fragments_in_chunk;
        max_stats_len       = 0;
      }
      max_stats_len = max(max_stats_len, minmax_len);
      num_dict_entries += frag_g.num_dict_vals;
      page_size += fragment_data_size;
      rows_in_page += frag_g.num_rows;
      values_in_page += frag_g.num_values;
      leaf_values_in_page += frag_g.num_leaf_values;
      num_rows += frag_g.num_rows;
      fragments_in_chunk++;
    } while (frag_g.num_rows != 0);
    __syncwarp();
    if (!t) {
      if (ck_g.ck_stat_size == 0 && ck_g.stats) {
        uint32_t ck_stat_size = util::round_up_unsafe(48 + 2 * ck_max_stats_len, page_align);
        page_offset += ck_stat_size;
        comp_page_offset += ck_stat_size;
        ck_g.ck_stat_size = ck_stat_size;
      }
      ck_g.num_pages          = num_pages;
      ck_g.bfr_size           = page_offset;
      ck_g.page_headers_size  = page_headers_size;
      ck_g.max_page_data_size = max_page_data_size;
      if (not comp_page_sizes.empty()) { ck_g.compressed_size = comp_page_offset; }
      pagestats_g.start_chunk = ck_g.first_page + ck_g.use_dictionary;  // Exclude dictionary
      pagestats_g.num_chunks  = num_pages - ck_g.use_dictionary;
    }
  }
  __syncthreads();
  if (t == 0) {
    if (not pages.empty()) ck_g.pages = &pages[ck_g.first_page];
    chunks[blockIdx.y][blockIdx.x] = ck_g;
    if (chunk_grstats) chunk_grstats[blockIdx.y * num_columns + blockIdx.x] = pagestats_g;
  }
}

/**
 * @brief Mask table representing how many consecutive repeats are needed to code a repeat run
 *[nbits-1]
 */
static __device__ __constant__ uint32_t kRleRunMask[24] = {
  0x00ff'ffff, 0x0fff, 0x00ff, 0x3f, 0x0f, 0x0f, 0x7, 0x7, 0x3, 0x3, 0x3, 0x3,
  0x1,         0x1,    0x1,    0x1,  0x1,  0x1,  0x1, 0x1, 0x1, 0x1, 0x1, 0x1};

/**
 * @brief Variable-length encode an integer
 */
inline __device__ uint8_t* VlqEncode(uint8_t* p, uint32_t v)
{
  while (v > 0x7f) {
    *p++ = (v | 0x80);
    v >>= 7;
  }
  *p++ = v;
  return p;
}

/**
 * @brief Pack literal values in output bitstream (1,2,3,4,5,6,8,10,12,16,20 or 24 bits per value)
 */
inline __device__ void PackLiteralsShuffle(
  uint8_t* dst, uint32_t v, uint32_t count, uint32_t w, uint32_t t)
{
  constexpr uint32_t MASK2T = 1;  // mask for 2 thread leader
  constexpr uint32_t MASK4T = 3;  // mask for 4 thread leader
  constexpr uint32_t MASK8T = 7;  // mask for 8 thread leader
  uint64_t v64;

  if (t > (count | 0x1f)) { return; }

  switch (w) {
    case 1:
      v |= shuffle_xor(v, 1) << 1;  // grab bit 1 from neighbor
      v |= shuffle_xor(v, 2) << 2;  // grab bits 2-3 from 2 lanes over
      v |= shuffle_xor(v, 4) << 4;  // grab bits 4-7 from 4 lanes over
      // sub-warp leader writes the combined bits
      if (t < count && !(t & MASK8T)) { dst[(t * w) >> 3] = v; }
      return;
    case 2:
      v |= shuffle_xor(v, 1) << 2;
      v |= shuffle_xor(v, 2) << 4;
      if (t < count && !(t & MASK4T)) { dst[(t * w) >> 3] = v; }
      return;
    case 3:
      v |= shuffle_xor(v, 1) << 3;
      v |= shuffle_xor(v, 2) << 6;
      v |= shuffle_xor(v, 4) << 12;
      if (t < count && !(t & MASK8T)) {
        dst[(t >> 3) * 3 + 0] = v;
        dst[(t >> 3) * 3 + 1] = v >> 8;
        dst[(t >> 3) * 3 + 2] = v >> 16;
      }
      return;
    case 4:
      v |= shuffle_xor(v, 1) << 4;
      if (t < count && !(t & MASK2T)) { dst[(t * w) >> 3] = v; }
      return;
    case 5:
      v |= shuffle_xor(v, 1) << 5;
      v |= shuffle_xor(v, 2) << 10;
      v64 = static_cast<uint64_t>(shuffle_xor(v, 4)) << 20 | v;
      if (t < count && !(t & MASK8T)) {
        dst[(t >> 3) * 5 + 0] = v64;
        dst[(t >> 3) * 5 + 1] = v64 >> 8;
        dst[(t >> 3) * 5 + 2] = v64 >> 16;
        dst[(t >> 3) * 5 + 3] = v64 >> 24;
        dst[(t >> 3) * 5 + 4] = v64 >> 32;
      }
      return;
    case 6:
      v |= shuffle_xor(v, 1) << 6;
      v |= shuffle_xor(v, 2) << 12;
      if (t < count && !(t & MASK4T)) {
        dst[(t >> 2) * 3 + 0] = v;
        dst[(t >> 2) * 3 + 1] = v >> 8;
        dst[(t >> 2) * 3 + 2] = v >> 16;
      }
      return;
    case 8:
      if (t < count) { dst[t] = v; }
      return;
    case 10:
      v |= shuffle_xor(v, 1) << 10;
      v64 = static_cast<uint64_t>(shuffle_xor(v, 2)) << 20 | v;
      if (t < count && !(t & MASK4T)) {
        dst[(t >> 2) * 5 + 0] = v64;
        dst[(t >> 2) * 5 + 1] = v64 >> 8;
        dst[(t >> 2) * 5 + 2] = v64 >> 16;
        dst[(t >> 2) * 5 + 3] = v64 >> 24;
        dst[(t >> 2) * 5 + 4] = v64 >> 32;
      }
      return;
    case 12:
      v |= shuffle_xor(v, 1) << 12;
      if (t < count && !(t & MASK2T)) {
        dst[(t >> 1) * 3 + 0] = v;
        dst[(t >> 1) * 3 + 1] = v >> 8;
        dst[(t >> 1) * 3 + 2] = v >> 16;
      }
      return;
    case 16:
      if (t < count) {
        dst[t * 2 + 0] = v;
        dst[t * 2 + 1] = v >> 8;
      }
      return;
    case 20:
      v64 = static_cast<uint64_t>(shuffle_xor(v, 1)) << 20 | v;
      if (t < count && !(t & MASK2T)) {
        dst[(t >> 1) * 5 + 0] = v64;
        dst[(t >> 1) * 5 + 1] = v64 >> 8;
        dst[(t >> 1) * 5 + 2] = v64 >> 16;
        dst[(t >> 1) * 5 + 3] = v64 >> 24;
        dst[(t >> 1) * 5 + 4] = v64 >> 32;
      }
      return;
    case 24:
      if (t < count) {
        dst[t * 3 + 0] = v;
        dst[t * 3 + 1] = v >> 8;
        dst[t * 3 + 2] = v >> 16;
      }
      return;

    default: CUDF_UNREACHABLE("Unsupported bit width");
  }
}

/**
 * @brief Pack literals of arbitrary bit-length in output bitstream.
 */
inline __device__ void PackLiteralsRoundRobin(
  uint8_t* dst, uint32_t v, uint32_t count, uint32_t w, uint32_t t)
{
  // Scratch space to temporarily write to. Needed because we will use atomics to write 32 bit
  // words but the destination mem may not be a multiple of 4 bytes.
  // TODO (dm): This assumes blockdim = 128. Reduce magic numbers.
  constexpr uint32_t NUM_THREADS  = 128;  // this needs to match gpuEncodePages block_size parameter
  constexpr uint32_t NUM_BYTES    = (NUM_THREADS * MAX_DICT_BITS) >> 3;
  constexpr uint32_t SCRATCH_SIZE = NUM_BYTES / sizeof(uint32_t);
  __shared__ uint32_t scratch[SCRATCH_SIZE];
  for (uint32_t i = t; i < SCRATCH_SIZE; i += NUM_THREADS) {
    scratch[i] = 0;
  }
  __syncthreads();

  if (t <= count) {
    // shift symbol left by up to 31 bits
    uint64_t v64 = v;
    v64 <<= (t * w) & 0x1f;

    // Copy 64 bit word into two 32 bit words while following C++ strict aliasing rules.
    uint32_t v32[2];
    memcpy(&v32, &v64, sizeof(uint64_t));

    // Atomically write result to scratch
    if (v32[0]) { atomicOr(scratch + ((t * w) >> 5), v32[0]); }
    if (v32[1]) { atomicOr(scratch + ((t * w) >> 5) + 1, v32[1]); }
  }
  __syncthreads();

  // Copy scratch data to final destination
  auto available_bytes = (count * w + 7) / 8;

  auto scratch_bytes = reinterpret_cast<char*>(&scratch[0]);
  for (uint32_t i = t; i < available_bytes; i += NUM_THREADS) {
    dst[i] = scratch_bytes[i];
  }
  __syncthreads();
}

/**
 * @brief Pack literal values in output bitstream
 */
inline __device__ void PackLiterals(
  uint8_t* dst, uint32_t v, uint32_t count, uint32_t w, uint32_t t)
{
  if (w > 24) { CUDF_UNREACHABLE("Unsupported bit width"); }
  switch (w) {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 8:
    case 10:
    case 12:
    case 16:
    case 20:
    case 24:
      // bit widths that lie on easy boundaries can be handled either directly
      // (8, 16, 24) or through fast shuffle operations.
      PackLiteralsShuffle(dst, v, count, w, t);
      return;
    default:
      // bit packing that uses atomics, but can handle arbitrary bit widths up to 24.
      PackLiteralsRoundRobin(dst, v, count, w, t);
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
  page_enc_state_s* s, uint32_t numvals, uint32_t nbits, uint32_t flush, uint32_t t)
{
  uint32_t rle_pos = s->rle_pos;
  uint32_t rle_run = s->rle_run;

  while (rle_pos < numvals || (flush && rle_run)) {
    uint32_t pos = rle_pos + t;
    if (rle_run > 0 && !(rle_run & 1)) {
      // Currently in a long repeat run
      uint32_t mask = ballot(pos < numvals && s->vals[pos & (rle_buffer_size - 1)] == s->run_val);
      uint32_t rle_rpt_count, max_rpt_count;
      if (!(t & 0x1f)) { s->rpt_map[t >> 5] = mask; }
      __syncthreads();
      if (t < 32) {
        uint32_t c32 = ballot(t >= 4 || s->rpt_map[t] != 0xffff'ffffu);
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
        if (t == 0) {
          uint32_t const run_val = s->run_val;
          uint8_t* dst           = VlqEncode(s->rle_out, rle_run);
          *dst++                 = run_val;
          if (nbits > 8) { *dst++ = run_val >> 8; }
          if (nbits > 16) { *dst++ = run_val >> 16; }
          s->rle_out = dst;
        }
        rle_run = 0;
      }
    } else {
      // New run or in a literal run
      uint32_t v0      = s->vals[pos & (rle_buffer_size - 1)];
      uint32_t v1      = s->vals[(pos + 1) & (rle_buffer_size - 1)];
      uint32_t mask    = ballot(pos + 1 < numvals && v0 == v1);
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
        mask                 = ballot((__funnelshift_r(m0, m1, pos8) & needed_mask) == needed_mask);
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
            uint8_t* dst = s->rle_out + 1 + (rle_run >> 1) * nbits;
            PackLiterals(dst, (rle_pos + t < numvals) ? v0 : 0, lit_div8 * 8, nbits, t);
            rle_run = (rle_run + lit_div8 * 2) | 1;
            rle_pos = min(rle_pos + lit_div8 * 8, numvals);
          }
        }
        if (rle_run >= ((rle_rpt_count != 0 || (flush && rle_pos == numvals)) ? 0x03 : 0x7f)) {
          __syncthreads();
          // Complete literal run
          if (!t) {
            uint8_t* dst = s->rle_out;
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
static __device__ void PlainBoolEncode(page_enc_state_s* s,
                                       uint32_t numvals,
                                       uint32_t flush,
                                       uint32_t t)
{
  uint32_t rle_pos = s->rle_pos;
  uint8_t* dst     = s->rle_out;

  while (rle_pos < numvals) {
    uint32_t pos    = rle_pos + t;
    uint32_t v      = (pos < numvals) ? s->vals[pos & (rle_buffer_size - 1)] : 0;
    uint32_t n      = min(numvals - rle_pos, 128);
    uint32_t nbytes = (n + ((flush) ? 7 : 0)) >> 3;
    if (!nbytes) { break; }
    v |= shuffle_xor(v, 1) << 1;
    v |= shuffle_xor(v, 2) << 2;
    v |= shuffle_xor(v, 4) << 4;
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

/**
 * @brief Determines the difference between the Proleptic Gregorian Calendar epoch (1970-01-01
 * 00:00:00 UTC) and the Julian date epoch (-4713-11-24 12:00:00 UTC).
 *
 * @return The difference between two epochs in `cuda::std::chrono::duration` format with a period
 * of hours.
 */
constexpr auto julian_calendar_epoch_diff()
{
  using namespace cuda::std::chrono;
  using namespace cuda::std::chrono_literals;
  return sys_days{January / 1 / 1970} - (sys_days{November / 24 / -4713} + 12h);
}

/**
 * @brief Converts a timestamp_ns into a pair with nanoseconds since midnight and number of Julian
 * days. Does not deal with time zones. Used by INT96 code.
 *
 * @param ns number of nanoseconds since epoch
 * @return std::pair<nanoseconds,days> where nanoseconds is the number of nanoseconds
 * elapsed in the day and days is the number of days from Julian epoch.
 */
static __device__ std::pair<duration_ns, duration_D> convert_nanoseconds(timestamp_ns const ns)
{
  using namespace cuda::std::chrono;
  auto const nanosecond_ticks = ns.time_since_epoch();
  auto const gregorian_days   = floor<days>(nanosecond_ticks);
  auto const julian_days      = gregorian_days + ceil<days>(julian_calendar_epoch_diff());

  auto const last_day_ticks = nanosecond_ticks - gregorian_days;
  return {last_day_ticks, julian_days};
}

// blockDim(128, 1, 1)
template <int block_size>
__global__ void __launch_bounds__(128, 8)
  gpuEncodePages(device_span<gpu::EncPage> pages,
                 device_span<device_span<uint8_t const>> comp_in,
                 device_span<device_span<uint8_t>> comp_out,
                 device_span<compression_result> comp_results)
{
  __shared__ __align__(8) page_enc_state_s state_g;
  using block_scan = cub::BlockScan<uint32_t, block_size>;
  __shared__ typename block_scan::TempStorage temp_storage;

  page_enc_state_s* const s = &state_g;
  uint32_t t                = threadIdx.x;

  if (t == 0) {
    state_g = page_enc_state_s{};
    s->page = pages[blockIdx.x];
    s->ck   = *s->page.chunk;
    s->col  = *s->ck.col_desc;
    s->cur  = s->page.page_data + s->page.max_hdr_size;
  }
  __syncthreads();

  // Encode Repetition and Definition levels
  if (s->page.page_type != PageType::DICTIONARY_PAGE &&
      (s->col.num_def_level_bits()) != 0 &&  // This means max definition level is not 0 (nullable)
      (s->col.num_rep_level_bits()) == 0     // This means there are no repetition levels (non-list)
  ) {
    // Calculate definition levels from validity
    uint32_t def_lvl_bits = s->col.num_def_level_bits();
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
        // Definition level encodes validity. Checks the valid map and if it is valid, then sets the
        // def_lvl accordingly and sets it in s->vals which is then given to RleEncode to encode
        uint32_t def_lvl = [&]() {
          bool within_bounds = rle_numvals + t < s->page.num_rows && row < s->col.num_rows;
          if (not within_bounds) { return 0u; }
          uint32_t def       = 0;
          size_type l        = 0;
          bool is_col_struct = false;
          auto col           = *s->col.parent_column;
          do {
            // If col not nullable then it does not contribute to def levels
            if (s->col.nullability[l]) {
              if (col.is_valid(row)) {
                ++def;
              } else {
                // We have found the shallowest level at which this row is null
                break;
              }
            }
            is_col_struct = (col.type().id() == type_id::STRUCT);
            if (is_col_struct) {
              row += col.offset();
              col = col.child(0);
              ++l;
            }
          } while (is_col_struct);
          return def;
        }();
        s->vals[(rle_numvals + t) & (rle_buffer_size - 1)] = def_lvl;
        __syncthreads();
        rle_numvals += nrows;
        RleEncode(s, rle_numvals, def_lvl_bits, (rle_numvals == s->page.num_rows), t);
        __syncthreads();
      }
      if (t < 32) {
        uint8_t* cur     = s->cur;
        uint8_t* rle_out = s->rle_out;
        if (t < 4) {
          uint32_t rle_bytes = (uint32_t)(rle_out - cur) - 4;
          cur[t]             = rle_bytes >> (t * 8);
        }
        __syncwarp();
        if (t == 0) { s->cur = rle_out; }
      }
    }
  } else if (s->page.page_type != PageType::DICTIONARY_PAGE &&
             s->col.num_rep_level_bits() != 0  // This means there ARE repetition levels (has list)
  ) {
    auto encode_levels = [&](uint8_t const* lvl_val_data, uint32_t nbits) {
      // For list types, the repetition and definition levels are pre-calculated. We just need to
      // encode and write them now.
      if (!t) {
        s->rle_run     = 0;
        s->rle_pos     = 0;
        s->rle_numvals = 0;
        s->rle_out     = s->cur + 4;
      }
      __syncthreads();
      size_type page_first_val_idx = s->col.level_offsets[s->page.start_row];
      size_type col_last_val_idx   = s->col.level_offsets[s->col.num_rows];
      while (s->rle_numvals < s->page.num_values) {
        uint32_t rle_numvals = s->rle_numvals;
        uint32_t nvals       = min(s->page.num_values - rle_numvals, 128);
        uint32_t idx         = page_first_val_idx + rle_numvals + t;
        uint32_t lvl_val =
          (rle_numvals + t < s->page.num_values && idx < col_last_val_idx) ? lvl_val_data[idx] : 0;
        s->vals[(rle_numvals + t) & (rle_buffer_size - 1)] = lvl_val;
        __syncthreads();
        rle_numvals += nvals;
        RleEncode(s, rle_numvals, nbits, (rle_numvals == s->page.num_values), t);
        __syncthreads();
      }
      if (t < 32) {
        uint8_t* cur     = s->cur;
        uint8_t* rle_out = s->rle_out;
        if (t < 4) {
          uint32_t rle_bytes = (uint32_t)(rle_out - cur) - 4;
          cur[t]             = rle_bytes >> (t * 8);
        }
        __syncwarp();
        if (t == 0) { s->cur = rle_out; }
      }
    };
    encode_levels(s->col.rep_values, s->col.num_rep_level_bits());
    __syncthreads();
    encode_levels(s->col.def_values, s->col.num_def_level_bits());
  }
  // Encode data values
  __syncthreads();
  auto const physical_type = s->col.physical_type;
  auto const type_id       = s->col.leaf_column->type().id();
  auto const dtype_len_out = physical_type_len(physical_type, type_id);
  auto const dtype_len_in  = [&]() -> uint32_t {
    if (physical_type == INT32) { return int32_logical_len(type_id); }
    if (physical_type == INT96) { return sizeof(int64_t); }
    return dtype_len_out;
  }();

  auto const dict_bits = (physical_type == BOOLEAN) ? 1
                         : (s->ck.use_dictionary and s->page.page_type != PageType::DICTIONARY_PAGE)
                           ? s->ck.dict_rle_bits
                           : -1;
  if (t == 0) {
    uint8_t* dst   = s->cur;
    s->rle_run     = 0;
    s->rle_pos     = 0;
    s->rle_numvals = 0;
    s->rle_out     = dst;
    if (dict_bits >= 0 && physical_type != BOOLEAN) {
      dst[0]     = dict_bits;
      s->rle_out = dst + 1;
    }
    s->page_start_val  = row_to_value_idx(s->page.start_row, s->col);
    s->chunk_start_val = row_to_value_idx(s->ck.start_row, s->col);
  }
  __syncthreads();
  for (uint32_t cur_val_idx = 0; cur_val_idx < s->page.num_leaf_values;) {
    uint32_t nvals = min(s->page.num_leaf_values - cur_val_idx, 128);
    uint32_t len, pos;

    auto [is_valid, val_idx] = [&]() {
      uint32_t val_idx;
      uint32_t is_valid;

      size_type val_idx_in_block = cur_val_idx + t;
      if (s->page.page_type == PageType::DICTIONARY_PAGE) {
        val_idx  = val_idx_in_block;
        is_valid = (val_idx < s->page.num_leaf_values);
        if (is_valid) { val_idx = s->ck.dict_data[val_idx]; }
      } else {
        size_type val_idx_in_leaf_col = s->page_start_val + val_idx_in_block;

        is_valid = (val_idx_in_leaf_col < s->col.leaf_column->size() &&
                    val_idx_in_block < s->page.num_leaf_values)
                     ? s->col.leaf_column->is_valid(val_idx_in_leaf_col)
                     : 0;
        val_idx =
          (s->ck.use_dictionary) ? val_idx_in_leaf_col - s->chunk_start_val : val_idx_in_leaf_col;
      }
      return std::make_tuple(is_valid, val_idx);
    }();

    cur_val_idx += nvals;
    if (dict_bits >= 0) {
      // Dictionary encoding
      if (dict_bits > 0) {
        uint32_t rle_numvals;
        uint32_t rle_numvals_in_block;
        block_scan(temp_storage).ExclusiveSum(is_valid, pos, rle_numvals_in_block);
        rle_numvals = s->rle_numvals;
        if (is_valid) {
          uint32_t v;
          if (physical_type == BOOLEAN) {
            v = s->col.leaf_column->element<uint8_t>(val_idx);
          } else {
            v = s->ck.dict_index[val_idx];
          }
          s->vals[(rle_numvals + pos) & (rle_buffer_size - 1)] = v;
        }
        rle_numvals += rle_numvals_in_block;
        __syncthreads();
        if ((!enable_bool_rle) && (physical_type == BOOLEAN)) {
          PlainBoolEncode(s, rle_numvals, (cur_val_idx == s->page.num_leaf_values), t);
        } else {
          RleEncode(s, rle_numvals, dict_bits, (cur_val_idx == s->page.num_leaf_values), t);
        }
        __syncthreads();
      }
      if (t == 0) { s->cur = s->rle_out; }
      __syncthreads();
    } else {
      // Non-dictionary encoding
      uint8_t* dst = s->cur;

      if (is_valid) {
        len = dtype_len_out;
        if (physical_type == BYTE_ARRAY) {
          if (type_id == type_id::STRING) {
            len += s->col.leaf_column->element<string_view>(val_idx).size_bytes();
          } else if (s->col.output_as_byte_array && type_id == type_id::LIST) {
            len +=
              get_element<statistics::byte_array_view>(*s->col.leaf_column, val_idx).size_bytes();
          }
        }
      } else {
        len = 0;
      }
      uint32_t total_len = 0;
      block_scan(temp_storage).ExclusiveSum(len, pos, total_len);
      __syncthreads();
      if (t == 0) { s->cur = dst + total_len; }
      if (is_valid) {
        switch (physical_type) {
          case INT32: [[fallthrough]];
          case FLOAT: {
            auto const v = [dtype_len = dtype_len_in,
                            idx       = val_idx,
                            col       = s->col.leaf_column,
                            scale     = s->col.ts_scale == 0 ? 1 : s->col.ts_scale]() -> int32_t {
              switch (dtype_len) {
                case 8: return col->element<int64_t>(idx) * scale;
                case 4: return col->element<int32_t>(idx) * scale;
                case 2: return col->element<int16_t>(idx) * scale;
                default: return col->element<int8_t>(idx) * scale;
              }
            }();

            dst[pos + 0] = v;
            dst[pos + 1] = v >> 8;
            dst[pos + 2] = v >> 16;
            dst[pos + 3] = v >> 24;
          } break;
          case INT64: {
            int64_t v        = s->col.leaf_column->element<int64_t>(val_idx);
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
          case INT96: {
            int64_t v        = s->col.leaf_column->element<int64_t>(val_idx);
            int32_t ts_scale = s->col.ts_scale;
            if (ts_scale != 0) {
              if (ts_scale < 0) {
                v /= -ts_scale;
              } else {
                v *= ts_scale;
              }
            }

            auto const ret = convert_nanoseconds([&]() {
              switch (s->col.leaf_column->type().id()) {
                case type_id::TIMESTAMP_SECONDS:
                case type_id::TIMESTAMP_MILLISECONDS: {
                  return timestamp_ns{duration_ms{v}};
                } break;
                case type_id::TIMESTAMP_MICROSECONDS:
                case type_id::TIMESTAMP_NANOSECONDS: {
                  return timestamp_ns{duration_us{v}};
                } break;
              }
              return timestamp_ns{duration_ns{0}};
            }());

            // the 12 bytes of fixed length data.
            v             = ret.first.count();
            dst[pos + 0]  = v;
            dst[pos + 1]  = v >> 8;
            dst[pos + 2]  = v >> 16;
            dst[pos + 3]  = v >> 24;
            dst[pos + 4]  = v >> 32;
            dst[pos + 5]  = v >> 40;
            dst[pos + 6]  = v >> 48;
            dst[pos + 7]  = v >> 56;
            uint32_t w    = ret.second.count();
            dst[pos + 8]  = w;
            dst[pos + 9]  = w >> 8;
            dst[pos + 10] = w >> 16;
            dst[pos + 11] = w >> 24;
          } break;

          case DOUBLE: {
            auto v = s->col.leaf_column->element<double>(val_idx);
            memcpy(dst + pos, &v, 8);
          } break;
          case BYTE_ARRAY: {
            auto const bytes = [](cudf::type_id const type_id,
                                  column_device_view const* leaf_column,
                                  uint32_t const val_idx) -> void const* {
              switch (type_id) {
                case type_id::STRING:
                  return reinterpret_cast<void const*>(
                    leaf_column->element<string_view>(val_idx).data());
                case type_id::LIST:
                  return reinterpret_cast<void const*>(
                    get_element<statistics::byte_array_view>(*(leaf_column), val_idx).data());
                default: CUDF_UNREACHABLE("invalid type id for byte array writing!");
              }
            }(type_id, s->col.leaf_column, val_idx);
            uint32_t v   = len - 4;  // string length
            dst[pos + 0] = v;
            dst[pos + 1] = v >> 8;
            dst[pos + 2] = v >> 16;
            dst[pos + 3] = v >> 24;
            if (v != 0) memcpy(dst + pos + 4, bytes, v);
          } break;
          case FIXED_LEN_BYTE_ARRAY: {
            if (type_id == type_id::DECIMAL128) {
              // When using FIXED_LEN_BYTE_ARRAY for decimals, the rep is encoded in big-endian
              auto const v = s->col.leaf_column->element<numeric::decimal128>(val_idx).value();
              auto const v_char_ptr = reinterpret_cast<char const*>(&v);
              thrust::copy(thrust::seq,
                           thrust::make_reverse_iterator(v_char_ptr + sizeof(v)),
                           thrust::make_reverse_iterator(v_char_ptr),
                           dst + pos);
            }
          } break;
        }
      }
      __syncthreads();
    }
  }
  if (t == 0) {
    uint8_t* base         = s->page.page_data + s->page.max_hdr_size;
    auto actual_data_size = static_cast<uint32_t>(s->cur - base);
    s->page.max_data_size = actual_data_size;
    if (not comp_in.empty()) {
      comp_in[blockIdx.x]  = {base, actual_data_size};
      comp_out[blockIdx.x] = {s->page.compressed_data + s->page.max_hdr_size, 0};  // size is unused
    }
    pages[blockIdx.x] = s->page;
    if (not comp_results.empty()) {
      comp_results[blockIdx.x]   = {0, compression_status::FAILURE};
      pages[blockIdx.x].comp_res = &comp_results[blockIdx.x];
    }
  }
}

// blockDim(128, 1, 1)
__global__ void __launch_bounds__(128) gpuDecideCompression(device_span<EncColumnChunk> chunks)
{
  // After changing the way structs are loaded from coop to normal, this kernel has no business
  // being launched with 128 thread block. It can easily be a single warp.
  __shared__ __align__(8) EncColumnChunk ck_g;
  __shared__ __align__(4) unsigned int error_count;
  using warp_reduce = cub::WarpReduce<uint32_t>;
  __shared__ typename warp_reduce::TempStorage temp_storage[2];
  __shared__ volatile bool has_compression;

  uint32_t t                      = threadIdx.x;
  uint32_t uncompressed_data_size = 0;
  uint32_t compressed_data_size   = 0;
  uint32_t num_pages;

  if (t == 0) {
    ck_g = chunks[blockIdx.x];
    atomicAnd(&error_count, 0);
    has_compression = false;
  }
  __syncthreads();
  if (t < 32) {
    num_pages = ck_g.num_pages;
    for (uint32_t page = t; page < num_pages; page += 32) {
      auto& curr_page         = ck_g.pages[page];
      uint32_t page_data_size = curr_page.max_data_size;
      uncompressed_data_size += page_data_size;
      if (auto comp_res = curr_page.comp_res; comp_res != nullptr) {
        has_compression = true;
        compressed_data_size += comp_res->bytes_written;
        if (comp_res->status != compression_status::SUCCESS) { atomicAdd(&error_count, 1); }
      }
    }
    uncompressed_data_size = warp_reduce(temp_storage[0]).Sum(uncompressed_data_size);
    compressed_data_size   = warp_reduce(temp_storage[1]).Sum(compressed_data_size);
  }
  __syncthreads();
  if (t == 0) {
    bool is_compressed;
    if (has_compression) {
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
 */
inline __device__ uint8_t* cpw_put_uint8(uint8_t* p, uint8_t v)
{
  *p++ = v;
  return p;
}

inline __device__ uint8_t* cpw_put_uint32(uint8_t* p, uint32_t v)
{
  while (v > 0x7f) {
    *p++ = v | 0x80;
    v >>= 7;
  }
  *p++ = v;
  return p;
}

inline __device__ uint8_t* cpw_put_uint64(uint8_t* p, uint64_t v)
{
  while (v > 0x7f) {
    *p++ = v | 0x80;
    v >>= 7;
  }
  *p++ = v;
  return p;
}

inline __device__ uint8_t* cpw_put_int32(uint8_t* p, int32_t v)
{
  int32_t s = (v < 0);
  return cpw_put_uint32(p, (v ^ -s) * 2 + s);
}

inline __device__ uint8_t* cpw_put_int64(uint8_t* p, int64_t v)
{
  int64_t s = (v < 0);
  return cpw_put_uint64(p, (v ^ -s) * 2 + s);
}

inline __device__ uint8_t* cpw_put_fldh(uint8_t* p, int f, int cur, int t)
{
  if (f > cur && f <= cur + 15) {
    *p++ = ((f - cur) << 4) | t;
    return p;
  } else {
    *p++ = t;
    return cpw_put_int32(p, f);
  }
}

class header_encoder {
  uint8_t* current_header_ptr;
  int current_field_index;

 public:
  inline __device__ header_encoder(uint8_t* header_start)
    : current_header_ptr(header_start), current_field_index(0)
  {
  }

  inline __device__ void field_struct_begin(int field)
  {
    current_header_ptr =
      cpw_put_fldh(current_header_ptr, field, current_field_index, ST_FLD_STRUCT);
    current_field_index = 0;
  }

  inline __device__ void field_struct_end(int field)
  {
    *current_header_ptr++ = 0;
    current_field_index   = field;
  }

  inline __device__ void field_list_begin(int field, size_t len, int type)
  {
    current_header_ptr = cpw_put_fldh(current_header_ptr, field, current_field_index, ST_FLD_LIST);
    current_header_ptr = cpw_put_uint8(
      current_header_ptr, static_cast<uint8_t>((std::min(len, size_t{0xfu}) << 4) | type));
    if (len >= 0xf) { current_header_ptr = cpw_put_uint32(current_header_ptr, len); }
    current_field_index = 0;
  }

  inline __device__ void field_list_end(int field) { current_field_index = field; }

  inline __device__ void put_bool(bool value)
  {
    current_header_ptr = cpw_put_uint8(current_header_ptr, value ? ST_FLD_TRUE : ST_FLD_FALSE);
  }

  inline __device__ void put_binary(const void* value, uint32_t length)
  {
    current_header_ptr = cpw_put_uint32(current_header_ptr, length);
    memcpy(current_header_ptr, value, length);
    current_header_ptr += length;
  }

  template <typename T>
  inline __device__ void put_int64(T value)
  {
    current_header_ptr = cpw_put_int64(current_header_ptr, static_cast<int64_t>(value));
  }

  template <typename T>
  inline __device__ void field_int32(int field, T value)
  {
    current_header_ptr  = cpw_put_fldh(current_header_ptr, field, current_field_index, ST_FLD_I32);
    current_header_ptr  = cpw_put_int32(current_header_ptr, static_cast<int32_t>(value));
    current_field_index = field;
  }

  template <typename T>
  inline __device__ void field_int64(int field, T value)
  {
    current_header_ptr  = cpw_put_fldh(current_header_ptr, field, current_field_index, ST_FLD_I64);
    current_header_ptr  = cpw_put_int64(current_header_ptr, static_cast<int64_t>(value));
    current_field_index = field;
  }

  inline __device__ void field_binary(int field, const void* value, uint32_t length)
  {
    current_header_ptr =
      cpw_put_fldh(current_header_ptr, field, current_field_index, ST_FLD_BINARY);
    current_header_ptr = cpw_put_uint32(current_header_ptr, length);
    memcpy(current_header_ptr, value, length);
    current_header_ptr += length;
    current_field_index = field;
  }

  inline __device__ void end(uint8_t** header_end, bool termination_flag = true)
  {
    if (not termination_flag) { *current_header_ptr++ = 0; }
    *header_end = current_header_ptr;
  }

  inline __device__ uint8_t* get_ptr() { return current_header_ptr; }

  inline __device__ void set_ptr(uint8_t* ptr) { current_header_ptr = ptr; }
};

namespace {

// byteswap 128 bit integer, placing result in dst in network byte order.
// dst must point to at least 16 bytes of memory.
__device__ void byte_reverse128(__int128_t v, void* dst)
{
  auto const v_char_ptr = reinterpret_cast<unsigned char const*>(&v);
  auto const d_char_ptr = static_cast<unsigned char*>(dst);
  thrust::copy(thrust::seq,
               thrust::make_reverse_iterator(v_char_ptr + sizeof(v)),
               thrust::make_reverse_iterator(v_char_ptr),
               d_char_ptr);
}

/**
 * @brief Test to see if a span contains all valid UTF-8 characters.
 *
 * @param span device_span to test.
 * @return true if the span contains all valid UTF-8 characters.
 */
__device__ bool is_valid_utf8(device_span<unsigned char const> span)
{
  auto idx = 0;
  while (idx < span.size_bytes()) {
    // UTF-8 character should start with valid beginning bit pattern
    if (not strings::detail::is_valid_begin_utf8_char(span[idx])) { return false; }
    // subsequent elements of the character should be continuation chars
    auto const width = strings::detail::bytes_in_utf8_byte(span[idx++]);
    for (size_type i = 1; i < width && idx < span.size_bytes(); i++, idx++) {
      if (not strings::detail::is_utf8_continuation_char(span[idx])) { return false; }
    }
  }

  return true;
}

/**
 * @brief Increment part of a UTF-8 character.
 *
 * Attempt to increment the char pointed to by ptr, which is assumed to be part of a valid UTF-8
 * character. Returns true if successful, false if the increment caused an overflow, in which case
 * the data at ptr will be set to the lowest valid UTF-8 bit pattern (start or continuation).
 * Will halt execution if passed invalid UTF-8.
 */
__device__ bool increment_utf8_at(unsigned char* ptr)
{
  unsigned char elem = *ptr;
  // elem is one of (no 5 or 6 byte chars allowed):
  //  0b0vvvvvvv a 1 byte character
  //  0b10vvvvvv a continuation byte
  //  0b110vvvvv start of a 2 byte character
  //  0b1110vvvv start of a 3 byte character
  //  0b11110vvv start of a 4 byte character

  // TODO(ets): starting at 4 byte and working down.  Should probably start low and work higher.
  uint8_t mask  = 0xF8;
  uint8_t valid = 0xF0;

  while (mask != 0) {
    if ((elem & mask) == valid) {
      elem++;
      if ((elem & mask) != mask) {  // no overflow
        *ptr = elem;
        return true;
      }
      *ptr = valid;
      return false;
    }
    mask <<= 1;
    valid <<= 1;
  }

  // should not reach here since we test for valid UTF-8 higher up the call chain
  CUDF_UNREACHABLE("Trying to increment non-utf8");
}

/**
 * @brief Attempt to truncate a span of UTF-8 characters to at most truncate_length_bytes.
 *
 * If is_min is false, then the final character (or characters if there is overflow) will be
 * incremented so that the resultant UTF-8 will still be a valid maximum. scratch is only used when
 * is_min is false, and must be at least truncate_length bytes in size. If the span cannot be
 * truncated, leave it untouched and return the original length.
 *
 * @return Pair object containing a pointer to the truncated data and its length.
 */
__device__ std::pair<const void*, uint32_t> truncate_utf8(device_span<unsigned char const> span,
                                                          bool is_min,
                                                          void* scratch,
                                                          size_type truncate_length)
{
  // we know at this point that truncate_length < size_bytes, so
  // there is data at [len]. work backwards until we find
  // the start of a UTF-8 encoded character, since UTF-8 characters may be multi-byte.
  auto len = truncate_length;
  while (not strings::detail::is_begin_utf8_char(span[len]) && len > 0) {
    len--;
  }

  if (len != 0) {
    if (is_min) { return {span.data(), len}; }
    memcpy(scratch, span.data(), len);
    // increment last byte, working backwards if the byte overflows
    auto const ptr = static_cast<unsigned char*>(scratch);
    for (int32_t i = len - 1; i >= 0; i--) {
      if (increment_utf8_at(&ptr[i])) {  // true if no overflow
        return {scratch, len};
      }
    }
    // cannot increment, so fall through
  }

  // couldn't truncate, return original value
  return {span.data(), span.size_bytes()};
}

/**
 * @brief Attempt to truncate a span of binary data to at most truncate_length bytes.
 *
 * If is_min is false, then the final byte (or bytes if there is overflow) will be
 * incremented so that the resultant binary will still be a valid maximum. scratch is only used when
 * is_min is false, and must be at least truncate_length bytes in size. If the span cannot be
 * truncated, leave it untouched and return the original length.
 *
 * @return Pair object containing a pointer to the truncated data and its length.
 */
__device__ std::pair<const void*, uint32_t> truncate_binary(device_span<uint8_t const> arr,
                                                            bool is_min,
                                                            void* scratch,
                                                            size_type truncate_length)
{
  if (is_min) { return {arr.data(), truncate_length}; }
  memcpy(scratch, arr.data(), truncate_length);
  // increment last byte, working backwards if the byte overflows
  auto const ptr = static_cast<uint8_t*>(scratch);
  for (int32_t i = truncate_length - 1; i >= 0; i--) {
    ptr[i]++;
    if (ptr[i] != 0) {  // no overflow
      return {scratch, i + 1};
    }
  }

  // couldn't truncate, return original value
  return {arr.data(), arr.size_bytes()};
}

// TODO (ets): the assumption here is that string columns might have UTF-8 or plain binary,
// while binary columns are assumed to be binary and will be treated as such.  If this assumption
// is incorrect, then truncate_byte_array() and truncate_string() should just be combined into
// a single function.
/**
 * @brief Attempt to truncate a UTF-8 string to at most truncate_length bytes.
 */
__device__ std::pair<const void*, uint32_t> truncate_string(const string_view& str,
                                                            bool is_min,
                                                            void* scratch,
                                                            size_type truncate_length)
{
  if (truncate_length == NO_TRUNC_STATS or str.size_bytes() <= truncate_length) {
    return {str.data(), str.size_bytes()};
  }

  // convert char to unsigned since UTF-8 is just bytes, not chars.  can't use std::byte because
  // that can't be incremented.
  auto const span = device_span<unsigned char const>(
    reinterpret_cast<unsigned char const*>(str.data()), str.size_bytes());

  // if str is all 8-bit chars, or is actually not UTF-8, then we can just use truncate_binary()
  if (str.size_bytes() != str.length() and is_valid_utf8(span.first(truncate_length))) {
    return truncate_utf8(span, is_min, scratch, truncate_length);
  }
  return truncate_binary(span, is_min, scratch, truncate_length);
}

/**
 * @brief Attempt to truncate a binary array to at most truncate_length bytes.
 */
__device__ std::pair<const void*, uint32_t> truncate_byte_array(
  const statistics::byte_array_view& arr, bool is_min, void* scratch, size_type truncate_length)
{
  if (truncate_length == NO_TRUNC_STATS or arr.size_bytes() <= truncate_length) {
    return {arr.data(), arr.size_bytes()};
  }

  // convert std::byte to uint8_t since bytes can't be incremented
  device_span<uint8_t const> const span{reinterpret_cast<uint8_t const*>(arr.data()),
                                        arr.size_bytes()};
  return truncate_binary(span, is_min, scratch, truncate_length);
}

/**
 * @brief Find a min or max value of the proper form to be included in Parquet statistics
 * structures.
 *
 * Given a statistics_val union and a data type, perform any transformations needed to produce a
 * valid min or max binary value.  String and byte array types will be truncated if they exceed
 * truncate_length.
 */
__device__ std::pair<const void*, uint32_t> get_extremum(const statistics_val* stats_val,
                                                         statistics_dtype dtype,
                                                         void* scratch,
                                                         bool is_min,
                                                         size_type truncate_length)
{
  switch (dtype) {
    case dtype_bool: return {stats_val, sizeof(bool)};
    case dtype_int8:
    case dtype_int16:
    case dtype_int32:
    case dtype_date32: return {stats_val, sizeof(int32_t)};
    case dtype_float32: {
      auto const fp_scratch = static_cast<float*>(scratch);
      fp_scratch[0]         = stats_val->fp_val;
      return {scratch, sizeof(float)};
    }
    case dtype_int64:
    case dtype_timestamp64:
    case dtype_float64:
    case dtype_decimal64: return {stats_val, sizeof(int64_t)};
    case dtype_decimal128:
      byte_reverse128(stats_val->d128_val, scratch);
      return {scratch, sizeof(__int128_t)};
    case dtype_string: return truncate_string(stats_val->str_val, is_min, scratch, truncate_length);
    case dtype_byte_array:
      return truncate_byte_array(stats_val->byte_val, is_min, scratch, truncate_length);
    default: CUDF_UNREACHABLE("Invalid statistics data type");
  }
}

}  // namespace

__device__ uint8_t* EncodeStatistics(uint8_t* start,
                                     const statistics_chunk* s,
                                     statistics_dtype dtype,
                                     void* scratch)
{
  uint8_t* end;
  header_encoder encoder(start);
  encoder.field_int64(3, s->null_count);
  if (s->has_minmax) {
    auto const [max_ptr, max_size] =
      get_extremum(&s->max_value, dtype, scratch, false, NO_TRUNC_STATS);
    encoder.field_binary(5, max_ptr, max_size);
    auto const [min_ptr, min_size] =
      get_extremum(&s->min_value, dtype, scratch, true, NO_TRUNC_STATS);
    encoder.field_binary(6, min_ptr, min_size);
  }
  encoder.end(&end);
  return end;
}

// blockDim(128, 1, 1)
__global__ void __launch_bounds__(128)
  gpuEncodePageHeaders(device_span<EncPage> pages,
                       device_span<compression_result const> comp_results,
                       device_span<statistics_chunk const> page_stats,
                       const statistics_chunk* chunk_stats)
{
  // When this whole kernel becomes single thread, the following variables need not be __shared__
  __shared__ __align__(8) parquet_column_device_view col_g;
  __shared__ __align__(8) EncColumnChunk ck_g;
  __shared__ __align__(8) EncPage page_g;
  __shared__ __align__(8) unsigned char scratch[MIN_STATS_SCRATCH_SIZE];

  uint32_t t = threadIdx.x;

  if (t == 0) {
    uint8_t *hdr_start, *hdr_end;
    uint32_t compressed_page_size, uncompressed_page_size;

    page_g = pages[blockIdx.x];
    ck_g   = *page_g.chunk;
    col_g  = *ck_g.col_desc;

    if (chunk_stats && &pages[blockIdx.x] == ck_g.pages) {  // Is this the first page in a chunk?
      hdr_start = (ck_g.is_compressed) ? ck_g.compressed_bfr : ck_g.uncompressed_bfr;
      hdr_end =
        EncodeStatistics(hdr_start, &chunk_stats[page_g.chunk_id], col_g.stats_dtype, scratch);
      page_g.chunk->ck_stat_size = static_cast<uint32_t>(hdr_end - hdr_start);
    }
    uncompressed_page_size = page_g.max_data_size;
    if (ck_g.is_compressed) {
      hdr_start            = page_g.compressed_data;
      compressed_page_size = (uint32_t)comp_results[blockIdx.x].bytes_written;
      page_g.max_data_size = compressed_page_size;
    } else {
      hdr_start            = page_g.page_data;
      compressed_page_size = uncompressed_page_size;
    }
    header_encoder encoder(hdr_start);
    PageType page_type = page_g.page_type;
    // NOTE: For dictionary encoding, parquet v2 recommends using PLAIN in dictionary page and
    // RLE_DICTIONARY in data page, but parquet v1 uses PLAIN_DICTIONARY in both dictionary and
    // data pages (actual encoding is identical).
    Encoding encoding;
    if (enable_bool_rle) {
      encoding = (col_g.physical_type == BOOLEAN) ? Encoding::RLE
                 : (page_type == PageType::DICTIONARY_PAGE || page_g.chunk->use_dictionary)
                   ? Encoding::PLAIN_DICTIONARY
                   : Encoding::PLAIN;
    } else {
      encoding = (page_type == PageType::DICTIONARY_PAGE || page_g.chunk->use_dictionary)
                   ? Encoding::PLAIN_DICTIONARY
                   : Encoding::PLAIN;
    }
    encoder.field_int32(1, page_type);
    encoder.field_int32(2, uncompressed_page_size);
    encoder.field_int32(3, compressed_page_size);
    if (page_type == PageType::DATA_PAGE) {
      // DataPageHeader
      encoder.field_struct_begin(5);
      encoder.field_int32(1, page_g.num_values);  // NOTE: num_values != num_rows for list types
      encoder.field_int32(2, encoding);           // encoding
      encoder.field_int32(3, Encoding::RLE);      // definition_level_encoding
      encoder.field_int32(4, Encoding::RLE);      // repetition_level_encoding
      // Optionally encode page-level statistics
      if (not page_stats.empty()) {
        encoder.field_struct_begin(5);
        encoder.set_ptr(
          EncodeStatistics(encoder.get_ptr(), &page_stats[blockIdx.x], col_g.stats_dtype, scratch));
        encoder.field_struct_end(5);
      }
      encoder.field_struct_end(5);
    } else {
      // DictionaryPageHeader
      encoder.field_struct_begin(7);
      encoder.field_int32(1, ck_g.num_dict_entries);  // number of values in dictionary
      encoder.field_int32(2, encoding);
      encoder.field_struct_end(7);
    }
    encoder.end(&hdr_end, false);
    page_g.hdr_size = (uint32_t)(hdr_end - hdr_start);
  }
  __syncthreads();
  if (t == 0) pages[blockIdx.x] = page_g;
}

// blockDim(1024, 1, 1)
__global__ void __launch_bounds__(1024)
  gpuGatherPages(device_span<EncColumnChunk> chunks, device_span<gpu::EncPage const> pages)
{
  __shared__ __align__(8) EncColumnChunk ck_g;
  __shared__ __align__(8) EncPage page_g;

  uint32_t t = threadIdx.x;
  uint8_t *dst, *dst_base;
  const EncPage* first_page;
  uint32_t num_pages, uncompressed_size;

  if (t == 0) ck_g = chunks[blockIdx.x];
  __syncthreads();

  first_page = ck_g.pages;
  num_pages  = ck_g.num_pages;
  dst        = (ck_g.is_compressed) ? ck_g.compressed_bfr : ck_g.uncompressed_bfr;
  dst += ck_g.ck_stat_size;  // Skip over chunk statistics
  dst_base          = dst;
  uncompressed_size = ck_g.bfr_size;
  for (uint32_t page = 0; page < num_pages; page++) {
    const uint8_t* src;
    uint32_t hdr_len, data_len;

    if (t == 0) { page_g = first_page[page]; }
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
    if (!t && page == 0 && ck_g.use_dictionary) { ck_g.dictionary_size = hdr_len + data_len; }
  }
  if (t == 0) {
    chunks[blockIdx.x].bfr_size        = uncompressed_size;
    chunks[blockIdx.x].compressed_size = (dst - dst_base);
    if (ck_g.use_dictionary) { chunks[blockIdx.x].dictionary_size = ck_g.dictionary_size; }
  }
}

namespace {

/**
 * @brief Tests if statistics are comparable given the column's
 * physical and converted types
 */
__device__ bool is_comparable(Type ptype, ConvertedType ctype)
{
  switch (ptype) {
    case Type::BOOLEAN:
    case Type::INT32:
    case Type::INT64:
    case Type::FLOAT:
    case Type::DOUBLE:
    case Type::BYTE_ARRAY: return true;
    case Type::FIXED_LEN_BYTE_ARRAY:
      if (ctype == ConvertedType::DECIMAL) { return true; }
      [[fallthrough]];
    default: return false;
  }
}

/**
 * @brief Compares two values.
 * @return -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
 */
template <typename T>
constexpr __device__ int32_t compare(T& v1, T& v2)
{
  return (v1 > v2) - (v1 < v2);
}

/**
 * @brief Compares two statistics_val structs.
 * @return < 0 if v1 < v2, 0 if v1 == v2, > 0 if v1 > v2
 */
__device__ int32_t compare_values(Type ptype,
                                  ConvertedType ctype,
                                  const statistics_val& v1,
                                  const statistics_val& v2)
{
  switch (ptype) {
    case Type::BOOLEAN: return compare(v1.u_val, v2.u_val);
    case Type::INT32:
    case Type::INT64:
      switch (ctype) {
        case ConvertedType::UINT_8:
        case ConvertedType::UINT_16:
        case ConvertedType::UINT_32:
        case ConvertedType::UINT_64: return compare(v1.u_val, v2.u_val);
        default:  // assume everything else is signed
          return compare(v1.i_val, v2.i_val);
      }
    case Type::FLOAT:
    case Type::DOUBLE: return compare(v1.fp_val, v2.fp_val);
    case Type::BYTE_ARRAY: return static_cast<string_view>(v1.str_val).compare(v2.str_val);
    case Type::FIXED_LEN_BYTE_ARRAY:
      if (ctype == ConvertedType::DECIMAL) { return compare(v1.d128_val, v2.d128_val); }
  }
  // calling is_comparable() should prevent reaching here
  CUDF_UNREACHABLE("Trying to compare non-comparable type");
  return 0;
}

/**
 * @brief Determine if a set of statstistics are in ascending order.
 */
__device__ bool is_ascending(const statistics_chunk* s,
                             Type ptype,
                             ConvertedType ctype,
                             uint32_t num_pages)
{
  for (uint32_t i = 1; i < num_pages; i++) {
    if (compare_values(ptype, ctype, s[i - 1].min_value, s[i].min_value) > 0 ||
        compare_values(ptype, ctype, s[i - 1].max_value, s[i].max_value) > 0) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Determine if a set of statstistics are in descending order.
 */
__device__ bool is_descending(const statistics_chunk* s,
                              Type ptype,
                              ConvertedType ctype,
                              uint32_t num_pages)
{
  for (uint32_t i = 1; i < num_pages; i++) {
    if (compare_values(ptype, ctype, s[i - 1].min_value, s[i].min_value) < 0 ||
        compare_values(ptype, ctype, s[i - 1].max_value, s[i].max_value) < 0) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Determine the ordering of a set of statistics.
 */
__device__ int32_t calculate_boundary_order(const statistics_chunk* s,
                                            Type ptype,
                                            ConvertedType ctype,
                                            uint32_t num_pages)
{
  if (not is_comparable(ptype, ctype)) { return BoundaryOrder::UNORDERED; }
  if (is_ascending(s, ptype, ctype, num_pages)) {
    return BoundaryOrder::ASCENDING;
  } else if (is_descending(s, ptype, ctype, num_pages)) {
    return BoundaryOrder::DESCENDING;
  }
  return BoundaryOrder::UNORDERED;
}

// align ptr to an 8-byte boundary. address returned will be <= ptr.
constexpr __device__ void* align8(void* ptr)
{
  // it's ok to round down because we have an extra 7 bytes in the buffer
  auto algn = 3 & reinterpret_cast<std::uintptr_t>(ptr);
  return static_cast<char*>(ptr) - algn;
}

}  // namespace

// blockDim(1, 1, 1)
__global__ void __launch_bounds__(1)
  gpuEncodeColumnIndexes(device_span<EncColumnChunk> chunks,
                         device_span<statistics_chunk const> column_stats,
                         size_type column_index_truncate_length)
{
  __align__(8) unsigned char s_scratch[MIN_STATS_SCRATCH_SIZE];
  uint8_t* col_idx_end;

  if (column_stats.empty()) { return; }

  EncColumnChunk* ck_g             = &chunks[blockIdx.x];
  uint32_t num_pages               = ck_g->num_pages;
  parquet_column_device_view col_g = *ck_g->col_desc;
  size_t first_data_page           = ck_g->use_dictionary ? 1 : 0;
  uint32_t pageidx                 = ck_g->first_page;

  header_encoder encoder(ck_g->column_index_blob);

  // make sure scratch is aligned properly. here column_index_size indicates
  // how much scratch space is available for this chunk, including space for
  // truncation scratch + padding for alignment.
  void* scratch =
    column_index_truncate_length < MIN_STATS_SCRATCH_SIZE
      ? s_scratch
      : align8(ck_g->column_index_blob + ck_g->column_index_size - column_index_truncate_length);

  // null_pages
  encoder.field_list_begin(1, num_pages - first_data_page, ST_FLD_TRUE);
  for (uint32_t page = first_data_page; page < num_pages; page++) {
    encoder.put_bool(column_stats[pageidx + page].non_nulls == 0);
  }
  encoder.field_list_end(1);
  // min_values
  encoder.field_list_begin(2, num_pages - first_data_page, ST_FLD_BINARY);
  for (uint32_t page = first_data_page; page < num_pages; page++) {
    auto const [min_ptr, min_size] = get_extremum(&column_stats[pageidx + page].min_value,
                                                  col_g.stats_dtype,
                                                  scratch,
                                                  true,
                                                  column_index_truncate_length);
    encoder.put_binary(min_ptr, min_size);
  }
  encoder.field_list_end(2);
  // max_values
  encoder.field_list_begin(3, num_pages - first_data_page, ST_FLD_BINARY);
  for (uint32_t page = first_data_page; page < num_pages; page++) {
    auto const [max_ptr, max_size] = get_extremum(&column_stats[pageidx + page].max_value,
                                                  col_g.stats_dtype,
                                                  scratch,
                                                  false,
                                                  column_index_truncate_length);
    encoder.put_binary(max_ptr, max_size);
  }
  encoder.field_list_end(3);
  // boundary_order
  encoder.field_int32(4,
                      calculate_boundary_order(&column_stats[first_data_page + pageidx],
                                               col_g.physical_type,
                                               col_g.converted_type,
                                               num_pages - first_data_page));
  // null_counts
  encoder.field_list_begin(5, num_pages - first_data_page, ST_FLD_I64);
  for (uint32_t page = first_data_page; page < num_pages; page++) {
    encoder.put_int64(column_stats[pageidx + page].null_count);
  }
  encoder.field_list_end(5);
  encoder.end(&col_idx_end, false);

  // now reset column_index_size to the actual size of the encoded column index blob
  ck_g->column_index_size = static_cast<uint32_t>(col_idx_end - ck_g->column_index_blob);
}

void InitPageFragments(device_2dspan<PageFragment> frag,
                       device_span<parquet_column_device_view const> col_desc,
                       device_span<partition_info const> partitions,
                       device_span<int const> part_frag_offset,
                       uint32_t fragment_size,
                       rmm::cuda_stream_view stream)
{
  auto const num_columns              = frag.size().first;
  auto const num_fragments_per_column = frag.size().second;
  auto const grid_y = std::min(static_cast<uint32_t>(num_fragments_per_column), MAX_GRID_Y_SIZE);
  dim3 const dim_grid(num_columns, grid_y);  // 1 threadblock per fragment
  gpuInitPageFragments<512><<<dim_grid, 512, 0, stream.value()>>>(
    frag, col_desc, partitions, part_frag_offset, fragment_size);
}

void InitFragmentStatistics(device_2dspan<statistics_group> groups,
                            device_2dspan<PageFragment const> fragments,
                            device_span<parquet_column_device_view const> col_desc,
                            rmm::cuda_stream_view stream)
{
  int const num_columns              = col_desc.size();
  int const num_fragments_per_column = fragments.size().second;
  auto const y_dim =
    util::div_rounding_up_safe(num_fragments_per_column, 128 / cudf::detail::warp_size);
  auto const grid_y = std::min(static_cast<uint32_t>(y_dim), MAX_GRID_Y_SIZE);
  dim3 const dim_grid(num_columns, grid_y);  // 1 warp per fragment
  gpuInitFragmentStats<<<dim_grid, 128, 0, stream.value()>>>(groups, fragments, col_desc);
}

void InitEncoderPages(device_2dspan<EncColumnChunk> chunks,
                      device_span<gpu::EncPage> pages,
                      device_span<size_type> page_sizes,
                      device_span<size_type> comp_page_sizes,
                      device_span<parquet_column_device_view const> col_desc,
                      int32_t num_columns,
                      size_t max_page_size_bytes,
                      size_type max_page_size_rows,
                      uint32_t page_align,
                      statistics_merge_group* page_grstats,
                      statistics_merge_group* chunk_grstats,
                      rmm::cuda_stream_view stream)
{
  auto num_rowgroups = chunks.size().first;
  dim3 dim_grid(num_columns, num_rowgroups);  // 1 threadblock per rowgroup
  gpuInitPages<<<dim_grid, 128, 0, stream.value()>>>(chunks,
                                                     pages,
                                                     page_sizes,
                                                     comp_page_sizes,
                                                     col_desc,
                                                     page_grstats,
                                                     chunk_grstats,
                                                     num_columns,
                                                     max_page_size_bytes,
                                                     max_page_size_rows,
                                                     page_align);
}

void EncodePages(device_span<gpu::EncPage> pages,
                 device_span<device_span<uint8_t const>> comp_in,
                 device_span<device_span<uint8_t>> comp_out,
                 device_span<compression_result> comp_results,
                 rmm::cuda_stream_view stream)
{
  auto num_pages = pages.size();
  // A page is part of one column. This is launching 1 block per page. 1 block will exclusively
  // deal with one datatype.
  gpuEncodePages<128>
    <<<num_pages, 128, 0, stream.value()>>>(pages, comp_in, comp_out, comp_results);
}

void DecideCompression(device_span<EncColumnChunk> chunks, rmm::cuda_stream_view stream)
{
  gpuDecideCompression<<<chunks.size(), 128, 0, stream.value()>>>(chunks);
}

void EncodePageHeaders(device_span<EncPage> pages,
                       device_span<compression_result const> comp_results,
                       device_span<statistics_chunk const> page_stats,
                       const statistics_chunk* chunk_stats,
                       rmm::cuda_stream_view stream)
{
  // TODO: single thread task. No need for 128 threads/block. Earlier it used to employ rest of the
  // threads to coop load structs
  gpuEncodePageHeaders<<<pages.size(), 128, 0, stream.value()>>>(
    pages, comp_results, page_stats, chunk_stats);
}

void GatherPages(device_span<EncColumnChunk> chunks,
                 device_span<gpu::EncPage const> pages,
                 rmm::cuda_stream_view stream)
{
  gpuGatherPages<<<chunks.size(), 1024, 0, stream.value()>>>(chunks, pages);
}

void EncodeColumnIndexes(device_span<EncColumnChunk> chunks,
                         device_span<statistics_chunk const> column_stats,
                         size_type column_index_truncate_length,
                         rmm::cuda_stream_view stream)
{
  gpuEncodeColumnIndexes<<<chunks.size(), 1, 0, stream.value()>>>(
    chunks, column_stats, column_index_truncate_length);
}

}  // namespace gpu
}  // namespace parquet
}  // namespace io
}  // namespace cudf
