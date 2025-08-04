/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "compact_protocol_reader.hpp"
#include "cudf/types.hpp"
#include "io/comp/decompression.hpp"
#include "reader_impl_chunking.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>

namespace cudf::io::parquet::detail {

using cudf::io::detail::codec_exec_result;
using cudf::io::detail::codec_status;
using cudf::io::detail::decompression_info;

// Forward declarations
struct cumulative_page_info;
struct page_span;

#if defined(CHUNKING_DEBUG)
void print_cumulative_page_info(device_span<PageInfo const> d_pages,
                                device_span<ColumnChunkDesc const> d_chunks,
                                device_span<cumulative_page_info const> d_c_info,
                                rmm::cuda_stream_view stream);
#endif  // CHUNKING_DEBUG

/**
 * @brief Return the required number of bits to store a value
 */
template <typename T = uint8_t>
[[nodiscard]] constexpr inline T required_bits(uint32_t max_level)
{
  return static_cast<T>(CompactProtocolReader::NumRequiredBits(max_level));
}

/**
 * @brief Returns the cudf compression type and whether it is supported by the parquet writer
 */
CUDF_HOST_DEVICE cuda::std::pair<compression_type, bool> parquet_compression_support(
  Compression compression);

/**
 * @brief Returns the string name of the Parquet compression type
 */
[[nodiscard]] std::string parquet_compression_name(Compression compression);

/**
 * @brief Converts a cudf Compression enum to a parquet compression type
 */
[[nodiscard]] compression_type from_parquet_compression(Compression compression);

/**
 * @brief Find the first entry in the aggreggated_info that corresponds to the specified row
 */
size_t find_start_index(cudf::host_span<cumulative_page_info const> aggregated_info,
                        size_t start_row);

/**
 * @brief Given a current position and row index, find the next split based on the
 * specified size limit
 *
 * @returns The inclusive index within `sizes` where the next split should happen
 */
int64_t find_next_split(int64_t cur_pos,
                        size_t cur_row_index,
                        size_t cur_cumulative_size,
                        cudf::host_span<cumulative_page_info const> sizes,
                        size_t size_limit,
                        size_t min_row_count);

/**
 * @brief Converts cuDF units to Parquet units
 *
 * @return A tuple of Parquet clock rate and Parquet decimal type.
 */
[[nodiscard]] std::tuple<int32_t, std::optional<LogicalType>> conversion_info(
  type_id column_type_id,
  type_id timestamp_type_id,
  Type physical,
  std::optional<LogicalType> logical_type);

/**
 * @brief return compressed and total size of the data in a row group
 *
 */
std::pair<size_t, size_t> get_row_group_size(RowGroup const& rg);

/**
 * @brief For a set of cumulative_page_info data, adjust the size_bytes field
 * such that it reflects the worst case for all pages that span the same rows
 *
 * By doing this, we can now look at row X and know the total
 * byte cost for all pages that span row X, not just the cost up to row X itself.
 *
 * This function is asynchronous. Call stream.synchronize() before using the
 * results.
 */
std::pair<rmm::device_uvector<cumulative_page_info>, rmm::device_uvector<int32_t>>
adjust_cumulative_sizes(device_span<cumulative_page_info const> c_info,
                        device_span<PageInfo const> pages,
                        rmm::cuda_stream_view stream);

/**
 * @brief Computes the next subpass within the current pass
 *
 * A subpass is a subset of the pages within the parent pass that is decompressed
 * as a batch and decoded.  Subpasses are the level at which we control memory intermediate
 * memory usage. A pass consists of >= 1 subpass.  We cannot compute all subpasses in one
 * shot because we do not know how many rows we actually have in the pages of list columns.
 * So we have to make an educated guess that fits within the memory limits, and then adjust
 * for subsequent subpasses when we see how many rows we actually receive.
 *
 * @param c_info The cumulative page size information (row count and byte size) per column
 * @param pages All of the pages in the pass
 * @param chunks All of the chunks in the pass
 * @param page_offsets Offsets into the pages array representing the first page for each column
 * @param start_row The row to start the subpass at
 * @param size_limit The size limit in bytes of the subpass
 * @param num_columns The number of columns
 * @param is_first_subpass Boolean indicating if this is the first subpass
 * @param has_page_index Boolean indicating if we have a page index
 * @param stream The stream to execute cuda operations on
 * @returns A tuple containing a vector of page_span structs indicating the page indices to include
 * for each column to be processed, the total number of pages over all columns, and the total
 * expected memory usage (including scratch space)
 *
 */
std::tuple<rmm::device_uvector<page_span>, size_t, size_t> compute_next_subpass(
  device_span<cumulative_page_info const> c_info,
  device_span<PageInfo const> pages,
  device_span<ColumnChunkDesc const> chunks,
  device_span<size_type const> page_offsets,
  size_t start_row,
  size_t size_limit,
  size_t num_columns,
  bool is_first_subpass,
  bool has_page_index,
  rmm::cuda_stream_view stream);

/**
 * @brief Computes the page splits for a given set of pages based on row count and size limit
 *
 * This function computes the page splits based on the cumulative page information, the number of
 * rows to skip, the total number of rows, and the size limit. It returns a vector of row_range
 * structs indicating the start and end rows for each split.
 *
 * @param c_info The cumulative page size information (row count and byte size) per column
 * @param pages All of the pages in the pass
 * @param skip_rows The number of rows to skip before starting the split computation
 * @param num_rows The total number of rows to consider for splitting
 * @param size_limit The size limit in bytes for each split
 * @param stream The stream to execute cuda operations on
 * @returns A vector of row_range structs indicating the start and end rows for each split
 */
std::vector<row_range> compute_page_splits_by_row(device_span<cumulative_page_info const> c_info,
                                                  device_span<PageInfo const> pages,
                                                  size_t skip_rows,
                                                  size_t num_rows,
                                                  size_t size_limit,
                                                  rmm::cuda_stream_view stream);

/**
 * @brief Decompresses a mix of dictionary and non-dictionary pages from a set of column chunks
 *
 * To avoid multiple calls to the decompression kernel, we batch pages by codec type, where the
 * batch can include both dictionary and non-dictionary pages. This allows us to decompress all
 * pages of a given codec type in one go.
 *
 * @param chunks List of column chunk descriptors
 * @param pass_pages List of page information for the pass
 * @param subpass_pages List of page information for the subpass
 * @param subpass_page_mask Boolean page mask indicating which subpass pages to decompress. Empty
 * span indicates all pages should be decompressed
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate buffers
 *
 * @return A pair of device buffers containing the decompressed data for dictionary and
 * non-dictionary pages, respectively.
 */
[[nodiscard]] std::pair<rmm::device_buffer, rmm::device_buffer> decompress_page_data(
  host_span<ColumnChunkDesc const> chunks,
  host_span<PageInfo> pass_pages,
  host_span<PageInfo> subpass_pages,
  host_span<bool const> subpass_page_mask,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Detect malformed parquet input data
 *
 * We have seen cases where parquet files can be oddly malformed. This function specifically
 * detects one case in particular:
 *
 * - When you have a file containing N rows
 * - For some reason, the sum total of the number of rows over all pages for a given column
 *   is != N
 *
 * @param pages All pages to be decoded
 * @param chunks Chunk data
 * @param expected_row_count Expected row count, if applicable
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void detect_malformed_pages(device_span<PageInfo const> pages,
                            device_span<ColumnChunkDesc const> chunks,
                            std::optional<size_t> expected_row_count,
                            rmm::cuda_stream_view stream);

/**
 * @brief Add the cost of decompression codec scratch space to the per-page cumulative
 * size information
 */
void include_decompression_scratch_size(device_span<ColumnChunkDesc const> chunks,
                                        device_span<PageInfo const> pages,
                                        device_span<cumulative_page_info> c_info,
                                        rmm::cuda_stream_view stream);

/**
 * @brief Struct to store split information
 */
struct split_info {
  row_range rows;
  int64_t split_pos;
};

/**
 * @brief
 */
struct cumulative_page_info {
  size_t end_row_index;  // end row index (start_row + num_rows for the corresponding page)
  size_t size_bytes;     // cumulative size in bytes
  cudf::size_type key;   // schema index
};

/**
 * @brief Struct to store the start and end of a page's row span
 */
struct page_span {
  size_t start;
  size_t end;
};

/**
 * @brief Functor which returns the size of a span
 */
struct get_span_size {
  CUDF_HOST_DEVICE inline size_t operator()(page_span const& s) const { return s.end - s.start; }
};

/**
 * @brief Functor which returns the size of a span in an array of spans, handling out-of-bounds
 * indices
 */
struct get_span_size_by_index {
  cudf::device_span<page_span const> page_indices;
  __device__ inline size_t operator()(size_t i) const
  {
    return i >= page_indices.size() ? 0 : page_indices[i].end - page_indices[i].start;
  }
};

/**
 * @brief Functor which returns the span of page indices for a given column index
 */
struct get_page_span_by_column {
  cudf::device_span<size_type const> page_offsets;
  __device__ inline page_span operator()(size_t i) const
  {
    return {static_cast<size_t>(page_offsets[i]), static_cast<size_t>(page_offsets[i + 1])};
  }
};

/**
 * @brief Functor which returns the end row index for a cumulative_page_info
 */
struct get_page_end_row_index {
  device_span<cumulative_page_info const> c_info;
  __device__ inline size_t operator()(size_t i) const { return c_info[i].end_row_index; }
};

/**
 * @brief Functor which reduces two cumulative_page_info structs of the same key
 */
struct cumulative_page_sum {
  __device__ inline cumulative_page_info operator()(cumulative_page_info const& a,
                                                    cumulative_page_info const& b) const
  {
    return cumulative_page_info{0, a.size_bytes + b.size_bytes, a.key};
  }
};

/**
 * @brief Functor which returns the compressed data size for a chunk
 */
struct get_chunk_compressed_size {
  __device__ inline size_t operator()(ColumnChunkDesc const& chunk) const
  {
    return chunk.compressed_size;
  }
};

/**
 * @brief Functor which returns the number of rows in a page for a flat column
 */
struct flat_column_num_rows {
  ColumnChunkDesc const* chunks;
  __device__ inline size_type operator()(PageInfo const& page) const
  {
    // Ignore dictionary pages and pages belonging to any column containing repetition (lists)
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) ||
        (chunks[page.chunk_idx].max_level[level_type::REPETITION] > 0)) {
      return 0;
    }
    return page.num_rows;
  }
};

/**
 * @brief Stores basic information about pages compressed with a specific codec
 */
struct codec_stats {
  Compression compression_type  = Compression::UNCOMPRESSED;
  size_t num_pages              = 0;
  int32_t max_decompressed_size = 0;
  size_t total_decomp_size      = 0;

  enum class page_selection { DICT_PAGES, NON_DICT_PAGES };

  void add_pages(host_span<ColumnChunkDesc const> chunks,
                 host_span<PageInfo> pages,
                 page_selection selection,
                 host_span<bool const> page_mask);
};

/**
 * @brief Functor which retrieves per-page decompression information.
 */
struct get_decomp_info {
  device_span<ColumnChunkDesc const> chunks;
  __device__ inline decompression_info operator()(PageInfo const& p) const
  {
    return {parquet_compression_support(chunks[p.chunk_idx].codec).first,
            1,
            static_cast<size_t>(p.uncompressed_page_size),
            static_cast<size_t>(p.uncompressed_page_size)};
  }
};

/**
 * @brief Functor which accumulates per-page decompression information.
 */
struct decomp_sum {
  __device__ inline decompression_info operator()(decompression_info const& a,
                                                  decompression_info const& b) const
  {
    return {a.type,
            a.num_pages + b.num_pages,
            cuda::std::max(a.max_page_decompressed_size, b.max_page_decompressed_size),
            a.total_decompressed_size + b.total_decompressed_size};
  }
};

/**
 * @brief Functor which check if row count is not zero
 */
struct row_counts_nonzero {
  __device__ inline bool operator()(size_type count) const { return count > 0; }
};

/**
 * @brief Functor which checks if row count is non zero and different from expected value
 */
struct row_counts_different {
  size_type const expected;
  __device__ inline bool operator()(size_type count) const
  {
    return (count != 0) && (count != expected);
  }
};

/**
 * @brief Functor which computes the total data size for a given type of cudf column
 *
 * In the case of strings, the return size does not include the chars themselves. That
 * information is tracked separately (see PageInfo::str_bytes).
 */
struct row_size_functor {
  __device__ inline size_t validity_size(size_t num_rows, bool nullable)
  {
    // TODO: Use a util from `null_mask.cuh` instead of this calculation when available
    return nullable ? cudf::util::div_rounding_up_safe<size_type>(
                        num_rows, cudf::detail::size_in_bits<bitmask_type>()) *
                        sizeof(bitmask_type)
                    : 0;
  }

  template <typename T>
  __device__ inline size_t operator()(size_t num_rows, bool nullable)
  {
    auto const element_size = sizeof(device_storage_type_t<T>);
    return (element_size * num_rows) + validity_size(num_rows, nullable);
  }
};

template <>
__device__ inline size_t row_size_functor::operator()<list_view>(size_t num_rows, bool nullable)
{
  auto const offset_size = sizeof(size_type);
  // NOTE: Adding the + 1 offset here isn't strictly correct.  There will only be 1 extra offset
  // for the entire column, whereas this is adding an extra offset per page.  So we will get a
  // small over-estimate of the real size of the order :  # of pages * 4 bytes. It seems better
  // to overestimate size somewhat than to underestimate it and potentially generate chunks
  // that are too large.
  return (offset_size * (num_rows + 1)) + validity_size(num_rows, nullable);
}

template <>
__device__ inline size_t row_size_functor::operator()<struct_view>(size_t num_rows, bool nullable)
{
  return validity_size(num_rows, nullable);
}

template <>
__device__ inline size_t row_size_functor::operator()<string_view>(size_t num_rows, bool nullable)
{
  // only returns the size of offsets and validity. the size of the actual string chars
  // is tracked separately.
  auto const offset_size = sizeof(size_type);
  // see note about offsets in the list_view template.
  return (offset_size * (num_rows + 1)) + validity_size(num_rows, nullable);
}

/**
 * @brief Functor which computes the total output cudf data size for all of
 * the data in this page
 *
 * Sums across all nesting levels.
 */
struct get_page_output_size {
  __device__ cumulative_page_info operator()(PageInfo const& page) const
  {
    if (page.flags & PAGEINFO_FLAGS_DICTIONARY) {
      return cumulative_page_info{0, 0, page.src_col_schema};
    }

    // total nested size, not counting string data
    auto iter = cudf::detail::make_counting_transform_iterator(
      0, cuda::proclaim_return_type<size_t>([page] __device__(size_type i) {
        auto const& pni = page.nesting[i];
        return cudf::type_dispatcher(
          data_type{pni.type}, row_size_functor{}, pni.size, pni.nullable);
      }));
    return {
      0,
      thrust::reduce(thrust::seq, iter, iter + page.num_output_nesting_levels) + page.str_bytes,
      page.src_col_schema};
  }
};

/**
 * @brief Functor which sets the (uncompressed) size of a page
 */
struct get_page_input_size {
  __device__ inline cumulative_page_info operator()(PageInfo const& page) const
  {
    // we treat dictionary page sizes as 0 for subpasses because we have already paid the price for
    // them at the pass level.
    if (page.flags & PAGEINFO_FLAGS_DICTIONARY) { return {0, 0, page.src_col_schema}; }
    return {0, static_cast<size_t>(page.uncompressed_page_size), page.src_col_schema};
  }
};

/**
 * @brief Functor which sets the absolute row index of a page in a cumulative_page_info struct
 */
struct set_row_index {
  device_span<ColumnChunkDesc const> chunks;
  device_span<PageInfo const> pages;
  device_span<cumulative_page_info> c_info;
  size_t max_row;

  __device__ inline void operator()(size_t i)
  {
    auto const& page          = pages[i];
    auto const& chunk         = chunks[page.chunk_idx];
    size_t const page_end_row = chunk.start_row + page.chunk_row + page.num_rows;
    // this cap is necessary because in the chunked reader, we use estimations for the row
    // counts for list columns, which can result in values > than the absolute number of rows.
    c_info[i].end_row_index = cuda::std::min(max_row, page_end_row);
  }
};

/**
 * @brief Functor which computes the effective size of all input columns by page
 *
 * For a given row, we want to find the cost of all pages for all columns involved
 * in loading up to that row.  The complication here is that not all pages are the
 * same size between columns. Example:
 *
 *              page row counts
 * Column A:    0 <----> 100 <----> 200
 * Column B:    0 <---------------> 200 <--------> 400
                          |
 * if we decide to split at row 100, we don't really know the actual amount of bytes in column B
 * at that point.  So we have to proceed as if we are taking the bytes from all 200 rows of that
 * page. Essentially, a conservative over-estimate of the real size.
 */
struct page_total_size {
  cumulative_page_info const* c_info;
  size_type const* key_offsets;
  size_t num_keys;

  __device__ cumulative_page_info operator()(cumulative_page_info const& i) const
  {
    // sum sizes for each input column at this row
    size_t sum = 0;
    for (auto idx = 0; std::cmp_less(idx, num_keys); idx++) {
      auto const start = key_offsets[idx];
      auto const end   = key_offsets[idx + 1];
      auto iter        = cudf::detail::make_counting_transform_iterator(
        0, cuda::proclaim_return_type<size_t>([&] __device__(size_type i) {
          return c_info[i].end_row_index;
        }));
      auto const page_index =
        thrust::lower_bound(thrust::seq, iter + start, iter + end, i.end_row_index) - iter;
      sum += c_info[page_index].size_bytes;
    }
    return {i.end_row_index, sum, i.key};
  }
};

/**
 * @brief Return the span of page indices for a given column index that spans start_row and end_row
 *
 */
template <typename RowIndexIter>
struct get_page_span {
  device_span<size_type const> page_offsets;
  device_span<ColumnChunkDesc const> chunks;
  RowIndexIter page_row_index;
  size_t const start_row;
  size_t const end_row;
  bool const is_first_subpass;
  bool const has_page_index;

  get_page_span(device_span<size_type const> _page_offsets,
                device_span<ColumnChunkDesc const> _chunks,
                RowIndexIter _page_row_index,
                size_t _start_row,
                size_t _end_row,
                bool _is_first_subpass,
                bool _has_page_index)
    : page_offsets(_page_offsets),
      chunks(_chunks),
      page_row_index(_page_row_index),
      start_row(_start_row),
      end_row(_end_row),
      is_first_subpass(_is_first_subpass),
      has_page_index(_has_page_index)
  {
  }

  __device__ page_span operator()(size_t column_index) const
  {
    auto const first_page_index  = page_offsets[column_index];
    auto const column_page_start = page_row_index + first_page_index;
    auto const column_page_end   = page_row_index + page_offsets[column_index + 1];
    auto const num_pages         = column_page_end - column_page_start;
    bool const is_list           = chunks[column_index].max_level[level_type::REPETITION] > 0;

    // For list columns, the row counts are estimates so we need all prefix pages to correctly
    // compute page bounds. For non-list columns, we can get an exact span of pages.
    auto start_page              = first_page_index;
    auto const update_start_page = has_page_index or (not is_list) or (not is_first_subpass);
    if (update_start_page) {
      start_page += cuda::std::distance(
        column_page_start,
        thrust::lower_bound(thrust::seq, column_page_start, column_page_end, start_row));
    }
    if (page_row_index[start_page] == start_row and (has_page_index or not is_list)) {
      start_page++;
    }

    auto end_page =
      cuda::std::distance(
        column_page_start,
        thrust::lower_bound(thrust::seq, column_page_start, column_page_end, end_row)) +
      first_page_index;
    if (end_page < (first_page_index + num_pages)) { end_page++; }

    return {static_cast<size_t>(start_page), static_cast<size_t>(end_page)};
  }
};

/**
 * @brief Copy page from appropriate source location (as defined by page_offsets) to the destination
 * location, and store the index mapping
 */
struct copy_subpass_page {
  cudf::device_span<PageInfo const> src_pages;
  cudf::device_span<PageInfo> dst_pages;
  cudf::device_span<size_t> page_src_index;
  cudf::device_span<size_t const> page_offsets;
  cudf::device_span<page_span const> page_indices;
  __device__ void operator()(size_t i) const
  {
    auto const index =
      thrust::lower_bound(thrust::seq, page_offsets.begin(), page_offsets.end(), i) -
      page_offsets.begin();
    auto const col_index = page_offsets[index] == i ? index : index - 1;
    // index within the pages for the column
    auto const col_page_index = i - page_offsets[col_index];
    auto const src_page_index = page_indices[col_index].start + col_page_index;

    dst_pages[i]      = src_pages[src_page_index];
    page_src_index[i] = src_page_index;
  }
};

}  // namespace cudf::io::parquet::detail
