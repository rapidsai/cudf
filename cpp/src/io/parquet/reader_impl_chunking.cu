/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "io/comp/nvcomp_adapter.hpp"
#include "io/utilities/config_utils.hpp"
#include "io/utilities/time_utils.cuh"
#include "reader_impl.hpp"
#include "reader_impl_chunking.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

#include <numeric>

namespace cudf::io::parquet::detail {

namespace {

struct split_info {
  row_range rows;
  int64_t split_pos;
};

struct cumulative_page_info {
  size_t end_row_index;  // end row index (start_row + num_rows for the corresponding page)
  size_t size_bytes;     // cumulative size in bytes
  int key;               // schema index
};

// the minimum amount of memory we can safely expect to be enough to
// do a subpass decode. if the difference between the user specified limit and
// the actual memory used for compressed/temp data is > than this value, we will still use
// at least this many additional bytes.
// Example:
// - user has specified 1 GB limit
// - we have read in 900 MB of compressed data
// - that leaves us 100 MB of space for decompression batches
// - to keep the gpu busy, we really don't want to do less than 200 MB at a time so we're just going
// to use 200 MB of space
//   even if that goes past the user-specified limit.
constexpr size_t minimum_subpass_expected_size = 200 * 1024 * 1024;

// percentage of the total available input read limit that should be reserved for compressed
// data vs uncompressed data.
constexpr float input_limit_compression_reserve = 0.3f;

#if defined(CHUNKING_DEBUG)
void print_cumulative_page_info(device_span<PageInfo const> d_pages,
                                device_span<ColumnChunkDesc const> d_chunks,
                                device_span<cumulative_page_info const> d_c_info,
                                rmm::cuda_stream_view stream)
{
  std::vector<PageInfo> pages              = cudf::detail::make_std_vector_sync(d_pages, stream);
  std::vector<ColumnChunkDesc> chunks      = cudf::detail::make_std_vector_sync(d_chunks, stream);
  std::vector<cumulative_page_info> c_info = cudf::detail::make_std_vector_sync(d_c_info, stream);

  printf("------------\nCumulative sizes by page\n");

  std::vector<int> schemas(pages.size());
  auto schema_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](size_type i) { return pages[i].src_col_schema; });
  thrust::copy(thrust::seq, schema_iter, schema_iter + pages.size(), schemas.begin());
  auto last = thrust::unique(thrust::seq, schemas.begin(), schemas.end());
  schemas.resize(last - schemas.begin());
  printf("Num schemas: %lu\n", schemas.size());

  for (size_t idx = 0; idx < schemas.size(); idx++) {
    printf("Schema %d\n", schemas[idx]);
    for (size_t pidx = 0; pidx < pages.size(); pidx++) {
      auto const& page = pages[pidx];
      if (page.flags & PAGEINFO_FLAGS_DICTIONARY || page.src_col_schema != schemas[idx]) {
        continue;
      }
      bool const is_list = chunks[page.chunk_idx].max_level[level_type::REPETITION] > 0;
      printf("\tP %s: {%lu, %lu, %lu}\n",
             is_list ? "(L)" : "",
             pidx,
             c_info[pidx].row_index,
             c_info[pidx].size_bytes);
    }
  }
}

void print_cumulative_row_info(host_span<cumulative_page_info const> sizes,
                               std::string const& label,
                               std::optional<std::vector<row_range>> splits = std::nullopt)
{
  if (splits.has_value()) {
    printf("------------\nSplits (skip_rows, num_rows)\n");
    for (size_t idx = 0; idx < splits->size(); idx++) {
      printf("{%lu, %lu}\n", splits.value()[idx].skip_rows, splits.value()[idx].num_rows);
    }
  }

  printf("------------\nCumulative sizes %s (index, row_index, size_bytes, page_key)\n",
         label.c_str());
  for (size_t idx = 0; idx < sizes.size(); idx++) {
    printf("{%lu, %lu, %lu, %d}", idx, sizes[idx].row_index, sizes[idx].size_bytes, sizes[idx].key);
    if (splits.has_value()) {
      // if we have a split at this row count and this is the last instance of this row count
      auto start             = thrust::make_transform_iterator(splits->begin(),
                                                   [](row_range const& i) { return i.skip_rows; });
      auto end               = start + splits->size();
      auto split             = std::find(start, end, sizes[idx].row_index);
      auto const split_index = [&]() -> int {
        if (split != end &&
            ((idx == sizes.size() - 1) || (sizes[idx + 1].row_index > sizes[idx].row_index))) {
          return static_cast<int>(std::distance(start, split));
        }
        return idx == 0 ? 0 : -1;
      }();
      if (split_index >= 0) {
        printf(" <-- split {%lu, %lu}",
               splits.value()[split_index].skip_rows,
               splits.value()[split_index].num_rows);
      }
    }
    printf("\n");
  }
}
#endif  // CHUNKING_DEBUG

/**
 * @brief Functor which reduces two cumulative_page_info structs of the same key.
 */
struct cumulative_page_sum {
  cumulative_page_info operator()
    __device__(cumulative_page_info const& a, cumulative_page_info const& b) const
  {
    return cumulative_page_info{0, a.size_bytes + b.size_bytes, a.key};
  }
};

/**
 * @brief Functor which computes the total data size for a given type of cudf column.
 *
 * In the case of strings, the return size does not include the chars themselves. That
 * information is tracked separately (see PageInfo::str_bytes).
 */
struct row_size_functor {
  __device__ size_t validity_size(size_t num_rows, bool nullable)
  {
    return nullable ? (cudf::util::div_rounding_up_safe(num_rows, size_t{32}) * 4) : 0;
  }

  template <typename T>
  __device__ size_t operator()(size_t num_rows, bool nullable)
  {
    auto const element_size = sizeof(device_storage_type_t<T>);
    return (element_size * num_rows) + validity_size(num_rows, nullable);
  }
};

template <>
__device__ size_t row_size_functor::operator()<list_view>(size_t num_rows, bool nullable)
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
__device__ size_t row_size_functor::operator()<struct_view>(size_t num_rows, bool nullable)
{
  return validity_size(num_rows, nullable);
}

template <>
__device__ size_t row_size_functor::operator()<string_view>(size_t num_rows, bool nullable)
{
  // only returns the size of offsets and validity. the size of the actual string chars
  // is tracked separately.
  auto const offset_size = sizeof(size_type);
  // see note about offsets in the list_view template.
  return (offset_size * (num_rows + 1)) + validity_size(num_rows, nullable);
}

/**
 * @brief Functor which computes the total output cudf data size for all of
 * the data in this page.
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
 * @brief Functor which sets the (uncompressed) size of a page.
 */
struct get_page_input_size {
  __device__ cumulative_page_info operator()(PageInfo const& page) const
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

  __device__ void operator()(size_t i)
  {
    auto const& page          = pages[i];
    auto const& chunk         = chunks[page.chunk_idx];
    size_t const page_end_row = chunk.start_row + page.chunk_row + page.num_rows;
    // if we have been passed in a cap, apply it
    c_info[i].end_row_index = max_row > 0 ? min(max_row, page_end_row) : page_end_row;
  }
};

/**
 * @brief Functor which computes the effective size of all input columns by page.
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
    for (int idx = 0; idx < num_keys; idx++) {
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
 * @brief Functor which returns the compressed data size for a chunk
 */
struct get_chunk_compressed_size {
  __device__ size_t operator()(ColumnChunkDesc const& chunk) const { return chunk.compressed_size; }
};

/**
 * @brief Find the first entry in the aggreggated_info that corresponds to the specified row
 *
 */
size_t find_start_index(cudf::host_span<cumulative_page_info const> aggregated_info,
                        size_t start_row)
{
  auto start = thrust::make_transform_iterator(
    aggregated_info.begin(), [&](cumulative_page_info const& i) { return i.end_row_index; });
  return thrust::lower_bound(thrust::host, start, start + aggregated_info.size(), start_row) -
         start;
}

/**
 * @brief Given a current position and row index, find the next split based on the
 * specified size limit
 *
 * @returns The inclusive index within `sizes` where the next split should happen
 *
 */
int64_t find_next_split(int64_t cur_pos,
                        size_t cur_row_index,
                        size_t cur_cumulative_size,
                        cudf::host_span<cumulative_page_info const> sizes,
                        size_t size_limit)
{
  auto const start = thrust::make_transform_iterator(
    sizes.begin(),
    [&](cumulative_page_info const& i) { return i.size_bytes - cur_cumulative_size; });
  auto const end = start + sizes.size();

  int64_t split_pos = thrust::lower_bound(thrust::seq, start + cur_pos, end, size_limit) - start;

  // if we're past the end, or if the returned bucket is > than the chunk_read_limit, move back
  // one as long as this doesn't put us before our starting point.
  if (static_cast<size_t>(split_pos) >= sizes.size() ||
      ((split_pos > cur_pos) && (sizes[split_pos].size_bytes - cur_cumulative_size > size_limit))) {
    split_pos--;
  }

  // move forward until we find the next group of pages that will actually advance our row count.
  // this guarantees that even if we cannot fit the set of rows represented by our where our cur_pos
  // is, we will still move forward instead of failing.
  while (split_pos < (static_cast<int64_t>(sizes.size()) - 1) &&
         (sizes[split_pos].end_row_index == cur_row_index)) {
    split_pos++;
  }

  return split_pos;
}

/**
 * @brief Converts cuDF units to Parquet units.
 *
 * @return A tuple of Parquet type width, Parquet clock rate and Parquet decimal type.
 */
[[nodiscard]] std::tuple<int32_t, int32_t, int8_t> conversion_info(
  type_id column_type_id,
  type_id timestamp_type_id,
  Type physical,
  cuda::std::optional<ConvertedType> converted,
  int32_t length)
{
  int32_t type_width = (physical == FIXED_LEN_BYTE_ARRAY) ? length : 0;
  int32_t clock_rate = 0;
  if (column_type_id == type_id::INT8 or column_type_id == type_id::UINT8) {
    type_width = 1;  // I32 -> I8
  } else if (column_type_id == type_id::INT16 or column_type_id == type_id::UINT16) {
    type_width = 2;  // I32 -> I16
  } else if (column_type_id == type_id::INT32) {
    type_width = 4;  // str -> hash32
  } else if (is_chrono(data_type{column_type_id})) {
    clock_rate = to_clockrate(timestamp_type_id);
  }

  int8_t converted_type = converted.value_or(UNKNOWN);
  if (converted_type == DECIMAL && column_type_id != type_id::FLOAT64 &&
      not cudf::is_fixed_point(data_type{column_type_id})) {
    converted_type = UNKNOWN;  // Not converting to float64 or decimal
  }
  return std::make_tuple(type_width, clock_rate, converted_type);
}

/**
 * @brief Return the required number of bits to store a value.
 */
template <typename T = uint8_t>
[[nodiscard]] T required_bits(uint32_t max_level)
{
  return static_cast<T>(CompactProtocolReader::NumRequiredBits(max_level));
}

struct row_count_less {
  __device__ bool operator()(cumulative_page_info const& a, cumulative_page_info const& b) const
  {
    return a.end_row_index < b.end_row_index;
  }
};

/**
 * @brief return compressed and total size of the data in a row group
 *
 */
std::pair<size_t, size_t> get_row_group_size(RowGroup const& rg)
{
  auto compressed_size_iter = thrust::make_transform_iterator(
    rg.columns.begin(), [](ColumnChunk const& c) { return c.meta_data.total_compressed_size; });

  // the trick is that total temp space needed is tricky to know
  auto const compressed_size =
    std::reduce(compressed_size_iter, compressed_size_iter + rg.columns.size());
  auto const total_size = compressed_size + rg.total_byte_size;
  return {compressed_size, total_size};
}

/**
 * @brief For a set of cumulative_page_info data, adjust the size_bytes field
 * such that it reflects the worst case for all pages that span the same rows.
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
                        rmm::cuda_stream_view stream)
{
  // sort by row count
  rmm::device_uvector<cumulative_page_info> c_info_sorted =
    make_device_uvector_async(c_info, stream, rmm::mr::get_current_device_resource());
  thrust::sort(
    rmm::exec_policy_nosync(stream), c_info_sorted.begin(), c_info_sorted.end(), row_count_less{});

  // page keys grouped by split.
  rmm::device_uvector<int32_t> page_keys_by_split{c_info.size(), stream};
  thrust::transform(rmm::exec_policy_nosync(stream),
                    c_info_sorted.begin(),
                    c_info_sorted.end(),
                    page_keys_by_split.begin(),
                    cuda::proclaim_return_type<int>(
                      [] __device__(cumulative_page_info const& c) { return c.key; }));

  // generate key offsets (offsets to the start of each partition of keys). worst case is 1 page per
  // key
  rmm::device_uvector<size_type> key_offsets(pages.size() + 1, stream);
  auto page_keys             = make_page_key_iterator(pages);
  auto const key_offsets_end = thrust::reduce_by_key(rmm::exec_policy(stream),
                                                     page_keys,
                                                     page_keys + pages.size(),
                                                     thrust::make_constant_iterator(1),
                                                     thrust::make_discard_iterator(),
                                                     key_offsets.begin())
                                 .second;
  size_t const num_unique_keys = key_offsets_end - key_offsets.begin();
  thrust::exclusive_scan(
    rmm::exec_policy_nosync(stream), key_offsets.begin(), key_offsets.end(), key_offsets.begin());

  // adjust the cumulative info such that for each row count, the size includes any pages that span
  // that row count. this is so that if we have this case:
  //              page row counts
  // Column A:    0 <----> 100 <----> 200
  // Column B:    0 <---------------> 200 <--------> 400
  //                        |
  // if we decide to split at row 100, we don't really know the actual amount of bytes in column B
  // at that point.  So we have to proceed as if we are taking the bytes from all 200 rows of that
  // page.
  //
  rmm::device_uvector<cumulative_page_info> aggregated_info(c_info.size(), stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    c_info_sorted.begin(),
                    c_info_sorted.end(),
                    aggregated_info.begin(),
                    page_total_size{c_info.data(), key_offsets.data(), num_unique_keys});
  return {std::move(aggregated_info), std::move(page_keys_by_split)};
}

struct page_span {
  size_t start, end;
};

struct get_page_end_row_index {
  device_span<cumulative_page_info const> c_info;

  __device__ size_t operator()(size_t i) const { return c_info[i].end_row_index; }
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

  get_page_span(device_span<size_type const> _page_offsets,
                device_span<ColumnChunkDesc const> _chunks,
                RowIndexIter _page_row_index,
                size_t _start_row,
                size_t _end_row)
    : page_offsets(_page_offsets),
      chunks(_chunks),
      page_row_index(_page_row_index),
      start_row(_start_row),
      end_row(_end_row)
  {
  }

  __device__ page_span operator()(size_t column_index) const
  {
    auto const first_page_index  = page_offsets[column_index];
    auto const column_page_start = page_row_index + first_page_index;
    auto const column_page_end   = page_row_index + page_offsets[column_index + 1];
    auto const num_pages         = column_page_end - column_page_start;
    bool const is_list           = chunks[column_index].max_level[level_type::REPETITION] > 0;

    auto start_page =
      (thrust::lower_bound(thrust::seq, column_page_start, column_page_end, start_row) -
       column_page_start) +
      first_page_index;
    // list rows can span page boundaries, so it is not always safe to assume that the row
    // represented by end_row_index starts on the subsequent page. It is possible that
    // the values for row end_row_index start within the page itself. so we must
    // include the page in that case.
    if (page_row_index[start_page] == start_row && !is_list) { start_page++; }

    auto end_page = (thrust::lower_bound(thrust::seq, column_page_start, column_page_end, end_row) -
                     column_page_start) +
                    first_page_index;
    if (end_page < (first_page_index + num_pages)) { end_page++; }

    return {static_cast<size_t>(start_page), static_cast<size_t>(end_page)};
  }
};

/**
 * @brief Return the span of page indices for a given column index

 */
struct get_page_span_by_column {
  cudf::device_span<size_type const> page_offsets;

  __device__ page_span operator()(size_t i) const
  {
    return {static_cast<size_t>(page_offsets[i]), static_cast<size_t>(page_offsets[i + 1])};
  }
};

/**
 * @brief Return the size of a span
 *
 */
struct get_span_size {
  CUDF_HOST_DEVICE size_t operator()(page_span const& s) const { return s.end - s.start; }
};

/**
 * @brief Return the size of a span in an array of spans, handling out-of-bounds indices.
 *
 */
struct get_span_size_by_index {
  cudf::device_span<page_span const> page_indices;

  __device__ size_t operator()(size_t i) const
  {
    return i >= page_indices.size() ? 0 : page_indices[i].end - page_indices[i].start;
  }
};

/**
 * @brief Copy page from appropriate source location (as defined by page_offsets) to the destination
 * location, and store the index mapping.
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

/**
 * @brief Computes the next subpass within the current pass.
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
  rmm::cuda_stream_view stream)
{
  auto [aggregated_info, page_keys_by_split] = adjust_cumulative_sizes(c_info, pages, stream);

  // bring back to the cpu
  auto const h_aggregated_info = cudf::detail::make_std_vector_sync(aggregated_info, stream);
  // print_cumulative_row_info(h_aggregated_info, "adjusted");

  // TODO: if the user has explicitly specified skip_rows/num_rows we could be more intelligent
  // about skipping subpasses/pages that do not fall within the range of values, but only if the
  // data does not contain lists (because our row counts are only estimates in that case)

  // find the next split
  auto const start_index = find_start_index(h_aggregated_info, start_row);
  auto const cumulative_size =
    start_row == 0 || start_index == 0 ? 0 : h_aggregated_info[start_index - 1].size_bytes;
  auto const end_index =
    find_next_split(start_index, start_row, cumulative_size, h_aggregated_info, size_limit);
  auto const end_row = h_aggregated_info[end_index].end_row_index;

  // for each column, collect the set of pages that spans start_row / end_row
  rmm::device_uvector<page_span> page_bounds(num_columns, stream);
  auto iter = thrust::make_counting_iterator(size_t{0});
  auto page_row_index =
    cudf::detail::make_counting_transform_iterator(0, get_page_end_row_index{c_info});
  thrust::transform(rmm::exec_policy_nosync(stream),
                    iter,
                    iter + num_columns,
                    page_bounds.begin(),
                    get_page_span{page_offsets, chunks, page_row_index, start_row, end_row});

  // total page count over all columns
  auto page_count_iter = thrust::make_transform_iterator(page_bounds.begin(), get_span_size{});
  size_t const total_pages =
    thrust::reduce(rmm::exec_policy(stream), page_count_iter, page_count_iter + num_columns);

  return {
    std::move(page_bounds), total_pages, h_aggregated_info[end_index].size_bytes - cumulative_size};
}

std::vector<row_range> compute_page_splits_by_row(device_span<cumulative_page_info const> c_info,
                                                  device_span<PageInfo const> pages,
                                                  size_t skip_rows,
                                                  size_t num_rows,
                                                  size_t size_limit,
                                                  rmm::cuda_stream_view stream)
{
  auto [aggregated_info, page_keys_by_split] = adjust_cumulative_sizes(c_info, pages, stream);

  // bring back to the cpu
  std::vector<cumulative_page_info> h_aggregated_info =
    cudf::detail::make_std_vector_sync(aggregated_info, stream);
  // print_cumulative_row_info(h_aggregated_info, "adjusted");

  std::vector<row_range> splits;
  // note: we are working with absolute row indices so skip_rows represents the absolute min row
  // index we care about
  size_t cur_pos             = find_start_index(h_aggregated_info, skip_rows);
  size_t cur_row_index       = skip_rows;
  size_t cur_cumulative_size = 0;
  auto const max_row         = min(skip_rows + num_rows, h_aggregated_info.back().end_row_index);
  while (cur_row_index < max_row) {
    auto const split_pos =
      find_next_split(cur_pos, cur_row_index, cur_cumulative_size, h_aggregated_info, size_limit);

    auto const start_row = cur_row_index;
    cur_row_index        = min(max_row, h_aggregated_info[split_pos].end_row_index);
    splits.push_back({start_row, cur_row_index - start_row});
    cur_pos             = split_pos;
    cur_cumulative_size = h_aggregated_info[split_pos].size_bytes;
  }
  // print_cumulative_row_info(h_aggregated_info, "adjusted w/splits", splits);

  return splits;
}

/**
 * @brief Decompresses a set of pages contained in the set of chunks.
 *
 * This function handles the case where `pages` is only a subset of all available
 * pages in `chunks`.
 *
 * @param chunks List of column chunk descriptors
 * @param pages List of page information
 * @param dict_pages If true, decompress dictionary pages only. Otherwise decompress non-dictionary
 * pages only.
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Device buffer to decompressed page data
 */
[[nodiscard]] rmm::device_buffer decompress_page_data(
  cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
  cudf::detail::hostdevice_span<PageInfo> pages,
  bool dict_pages,
  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  auto for_each_codec_page = [&](Compression codec, std::function<void(size_t)> const& f) {
    for (size_t p = 0; p < pages.size(); p++) {
      if (chunks[pages[p].chunk_idx].codec == codec &&
          ((dict_pages && (pages[p].flags & PAGEINFO_FLAGS_DICTIONARY)) ||
           (!dict_pages && !(pages[p].flags & PAGEINFO_FLAGS_DICTIONARY)))) {
        f(p);
      }
    }
  };

  // Brotli scratch memory for decompressing
  rmm::device_buffer debrotli_scratch;

  // Count the exact number of compressed pages
  size_t num_comp_pages    = 0;
  size_t total_decomp_size = 0;

  struct codec_stats {
    Compression compression_type  = UNCOMPRESSED;
    size_t num_pages              = 0;
    int32_t max_decompressed_size = 0;
    size_t total_decomp_size      = 0;
  };

  std::array codecs{codec_stats{GZIP},
                    codec_stats{SNAPPY},
                    codec_stats{BROTLI},
                    codec_stats{ZSTD},
                    codec_stats{LZ4_RAW}};

  auto is_codec_supported = [&codecs](int8_t codec) {
    if (codec == UNCOMPRESSED) return true;
    return std::find_if(codecs.begin(), codecs.end(), [codec](auto& cstats) {
             return codec == cstats.compression_type;
           }) != codecs.end();
  };
  CUDF_EXPECTS(std::all_of(chunks.host_begin(),
                           chunks.host_end(),
                           [&is_codec_supported](auto const& chunk) {
                             return is_codec_supported(chunk.codec);
                           }),
               "Unsupported compression type");

  for (auto& codec : codecs) {
    for_each_codec_page(codec.compression_type, [&](size_t page) {
      auto page_uncomp_size = pages[page].uncompressed_page_size;
      total_decomp_size += page_uncomp_size;
      codec.total_decomp_size += page_uncomp_size;
      codec.max_decompressed_size = std::max(codec.max_decompressed_size, page_uncomp_size);
      codec.num_pages++;
      num_comp_pages++;
    });
    if (codec.compression_type == BROTLI && codec.num_pages > 0) {
      debrotli_scratch.resize(get_gpu_debrotli_scratch_size(codec.num_pages), stream);
    }
  }

  // Dispatch batches of pages to decompress for each codec.
  // Buffer needs to be padded, required by `gpuDecodePageData`.
  rmm::device_buffer decomp_pages(
    cudf::util::round_up_safe(total_decomp_size, BUFFER_PADDING_MULTIPLE), stream);

  std::vector<device_span<uint8_t const>> comp_in;
  comp_in.reserve(num_comp_pages);
  std::vector<device_span<uint8_t>> comp_out;
  comp_out.reserve(num_comp_pages);

  // vectors to save v2 def and rep level data, if any
  std::vector<device_span<uint8_t const>> copy_in;
  copy_in.reserve(num_comp_pages);
  std::vector<device_span<uint8_t>> copy_out;
  copy_out.reserve(num_comp_pages);

  rmm::device_uvector<compression_result> comp_res(num_comp_pages, stream);
  thrust::fill(rmm::exec_policy_nosync(stream),
               comp_res.begin(),
               comp_res.end(),
               compression_result{0, compression_status::FAILURE});

  size_t decomp_offset = 0;
  int32_t start_pos    = 0;
  for (auto const& codec : codecs) {
    if (codec.num_pages == 0) { continue; }

    for_each_codec_page(codec.compression_type, [&](size_t page_idx) {
      auto const dst_base = static_cast<uint8_t*>(decomp_pages.data()) + decomp_offset;
      auto& page          = pages[page_idx];
      // offset will only be non-zero for V2 pages
      auto const offset =
        page.lvl_bytes[level_type::DEFINITION] + page.lvl_bytes[level_type::REPETITION];
      // for V2 need to copy def and rep level info into place, and then offset the
      // input and output buffers. otherwise we'd have to keep both the compressed
      // and decompressed data.
      if (offset != 0) {
        copy_in.emplace_back(page.page_data, offset);
        copy_out.emplace_back(dst_base, offset);
      }
      comp_in.emplace_back(page.page_data + offset,
                           static_cast<size_t>(page.compressed_page_size - offset));
      comp_out.emplace_back(dst_base + offset,
                            static_cast<size_t>(page.uncompressed_page_size - offset));
      page.page_data = dst_base;
      decomp_offset += page.uncompressed_page_size;
    });

    host_span<device_span<uint8_t const> const> comp_in_view{comp_in.data() + start_pos,
                                                             codec.num_pages};
    auto const d_comp_in = cudf::detail::make_device_uvector_async(
      comp_in_view, stream, rmm::mr::get_current_device_resource());
    host_span<device_span<uint8_t> const> comp_out_view(comp_out.data() + start_pos,
                                                        codec.num_pages);
    auto const d_comp_out = cudf::detail::make_device_uvector_async(
      comp_out_view, stream, rmm::mr::get_current_device_resource());
    device_span<compression_result> d_comp_res_view(comp_res.data() + start_pos, codec.num_pages);

    switch (codec.compression_type) {
      case GZIP:
        gpuinflate(d_comp_in, d_comp_out, d_comp_res_view, gzip_header_included::YES, stream);
        break;
      case SNAPPY:
        if (cudf::io::detail::nvcomp_integration::is_stable_enabled()) {
          nvcomp::batched_decompress(nvcomp::compression_type::SNAPPY,
                                     d_comp_in,
                                     d_comp_out,
                                     d_comp_res_view,
                                     codec.max_decompressed_size,
                                     codec.total_decomp_size,
                                     stream);
        } else {
          gpu_unsnap(d_comp_in, d_comp_out, d_comp_res_view, stream);
        }
        break;
      case ZSTD:
        nvcomp::batched_decompress(nvcomp::compression_type::ZSTD,
                                   d_comp_in,
                                   d_comp_out,
                                   d_comp_res_view,
                                   codec.max_decompressed_size,
                                   codec.total_decomp_size,
                                   stream);
        break;
      case BROTLI:
        gpu_debrotli(d_comp_in,
                     d_comp_out,
                     d_comp_res_view,
                     debrotli_scratch.data(),
                     debrotli_scratch.size(),
                     stream);
        break;
      case LZ4_RAW:
        nvcomp::batched_decompress(nvcomp::compression_type::LZ4,
                                   d_comp_in,
                                   d_comp_out,
                                   d_comp_res_view,
                                   codec.max_decompressed_size,
                                   codec.total_decomp_size,
                                   stream);
        break;
      default: CUDF_FAIL("Unexpected decompression dispatch"); break;
    }
    start_pos += codec.num_pages;
  }

  CUDF_EXPECTS(thrust::all_of(rmm::exec_policy(stream),
                              comp_res.begin(),
                              comp_res.end(),
                              cuda::proclaim_return_type<bool>([] __device__(auto const& res) {
                                return res.status == compression_status::SUCCESS;
                              })),
               "Error during decompression");

  // now copy the uncompressed V2 def and rep level data
  if (not copy_in.empty()) {
    auto const d_copy_in = cudf::detail::make_device_uvector_async(
      copy_in, stream, rmm::mr::get_current_device_resource());
    auto const d_copy_out = cudf::detail::make_device_uvector_async(
      copy_out, stream, rmm::mr::get_current_device_resource());

    gpu_copy_uncompressed_blocks(d_copy_in, d_copy_out, stream);
    stream.synchronize();
  }

  pages.host_to_device_async(stream);

  stream.synchronize();
  return decomp_pages;
}

struct flat_column_num_rows {
  ColumnChunkDesc const* chunks;

  __device__ size_type operator()(PageInfo const& page) const
  {
    // ignore dictionary pages and pages belonging to any column containing repetition (lists)
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) ||
        (chunks[page.chunk_idx].max_level[level_type::REPETITION] > 0)) {
      return 0;
    }
    return page.num_rows;
  }
};

struct row_counts_nonzero {
  __device__ bool operator()(size_type count) const { return count > 0; }
};

struct row_counts_different {
  size_type const expected;
  __device__ bool operator()(size_type count) const { return (count != 0) && (count != expected); }
};

/**
 * @brief Detect malformed parquet input data.
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
                            rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  // sum row counts for all non-dictionary, non-list columns. other columns will be indicated as 0
  rmm::device_uvector<size_type> row_counts(pages.size(),
                                            stream);  // worst case:  num keys == num pages
  auto const size_iter =
    thrust::make_transform_iterator(pages.begin(), flat_column_num_rows{chunks.data()});
  auto const row_counts_begin = row_counts.begin();
  auto page_keys              = make_page_key_iterator(pages);
  auto const row_counts_end   = thrust::reduce_by_key(rmm::exec_policy(stream),
                                                    page_keys,
                                                    page_keys + pages.size(),
                                                    size_iter,
                                                    thrust::make_discard_iterator(),
                                                    row_counts_begin)
                                .second;

  // make sure all non-zero row counts are the same
  rmm::device_uvector<size_type> compacted_row_counts(pages.size(), stream);
  auto const compacted_row_counts_begin = compacted_row_counts.begin();
  auto const compacted_row_counts_end   = thrust::copy_if(rmm::exec_policy(stream),
                                                        row_counts_begin,
                                                        row_counts_end,
                                                        compacted_row_counts_begin,
                                                        row_counts_nonzero{});
  if (compacted_row_counts_end != compacted_row_counts_begin) {
    size_t const found_row_count = static_cast<size_t>(compacted_row_counts.element(0, stream));

    // if we somehow don't match the expected row count from the row groups themselves
    if (expected_row_count.has_value()) {
      CUDF_EXPECTS(expected_row_count.value() == found_row_count,
                   "Encountered malformed parquet page data (unexpected row count in page data)");
    }

    // all non-zero row counts must be the same
    auto const chk =
      thrust::count_if(rmm::exec_policy(stream),
                       compacted_row_counts_begin,
                       compacted_row_counts_end,
                       row_counts_different{static_cast<size_type>(found_row_count)});
    CUDF_EXPECTS(chk == 0,
                 "Encountered malformed parquet page data (row count mismatch in page data)");
  }
}

struct decompression_info {
  Compression codec;
  size_t num_pages;
  size_t max_page_decompressed_size;
  size_t total_decompressed_size;
};

/**
 * @brief Functor which retrieves per-page decompression information.
 *
 */
struct get_decomp_info {
  device_span<const ColumnChunkDesc> chunks;

  __device__ decompression_info operator()(PageInfo const& p) const
  {
    return {static_cast<Compression>(chunks[p.chunk_idx].codec),
            1,
            static_cast<size_t>(p.uncompressed_page_size),
            static_cast<size_t>(p.uncompressed_page_size)};
  }
};

/**
 * @brief Functor which accumulates per-page decompression information.
 *
 */
struct decomp_sum {
  __device__ decompression_info operator()(decompression_info const& a,
                                           decompression_info const& b) const
  {
    return {a.codec,
            a.num_pages + b.num_pages,
            std::max(a.max_page_decompressed_size, b.max_page_decompressed_size),
            a.total_decompressed_size + b.total_decompressed_size};
  }
};

/**
 * @brief Functor which returns total scratch space required based on computed decompression_info
 * data.
 *
 */
struct get_decomp_scratch {
  size_t operator()(decompression_info const& di) const
  {
    switch (di.codec) {
      case UNCOMPRESSED:
      case GZIP: return 0;

      case BROTLI: return get_gpu_debrotli_scratch_size(di.num_pages);

      case SNAPPY:
        if (cudf::io::detail::nvcomp_integration::is_stable_enabled()) {
          return cudf::io::nvcomp::batched_decompress_temp_size(
            cudf::io::nvcomp::compression_type::SNAPPY,
            di.num_pages,
            di.max_page_decompressed_size,
            di.total_decompressed_size);
        } else {
          return 0;
        }
        break;

      case ZSTD:
        return cudf::io::nvcomp::batched_decompress_temp_size(
          cudf::io::nvcomp::compression_type::ZSTD,
          di.num_pages,
          di.max_page_decompressed_size,
          di.total_decompressed_size);
      case LZ4_RAW:
        return cudf::io::nvcomp::batched_decompress_temp_size(
          cudf::io::nvcomp::compression_type::LZ4,
          di.num_pages,
          di.max_page_decompressed_size,
          di.total_decompressed_size);

      default: CUDF_FAIL("Invalid compression codec for parquet decompression");
    }
  }
};

/**
 * @brief Add the cost of decompression codec scratch space to the per-page cumulative
 * size information.
 *
 */
void include_decompression_scratch_size(device_span<ColumnChunkDesc const> chunks,
                                        device_span<PageInfo const> pages,
                                        device_span<cumulative_page_info> c_info,
                                        rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() == c_info.size(),
               "Encountered page/cumulative_page_info size mismatch");

  auto page_keys = make_page_key_iterator(pages);

  // per-codec page counts and decompression sizes
  rmm::device_uvector<decompression_info> decomp_info(pages.size(), stream);
  auto decomp_iter = thrust::make_transform_iterator(pages.begin(), get_decomp_info{chunks});
  thrust::inclusive_scan_by_key(rmm::exec_policy_nosync(stream),
                                page_keys,
                                page_keys + pages.size(),
                                decomp_iter,
                                decomp_info.begin(),
                                thrust::equal_to<int32_t>{},
                                decomp_sum{});

  // retrieve to host so we can call nvcomp to get compression scratch sizes
  std::vector<decompression_info> h_decomp_info =
    cudf::detail::make_std_vector_sync(decomp_info, stream);
  std::vector<size_t> temp_cost(pages.size());
  thrust::transform(thrust::host,
                    h_decomp_info.begin(),
                    h_decomp_info.end(),
                    temp_cost.begin(),
                    get_decomp_scratch{});

  // add to the cumulative_page_info data
  rmm::device_uvector<size_t> d_temp_cost = cudf::detail::make_device_uvector_async(
    temp_cost, stream, rmm::mr::get_current_device_resource());
  auto iter = thrust::make_counting_iterator(size_t{0});
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   iter,
                   iter + pages.size(),
                   [temp_cost = d_temp_cost.begin(), c_info = c_info.begin()] __device__(size_t i) {
                     c_info[i].size_bytes += temp_cost[i];
                   });
  stream.synchronize();
}

}  // anonymous namespace

void reader::impl::handle_chunking(bool uses_custom_row_bounds)
{
  // if this is our first time in here, setup the first pass.
  if (!_pass_itm_data) {
    // setup the next pass
    setup_next_pass(uses_custom_row_bounds);
  }

  auto& pass = *_pass_itm_data;

  // if we already have a subpass in flight.
  if (pass.subpass != nullptr) {
    // if it still has more chunks in flight, there's nothing more to do
    if (pass.subpass->current_output_chunk < pass.subpass->output_chunk_read_info.size()) {
      return;
    }

    // increment rows processed
    pass.processed_rows += pass.subpass->num_rows;

    // release the old subpass (will free memory)
    pass.subpass.reset();

    // otherwise we are done with the pass entirely
    if (pass.processed_rows == pass.num_rows) {
      // release the old pass
      _pass_itm_data.reset();

      _file_itm_data._current_input_pass++;
      // no more passes. we are absolutely done with this file.
      if (_file_itm_data._current_input_pass == _file_itm_data.num_passes()) { return; }

      // setup the next pass
      setup_next_pass(uses_custom_row_bounds);
    }
  }

  // setup the next sub pass
  setup_next_subpass(uses_custom_row_bounds);
}

void reader::impl::setup_next_pass(bool uses_custom_row_bounds)
{
  auto const num_passes = _file_itm_data.num_passes();

  // always create the pass struct, even if we end up with no work.
  // this will also cause the previous pass information to be deleted
  _pass_itm_data = std::make_unique<pass_intermediate_data>();

  if (_file_itm_data.global_num_rows > 0 && not _file_itm_data.row_groups.empty() &&
      not _input_columns.empty() && _file_itm_data._current_input_pass < num_passes) {
    auto& pass = *_pass_itm_data;

    // setup row groups to be loaded for this pass
    auto const row_group_start =
      _file_itm_data.input_pass_row_group_offsets[_file_itm_data._current_input_pass];
    auto const row_group_end =
      _file_itm_data.input_pass_row_group_offsets[_file_itm_data._current_input_pass + 1];
    auto const num_row_groups = row_group_end - row_group_start;
    pass.row_groups.resize(num_row_groups);
    std::copy(_file_itm_data.row_groups.begin() + row_group_start,
              _file_itm_data.row_groups.begin() + row_group_end,
              pass.row_groups.begin());

    CUDF_EXPECTS(_file_itm_data._current_input_pass < num_passes,
                 "Encountered an invalid read pass index");

    auto const chunks_per_rowgroup = _input_columns.size();
    auto const num_chunks          = chunks_per_rowgroup * num_row_groups;

    auto chunk_start = _file_itm_data.chunks.begin() + (row_group_start * chunks_per_rowgroup);
    auto chunk_end   = _file_itm_data.chunks.begin() + (row_group_end * chunks_per_rowgroup);

    pass.chunks = cudf::detail::hostdevice_vector<ColumnChunkDesc>(num_chunks, _stream);
    std::copy(chunk_start, chunk_end, pass.chunks.begin());

    // compute skip_rows / num_rows for this pass.
    if (num_passes == 1) {
      pass.skip_rows = _file_itm_data.global_skip_rows;
      pass.num_rows  = _file_itm_data.global_num_rows;
    } else {
      auto const global_start_row = _file_itm_data.global_skip_rows;
      auto const global_end_row   = global_start_row + _file_itm_data.global_num_rows;
      auto const start_row =
        std::max(_file_itm_data.input_pass_start_row_count[_file_itm_data._current_input_pass],
                 global_start_row);
      auto const end_row =
        std::min(_file_itm_data.input_pass_start_row_count[_file_itm_data._current_input_pass + 1],
                 global_end_row);

      // skip_rows is always global in the sense that it is relative to the first row of
      // everything we will be reading, regardless of what pass we are on.
      // num_rows is how many rows we are reading this pass.
      pass.skip_rows =
        global_start_row +
        _file_itm_data.input_pass_start_row_count[_file_itm_data._current_input_pass];
      pass.num_rows = end_row - start_row;
    }

    // load page information for the chunk. this retrieves the compressed bytes for all the
    // pages, and their headers (which we can access without decompressing)
    read_compressed_data();

    // detect malformed columns.
    // - we have seen some cases in the wild where we have a row group containing N
    //   rows, but the total number of rows in the pages for column X is != N. while it
    //   is possible to load this by just capping the number of rows read, we cannot tell
    //   which rows are invalid so we may be returning bad data. in addition, this mismatch
    //   confuses the chunked reader
    detect_malformed_pages(
      pass.pages,
      pass.chunks,
      uses_custom_row_bounds ? std::nullopt : std::make_optional(pass.num_rows),
      _stream);

    // decompress dictionary data if applicable.
    if (pass.has_compressed_data) {
      pass.decomp_dict_data = decompress_page_data(pass.chunks, pass.pages, true, _stream);
    }

    // store off how much memory we've used so far. This includes the compressed page data and the
    // decompressed dictionary data. we will subtract this from the available total memory for the
    // subpasses
    auto chunk_iter =
      thrust::make_transform_iterator(pass.chunks.d_begin(), get_chunk_compressed_size{});
    pass.base_mem_size =
      pass.decomp_dict_data.size() +
      thrust::reduce(rmm::exec_policy(_stream), chunk_iter, chunk_iter + pass.chunks.size());

    // since there is only ever 1 dictionary per chunk (the first page), do it at the
    // pass level.
    build_string_dict_indices();

    // if we are doing subpass reading, generate more accurate num_row estimates for list columns.
    // this helps us to generate more accurate subpass splits.
    if (pass.has_compressed_data && _input_pass_read_limit != 0) {
      generate_list_column_row_count_estimates();
    }

#if defined(PARQUET_CHUNK_LOGGING)
    printf("Pass: row_groups(%'lu), chunks(%'lu), pages(%'lu)\n",
           pass.row_groups.size(),
           pass.chunks.size(),
           pass.pages.size());
    printf("\tskip_rows: %'lu\n", pass.skip_rows);
    printf("\tnum_rows: %'lu\n", pass.num_rows);
    printf("\tbase mem usage: %'lu\n", pass.base_mem_size);
    auto const num_columns = _input_columns.size();
    for (size_t c_idx = 0; c_idx < num_columns; c_idx++) {
      printf("\t\tColumn %'lu: num_pages(%'d)\n",
             c_idx,
             pass.page_offsets[c_idx + 1] - pass.page_offsets[c_idx]);
    }
#endif

    _stream.synchronize();
  }
}

void reader::impl::setup_next_subpass(bool uses_custom_row_bounds)
{
  auto& pass    = *_pass_itm_data;
  pass.subpass  = std::make_unique<subpass_intermediate_data>();
  auto& subpass = *pass.subpass;

  auto const num_columns = _input_columns.size();

  // if the user has passed a very small value (under the hardcoded minimum_subpass_expected_size),
  // respect it.
  auto const min_subpass_size = std::min(_input_pass_read_limit, minimum_subpass_expected_size);

  // what do we do if the base memory size (the compressed data) itself is approaching or larger
  // than the overall read limit? we are still going to be decompressing in subpasses, but we have
  // to assume some reasonable minimum size needed to safely decompress a single subpass. so always
  // reserve at least that much space. this can result in using up to 2x the specified user limit
  // but should only ever happen with unrealistically low numbers.
  size_t const remaining_read_limit =
    _input_pass_read_limit == 0 ? 0
    : pass.base_mem_size + min_subpass_size >= _input_pass_read_limit
      ? min_subpass_size
      : _input_pass_read_limit - pass.base_mem_size;

  // page_indices is an array of spans where each element N is the
  // indices into the pass.pages array that represents the subset of pages
  // for column N to use for the subpass.
  auto [page_indices, total_pages, total_expected_size] =
    [&]() -> std::tuple<rmm::device_uvector<page_span>, size_t, size_t> {
    if (!pass.has_compressed_data || _input_pass_read_limit == 0) {
      rmm::device_uvector<page_span> page_indices(
        num_columns, _stream, rmm::mr::get_current_device_resource());
      auto iter = thrust::make_counting_iterator(0);
      thrust::transform(rmm::exec_policy_nosync(_stream),
                        iter,
                        iter + num_columns,
                        page_indices.begin(),
                        get_page_span_by_column{pass.page_offsets});
      return {std::move(page_indices), pass.pages.size(), size_t{0}};
    }
    // otherwise we have to look forward and choose a batch of pages

    // as subpasses get decoded, the initial estimates we have for list row counts
    // get updated with accurate data, so regenerate cumulative size info and row
    // indices
    rmm::device_uvector<cumulative_page_info> c_info(pass.pages.size(), _stream);
    auto page_keys = make_page_key_iterator(pass.pages);
    auto page_size = thrust::make_transform_iterator(pass.pages.d_begin(), get_page_input_size{});
    thrust::inclusive_scan_by_key(rmm::exec_policy_nosync(_stream),
                                  page_keys,
                                  page_keys + pass.pages.size(),
                                  page_size,
                                  c_info.begin(),
                                  thrust::equal_to{},
                                  cumulative_page_sum{});

    // include scratch space needed for decompression. for certain codecs (eg ZSTD) this
    // can be considerable.
    include_decompression_scratch_size(pass.chunks, pass.pages, c_info, _stream);

    auto iter = thrust::make_counting_iterator(0);
    thrust::for_each(rmm::exec_policy_nosync(_stream),
                     iter,
                     iter + pass.pages.size(),
                     set_row_index{pass.chunks, pass.pages, c_info, 0});
    // print_cumulative_page_info(pass.pages, pass.chunks, c_info, _stream);

    // get the next batch of pages
    return compute_next_subpass(c_info,
                                pass.pages,
                                pass.chunks,
                                pass.page_offsets,
                                pass.processed_rows + pass.skip_rows,
                                remaining_read_limit,
                                num_columns,
                                _stream);
  }();

  // check to see if we are processing the entire pass (enabling us to skip a bunch of work)
  subpass.single_subpass = total_pages == pass.pages.size();

  // in the single pass case, no page copying is necessary - just use what's in the pass itself
  if (subpass.single_subpass) {
    subpass.pages = pass.pages;
  }
  // copy the appropriate subset of pages from each column and store the mapping back to the source
  // (pass) pages
  else {
    subpass.page_buf = cudf::detail::hostdevice_vector<PageInfo>(total_pages, total_pages, _stream);
    subpass.page_src_index = rmm::device_uvector<size_t>(total_pages, _stream);
    auto iter              = thrust::make_counting_iterator(0);
    rmm::device_uvector<size_t> dst_offsets(num_columns + 1, _stream);
    thrust::transform_exclusive_scan(rmm::exec_policy_nosync(_stream),
                                     iter,
                                     iter + num_columns + 1,
                                     dst_offsets.begin(),
                                     get_span_size_by_index{page_indices},
                                     0,
                                     thrust::plus<size_t>{});
    thrust::for_each(
      rmm::exec_policy_nosync(_stream),
      iter,
      iter + total_pages,
      copy_subpass_page{
        pass.pages, subpass.page_buf, subpass.page_src_index, dst_offsets, page_indices});
    subpass.pages = subpass.page_buf;
  }

  std::vector<page_span> h_spans = cudf::detail::make_std_vector_async(page_indices, _stream);
  subpass.pages.device_to_host_async(_stream);

  _stream.synchronize();

  subpass.column_page_count = std::vector<size_t>(num_columns);
  std::transform(
    h_spans.begin(), h_spans.end(), subpass.column_page_count.begin(), get_span_size{});

  // decompress the data for the pages in this subpass.
  if (pass.has_compressed_data) {
    subpass.decomp_page_data = decompress_page_data(pass.chunks, subpass.pages, false, _stream);
  }

  // buffers needed by the decode kernels
  {
    // nesting information (sizes, etc) stored -per page-
    // note : even for flat schemas, we allocate 1 level of "nesting" info
    allocate_nesting_info();

    // level decode space
    allocate_level_decode_space();
  }
  subpass.pages.host_to_device_async(_stream);

  // preprocess pages (computes row counts for lists, computes output chunks and computes
  // the actual row counts we will be able load out of this subpass)
  preprocess_subpass_pages(uses_custom_row_bounds, _output_chunk_read_limit);

#if defined(PARQUET_CHUNK_LOGGING)
  printf("\tSubpass: skip_rows(%'lu), num_rows(%'lu), remaining read limit(%'lu)\n",
         subpass.skip_rows,
         subpass.num_rows,
         remaining_read_limit);
  printf("\t\tDecompressed size: %'lu\n", subpass.decomp_page_data.size());
  printf("\t\tTotal expected usage: %'lu\n",
         total_expected_size == 0 ? subpass.decomp_page_data.size() + pass.base_mem_size
                                  : total_expected_size + pass.base_mem_size);
  for (size_t c_idx = 0; c_idx < num_columns; c_idx++) {
    printf("\t\tColumn %'lu: pages(%'lu - %'lu)\n",
           c_idx,
           page_indices[c_idx].start,
           page_indices[c_idx].end);
  }
  printf("\t\tOutput chunks:\n");
  for (size_t idx = 0; idx < subpass.output_chunk_read_info.size(); idx++) {
    printf("\t\t\t%'lu: skip_rows(%'lu) num_rows(%'lu)\n",
           idx,
           subpass.output_chunk_read_info[idx].skip_rows,
           subpass.output_chunk_read_info[idx].num_rows);
  }
#endif
}

void reader::impl::create_global_chunk_info()
{
  auto const num_rows         = _file_itm_data.global_num_rows;
  auto const& row_groups_info = _file_itm_data.row_groups;
  auto& chunks                = _file_itm_data.chunks;

  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = _input_columns.size();
  auto const num_chunks        = row_groups_info.size() * num_input_columns;

  // Mapping of input column to page index column
  std::vector<size_type> column_mapping;

  if (_has_page_index and not row_groups_info.empty()) {
    // use first row group to define mappings (assumes same schema for each file)
    auto const& rg      = row_groups_info[0];
    auto const& columns = _metadata->get_row_group(rg.index, rg.source_index).columns;
    column_mapping.resize(num_input_columns);
    std::transform(
      _input_columns.begin(), _input_columns.end(), column_mapping.begin(), [&](auto const& col) {
        // translate schema_idx into something we can use for the page indexes
        if (auto it = std::find_if(
              columns.begin(),
              columns.end(),
              [&col](auto const& col_chunk) { return col_chunk.schema_idx == col.schema_idx; });
            it != columns.end()) {
          return std::distance(columns.begin(), it);
        }
        CUDF_FAIL("cannot find column mapping");
      });
  }

  // Initialize column chunk information
  auto remaining_rows = num_rows;
  for (auto const& rg : row_groups_info) {
    auto const& row_group      = _metadata->get_row_group(rg.index, rg.source_index);
    auto const row_group_start = rg.start_row;
    auto const row_group_rows  = std::min<int>(remaining_rows, row_group.num_rows);

    // generate ColumnChunkDesc objects for everything to be decoded (all input columns)
    for (size_t i = 0; i < num_input_columns; ++i) {
      auto col = _input_columns[i];
      // look up metadata
      auto& col_meta = _metadata->get_column_metadata(rg.index, rg.source_index, col.schema_idx);
      auto& schema   = _metadata->get_schema(col.schema_idx);

      auto [type_width, clock_rate, converted_type] =
        conversion_info(to_type_id(schema, _strings_to_categorical, _timestamp_type.id()),
                        _timestamp_type.id(),
                        schema.type,
                        schema.converted_type,
                        schema.type_length);

      // for lists, estimate the number of bytes per row. this is used by the subpass reader to
      // determine where to split the decompression boundaries
      float const list_bytes_per_row_est =
        schema.max_repetition_level > 0 && row_group.num_rows > 0
          ? static_cast<float>(col_meta.total_uncompressed_size) /
              static_cast<float>(row_group.num_rows)
          : 0.0f;

      // grab the column_chunk_info for each chunk (if it exists)
      column_chunk_info const* const chunk_info =
        _has_page_index ? &rg.column_chunks.value()[column_mapping[i]] : nullptr;

      chunks.push_back(ColumnChunkDesc(col_meta.total_compressed_size,
                                       nullptr,
                                       col_meta.num_values,
                                       schema.type,
                                       type_width,
                                       row_group_start,
                                       row_group_rows,
                                       schema.max_definition_level,
                                       schema.max_repetition_level,
                                       _metadata->get_output_nesting_depth(col.schema_idx),
                                       required_bits(schema.max_definition_level),
                                       required_bits(schema.max_repetition_level),
                                       col_meta.codec,
                                       converted_type,
                                       schema.logical_type,
                                       schema.decimal_precision,
                                       clock_rate,
                                       i,
                                       col.schema_idx,
                                       chunk_info,
                                       list_bytes_per_row_est));
    }

    remaining_rows -= row_group_rows;
  }
}

void reader::impl::compute_input_passes()
{
  // at this point, row_groups has already been filtered down to just the row groups we need to
  // handle optional skip_rows/num_rows parameters.
  auto const& row_groups_info = _file_itm_data.row_groups;

  // if the user hasn't specified an input size limit, read everything in a single pass.
  if (_input_pass_read_limit == 0) {
    _file_itm_data.input_pass_row_group_offsets.push_back(0);
    _file_itm_data.input_pass_row_group_offsets.push_back(row_groups_info.size());
    _file_itm_data.input_pass_start_row_count.push_back(0);
    auto rg_row_count = cudf::detail::make_counting_transform_iterator(0, [&](size_t i) {
      auto const& rgi       = row_groups_info[i];
      auto const& row_group = _metadata->get_row_group(rgi.index, rgi.source_index);
      return row_group.num_rows;
    });
    _file_itm_data.input_pass_start_row_count.push_back(
      std::reduce(rg_row_count, rg_row_count + row_groups_info.size()));
    return;
  }

  // generate passes. make sure to account for the case where a single row group doesn't fit within
  //
  std::size_t const comp_read_limit =
    _input_pass_read_limit > 0
      ? static_cast<size_t>(_input_pass_read_limit * input_limit_compression_reserve)
      : std::numeric_limits<std::size_t>::max();
  std::size_t cur_pass_byte_size = 0;
  std::size_t cur_rg_start       = 0;
  std::size_t cur_row_count      = 0;
  _file_itm_data.input_pass_row_group_offsets.push_back(0);
  _file_itm_data.input_pass_start_row_count.push_back(0);

  for (size_t cur_rg_index = 0; cur_rg_index < row_groups_info.size(); cur_rg_index++) {
    auto const& rgi       = row_groups_info[cur_rg_index];
    auto const& row_group = _metadata->get_row_group(rgi.index, rgi.source_index);

    // total compressed size and total size (compressed + uncompressed) for
    auto const [compressed_rg_size, _ /*compressed + uncompressed*/] =
      get_row_group_size(row_group);

    // can we add this row group
    if (cur_pass_byte_size + compressed_rg_size >= comp_read_limit) {
      // A single row group (the current one) is larger than the read limit:
      // We always need to include at least one row group, so end the pass at the end of the current
      // row group
      if (cur_rg_start == cur_rg_index) {
        _file_itm_data.input_pass_row_group_offsets.push_back(cur_rg_index + 1);
        _file_itm_data.input_pass_start_row_count.push_back(cur_row_count + row_group.num_rows);
        cur_rg_start       = cur_rg_index + 1;
        cur_pass_byte_size = 0;
      }
      // End the pass at the end of the previous row group
      else {
        _file_itm_data.input_pass_row_group_offsets.push_back(cur_rg_index);
        _file_itm_data.input_pass_start_row_count.push_back(cur_row_count);
        cur_rg_start       = cur_rg_index;
        cur_pass_byte_size = compressed_rg_size;
      }
    } else {
      cur_pass_byte_size += compressed_rg_size;
    }
    cur_row_count += row_group.num_rows;
  }

  // add the last pass if necessary
  if (_file_itm_data.input_pass_row_group_offsets.back() != row_groups_info.size()) {
    _file_itm_data.input_pass_row_group_offsets.push_back(row_groups_info.size());
    _file_itm_data.input_pass_start_row_count.push_back(cur_row_count);
  }
}

void reader::impl::compute_output_chunks_for_subpass()
{
  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  // simple case : no chunk size, no splits
  if (_output_chunk_read_limit <= 0) {
    subpass.output_chunk_read_info.push_back({subpass.skip_rows, subpass.num_rows});
    return;
  }

  // generate row_indices and cumulative output sizes for all pages
  rmm::device_uvector<cumulative_page_info> c_info(subpass.pages.size(), _stream);
  auto page_input =
    thrust::make_transform_iterator(subpass.pages.device_begin(), get_page_output_size{});
  auto page_keys = make_page_key_iterator(subpass.pages);
  thrust::inclusive_scan_by_key(rmm::exec_policy_nosync(_stream),
                                page_keys,
                                page_keys + subpass.pages.size(),
                                page_input,
                                c_info.begin(),
                                thrust::equal_to{},
                                cumulative_page_sum{});
  auto iter = thrust::make_counting_iterator(0);
  // cap the max row in all pages by the max row we expect in the subpass. input chunking
  // can cause "dangling" row counts where for example, only 1 column has a page whose
  // maximum row is beyond our expected subpass max row, which will cause an out of
  // bounds index in compute_page_splits_by_row.
  auto const subpass_max_row = subpass.skip_rows + subpass.num_rows;
  thrust::for_each(rmm::exec_policy_nosync(_stream),
                   iter,
                   iter + subpass.pages.size(),
                   set_row_index{pass.chunks, subpass.pages, c_info, subpass_max_row});
  // print_cumulative_page_info(subpass.pages, c_info, _stream);

  // compute the splits
  subpass.output_chunk_read_info = compute_page_splits_by_row(
    c_info, subpass.pages, subpass.skip_rows, subpass.num_rows, _output_chunk_read_limit, _stream);
}

}  // namespace cudf::io::parquet::detail
