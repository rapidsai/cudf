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

#include "hybrid_scan_helpers.hpp"
#include "io/parquet/reader_impl_helpers.hpp"
#include "io/parquet/stats_filter_helpers.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/types.hpp>
#include <cudf/logger.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/unique.h>

#include <algorithm>
#include <limits>
#include <numeric>

namespace cudf::io::parquet::experimental::detail {

using metadata_base     = parquet::detail::metadata;
using stats_caster_base = parquet::detail::stats_caster_base;
using string_index_pair = parquet::detail::string_index_pair;

namespace {

/**
 * @brief Make a device vector where each row contains the index of the page it belongs to
 */
[[nodiscard]] rmm::device_uvector<size_type> make_page_indices_async(
  cudf::host_span<cudf::size_type const> page_row_counts,
  cudf::host_span<cudf::size_type const> page_row_offsets,
  cudf::size_type total_rows,
  rmm::cuda_stream_view stream)
{
  auto mr = cudf::get_current_device_resource_ref();

  // Copy page-level row counts and offsets to device
  auto row_counts  = cudf::detail::make_device_uvector_async(page_row_counts, stream, mr);
  auto row_offsets = cudf::detail::make_device_uvector_async(page_row_offsets, stream, mr);

  // Make a zeroed device vector to store page indices of each row
  auto page_indices =
    cudf::detail::make_zeroed_device_uvector_async<cudf::size_type>(total_rows, stream, mr);

  // Scatter page indices across the their first row's index
  thrust::scatter_if(rmm::exec_policy_nosync(stream),
                     thrust::counting_iterator<size_type>(0),
                     thrust::counting_iterator<size_type>(row_counts.size()),
                     row_offsets.begin(),
                     row_counts.begin(),
                     page_indices.begin());

  // Inclusive scan with maximum to replace zeros with the (increasing) page index it belongs to.
  // Page indices are scattered at their first row's index.
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream),
                         page_indices.begin(),
                         page_indices.end(),
                         page_indices.begin(),
                         cuda::maximum<cudf::size_type>());
  return page_indices;
}

/**
 * @brief Compute page row counts and page row offsets and column chunk page (count) offsets for a
 * given column schema index
 */
[[nodiscard]] auto make_page_row_counts_and_offsets(
  cudf::host_span<metadata_base const> per_file_metadata,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  size_type schema_idx,
  rmm::cuda_stream_view stream)
{
  // Compute total number of row groups
  auto const total_row_groups =
    std::accumulate(row_group_indices.begin(),
                    row_group_indices.end(),
                    size_t{0},
                    [](auto sum, auto const& rg_indices) { return sum + rg_indices.size(); });

  // Vector to store how many rows are present in each page - set initial capacity to two data pages
  // per row group
  auto page_row_counts =
    cudf::detail::make_empty_host_vector<size_type>(2 * total_row_groups, stream);
  // Vector to store the cumulative number of rows in each page - - set initial capacity to two data
  // pages per row group
  auto page_row_offsets =
    cudf::detail::make_empty_host_vector<size_type>((2 * total_row_groups) + 1, stream);
  // Vector to store the cumulative number of pages in each column chunk
  auto col_chunk_page_offsets =
    cudf::detail::make_empty_host_vector<size_type>(total_row_groups + 1, stream);

  page_row_offsets.push_back(0);
  col_chunk_page_offsets.push_back(0);

  // For all data sources
  std::for_each(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(row_group_indices.size()),
    [&](auto src_idx) {
      auto const& rg_indices = row_group_indices[src_idx];
      // For all column chunks in this data source
      std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto rg_idx) {
        auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
        // Find the column chunk with the given schema index
        auto colchunk_iter = std::find_if(
          row_group.columns.begin(), row_group.columns.end(), [schema_idx](ColumnChunk const& col) {
            return col.schema_idx == schema_idx;
          });

        CUDF_EXPECTS(colchunk_iter != row_group.columns.end(),
                     "Column chunk with schema index " + std::to_string(schema_idx) +
                       " not found in row group",
                     std::invalid_argument);

        // Compute page row counts and offsets if this column chunk has column and offset indexes
        if (colchunk_iter->offset_index.has_value()) {
          CUDF_EXPECTS(colchunk_iter->column_index.has_value(),
                       "Both offset and column indexes must be present");
          // Get the offset and column indexes of the column chunk
          auto const& offset_index = colchunk_iter->offset_index.value();
          auto const& column_index = colchunk_iter->column_index.value();

          // Number of pages in this column chunk
          auto const row_group_num_pages = offset_index.page_locations.size();

          CUDF_EXPECTS(column_index.min_values.size() == column_index.max_values.size(),
                       "page min and max values should be of same size");
          CUDF_EXPECTS(column_index.min_values.size() == row_group_num_pages,
                       "mismatch between size of min/max page values and the size of page "
                       "locations");
          // Update the cumulative number of pages in this column chunk
          col_chunk_page_offsets.push_back(col_chunk_page_offsets.back() + row_group_num_pages);

          // For all pages in this column chunk, update page row counts and offsets.
          std::for_each(
            thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator(row_group_num_pages),
            [&](auto const page_idx) {
              int64_t const first_row_idx = offset_index.page_locations[page_idx].first_row_index;
              // For the last page, this is simply the total number of rows in the column chunk
              int64_t const last_row_idx =
                (page_idx < row_group_num_pages - 1)
                  ? offset_index.page_locations[page_idx + 1].first_row_index
                  : row_group.num_rows;

              // Update the page row counts and offsets
              page_row_counts.push_back(last_row_idx - first_row_idx);
              page_row_offsets.push_back(page_row_offsets.back() + page_row_counts.back());
            });
        }
      });
    });

  return std::tuple{
    std::move(page_row_counts), std::move(page_row_offsets), std::move(col_chunk_page_offsets)};
}

/**
 * @brief Compute if the page index is present in all parquet data sources for all output columns
 */
[[nodiscard]] bool compute_has_page_index(
  cudf::host_span<metadata_base const> file_metadatas,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<size_type const> output_column_schemas)
{
  // For all output columns, check all parquet data sources
  return std::all_of(
    output_column_schemas.begin(), output_column_schemas.end(), [&](auto const schema_idx) {
      // For all parquet data sources
      return std::all_of(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator(row_group_indices.size()),
        [&](auto const src_index) {
          // For all row groups in this parquet data source
          auto const& rg_indices = row_group_indices[src_index];
          return std::all_of(rg_indices.begin(), rg_indices.end(), [&](auto const& rg_index) {
            auto const& row_group = file_metadatas[src_index].row_groups[rg_index];
            auto col              = std::find_if(
              row_group.columns.begin(),
              row_group.columns.end(),
              [schema_idx](ColumnChunk const& col) { return col.schema_idx == schema_idx; });
            // Check if the offset_index and column_index are present
            return col != file_metadatas[src_index].row_groups[rg_index].columns.end() and
                   col->offset_index.has_value() and col->column_index.has_value();
          });
        });
    });
}
/**
 * @brief Construct a vector of all required data pages from the page row counts
 */
[[nodiscard]] auto all_required_data_pages(
  cudf::host_span<cudf::detail::host_vector<size_type> const> page_row_counts)
{
  std::vector<std::vector<bool>> all_required_data_pages;
  all_required_data_pages.reserve(page_row_counts.size());
  std::transform(
    page_row_counts.begin(),
    page_row_counts.end(),
    std::back_inserter(all_required_data_pages),
    [&](auto const& col_page_counts) { return std::vector<bool>(col_page_counts.size(), true); });

  return all_required_data_pages;
};

/**
 * @brief Converts page-level statistics of a column to 2 device columns - min, max values. Each
 * column has number of rows equal to the total rows in all row groups.
 */
struct page_stats_caster : public stats_caster_base {
  cudf::size_type total_rows;
  cudf::host_span<metadata_base const> per_file_metadata;
  cudf::host_span<std::vector<size_type> const> row_group_indices;

  page_stats_caster(size_type total_rows,
                    cudf::host_span<metadata_base const> per_file_metadata,
                    cudf::host_span<std::vector<size_type> const> row_group_indices)
    : total_rows{total_rows},
      per_file_metadata{per_file_metadata},
      row_group_indices{row_group_indices}
  {
  }

  /**
   * @brief Transforms a page-level stats column to a row-level stats column for non-string types
   *
   * @tparam T The data type of the column - must be non-compound
   * @param column Mutable view of input page-level device column
   * @param page_nullmask Host nullmask of the input page-level column
   * @param page_indices Device vector containing the page index for each row index
   * @param page_row_offsets Host vector row offsets of each page
   * @param dtype The data type of the column
   * @param stream CUDA stream
   * @param mr Device memory resource
   *
   * @return A pair containing the output data buffer and nullmask
   */
  template <typename T>
  [[nodiscard]] std::pair<rmm::device_buffer, rmm::device_buffer> build_data_and_nullmask(
    mutable_column_view input_column,
    bitmask_type const* page_nullmask,
    cudf::device_span<size_type const> page_indices,
    cudf::host_span<size_type const> page_row_offsets,
    cudf::data_type dtype,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
    requires(not cudf::is_compound<T>())
  {
    // Total number of pages in the column
    size_type const total_pages = page_row_offsets.size() - 1;

    // Buffer for output data
    auto output_data = rmm::device_buffer(cudf::size_of(dtype) * total_rows, stream, mr);

    // For each row index, copy over the min/max page stat value from the corresponding page.
    thrust::gather(rmm::exec_policy_nosync(stream),
                   page_indices.begin(),
                   page_indices.end(),
                   input_column.template begin<T>(),
                   reinterpret_cast<T*>(output_data.data()));

    // Buffer for output bitmask. Set all bits valid
    auto output_nullmask = cudf::create_null_mask(total_rows, mask_state::ALL_VALID, stream, mr);

    // For each input page, invalidate the null mask for corresponding rows if needed.
    std::for_each(thrust::counting_iterator(0),
                  thrust::counting_iterator(total_pages),
                  [&](auto const page_idx) {
                    if (not bit_is_set(page_nullmask, page_idx)) {
                      cudf::set_null_mask(static_cast<bitmask_type*>(output_nullmask.data()),
                                          page_row_offsets[page_idx],
                                          page_row_offsets[page_idx + 1],
                                          false,
                                          stream);
                    }
                  });

    return {std::move(output_data), std::move(output_nullmask)};
  }

  /**
   * @brief Transforms a page-level stats column to a row-level stats column for string type
   *
   * @param host_strings Host span of cudf::string_view values in the input page-level host column
   * @param host_chars Host span of string data of the input page-level host column
   * @param host_nullmask Nullmask of the input page-level host column
   * @param page_indices Device vector containing the page index for each row index
   * @param page_row_offsets Host vector row offsets of each page
   * @param stream CUDA stream
   * @param mr Device memory resource
   *
   * @return A pair containing the output data buffer and nullmask
   */
  [[nodiscard]] std::
    tuple<rmm::device_buffer, rmm::device_uvector<cudf::size_type>, rmm::device_buffer>
    build_string_data_and_nullmask(cudf::host_span<cudf::string_view const> host_strings,
                                   cudf::host_span<char const> host_chars,
                                   bitmask_type const* host_page_nullmask,
                                   cudf::device_span<size_type const> page_indices,
                                   cudf::host_span<size_type const> page_row_offsets,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr) const
  {
    // Total number of pages in the column
    size_type const total_pages = page_row_offsets.size() - 1;

    // Construct device vectors containing page-level (input) string data, and offsets and sizes
    auto [page_str_chars, page_str_offsets, page_str_sizes] =
      host_column<cudf::string_view>::make_strings_children(host_strings, host_chars, stream, mr);

    // Buffer for row-level string sizes (output).
    auto row_str_sizes = rmm::device_uvector<size_t>(total_rows, stream, mr);
    // Gather string sizes from page to row level
    thrust::gather(rmm::exec_policy_nosync(stream),
                   page_indices.begin(),
                   page_indices.end(),
                   page_str_sizes.begin(),
                   row_str_sizes.begin());

    // Total bytes in the output chars buffer
    auto const total_bytes = thrust::reduce(rmm::exec_policy(stream),
                                            row_str_sizes.begin(),
                                            row_str_sizes.end(),
                                            size_t{0},
                                            cuda::std::plus<size_t>());

    CUDF_EXPECTS(
      total_bytes <= cuda::std::numeric_limits<cudf::size_type>::max(),
      "The strings child of the page statistics column cannot exceed the column size limit");

    // page-level strings nullmask (input)
    auto const input_nullmask = host_page_nullmask;

    // Buffer for row-level strings nullmask (output). Initialize to all bits set.
    auto output_nullmask = cudf::create_null_mask(total_rows, mask_state::ALL_VALID, stream, mr);

    // For each input page, invalidate the null mask for corresponding rows if needed.
    std::for_each(thrust::counting_iterator(0),
                  thrust::counting_iterator(total_pages),
                  [&](auto const page_idx) {
                    if (not bit_is_set(input_nullmask, page_idx)) {
                      cudf::set_null_mask(static_cast<bitmask_type*>(output_nullmask.data()),
                                          page_row_offsets[page_idx],
                                          page_row_offsets[page_idx + 1],
                                          false,
                                          stream);
                    }
                  });

    // Buffer for row-level string offsets (output).
    auto row_str_offsets =
      cudf::detail::make_zeroed_device_uvector_async<cudf::size_type>(total_rows + 1, stream, mr);
    thrust::inclusive_scan(rmm::exec_policy_nosync(stream),
                           row_str_sizes.begin(),
                           row_str_sizes.end(),
                           row_str_offsets.begin() + 1);

    // Buffer for row-level string chars (output).
    auto row_str_chars = rmm::device_buffer(total_bytes, stream, mr);

    // Iterator for input (page-level) string chars
    auto src_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_t>(0),
      cuda::proclaim_return_type<char*>(
        [chars        = page_str_chars.begin(),
         offsets      = page_str_offsets.begin(),
         page_indices = page_indices.begin()] __device__(size_t index) {
          auto const page_index = page_indices[index];
          return chars + offsets[page_index];
        }));

    // Iterator for output (row-level) string chars
    auto dst_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_t>(0),
      cuda::proclaim_return_type<char*>(
        [chars   = reinterpret_cast<char*>(row_str_chars.data()),
         offsets = row_str_offsets.begin()] __device__(size_t index) {
          return chars + offsets[index];
        }));

    // Iterator for string sizes
    auto size_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_t>(0),
      cuda::proclaim_return_type<size_t>(
        [sizes = row_str_sizes.begin()] __device__(size_t index) { return sizes[index]; }));

    // Gather page-level string chars to row-level string chars
    cudf::detail::batched_memcpy_async(src_iter, dst_iter, size_iter, total_rows, stream);

    // Return row-level (output) strings children and the nullmask
    return std::tuple{
      std::move(row_str_chars), std::move(row_str_offsets), std::move(output_nullmask)};
  }

  /**
   * @brief Builds two device columns storing the corresponding page-level statistics (min, max)
   *        respectively of a column at each row index.
   *
   * @tparam T underlying type of the column
   * @param schema_idx Column schema index
   * @param dtype Column data type
   * @param stream CUDA stream
   * @param mr Device memory resource
   *
   * @return A pair of device columns with min and max value from page statistics for each row
   */
  template <typename T>
  std::pair<std::unique_ptr<column>, std::unique_ptr<column>> operator()(
    cudf::size_type schema_idx,
    cudf::data_type dtype,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    // List, Struct, Dictionary types are not supported
    if constexpr (cudf::is_compound<T>() and not cuda::std::is_same_v<T, string_view>) {
      CUDF_FAIL("Compound types other than strings do not have statistics");
    } else {
      // Compute column chunk level page count offsets, and page level row counts and row offsets.
      auto const [page_row_counts, page_row_offsets, col_chunk_page_offsets] =
        make_page_row_counts_and_offsets(per_file_metadata, row_group_indices, schema_idx, stream);

      CUDF_EXPECTS(
        page_row_offsets.back() == total_rows,
        "The number of rows must be equal across row groups and pages within row groups");

      auto const total_pages = col_chunk_page_offsets.back();

      // Create host columns with page-level min, max values
      host_column<T> min(total_pages, stream);
      host_column<T> max(total_pages, stream);

      // Populate the host columns with page-level min, max statistics from the page index
      auto page_offset_idx = 0;
      // For all row data sources
      std::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator(row_group_indices.size()),
        [&](auto src_idx) {
          // For all column chunks in this source
          auto const& rg_indices = row_group_indices[src_idx];
          std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto rg_idx) {
            auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
            // Find colchunk_iter in row_group.columns. Guaranteed to be found as already verified
            // in make_page_row_counts_and_offsets()
            auto colchunk_iter = std::find_if(
              row_group.columns.begin(),
              row_group.columns.end(),
              [schema_idx](ColumnChunk const& col) { return col.schema_idx == schema_idx; });

            auto const& colchunk               = *colchunk_iter;
            auto const& column_index           = colchunk.column_index.value();
            auto const num_pages_in_colchunk   = column_index.min_values.size();
            auto const page_offset_in_colchunk = col_chunk_page_offsets[page_offset_idx++];

            // For all pages in this column chunk
            std::for_each(
              thrust::counting_iterator<size_t>(0),
              thrust::counting_iterator(num_pages_in_colchunk),
              [&](auto page_idx) {
                auto const& min_value = column_index.min_values[page_idx];
                auto const& max_value = column_index.max_values[page_idx];
                // Translate binary data to Type then to <T>
                min.set_index(
                  page_offset_in_colchunk + page_idx, min_value, colchunk.meta_data.type);
                max.set_index(
                  page_offset_in_colchunk + page_idx, max_value, colchunk.meta_data.type);
              });
          });
        });

      // Construct a row indices mapping based on page row counts and offsets
      auto const page_indices =
        make_page_indices_async(page_row_counts, page_row_offsets, total_rows, stream);

      // For non-strings columns, directly gather the page-level column data and bitmask to the
      // row-level.
      if constexpr (not cuda::std::is_same_v<T, cudf::string_view>) {
        // Move host columns to device
        auto mincol = min.to_device(dtype, stream, mr);
        auto maxcol = max.to_device(dtype, stream, mr);

        // Convert page-level min and max columns to row-level min and max columns by gathering
        // values based on page-level row offsets
        auto [min_data, min_bitmask] = build_data_and_nullmask<T>(mincol->mutable_view(),
                                                                  min.null_mask.data(),
                                                                  page_indices,
                                                                  page_row_offsets,
                                                                  dtype,
                                                                  stream,
                                                                  mr);
        auto [max_data, max_bitmask] = build_data_and_nullmask<T>(maxcol->mutable_view(),
                                                                  max.null_mask.data(),
                                                                  page_indices,
                                                                  page_row_offsets,
                                                                  dtype,
                                                                  stream,
                                                                  mr);

        // Count nulls in min and max columns
        auto const min_nulls = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(min_bitmask.data()), 0, total_rows, stream);
        auto const max_nulls = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(max_bitmask.data()), 0, total_rows, stream);

        // Return min and max device columns
        return {std::make_unique<column>(
                  dtype, total_rows, std::move(min_data), std::move(min_bitmask), min_nulls),
                std::make_unique<column>(
                  dtype, total_rows, std::move(max_data), std::move(max_bitmask), max_nulls)};
      }
      // For strings columns, gather the page-level string offsets and bitmask to row-level
      // directly and gather string chars using a batched memcpy.
      else {
        auto [min_data, min_offsets, min_nullmask] = build_string_data_and_nullmask(
          min.val, min.chars, min.null_mask.data(), page_indices, page_row_offsets, stream, mr);
        auto [max_data, max_offsets, max_nullmask] = build_string_data_and_nullmask(
          max.val, max.chars, max.null_mask.data(), page_indices, page_row_offsets, stream, mr);

        // Count nulls in min and max columns
        auto const min_nulls = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(min_nullmask.data()), 0, total_rows, stream);
        auto const max_nulls = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(max_nullmask.data()), 0, total_rows, stream);

        // Return min and max device strings columns
        return {
          cudf::make_strings_column(
            total_rows,
            std::make_unique<column>(std::move(min_offsets), rmm::device_buffer{0, stream, mr}, 0),
            std::move(min_data),
            min_nulls,
            std::move(min_nullmask)),
          cudf::make_strings_column(
            total_rows,
            std::make_unique<column>(std::move(max_offsets), rmm::device_buffer{0, stream, mr}, 0),
            std::move(max_data),
            max_nulls,
            std::move(max_nullmask))};
      }
    }
  }
};

/**
 * @brief Functor to compute if a row in the row mask is required
 *
 * The row is required if the row mask value at row_index is either invalid or a valid `true`
 *
 * @param is_nullable Whether the row mask is nullable
 * @param nullmask The nullmask of the row mask
 * @param row_mask_data The row mask data values
 *
 * @return True if the row is valid, false otherwise.
 */
struct is_row_required_fn {
  bool is_nullable;
  bitmask_type const* nullmask;
  bool const* row_mask_data;

  __device__ bool operator()(size_type row_index) const
  {
    auto const is_invalid = is_nullable and not bit_is_set(nullmask, row_index);
    return is_invalid or row_mask_data[row_index];
  }
};

}  // namespace

std::unique_ptr<cudf::column> aggregate_reader_metadata::build_row_mask_with_page_index_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<cudf::data_type const> output_dtypes,
  cudf::host_span<cudf::size_type const> output_column_schemas,
  std::reference_wrapper<ast::expression const> filter,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  // Return if empty row group indices
  if (row_group_indices.empty()) { return cudf::make_empty_column(cudf::type_id::BOOL8); }

  // Check if we have page index for all columns in all row groups
  auto const has_page_index =
    compute_has_page_index(per_file_metadata, row_group_indices, output_column_schemas);

  // Return if page index is not present
  CUDF_EXPECTS(has_page_index,
               "Page pruning requires the Parquet page index for all output columns",
               std::runtime_error);

  // Total number of rows
  auto const total_rows = std::accumulate(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(row_group_indices.size()),
    size_t{0},
    [&](auto sum, auto const src_index) {
      auto const& rg_indices = row_group_indices[src_index];
      return std::accumulate(
        rg_indices.begin(), rg_indices.end(), sum, [&](auto subsum, auto const rg_index) {
          CUDF_EXPECTS(subsum + per_file_metadata[src_index].row_groups[rg_index].num_rows <=
                         std::numeric_limits<size_type>::max(),
                       "Total rows exceed the maximum value");
          return subsum + per_file_metadata[src_index].row_groups[rg_index].num_rows;
        });
    });

  auto const num_columns = output_dtypes.size();

  // Get a boolean mask indicating which columns will participate in stats based filtering
  auto const stats_columns_mask =
    parquet::detail::stats_columns_collector{filter.get(),
                                             static_cast<size_type>(output_dtypes.size())}
      .get_stats_columns_mask();

  // Convert page statistics to a table
  // where min(col[i]) = columns[i*2], max(col[i])=columns[i*2+1]
  // For each column, it contains total number of rows from all row groups.
  page_stats_caster const stats_col{
    static_cast<size_type>(total_rows), per_file_metadata, row_group_indices};

  std::vector<std::unique_ptr<column>> columns;
  std::for_each(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(num_columns),
    [&](auto col_idx) {
      auto const schema_idx = output_column_schemas[col_idx];
      auto const& dtype     = output_dtypes[col_idx];
      // Only participating columns and comparable types except fixed point are supported
      if (not stats_columns_mask[col_idx] or
          (cudf::is_compound(dtype) && dtype.id() != cudf::type_id::STRING)) {
        // Placeholder for unsupported types and non-participating columns
        columns.push_back(cudf::make_numeric_column(
          data_type{cudf::type_id::BOOL8}, total_rows, rmm::device_buffer{}, 0, stream, mr));
        columns.push_back(cudf::make_numeric_column(
          data_type{cudf::type_id::BOOL8}, total_rows, rmm::device_buffer{}, 0, stream, mr));
        return;
      }
      auto [min_col, max_col] = cudf::type_dispatcher<dispatch_storage_type>(
        dtype, stats_col, schema_idx, dtype, stream, mr);
      columns.push_back(std::move(min_col));
      columns.push_back(std::move(max_col));
    });

  auto stats_table = cudf::table(std::move(columns));

  // Converts AST to StatsAST with reference to min, max columns in above `stats_table`.
  parquet::detail::stats_expression_converter const stats_expr{
    filter.get(), static_cast<size_type>(output_dtypes.size())};

  // Filter the input table using AST expression and return the (BOOL8) predicate column.
  return cudf::detail::compute_column(stats_table, stats_expr.get_stats_expr().get(), stream, mr);
}

std::vector<std::vector<bool>> aggregate_reader_metadata::compute_data_page_mask(
  cudf::column_view row_mask,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<cudf::data_type const> output_dtypes,
  cudf::host_span<cudf::size_type const> output_column_schemas,
  rmm::cuda_stream_view stream) const
{
  CUDF_EXPECTS(row_mask.type().id() == cudf::type_id::BOOL8,
               "Input row bitmask should be of type BOOL8");

  auto const total_rows  = row_mask.size();
  auto const num_columns = output_dtypes.size();

  auto const has_page_index =
    compute_has_page_index(per_file_metadata, row_group_indices, output_column_schemas);

  // TODO: Don't use page pruning in case of lists and structs until we support them
  if (not has_page_index) {
    CUDF_LOG_WARN("Encountered missing Parquet page index for one or more output columns");
    return {};  // An empty data page mask indicates all pages are required
  }

  // Compute page row counts, offsets, and column chunk page offsets for each column
  std::vector<cudf::detail::host_vector<size_type>> page_row_counts;
  std::vector<cudf::detail::host_vector<size_type>> page_row_offsets;
  std::vector<cudf::detail::host_vector<size_type>> col_chunk_page_offsets;
  page_row_counts.reserve(num_columns);
  page_row_offsets.reserve(num_columns);
  col_chunk_page_offsets.reserve(num_columns);

  if (num_columns == 1) {
    auto const schema_idx = output_column_schemas[0];
    auto [counts, offsets, chunk_offsets] =
      make_page_row_counts_and_offsets(per_file_metadata, row_group_indices, schema_idx, stream);
    page_row_counts.emplace_back(std::move(counts));
    page_row_offsets.emplace_back(std::move(offsets));
  } else {
    std::vector<std::future<std::tuple<cudf::detail::host_vector<size_type>,
                                       cudf::detail::host_vector<size_type>,
                                       cudf::detail::host_vector<size_type>>>>
      page_row_counts_and_offsets_tasks;
    page_row_counts_and_offsets_tasks.reserve(num_columns);

    auto streams = cudf::detail::fork_streams(stream, num_columns);

    std::for_each(thrust::counting_iterator<size_t>(0),
                  thrust::counting_iterator(num_columns),
                  [&](auto const col_idx) {
                    page_row_counts_and_offsets_tasks.emplace_back(
                      cudf::detail::host_worker_pool().submit_task([&, col_idx = col_idx] {
                        auto const schema_idx = output_column_schemas[col_idx];
                        return make_page_row_counts_and_offsets(
                          per_file_metadata, row_group_indices, schema_idx, streams[col_idx]);
                      }));
                  });

    // Collect results from all tasks
    std::for_each(page_row_counts_and_offsets_tasks.begin(),
                  page_row_counts_and_offsets_tasks.end(),
                  [&](auto& task) {
                    auto [counts, offsets, chunk_offsets] = std::move(task).get();
                    page_row_counts.emplace_back(std::move(counts));
                    page_row_offsets.emplace_back(std::move(offsets));
                    col_chunk_page_offsets.emplace_back(std::move(chunk_offsets));
                  });
  }

  CUDF_EXPECTS(page_row_offsets.back().back() == total_rows,
               "Mismatch in total rows in input row mask and row groups",
               std::invalid_argument);

  // Return if all rows are required or all are invalid.
  if (row_mask.null_count() == row_mask.size() or thrust::all_of(rmm::exec_policy(stream),
                                                                 row_mask.begin<bool>(),
                                                                 row_mask.end<bool>(),
                                                                 cuda::std::identity{})) {
    return all_required_data_pages(page_row_counts);
  }

  auto const mr = cudf::get_current_device_resource_ref();

  // Vector to hold data page mask for each column
  auto data_page_mask = std::vector<std::vector<bool>>();
  data_page_mask.reserve(num_columns);

  auto total_surviving_pages = size_t{0};

  // For all columns, look up which pages contain at least one required row. i.e.
  // !validity_it[row_idx] or is_row_required[row_idx] satisfies, and add its byte range to the
  // output list of byte ranges for the column.
  std::for_each(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(num_columns),
    [&](auto const col_idx) {
      // Construct a row indices mapping based on page row counts and offsets
      auto const total_pages_in_this_column = page_row_counts[col_idx].size();

      auto const page_indices = make_page_indices_async(
        page_row_counts[col_idx], page_row_offsets[col_idx], total_rows, stream);

      // Device vector to hold page indices with at least one required row
      rmm::device_uvector<size_type> select_page_indices(total_rows, stream, mr);

      // Copy page indices with at least one required row
      auto const filtered_pages_end_iter = thrust::copy_if(
        rmm::exec_policy_nosync(stream),
        page_indices.begin(),
        page_indices.end(),
        thrust::counting_iterator<size_type>(0),
        select_page_indices.begin(),
        is_row_required_fn{row_mask.nullable(), row_mask.null_mask(), row_mask.data<bool>()});

      // Remove duplicate page indices across (presorted) rows
      auto const filtered_uniq_page_end_iter = thrust::unique(
        rmm::exec_policy_nosync(stream), select_page_indices.begin(), filtered_pages_end_iter);

      // Number of final filtered pages for this column
      size_t const num_surviving_pages_this_column =
        thrust::distance(select_page_indices.begin(), filtered_uniq_page_end_iter);

      total_surviving_pages += num_surviving_pages_this_column;

      // Copy the filtered page indices for this column to host
      auto host_select_page_indices = cudf::detail::make_host_vector(
        cudf::device_span<cudf::size_type const>{select_page_indices.data(),
                                                 num_surviving_pages_this_column},
        stream);

      // Vector to data page mask the this column
      auto valid_pages = std::vector<bool>(total_pages_in_this_column, false);
      std::for_each(host_select_page_indices.begin(),
                    host_select_page_indices.end(),
                    [&](auto const page_idx) { valid_pages[page_idx] = true; });

      data_page_mask.push_back(std::move(valid_pages));
    });

  // Total number of input pages across all columns
  auto const total_pages = std::accumulate(
    page_row_counts.cbegin(),
    page_row_counts.cend(),
    size_t{0},
    [](auto sum, auto const& page_row_counts) { return sum + page_row_counts.size(); });

  CUDF_EXPECTS(
    total_surviving_pages <= total_pages,
    "Number of surviving pages must be less than or equal to the total number of input pages");

  return data_page_mask;
}

}  // namespace cudf::io::parquet::experimental::detail
