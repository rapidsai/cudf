
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

#include "page_index_filter_utils.hpp"

#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/gather.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace cudf::io::parquet::experimental::detail {

bool compute_has_page_index(cudf::host_span<metadata_base const> file_metadatas,
                            cudf::host_span<std::vector<size_type> const> row_group_indices,
                            cudf::host_span<size_type const> column_schema_indices)
{
  // For all output columns, check all parquet data sources
  return std::all_of(
    column_schema_indices.begin(), column_schema_indices.end(), [&](auto const schema_idx) {
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

std::tuple<cudf::detail::host_vector<size_type>,
           cudf::detail::host_vector<size_type>,
           cudf::detail::host_vector<size_type>>
compute_page_row_counts_and_offsets(cudf::host_span<metadata_base const> per_file_metadata,
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
    cudf::detail::make_empty_host_vector<cudf::size_type>(2 * total_row_groups, stream);
  // Vector to store the cumulative number of rows in each page - - set initial capacity to two data
  // pages per row group
  auto page_row_offsets =
    cudf::detail::make_empty_host_vector<cudf::size_type>((2 * total_row_groups) + 1, stream);
  // Vector to store the cumulative number of pages in each column chunk
  auto col_chunk_page_offsets =
    cudf::detail::make_empty_host_vector<cudf::size_type>(total_row_groups + 1, stream);

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
          // Get the offset index of the column chunk
          auto const& offset_index       = colchunk_iter->offset_index.value();
          auto const row_group_num_pages = offset_index.page_locations.size();

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

  return {
    std::move(page_row_counts), std::move(page_row_offsets), std::move(col_chunk_page_offsets)};
}

std::tuple<std::vector<size_type>, size_type, size_type> compute_page_row_offsets(
  cudf::host_span<metadata_base const> per_file_metadata,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  size_type schema_idx)
{
  // Compute total number of row groups
  auto const total_row_groups =
    std::accumulate(row_group_indices.begin(),
                    row_group_indices.end(),
                    size_t{0},
                    [](auto sum, auto const& rg_indices) { return sum + rg_indices.size(); });

  std::vector<size_type> page_row_offsets;
  page_row_offsets.push_back(0);
  size_type max_page_size = 0;
  size_type num_pages     = 0;

  std::for_each(thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator(row_group_indices.size()),
                [&](auto const src_idx) {
                  auto const& rg_indices = row_group_indices[src_idx];
                  // For all row groups in this source
                  std::for_each(rg_indices.begin(), rg_indices.end(), [&](auto const& rg_idx) {
                    auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
                    // Find the column chunk with the given schema index
                    auto colchunk_iter = std::find_if(
                      row_group.columns.begin(),
                      row_group.columns.end(),
                      [schema_idx](auto const& col) { return col.schema_idx == schema_idx; });
                    CUDF_EXPECTS(colchunk_iter != row_group.columns.end(),
                                 "Column chunk with schema index " + std::to_string(schema_idx) +
                                   " not found in row group",
                                 std::invalid_argument);
                    auto const& offset_index       = colchunk_iter->offset_index.value();
                    auto const row_group_num_pages = offset_index.page_locations.size();
                    num_pages += static_cast<size_type>(row_group_num_pages);
                    std::for_each(thrust::counting_iterator<size_t>(0),
                                  thrust::counting_iterator(row_group_num_pages),
                                  [&](auto const page_idx) {
                                    int64_t const first_row_idx =
                                      offset_index.page_locations[page_idx].first_row_index;
                                    int64_t const last_row_idx =
                                      (page_idx < row_group_num_pages - 1)
                                        ? offset_index.page_locations[page_idx + 1].first_row_index
                                        : row_group.num_rows;
                                    auto const page_size = last_row_idx - first_row_idx;
                                    max_page_size = std::max<size_type>(max_page_size, page_size);
                                    page_row_offsets.push_back(page_row_offsets.back() + page_size);
                                  });
                  });
                });

  return {std::move(page_row_offsets), num_pages, max_page_size};
}

rmm::device_uvector<size_type> make_page_indices_async(
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

}  // namespace cudf::io::parquet::experimental::detail