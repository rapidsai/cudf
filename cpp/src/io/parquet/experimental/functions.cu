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

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda/functional>
#include <thrust/host_vector.h>
#include <thrust/scatter.h>

#include <numeric>

using namespace roaring::api;

namespace cudf::io::parquet::experimental {

table_with_metadata read_parquet_and_apply_deletion_vector(
  parquet_reader_options const& options,
  roaring64_bitmap_t const* deletion_vector,
  cudf::host_span<size_type const> row_group_row_offsets,
  cudf::host_span<size_type const> row_group_num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Total number of row groups
  auto const num_row_groups = row_group_num_rows.size();

  CUDF_EXPECTS(row_group_row_offsets.size() == num_row_groups,
               "Mismatch in size of row group offsets and number of rows");
  CUDF_EXPECTS(not options.get_filter().has_value(),
               "AST filter is not supported. Use the deletion vector instead");

  // Compute row group segment offsets
  auto row_group_segment_offsets =
    cudf::detail::make_host_vector<size_type>(num_row_groups + 1, stream);
  row_group_segment_offsets[0] = 0;
  std::inclusive_scan(
    row_group_num_rows.begin(), row_group_num_rows.end(), row_group_segment_offsets.begin() + 1);

  // Total number of rows in the table
  auto const num_rows = row_group_segment_offsets.back();

  // Host vector of row indices
  auto host_row_indices = cudf::detail::make_host_vector<size_t>(num_rows, stream);
  std::fill(host_row_indices.begin(), host_row_indices.end(), 1);

  // Scatter the row group offsets to corresponding row indices
  thrust::scatter(row_group_row_offsets.begin(),
                  row_group_row_offsets.end(),
                  row_group_segment_offsets.begin(),
                  host_row_indices.begin());

  // Segmented inclusive scan to compute the rest of the row indices
  std::for_each(thrust::counting_iterator<size_type>(0),
                thrust::counting_iterator<size_type>(num_row_groups),
                [&](auto i) {
                  auto start_row_index = row_group_segment_offsets[i];
                  auto end_row_index   = row_group_segment_offsets[i + 1];
                  std::inclusive_scan(host_row_indices.begin() + start_row_index,
                                      host_row_indices.begin() + end_row_index,
                                      host_row_indices.begin() + start_row_index);
                });

  auto row_index_buffer = rmm::device_buffer{host_row_indices.size() * sizeof(size_t), stream, mr};
  cudf::detail::cuda_memcpy_async<size_t>(
    device_span<size_t>{static_cast<size_t*>(row_index_buffer.data()), host_row_indices.size()},
    host_row_indices,
    stream);
  auto row_index_column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT64},
                                                         num_rows,
                                                         std::move(row_index_buffer),
                                                         rmm::device_buffer{},
                                                         cudf::size_type{0});

  // Read the parquet table
  auto [table, metadata] = cudf::io::read_parquet(options, stream, mr);

  CUDF_EXPECTS(table->num_rows() == num_rows,
               "Mismatch in number of rows in table and row offsets");

  auto table_columns_and_index = std::vector<std::unique_ptr<cudf::column>>{};
  table_columns_and_index.reserve(table->num_columns() + 1);
  table_columns_and_index.push_back(std::move(row_index_column));
  auto table_contents = table->release();
  table_columns_and_index.insert(table_columns_and_index.end(),
                                 std::make_move_iterator(table_contents.begin()),
                                 std::make_move_iterator(table_contents.end()));

  // Add `index` column to the table schema information
  auto updated_schema_info = std::vector<cudf::io::column_name_info>{};
  updated_schema_info.reserve(metadata.schema_info.size() + 1);
  updated_schema_info.emplace_back("index");
  updated_schema_info.insert(updated_schema_info.end(),
                             std::make_move_iterator(metadata.schema_info.begin()),
                             std::make_move_iterator(metadata.schema_info.end()));
  metadata.schema_info = std::move(updated_schema_info);

  // Move index and other table columns to a new table
  auto table_with_index = std::make_unique<cudf::table>(std::move(table_columns_and_index));

  // If deletion vector doesn't exist or is empty, return early with the table with index column and
  // the updated metadata
  if (not deletion_vector or roaring64_bitmap_get_cardinality(deletion_vector)) {
    return table_with_metadata{std::move(table_with_index), std::move(metadata)};
  }

  // Host vector to store the row mask
  auto host_row_mask = cudf::detail::make_host_vector<bool>(num_rows, stream);

  // Context for the roaring64 bitmap for faster (bulk) contains operations
  auto roaring64_context =
    roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0}, .leaf = nullptr};

  // Fill up the row mask using the deletion vector
  std::transform(
    host_row_indices.begin(), host_row_indices.end(), host_row_mask.begin(), [&](auto i) {
      // Check if each row index is not present in the deletion vector
      return not roaring64_bitmap_contains_bulk(
        deletion_vector, &roaring64_context, host_row_indices[i]);
    });

  // Device buffer for the row mask column
  auto row_mask = rmm::device_buffer{num_rows * sizeof(bool), stream};

  // Copy the row mask to the device
  cudf::detail::cuda_memcpy_async<bool>(
    device_span<bool>{static_cast<bool*>(row_mask.data()), static_cast<size_t>(num_rows)},
    host_row_mask,
    stream);

  // Convert the row mask buffer into a BOOL8 column
  auto row_mask_column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::BOOL8},
                                                        num_rows,
                                                        std::move(row_mask),
                                                        rmm::device_buffer{},
                                                        cudf::size_type{0});

  // Apply the row mask to the table and return the resultant table along with the updated metadata
  return table_with_metadata{
    cudf::apply_boolean_mask(table_with_index->view(), row_mask_column->view(), stream, mr),
    std::move(metadata)};
}

}  // namespace cudf::io::parquet::experimental
