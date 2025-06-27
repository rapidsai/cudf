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
  rmm::cuda_stream_view stream)
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

  auto row_index_buffer = rmm::device_buffer{host_row_indices.size() * sizeof(size_t), stream};
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
  auto [table, metadata] = cudf::io::read_parquet(options, stream);

  CUDF_EXPECTS(table->num_rows() == num_rows,
               "Mismatch in number of rows in table and row offsets");

  auto columns_with_index = std::vector<std::unique_ptr<cudf::column>>{};
  columns_with_index.reserve(table->num_columns() + 1);
  columns_with_index.push_back(std::move(row_index_column));
  auto parquet_columns = table->release();
  columns_with_index.insert(columns_with_index.end(),
                            std::make_move_iterator(parquet_columns.begin()),
                            std::make_move_iterator(parquet_columns.end()));

  auto table_with_index = std::make_unique<cudf::table>(std::move(columns_with_index));

  // Add `index` column to the table schema infomation
  auto schema_info_with_index = std::vector<cudf::io::column_name_info>{};
  schema_info_with_index.reserve(metadata.schema_info.size() + 1);
  schema_info_with_index.emplace_back("index");
  schema_info_with_index.insert(schema_info_with_index.end(),
                                std::make_move_iterator(metadata.schema_info.begin()),
                                std::make_move_iterator(metadata.schema_info.end()));
  metadata.schema_info = std::move(schema_info_with_index);

  if (deletion_vector and roaring64_bitmap_get_cardinality(deletion_vector)) {
    auto host_row_mask = cudf::detail::make_host_vector<bool>(num_rows, stream);
    std::transform(
      host_row_indices.begin(), host_row_indices.end(), host_row_mask.begin(), [&](auto i) {
        // If the row is in the deletion vector, set the mask to false
        return not roaring64_bitmap_contains(deletion_vector, host_row_indices[i]);
      });

    auto row_mask = rmm::device_buffer{num_rows * sizeof(bool), stream};
    cudf::detail::cuda_memcpy_async<bool>(
      device_span<bool>{static_cast<bool*>(row_mask.data()), static_cast<size_t>(num_rows)},
      host_row_mask,
      stream);
    auto row_mask_column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::BOOL8},
                                                          num_rows,
                                                          std::move(row_mask),
                                                          rmm::device_buffer{},
                                                          cudf::size_type{0});

    auto table_with_index_and_row_mask =
      cudf::apply_boolean_mask(table_with_index->view(), row_mask_column->view());
    return table_with_metadata{std::move(table_with_index_and_row_mask), std::move(metadata)};
  } else {
    return table_with_metadata{std::move(table_with_index), std::move(metadata)};
  }
}

}  // namespace cudf::io::parquet::experimental