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
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda/functional>
#include <thrust/host_vector.h>
#include <thrust/scatter.h>

#include <numeric>

namespace cudf::io::parquet::experimental {

table_with_metadata read_parquet_and_apply_deletion_vector(
  parquet_reader_options const& options,
  roaring64_bitmap_t const* deletion_vector,
  cudf::host_span<size_type const> rg_offsets,
  cudf::host_span<size_type const> rg_num_rows,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(deletion_vector != nullptr, "Deletion vector cannot be null");
  CUDF_EXPECTS(rg_offsets.size() == rg_num_rows.size(),
               "Mismatch in size of row group offsets and number of rows");
  CUDF_EXPECTS(not options.get_filter().has_value(),
               "AST filter is not supported. Use the deletion vector instead");

  auto row_group_offsets =
    cudf::detail::make_host_vector<size_type>(rg_num_rows.size() + 1, stream);
  row_group_offsets[0] = 0;
  std::inclusive_scan(rg_num_rows.begin(), rg_num_rows.end(), row_group_offsets.begin() + 1);

  auto const num_rows = row_group_offsets.back();

  auto row_indices = cudf::detail::make_host_vector<size_type>(num_rows, stream);
  thrust::scatter(
    rg_offsets.begin(), rg_offsets.end(), row_group_offsets.begin(), row_indices.begin());

  auto [table, metadata] = cudf::io::read_parquet(options, stream);

  CUDF_EXPECTS(table->num_rows() == num_rows,
               "Mismatch in number of rows in table and row offsets");

  // auto row_indices_column =
  // std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT64},
  //                                                          num_rows,
  //                                                          std::move(row_indices),
  //                                                          rmm::device_buffer{},
  //                                                          0,
  //                                                          stream);
  //
  // auto parquet_columns = table->release();

  // Create a column with row indices using row group offsets and number of rows
  return table_with_metadata{std::move(table), std::move(metadata)};
}

}  // namespace cudf::io::parquet::experimental