/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/experimental/deletion_vectors.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuco/roaring_bitmap.cuh>
#include <cuda/functional>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>

#include <numeric>

namespace cudf::io::parquet::experimental {

namespace {

/**
 * @brief Computes a row index column from the specified row group row offsets and counts
 *
 * @param row_group_offsets Host span of row group offsets
 * @param row_group_num_rows Host span of number of rows in each row group
 * @param num_rows Total number of table rows
 * @param stream CUDA stream for kernel launches and data transfers
 * @param mr Device memory resource to allocate device memory for the row index column
 *
 * @return UINT64 column containing row indices
 */
std::unique_ptr<cudf::column> compute_row_index_column(
  cudf::host_span<size_t const> row_group_offsets,
  cudf::host_span<size_type const> row_group_num_rows,
  size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_row_groups = static_cast<size_type>(row_group_num_rows.size());

  if (row_group_offsets.empty()) {
    auto row_indices      = rmm::device_buffer(num_rows * sizeof(size_t), stream, mr);
    auto row_indices_iter = static_cast<size_t*>(row_indices.data());
    thrust::sequence(
      rmm::exec_policy_nosync(stream), row_indices_iter, row_indices_iter + num_rows, 0);
    return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT64},
                                          num_rows,
                                          std::move(row_indices),
                                          rmm::device_buffer{},
                                          0);
  }

  // Convert number of rows in each row group to row group span offsets. Here, a span is the range
  // of (table) rows that belong to a row group
  auto row_group_span_offsets =
    cudf::detail::make_host_vector<size_type>(num_row_groups + 1, stream);
  row_group_span_offsets[0] = 0;
  std::inclusive_scan(
    row_group_num_rows.begin(), row_group_num_rows.end(), row_group_span_offsets.begin() + 1);

  auto row_indices      = rmm::device_buffer(num_rows * sizeof(size_t), stream, mr);
  auto row_indices_iter = static_cast<size_t*>(row_indices.data());
  thrust::fill(rmm::exec_policy_nosync(stream), row_indices_iter, row_indices_iter + num_rows, 1);

  auto row_group_keys = rmm::device_uvector<size_type>(num_rows, stream);
  thrust::fill(rmm::exec_policy_nosync(stream), row_group_keys.begin(), row_group_keys.end(), 0);

  // Scatter row group offsets and row group indices (or span indices) to their corresponding
  // row group span offsets
  auto d_row_group_offsets = cudf::detail::make_device_uvector_async(
    row_group_offsets, stream, cudf::get_current_device_resource_ref());
  auto d_row_group_span_offsets = cudf::detail::make_device_uvector_async(
    row_group_span_offsets, stream, cudf::get_current_device_resource_ref());
  auto in_iter =
    thrust::make_zip_iterator(d_row_group_offsets.begin(), thrust::counting_iterator<size_type>(0));
  auto out_iter = thrust::make_zip_iterator(row_indices_iter, row_group_keys.begin());
  thrust::scatter(rmm::exec_policy_nosync(stream),
                  in_iter,
                  in_iter + num_row_groups,
                  d_row_group_span_offsets.begin(),
                  out_iter);

  // Fill in the the rest of the row group span indices
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream),
                         row_group_keys.begin(),
                         row_group_keys.end(),
                         row_group_keys.begin(),
                         cuda::maximum<cudf::size_type>());

  // Segmented inclusive scan to compute the rest of the row indices
  thrust::inclusive_scan_by_key(rmm::exec_policy_nosync(stream),
                                row_group_keys.begin(),
                                row_group_keys.end(),
                                row_indices_iter,
                                row_indices_iter);

  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT64},
                                        num_rows,
                                        std::move(row_indices),
                                        rmm::device_buffer{},
                                        0);
}

/**
 * @brief Builds a BOOL8 row mask column from the specified host span of row indices and the
 * roaring64 deletion vector
 *
 * @param row_indices Host span of row indices
 * @param deletion_vector Pointer to the roaring64 deletion vector
 * @param num_rows Number of rows in the column
 * @param stream CUDA stream for kernel launches and data transfers
 * @param mr Device memory resource to allocate device memory for the row mask column
 *
 * @return Unique pointer to the row mask column
 */
std::unique_ptr<cudf::column> build_row_mask_column(
  cudf::column_view const& row_index_column,
  cudf::host_span<cuda::std::byte const> deletion_vector,
  size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto row_mask = rmm::device_buffer(num_rows, stream, mr);
  cuco::experimental::roaring_bitmap<cuda::std::uint64_t> roaring_bitmap(
    deletion_vector.data(), {}, stream);

  // Iterator to negate and store the output value from `contains_async`
  auto row_mask_iter = thrust::make_transform_output_iterator(
    static_cast<bool*>(row_mask.data()), [] __device__(auto b) { return not b; });
  roaring_bitmap.contains_async(
    row_index_column.begin<size_t>(), row_index_column.end<size_t>(), row_mask_iter, stream);

  return std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::BOOL8}, num_rows, std::move(row_mask), rmm::device_buffer{}, 0);
}

}  // namespace

table_with_metadata read_parquet_and_apply_deletion_vector(
  parquet_reader_options const& options,
  cudf::host_span<cuda::std::byte const> serialized_roaring64,
  cudf::host_span<size_t const> row_group_offsets,
  cudf::host_span<size_type const> row_group_num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    row_group_offsets.size() == row_group_num_rows.size(),
    "Encountered a mismatch in the number of row group offsets and row group row counts");
  CUDF_EXPECTS(
    not options.get_filter().has_value(),
    "Encountered a non-empty AST filter expression. Use a roaring64 bitmap deletion vector to "
    "filter the table instead");

  // Use default mr to read parquet table and build row index column if we will be applying the
  // deletion vector to produce a new table later
  auto const table_mr =
    serialized_roaring64.empty() ? mr : rmm::mr::get_current_device_resource_ref();

  // Read the parquet table
  auto [table, metadata] = cudf::io::read_parquet(options, stream, table_mr);
  auto const num_rows    = table->num_rows();

  CUDF_EXPECTS(
    row_group_num_rows.empty() or
      std::cmp_equal(metadata.num_input_row_groups, row_group_num_rows.size()),
    "Encountered a mismatch in the number of row groups in parquet file and the specified "
    "row group offsets/row counts vectors");

  // Compute a row index column from the specified row group offsets and counts
  auto row_index_column =
    compute_row_index_column(row_group_offsets, row_group_num_rows, num_rows, stream, table_mr);

  // Prepend row index column to the table columns
  auto index_and_table_columns = std::vector<std::unique_ptr<cudf::column>>{};
  index_and_table_columns.reserve(table->num_columns() + 1);
  index_and_table_columns.push_back(std::move(row_index_column));
  auto table_columns = table->release();
  index_and_table_columns.insert(index_and_table_columns.end(),
                                 std::make_move_iterator(table_columns.begin()),
                                 std::make_move_iterator(table_columns.end()));

  // Table with the index and table columns
  auto table_with_index = std::make_unique<cudf::table>(std::move(index_and_table_columns));

  // Also prepend the row index column's metadata to the table schema
  auto updated_schema_info = std::vector<cudf::io::column_name_info>{};
  updated_schema_info.reserve(metadata.schema_info.size() + 1);
  updated_schema_info.emplace_back("index");
  updated_schema_info.insert(updated_schema_info.end(),
                             std::make_move_iterator(metadata.schema_info.begin()),
                             std::make_move_iterator(metadata.schema_info.end()));
  metadata.schema_info = std::move(updated_schema_info);

  if (serialized_roaring64.empty()) {
    return table_with_metadata{std::move(table_with_index), std::move(metadata)};
  } else {
    // Filter the table using the deletion vector
    auto row_mask = build_row_mask_column(table_with_index->get_column(0).view(),
                                          serialized_roaring64,
                                          num_rows,
                                          stream,
                                          cudf::get_current_device_resource_ref());
    return table_with_metadata{
      // Supply user-provided mr to apply_boolean_mask to allocate output table's memory
      cudf::apply_boolean_mask(table_with_index->view(), row_mask->view(), stream, mr),
      std::move(metadata)};
  }
}

}  // namespace cudf::io::parquet::experimental
