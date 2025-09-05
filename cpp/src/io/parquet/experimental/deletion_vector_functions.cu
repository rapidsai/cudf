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

#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
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
#include <thrust/scatter.h>
#include <thrust/sequence.h>

#include <roaring/roaring64.h>

#include <numeric>

namespace cudf::io::parquet::experimental {

namespace {

/**
 * @brief Computes a host vector of row indices from the specified row group row offsets and counts
 *
 * @param row_group_offsets Host span of row group offsets
 * @param row_group_num_rows Host span of number of rows in each row group
 * @param num_rows Total number of table rows
 * @param stream CUDA stream for kernel launches and data transfers
 *
 * @return Host vector of row indices
 */
auto compute_row_indices(cudf::host_span<size_t const> row_group_offsets,
                         cudf::host_span<size_type const> row_group_num_rows,
                         size_type num_rows,
                         rmm::cuda_stream_view stream)
{
  auto const num_row_groups = static_cast<size_type>(row_group_num_rows.size());

  if (row_group_offsets.empty()) {
    auto host_row_indices = cudf::detail::make_host_vector<size_t>(num_rows, stream);
    std::iota(host_row_indices.begin(), host_row_indices.end(), 0);
    return host_row_indices;
  }

  // Convert number of rows in each row group to row group span offsets. Here, a span is the range
  // of (table) rows that belong to a row group
  auto row_group_span_offsets =
    cudf::detail::make_host_vector<size_type>(num_row_groups + 1, stream);
  row_group_span_offsets[0] = 0;
  std::inclusive_scan(
    row_group_num_rows.begin(), row_group_num_rows.end(), row_group_span_offsets.begin() + 1);

  // Check if the last span ends at the total number of table rows
  CUDF_EXPECTS(row_group_span_offsets.back() == num_rows,
               "Mismatch in number of rows in table and specified row group offsets");

  auto row_indices = cudf::detail::make_host_vector<size_t>(num_rows, stream);
  std::fill(row_indices.begin(), row_indices.end(), 1);

  auto row_group_keys = cudf::detail::make_host_vector<size_type>(num_rows, stream);
  std::fill(row_group_keys.begin(), row_group_keys.end(), 0);

  // Scatter row group offsets and row group indices (or span indices) to their corresponding
  // row group span offsets
  auto in_iter =
    thrust::make_zip_iterator(row_group_offsets.begin(), thrust::counting_iterator<size_type>(0));
  auto out_iter = thrust::make_zip_iterator(row_indices.begin(), row_group_keys.begin());
  thrust::scatter(in_iter, in_iter + num_row_groups, row_group_span_offsets.begin(), out_iter);

  // Fill in the the rest of the row group span indices
  thrust::inclusive_scan(row_group_keys.begin(),
                         row_group_keys.end(),
                         row_group_keys.begin(),
                         cuda::maximum<cudf::size_type>());

  // Segmented inclusive scan to compute the rest of the row indices
  thrust::inclusive_scan_by_key(
    row_group_keys.begin(), row_group_keys.end(), row_indices.begin(), row_indices.begin());

  return row_indices;
}

/**
 * @brief Computes a host vector of row indices from the specified row group row offsets and counts
 *
 * @param row_group_offsets Host span of row group offsets
 * @param row_group_num_rows Host span of number of rows in each row group
 * @param num_rows Total number of table rows
 * @param stream CUDA stream for kernel launches and data transfers
 *
 * @return Host vector of row indices
 */
auto compute_row_indices_gpu(cudf::host_span<size_t const> row_group_offsets,
                             cudf::host_span<size_type const> row_group_num_rows,
                             size_type num_rows,
                             rmm::cuda_stream_view stream)
{
  auto const num_row_groups = static_cast<size_type>(row_group_num_rows.size());

  if (row_group_offsets.empty()) {
    auto row_indices      = rmm::device_buffer(num_rows * sizeof(size_t), stream);
    auto row_indices_iter = static_cast<size_t*>(row_indices.data());
    thrust::sequence(
      rmm::exec_policy_nosync(stream), row_indices_iter, row_indices_iter + num_rows, 0);
    return row_indices;
  }

  // Convert number of rows in each row group to row group span offsets. Here, a span is the range
  // of (table) rows that belong to a row group
  auto row_group_span_offsets =
    cudf::detail::make_host_vector<size_type>(num_row_groups + 1, stream);
  row_group_span_offsets[0] = 0;
  thrust::inclusive_scan(
    row_group_num_rows.begin(), row_group_num_rows.end(), row_group_span_offsets.begin() + 1);

  auto row_indices =
    rmm::device_buffer(num_rows * sizeof(size_t), stream, cudf::get_current_device_resource_ref());
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

  return row_indices;
}

/**
 * @brief Builds a cudf column from a span of host data
 *
 * @param host_data Span of host data
 * @param data_type The data type of the column
 * @param stream The stream to use for the operation
 * @param mr The memory resource to use for the operation
 *
 * @return A unique pointer to a column containing the row mask
 */
template <typename T>
auto build_column_from_host_data(cudf::host_span<T const> host_data,
                                 cudf::type_id data_type,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not host_data.empty(), "Encountered an empty host column data span");

  auto const num_rows = host_data.size();
  rmm::device_buffer buffer{num_rows * sizeof(T), stream, mr};
  cudf::detail::cuda_memcpy_async<T>(
    cudf::device_span<T>{static_cast<T*>(buffer.data()), num_rows}, host_data, stream);
  return std::make_unique<cudf::column>(
    cudf::data_type{data_type}, num_rows, std::move(buffer), rmm::device_buffer{}, 0);
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
auto build_row_mask_column(cudf::host_span<size_t const> row_indices,
                           roaring64_bitmap_t const* deletion_vector,
                           size_type num_rows,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  auto row_mask = cudf::detail::make_host_vector<bool>(num_rows, stream);

  auto constexpr thread_pool_size = 16;

  // Use a thread pool to bulk query the deletion vector and fill up the host row mask
  std::vector<std::future<void>> row_mask_tasks;
  row_mask_tasks.reserve(thread_pool_size);
  std::for_each(
    thrust::counting_iterator(0),
    thrust::counting_iterator(thread_pool_size),
    [&](auto const thread_idx) {
      row_mask_tasks.emplace_back(
        cudf::detail::host_worker_pool().submit_task([&, thread_idx = thread_idx] {
          // Thread-local roaring64 context for faster (bulk) contains operations
          auto roaring64_context =
            roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0}, .leaf = nullptr};

          for (auto row_idx = thread_idx; row_idx < num_rows; row_idx += thread_pool_size) {
            row_mask[row_idx] = not roaring::api::roaring64_bitmap_contains_bulk(
              deletion_vector, &roaring64_context, static_cast<uint64_t>(row_indices[row_idx]));
          }
        }));
    });

  std::for_each(
    row_mask_tasks.begin(), row_mask_tasks.end(), [&](auto& task) { std::move(task).get(); });

  // Convert the row mask buffer into a BOOL8 column
  return build_column_from_host_data<bool>(row_mask, cudf::type_id::BOOL8, stream, mr);
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
std::unique_ptr<cudf::column> build_row_mask_column_gpu(
  cudf::device_span<size_t const> row_indices,
  cudf::host_span<cuda::std::byte const> deletion_vector,
  size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto row_mask = rmm::device_buffer(num_rows, stream, mr);
  cuco::experimental::roaring_bitmap<uint64_t> roaring_bitmap(deletion_vector.data());

  auto row_mask_iter = static_cast<bool*>(row_mask.data());
  roaring_bitmap.contains_async(row_indices.begin(), row_indices.end(), row_mask_iter, stream);
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   row_mask_iter,
                   row_mask_iter + num_rows,
                   [] __device__(auto& b) { b = not b; });

  return std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::BOOL8}, num_rows, std::move(row_mask), rmm::device_buffer{}, 0);
}

}  // namespace

table_with_metadata read_parquet_and_apply_deletion_vector(
  parquet_reader_options const& options,
  cudf::host_span<cuda::std::byte const> serialized_roaring64_bytes,
  cudf::host_span<size_t const> row_group_offsets,
  cudf::host_span<size_type const> row_group_num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(row_group_offsets.size() == row_group_num_rows.size(),
               "Encountered a mismatch in the size of row group offsets and row counts vectors");
  CUDF_EXPECTS(
    not options.get_filter().has_value(),
    "Encountered a non-null AST filter expression. Use a roaring64 bitmap deletion vector to "
    "filter the table instead");

  auto const deletion_vector = [&]() -> roaring64_bitmap_t* {
    if (serialized_roaring64_bytes.empty()) { return nullptr; }

    // Check if we can deserialize the roaring64 bitmap from the serialized bytes
    CUDF_EXPECTS(roaring::api::roaring64_bitmap_portable_deserialize_size(
                   reinterpret_cast<char const*>(serialized_roaring64_bytes.data()),
                   static_cast<size_t>(serialized_roaring64_bytes.size())),
                 "Failed to deserialize the roaring64 bitmap");

    // Deserialize the roaring64 bitmap from the frozen serialized bytes
    auto deletion_vector = roaring::api::roaring64_bitmap_portable_deserialize_safe(
      reinterpret_cast<char const*>(serialized_roaring64_bytes.data()),
      static_cast<size_t>(serialized_roaring64_bytes.size()));

    // Validate the deserialized roaring64 bitmap
    CUDF_EXPECTS(deletion_vector, "Failed to deserialize the portable roaring64 bitmap");
    CUDF_EXPECTS(roaring::api::roaring64_bitmap_internal_validate(deletion_vector, nullptr),
                 "Encountered an inconsistent roaring64 bitmap after deserialization");

    return deletion_vector;
  }();

  // Read the parquet table
  auto [table, metadata] = cudf::io::read_parquet(options, stream, mr);
  auto const num_rows    = table->num_rows();

  CUDF_EXPECTS(
    row_group_num_rows.empty() or
      std::cmp_equal(metadata.num_input_row_groups, row_group_num_rows.size()),
    "Encountered a mismatch in the number of row groups in the file and row group row counts");

  // Compute a row index host vector from the specified row group offsets and counts
  auto row_indices = compute_row_indices(row_group_offsets, row_group_num_rows, num_rows, stream);

  // Build and prepend the index column to the table columns
  auto row_index_column =
    build_column_from_host_data<size_t>(row_indices, cudf::type_id::UINT64, stream, mr);
  auto index_and_table_columns = std::vector<std::unique_ptr<cudf::column>>{};
  index_and_table_columns.reserve(table->num_columns() + 1);
  index_and_table_columns.push_back(std::move(row_index_column));
  auto table_columns = table->release();
  index_and_table_columns.insert(index_and_table_columns.end(),
                                 std::make_move_iterator(table_columns.begin()),
                                 std::make_move_iterator(table_columns.end()));

  // Table with the index and table columns
  auto table_with_index = std::make_unique<cudf::table>(std::move(index_and_table_columns));

  // Likewise, prepend the `index` column metadata to the schema information
  auto updated_schema_info = std::vector<cudf::io::column_name_info>{};
  updated_schema_info.reserve(metadata.schema_info.size() + 1);
  updated_schema_info.emplace_back("index");
  updated_schema_info.insert(updated_schema_info.end(),
                             std::make_move_iterator(metadata.schema_info.begin()),
                             std::make_move_iterator(metadata.schema_info.end()));
  metadata.schema_info = std::move(updated_schema_info);

  // Roaring bitmap is nullptr or empty, return early
  if (not deletion_vector or not roaring::api::roaring64_bitmap_get_cardinality(deletion_vector)) {
    return table_with_metadata{std::move(table_with_index), std::move(metadata)};
  }

  // Build the row mask column using row indices and the deletion vector
  auto row_mask_column = build_row_mask_column(
    row_indices, deletion_vector, num_rows, stream, cudf::get_current_device_resource_ref());

  // Apply the row mask to the table and return the resultant table
  return table_with_metadata{
    cudf::apply_boolean_mask(table_with_index->view(), row_mask_column->view(), stream, mr),
    std::move(metadata)};
}

table_with_metadata read_parquet_and_apply_deletion_vector_gpu(
  parquet_reader_options const& options,
  cudf::host_span<cuda::std::byte const> serialized_roaring64_bytes,
  cudf::host_span<size_t const> row_group_offsets,
  cudf::host_span<size_type const> row_group_num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(row_group_offsets.size() == row_group_num_rows.size(),
               "Encountered a mismatch in the size of row group offsets and row counts vectors");
  CUDF_EXPECTS(
    not options.get_filter().has_value(),
    "Encountered a non-null AST filter expression. Use a roaring64 bitmap deletion vector to "
    "filter the table instead");

  // Read the parquet table
  auto [table, metadata] = cudf::io::read_parquet(options, stream, mr);
  auto const num_rows    = table->num_rows();

  CUDF_EXPECTS(row_group_num_rows.empty() or
                 std::cmp_equal(metadata.num_input_row_groups, row_group_num_rows.size()),
               "Encountered a mismatch in the number of input row groups and row counts vector");

  // Compute a row index host vector from the specified row group offsets and counts
  auto row_indices =
    compute_row_indices_gpu(row_group_offsets, row_group_num_rows, num_rows, stream);

  if (serialized_roaring64_bytes.empty()) {
    // Build and prepend the index column to the table columns
    auto row_index_column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT64},
                                                           num_rows,
                                                           std::move(row_indices),
                                                           rmm::device_buffer{},
                                                           0);
    auto index_and_table_columns = std::vector<std::unique_ptr<cudf::column>>{};
    index_and_table_columns.reserve(table->num_columns() + 1);
    index_and_table_columns.push_back(std::move(row_index_column));
    auto table_columns = table->release();
    index_and_table_columns.insert(index_and_table_columns.end(),
                                   std::make_move_iterator(table_columns.begin()),
                                   std::make_move_iterator(table_columns.end()));

    // Table with the index and table columns
    auto table_with_index = std::make_unique<cudf::table>(std::move(index_and_table_columns));

    // Likewise, prepend the `index` column metadata to the schema information
    auto updated_schema_info = std::vector<cudf::io::column_name_info>{};
    updated_schema_info.reserve(metadata.schema_info.size() + 1);
    updated_schema_info.emplace_back("index");
    updated_schema_info.insert(updated_schema_info.end(),
                               std::make_move_iterator(metadata.schema_info.begin()),
                               std::make_move_iterator(metadata.schema_info.end()));
    metadata.schema_info = std::move(updated_schema_info);

    return table_with_metadata{std::move(table_with_index), std::move(metadata)};
  } else {
    // Build the row mask column using row indices and the deletion vector
    auto row_mask_column = build_row_mask_column_gpu(
      cudf::device_span<size_t const>{static_cast<size_t*>(row_indices.data()),
                                      static_cast<size_t>(num_rows)},
      serialized_roaring64_bytes,
      num_rows,
      stream,
      cudf::get_current_device_resource_ref());

    // Build and prepend the index column to the table columns
    auto row_index_column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT64},
                                                           num_rows,
                                                           std::move(row_indices),
                                                           rmm::device_buffer{},
                                                           0);
    auto index_and_table_columns = std::vector<std::unique_ptr<cudf::column>>{};
    index_and_table_columns.reserve(table->num_columns() + 1);
    index_and_table_columns.push_back(std::move(row_index_column));
    auto table_columns = table->release();
    index_and_table_columns.insert(index_and_table_columns.end(),
                                   std::make_move_iterator(table_columns.begin()),
                                   std::make_move_iterator(table_columns.end()));

    // Table with the index and table columns
    auto table_with_index = std::make_unique<cudf::table>(std::move(index_and_table_columns));

    // Likewise, prepend the `index` column metadata to the schema information
    auto updated_schema_info = std::vector<cudf::io::column_name_info>{};
    updated_schema_info.reserve(metadata.schema_info.size() + 1);
    updated_schema_info.emplace_back("index");
    updated_schema_info.insert(updated_schema_info.end(),
                               std::make_move_iterator(metadata.schema_info.begin()),
                               std::make_move_iterator(metadata.schema_info.end()));
    metadata.schema_info = std::move(updated_schema_info);

    // Apply the row mask to the table and return the resultant table
    return table_with_metadata{
      cudf::apply_boolean_mask(table_with_index->view(), row_mask_column->view(), stream, mr),
      std::move(metadata)};
  }
}

}  // namespace cudf::io::parquet::experimental
