/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/io/experimental/deletion_vectors.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/roaring_bitmap.cuh>
#include <cuda/std/cstdint>

#include <functional>
#include <memory>
#include <optional>
#include <queue>

namespace cudf::io::parquet::experimental {

// Type alias for the cuco 64-bit roaring bitmap
using roaring_bitmap_type =
  cuco::experimental::roaring_bitmap<cuda::std::uint64_t, rmm::mr::polymorphic_allocator<char>>;

/**
 * @brief Opaque wrapper class for cuco's 64-bit roaring bitmap
 */
struct chunked_parquet_reader::roaring_bitmap_impl {
  /// Unique pointer to the roaring bitmap
  std::unique_ptr<roaring_bitmap_type> roaring_bitmap;
  /// Host span of the serialized roaring bitmap data
  cudf::host_span<cuda::std::byte const> const roaring_bitmap_data;

  explicit roaring_bitmap_impl(
    cudf::host_span<cuda::std::byte const> const& serialized_roaring_bitmap)
    : roaring_bitmap_data{serialized_roaring_bitmap}
  {
  }

  roaring_bitmap_impl(roaring_bitmap_impl&&)      = default;
  roaring_bitmap_impl(roaring_bitmap_impl const&) = delete;

  /**
   * @brief Constructs a roaring bitmap from the serialized data
   *
   * @param allocator Memory allocator
   * @param stream CUDA stream to launch the query kernel
   */
  void construct_roaring_bitmap(rmm::mr::polymorphic_allocator<char> const& allocator,
                                rmm::cuda_stream_view stream)
  {
    if (roaring_bitmap == nullptr) {
      CUDF_EXPECTS(not roaring_bitmap_data.empty(),
                   "Encountered empty data while constructing roaring bitmap");
      roaring_bitmap = std::make_unique<roaring_bitmap_type>(
        static_cast<cuda::std::byte const*>(roaring_bitmap_data.data()), allocator, stream);
    }
  }
};

/**
 * @brief Prepends the index column information to the table metadata
 *
 * @param metadata Table metadata
 */
void prepend_index_column_to_table_metadata(table_metadata& metadata);

/**
 * @brief Prepends the index column to the table columns
 *
 * @param table Table
 * @param row_index_column Row index column
 * @return A unique pointer to the prepended table
 */
std::unique_ptr<cudf::table> prepend_index_column_to_table(
  std::unique_ptr<cudf::table>&& table, std::unique_ptr<cudf::column>&& row_index_column);

/**
 * @brief Computes a row index column from the specified row group row offsets and counts
 *
 * @param row_group_offsets Row group offsets
 * @param row_group_num_rows Row group row counts
 * @param start_row Start row
 * @param num_rows Number of rows
 * @param stream CUDA stream to launch the query kernel
 * @param mr Memory resource to allocate the output column's memory
 * @return A unique pointer to the computed row index column
 */
std::unique_ptr<cudf::column> compute_row_index_column(
  cudf::host_span<size_t const> row_group_offsets,
  cudf::host_span<size_type const> row_group_num_rows,
  std::optional<size_t> start_row,
  size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Computes a chunk of the row index column from the specified row group offsets and counts
 */
std::unique_ptr<cudf::column> compute_partial_row_index_column(
  std::queue<size_t>& row_group_offsets,
  std::queue<size_type>& row_group_num_rows,
  size_t start_row,
  size_type num_rows,
  bool is_unspecified_row_group_data,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Computes a BOOL8 row mask column from the specified row index column and deletion vectors
 *
 * @param row_index_column Row index column
 * @param deletion_vector_refs Deletion vector refs
 * @param rows_per_deletion_vector Rows per deletion vector
 * @param stream CUDA stream to launch the query kernel
 * @param mr Memory resource to allocate the output column's memory
 * @return A unique pointer to the computed BOOL8 row mask column
 */
std::unique_ptr<cudf::column> compute_row_mask_column(
  cudf::column_view const& row_index_column,
  cudf::host_span<std::reference_wrapper<roaring_bitmap_type> const> deletion_vector_refs,
  cudf::host_span<size_type const> rows_per_deletion_vector,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Computes the number of rows deleted by the deletion vectors
 *
 * @param row_index_column Row index column
 * @param deletion_vector_refs Deletion vector refs
 * @param deletion_vector_row_counts Rows per deletion vector
 * @param stream CUDA stream to launch the query kernel
 * @return The number of deleted rows
 */
size_t compute_deleted_row_count(
  cudf::column_view const& row_index_column,
  cudf::host_span<std::reference_wrapper<roaring_bitmap_type> const> deletion_vector_refs,
  cudf::host_span<size_type const> deletion_vector_row_counts,
  rmm::cuda_stream_view stream);

/**
 * @brief Computes a chunk of the BOOL8 row mask column from the row index column and the deletion
 * vectors
 *
 * @param row_index_column Row index column
 * @param deletion_vectors Deletion vectors
 * @param deletion_vector_row_counts Rows per deletion vector
 * @param stream CUDA stream to launch the query kernel
 * @param mr Memory resource to allocate the output column's memory
 * @return A unique pointer to the computed BOOL8 row mask column
 */
std::unique_ptr<cudf::column> compute_partial_row_mask_column(
  cudf::column_view const& row_index_column,
  std::queue<chunked_parquet_reader::roaring_bitmap_impl>& deletion_vectors,
  std::queue<size_type>& deletion_vector_row_counts,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Computes the number of deleted rows for a chunk of the row index column
 *
 * @param row_index_column Row index column
 * @param deletion_vectors Deletion vectors
 * @param deletion_vector_row_counts Rows per deletion vector
 * @param stream CUDA stream to launch the query kernel
 * @return The number of deleted rows
 */
size_t compute_partial_deleted_row_count(
  cudf::column_view const& row_index_column,
  std::queue<chunked_parquet_reader::roaring_bitmap_impl>& deletion_vectors,
  std::queue<size_type>& deletion_vector_row_counts,
  rmm::cuda_stream_view stream);

}  // namespace cudf::io::parquet::experimental
