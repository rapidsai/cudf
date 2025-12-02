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
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/roaring_bitmap.cuh>
#include <cuda/functional>
#include <cuda/std/tuple>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>

#include <numeric>
#include <utility>

namespace cudf::io::parquet::experimental {

// Type alias for the cuco 64-bit roaring bitmap
using roaring_bitmap_type =
  cuco::experimental::roaring_bitmap<cuda::std::uint64_t, rmm::mr::polymorphic_allocator<char>>;

namespace {

/**
 * @brief Prepends the index column information to the table metadata
 *
 * @param[in,out] metadata Table metadata
 */
void prepend_index_column_to_table_metadata(table_metadata& metadata)
{
  auto updated_schema_info = std::vector<cudf::io::column_name_info>{};
  updated_schema_info.reserve(metadata.schema_info.size() + 1);
  updated_schema_info.emplace_back("index");
  updated_schema_info.insert(updated_schema_info.end(),
                             std::make_move_iterator(metadata.schema_info.begin()),
                             std::make_move_iterator(metadata.schema_info.end()));
  metadata.schema_info = std::move(updated_schema_info);
}

/**
 * @brief Prepends the index column to the table columns
 *
 * @param table Input table
 * @param row_index_column row index column to prepend
 *
 * @return A table with the index column prepended to the table columns
 */
std::unique_ptr<cudf::table> prepend_index_column_to_table(
  std::unique_ptr<cudf::table>&& table, std::unique_ptr<cudf::column>&& row_index_column)
{
  auto index_and_table_columns = std::vector<std::unique_ptr<cudf::column>>{};
  index_and_table_columns.reserve(table->num_columns() + 1);
  index_and_table_columns.push_back(std::move(row_index_column));
  auto table_columns = table->release();
  index_and_table_columns.insert(index_and_table_columns.end(),
                                 std::make_move_iterator(table_columns.begin()),
                                 std::make_move_iterator(table_columns.end()));
  return std::make_unique<cudf::table>(std::move(index_and_table_columns));
}

/**
 * @brief Computes a row index column from the specified row group row offsets and counts
 *
 * @param row_group_offsets Host span of row offsets of each row group
 * @param row_group_num_rows Host span of number of rows in each row group
 * @param start_row Starting row index if the row group offsets and counts are empty
 * @param num_rows Number of rows in the table
 * @param stream CUDA stream for kernel launches and data transfers
 * @param mr Device memory resource to allocate device memory for the row index column
 *
 * @return UINT64 column containing row indices
 */
std::unique_ptr<cudf::column> compute_row_index_column(
  cudf::host_span<size_t const> row_group_offsets,
  cudf::host_span<size_type const> row_group_num_rows,
  std::optional<size_t> start_row,
  size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_row_groups = static_cast<size_type>(row_group_num_rows.size());

  if (row_group_offsets.empty()) {
    auto row_indices      = rmm::device_buffer(num_rows * sizeof(size_t), stream, mr);
    auto row_indices_iter = static_cast<size_t*>(row_indices.data());
    thrust::sequence(rmm::exec_policy_nosync(stream),
                     row_indices_iter,
                     row_indices_iter + num_rows,
                     start_row.value_or(size_t{0}));
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
 * @brief Computes a chunk of the row index column from the specified row group offsets and counts
 *
 * @param row_group_offsets Queue of row offsets of eachrow group
 * @param row_group_num_rows Queue of number of rows in each row group
 * @param start_row Starting row index of the current table chunk
 * @param num_rows Total number of rows in the current table chunk
 * @param is_unspecified_row_group_data Whether the row group offsets and counts are unspecified
 * @param stream CUDA stream for kernel launches and data transfers
 * @param mr Device memory resource to allocate device memory for the row index column
 *
 * @return UINT64 column containing row indices
 */
std::unique_ptr<cudf::column> compute_partial_row_index_column(
  std::queue<size_t>& row_group_offsets,
  std::queue<size_type>& row_group_num_rows,
  size_t start_row,
  size_type num_rows,
  bool is_unspecified_row_group_data,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Build a simple row index column if the row group offsets and counts are unspecified
  if (is_unspecified_row_group_data) {
    return compute_row_index_column({}, {}, start_row, num_rows, stream, mr);
  }

  // Compute current table chunk's vectors of row group row offsets and counts from the input queues
  std::vector<size_t> row_offsets;
  std::vector<size_type> row_counts;
  size_type rows_filled = 0;
  while (std::cmp_less(rows_filled, num_rows)) {
    CUDF_EXPECTS(
      not(row_group_offsets.empty() or row_group_num_rows.empty()),
      "Unable to compute the row index column from the specified row group offsets and counts");

    // Compute how many rows can be extracted from the current row group
    auto const row_count = std::min<size_type>(num_rows - rows_filled, row_group_num_rows.front());
    row_counts.emplace_back(row_count);
    row_offsets.emplace_back(row_group_offsets.front());

    // If we still have remaining rows in the current row group, update its offset and row count
    if (std::cmp_less(row_count, row_group_num_rows.front())) {
      row_group_offsets.front()  = row_group_offsets.front() + row_count;
      row_group_num_rows.front() = row_group_num_rows.front() - row_count;
    } else {
      // Else if the row group is fully consumed, pop it from the queues
      row_group_offsets.pop();
      row_group_num_rows.pop();
    }

    rows_filled += row_count;
  }

  // Compute the row index column with the computed row group row offsets and counts
  return compute_row_index_column(row_offsets, row_counts, std::nullopt, num_rows, stream, mr);
}

/**
 * @brief Builds a BOOL8 row mask column from the specified host span of row indices and the
 * roaring64 deletion vector
 *
 * @param row_indices Host span of row indices
 * @param deletion_vector The cuco roaring bitmap
 * @param num_rows Number of rows in the column
 * @param stream CUDA stream for kernel launches and data transfers
 * @param mr Device memory resource to allocate device memory for the row mask column
 *
 * @return Unique pointer to the row mask column
 */
std::unique_ptr<cudf::column> build_row_mask_column(cudf::column_view const& row_index_column,
                                                    roaring_bitmap_type const& deletion_vector,
                                                    size_type num_rows,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  auto row_mask = rmm::device_buffer(num_rows, stream, mr);

  // Iterator to negate and store the output value from `contains_async`
  auto row_mask_iter = thrust::make_transform_output_iterator(
    static_cast<bool*>(row_mask.data()), [] __device__(auto b) { return not b; });
  deletion_vector.contains_async(
    row_index_column.begin<size_t>(), row_index_column.end<size_t>(), row_mask_iter, stream);

  return std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::BOOL8}, num_rows, std::move(row_mask), rmm::device_buffer{}, 0);
}

}  // namespace

/**
 * @copydoc
 * cudf::io::parquet::experimental::chunked_parquet_reader::chunked_parquet_reader
 */
chunked_parquet_reader::chunked_parquet_reader(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  parquet_reader_options const& options,
  cudf::host_span<cuda::std::byte const> serialized_roaring64,
  cudf::host_span<size_t const> row_group_offsets,
  cudf::host_span<size_type const> row_group_num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
  : _start_row{0},
    _is_unspecified_row_group_data{row_group_offsets.empty()},
    _stream{stream},
    _mr{mr},
    // Use default mr for the internal chunked reader and row index column if we will
    // be applying the deletion vector to produce the output table
    _table_mr{serialized_roaring64.empty() ? mr : rmm::mr::get_current_device_resource_ref()}
{
  CUDF_EXPECTS(
    row_group_offsets.size() == row_group_num_rows.size(),
    "Encountered a mismatch in the number of row group offsets and row group row counts");
  CUDF_EXPECTS(
    not options.get_filter().has_value(),
    "Encountered a non-empty AST filter expression. Use a roaring64 bitmap deletion vector to "
    "filter the table instead");

  // Initialize the internal chunked parquet reader
  _reader = std::make_unique<cudf::io::chunked_parquet_reader>(
    chunk_read_limit, pass_read_limit, options, _stream, _table_mr);

  auto iter = thrust::make_zip_iterator(row_group_offsets.begin(), row_group_num_rows.begin());
  std::for_each(iter, iter + row_group_offsets.size(), [&](auto const& elem) {
    _row_group_row_offsets.push(cuda::std::get<0>(elem));
    _row_group_row_counts.push(cuda::std::get<1>(elem));
  });

  if (not serialized_roaring64.empty()) {
    _deletion_vector = std::make_unique<roaring_bitmap_impl>(
      serialized_roaring64.data(), rmm::mr::polymorphic_allocator<char>{}, _stream);
  }
}

/**
 * @brief Opaque wrapper class for cuco's 64-bit roaring bitmap
 */
struct chunked_parquet_reader::roaring_bitmap_impl {
  roaring_bitmap_type roaring_bitmap;
  roaring_bitmap_impl(cuda::std::byte const* const serialized_roaring64_data,
                      rmm::mr::polymorphic_allocator<char> const& allocator,
                      rmm::cuda_stream_view stream)
    : roaring_bitmap(serialized_roaring64_data, allocator, stream)
  {
  }
};

/**
 * @copydoc
 * cudf::io::parquet::experimental::chunked_parquet_reader::chunked_parquet_reader
 */
chunked_parquet_reader::chunked_parquet_reader(
  std::size_t chunk_read_limit,
  parquet_reader_options const& options,
  cudf::host_span<cuda::std::byte const> serialized_roaring64,
  cudf::host_span<size_t const> row_group_offsets,
  cudf::host_span<size_type const> row_group_num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
  : chunked_parquet_reader(chunk_read_limit,
                           std::size_t{0},
                           options,
                           serialized_roaring64,
                           row_group_offsets,
                           row_group_num_rows,
                           stream,
                           mr)
{
}

/**
 * @copydoc cudf::io::parquet::experimental::chunked_parquet_reader::~chunked_parquet_reader
 */
chunked_parquet_reader::~chunked_parquet_reader() = default;

/**
 * @copydoc cudf::io::parquet::experimental::chunked_parquet_reader::has_next
 */
bool chunked_parquet_reader::has_next() const { return _reader->has_next(); }

/**
 * @copydoc cudf::io::parquet::experimental::chunked_parquet_reader::read_chunk
 */
table_with_metadata chunked_parquet_reader::read_chunk()
{
  // Read a chunk of the parquet table
  auto [table, metadata] = _reader->read_chunk();
  auto const num_rows    = table->num_rows();

  // Compute a chunk of the row index column from the specified row group offsets and counts
  auto row_index_column = compute_partial_row_index_column(_row_group_row_offsets,
                                                           _row_group_row_counts,
                                                           _start_row,
                                                           num_rows,
                                                           _is_unspecified_row_group_data,
                                                           _stream,
                                                           _table_mr);
  // Update the start row index for the next chunk
  _start_row += num_rows;

  // Prepend row index column to the table columns
  auto table_with_index =
    prepend_index_column_to_table(std::move(table), std::move(row_index_column));

  // Also prepend the row index column's metadata to the table schema
  prepend_index_column_to_table_metadata(metadata);

  // Return early if deletion vector is not present
  if (not _deletion_vector) {
    return table_with_metadata{std::move(table_with_index), std::move(metadata)};
  }

  // Filter the table using the deletion vector
  auto row_mask = build_row_mask_column(table_with_index->get_column(0).view(),
                                        _deletion_vector->roaring_bitmap,
                                        num_rows,
                                        _stream,
                                        cudf::get_current_device_resource_ref());
  return table_with_metadata{
    // Supply user-provided mr to apply_boolean_mask to allocate output table's memory
    cudf::apply_boolean_mask(table_with_index->view(), row_mask->view(), _stream, _mr),
    std::move(metadata)};
}

/**
 * @copydoc cudf::io::parquet::experimental::read_parquet
 */
table_with_metadata read_parquet(parquet_reader_options const& options,
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
  auto row_index_column = compute_row_index_column(
    row_group_offsets, row_group_num_rows, std::nullopt, num_rows, stream, table_mr);

  // Prepend row index column to the table columns
  auto table_with_index =
    prepend_index_column_to_table(std::move(table), std::move(row_index_column));

  // Also prepend the row index column's metadata to the table schema
  prepend_index_column_to_table_metadata(metadata);

  // Return early if roaring64 data is empty
  if (serialized_roaring64.empty()) {
    return table_with_metadata{std::move(table_with_index), std::move(metadata)};
  }

  // Filter the table using the deletion vector
  auto deletion_vector = roaring_bitmap_type{
    serialized_roaring64.data(), rmm::mr::polymorphic_allocator<char>{}, stream};
  auto row_mask = build_row_mask_column(table_with_index->get_column(0).view(),
                                        deletion_vector,
                                        num_rows,
                                        stream,
                                        cudf::get_current_device_resource_ref());
  return table_with_metadata{
    // Supply user-provided mr to apply_boolean_mask to allocate output table's memory
    cudf::apply_boolean_mask(table_with_index->view(), row_mask->view(), stream, mr),
    std::move(metadata)};
}

}  // namespace cudf::io::parquet::experimental
