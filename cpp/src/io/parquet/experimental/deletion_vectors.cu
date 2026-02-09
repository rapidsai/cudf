/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/stream_pool.hpp>
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

/**
 * @brief Opaque wrapper class for cuco's 64-bit roaring bitmap
 */
struct chunked_parquet_reader::roaring_bitmap_impl {
  std::unique_ptr<roaring_bitmap_type> roaring_bitmap;
  cudf::host_span<cuda::std::byte const> const roaring_bitmap_data;

  explicit roaring_bitmap_impl(
    cudf::host_span<cuda::std::byte const> const& serialized_roaring_bitmap)
    : roaring_bitmap_data{serialized_roaring_bitmap}
  {
  }

  roaring_bitmap_impl(roaring_bitmap_impl&&)      = default;
  roaring_bitmap_impl(roaring_bitmap_impl const&) = delete;

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

  CUDF_EXPECTS(row_group_span_offsets.back() == num_rows,
               "Encountered a mismatch in the number of rows in the row index column and the "
               "number of rows in the row group(s)");

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
 * @brief Computes a BOOL8 row mask column from the specified row index column and deletion vectors
 *
 * @note This function synchronizes the stream before returning the row mask column
 *
 * @param row_index_column View of the row index column
 * @param deletion_vector_refs Host span of cuco roaring bitmap references
 * @param rows_per_deletion_vector Host span of number of rows per deletion vector
 * @param stream CUDA stream for kernel launches and data transfers
 * @param mr Device memory resource to allocate device memory for the row mask column
 *
 * @return Unique pointer to the row mask column
 */
std::unique_ptr<cudf::column> compute_row_mask_column(
  cudf::column_view const& row_index_column,
  cudf::host_span<std::reference_wrapper<roaring_bitmap_type> const> deletion_vector_refs,
  cudf::host_span<size_type const> rows_per_deletion_vector,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_rows             = row_index_column.size();
  auto const num_deletion_vectors = static_cast<cudf::size_type>(deletion_vector_refs.size());
  auto row_mask                   = rmm::device_buffer(num_rows * sizeof(bool), stream, mr);

  // Iterator to negate and store the output value from `contains_async`
  auto row_mask_iter = thrust::make_transform_output_iterator(
    static_cast<bool*>(row_mask.data()), [] __device__(auto b) { return not b; });

  if (num_deletion_vectors == 1) {
    deletion_vector_refs.front().get().contains(
      row_index_column.begin<size_t>(), row_index_column.end<size_t>(), row_mask_iter, stream);
    return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::BOOL8},
                                          num_rows,
                                          std::move(row_mask),
                                          rmm::device_buffer{},
                                          0);
  }

  auto deletion_vector_row_offsets = std::vector<size_type>(deletion_vector_refs.size() + 1);
  deletion_vector_row_offsets[0]   = 0;
  std::inclusive_scan(rows_per_deletion_vector.begin(),
                      rows_per_deletion_vector.end(),
                      deletion_vector_row_offsets.begin() + 1);

  CUDF_EXPECTS(deletion_vector_row_offsets.back() == num_rows,
               "Encountered a mismatch in the number of rows in the row index column and the "
               "number of rows in the deletion vector(s)");

  // Fork the stream if the number of deletion vectors is greater than the threshold
  constexpr auto stream_fork_threshold = 8;
  if (num_deletion_vectors >= stream_fork_threshold) {
    auto streams = cudf::detail::fork_streams(stream, num_deletion_vectors);
    std::for_each(thrust::counting_iterator(0),
                  thrust::counting_iterator(num_deletion_vectors),
                  [&](auto const dv_idx) {
                    deletion_vector_refs[dv_idx].get().contains_async(
                      row_index_column.begin<size_t>() + deletion_vector_row_offsets[dv_idx],
                      row_index_column.begin<size_t>() + deletion_vector_row_offsets[dv_idx + 1],
                      row_mask_iter + deletion_vector_row_offsets[dv_idx],
                      streams[dv_idx]);
                  });
    cudf::detail::join_streams(streams, stream);
  } else {
    // Otherwise, launch the queries on the same stream
    std::for_each(thrust::counting_iterator(0),
                  thrust::counting_iterator(num_deletion_vectors),
                  [&](auto const dv_idx) {
                    deletion_vector_refs[dv_idx].get().contains_async(
                      row_index_column.begin<size_t>() + deletion_vector_row_offsets[dv_idx],
                      row_index_column.begin<size_t>() + deletion_vector_row_offsets[dv_idx + 1],
                      row_mask_iter + deletion_vector_row_offsets[dv_idx],
                      stream);
                  });
    stream.synchronize();
  }

  return std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::BOOL8}, num_rows, std::move(row_mask), rmm::device_buffer{}, 0);
}

/**
 * @brief Computes a chunk of the BOOL8 row mask column from the row index column and the deletion
 * vectors
 *
 * @param row_index_column View of the row index column
 * @param deletion_vector_refs Queue of roaring bitmap wrappers
 * @param deletion_vector_row_counts Queue of number of rows in eachdeletion vector
 * @param start_row Starting row index of the current table chunk
 * @param stream CUDA stream for kernel launches and data transfers
 * @param mr Device memory resource to allocate device memory for the row mask column
 *
 * @return Unique pointer to the row mask column
 */
std::unique_ptr<cudf::column> compute_partial_row_mask_column(
  cudf::column_view const& row_index_column,
  std::queue<chunked_parquet_reader::roaring_bitmap_impl>& deletion_vectors,
  std::queue<size_type>& deletion_vector_row_counts,
  size_t start_row,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_rows = row_index_column.size();

  std::vector<size_type> row_counts;
  std::vector<chunked_parquet_reader::roaring_bitmap_impl> deletion_vectors_impls;
  std::vector<std::reference_wrapper<roaring_bitmap_type>> deletion_vector_refs;
  size_type rows_filled = 0;

  while (std::cmp_less(rows_filled, num_rows)) {
    CUDF_EXPECTS(
      not(deletion_vector_row_counts.empty() or deletion_vectors.empty()),
      "Encountered insufficient number of deletion vector row counts or deletion vectors: " +
        std::to_string(deletion_vector_row_counts.size()) + " " +
        std::to_string(deletion_vectors.size()));

    // Compute how many rows can be queried from the current roaring bitmap
    auto const row_count =
      std::min<size_type>(num_rows - rows_filled, deletion_vector_row_counts.front());
    row_counts.emplace_back(row_count);

    auto& deletion_vector = deletion_vectors.front();
    deletion_vector.construct_roaring_bitmap(rmm::mr::polymorphic_allocator<char>{}, stream);
    CUDF_EXPECTS(deletion_vector.roaring_bitmap != nullptr, "Failed to construct roaring_bitmap");
    deletion_vector_refs.emplace_back(std::ref(*(deletion_vector.roaring_bitmap)));

    // If we still have remaining rows to query from the current roaring bitmap, update its row
    // count
    if (std::cmp_less(row_count, deletion_vector_row_counts.front())) {
      deletion_vector_row_counts.front() = deletion_vector_row_counts.front() - row_count;
    } else {
      // Else if the deletion vector is fully queried, move it to the temporary vector and pop it
      // from the queue
      deletion_vectors_impls.emplace_back(std::move(deletion_vectors.front()));
      deletion_vectors.pop();
      deletion_vector_row_counts.pop();
    }

    rows_filled += row_count;
  }

  // Compute the row index column with the computed row group row offsets and counts
  return compute_row_mask_column(row_index_column, deletion_vector_refs, row_counts, stream, mr);
}

}  // namespace

/**
 * @copydoc
 * cudf::io::parquet::experimental::chunked_parquet_reader::chunked_parquet_reader
 */
chunked_parquet_reader::chunked_parquet_reader(std::size_t chunk_read_limit,
                                               std::size_t pass_read_limit,
                                               parquet_reader_options const& options,
                                               deletion_vector_info const& deletion_vector_info,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
  : _start_row{0},
    _is_unspecified_row_group_data{deletion_vector_info.row_group_offsets.empty()},
    _stream{stream},
    _mr{mr},
    // Use default mr for the internal chunked reader and row index column if we will
    // be applying the deletion vector to produce the output table
    _table_mr{deletion_vector_info.serialized_roaring_bitmaps.empty()
                ? mr
                : rmm::mr::get_current_device_resource_ref()}
{
  auto const& serialized_roaring_bitmaps = deletion_vector_info.serialized_roaring_bitmaps;
  auto const& deletion_vector_row_counts = deletion_vector_info.deletion_vector_row_counts;
  auto const& row_group_offsets          = deletion_vector_info.row_group_offsets;
  auto const& row_group_num_rows         = deletion_vector_info.row_group_num_rows;

  CUDF_EXPECTS(
    row_group_offsets.size() == row_group_num_rows.size(),
    "Encountered a mismatch in the number of row group offsets and row group row counts");
  CUDF_EXPECTS(
    not options.get_filter().has_value(),
    "Encountered a non-empty AST filter expression. Use a roaring64 bitmap deletion vector to "
    "filter the table instead");
  CUDF_EXPECTS(serialized_roaring_bitmaps.empty() or
                 serialized_roaring_bitmaps.size() == deletion_vector_row_counts.size(),
               "Encountered a mismatch in the number of deletion vectors and the number of rows "
               "per deletion vector");

  // Initialize the internal chunked parquet reader
  _reader = std::make_unique<cudf::io::chunked_parquet_reader>(
    chunk_read_limit, pass_read_limit, options, _stream, _table_mr);

  // Push row group offsets and counts to the internal queues
  if (not row_group_offsets.empty()) {
    auto iter = thrust::make_zip_iterator(row_group_offsets.begin(), row_group_num_rows.begin());
    std::for_each(iter, iter + row_group_offsets.size(), [&](auto const& elem) {
      _row_group_row_offsets.push(cuda::std::get<0>(elem));
      _row_group_row_counts.push(cuda::std::get<1>(elem));
    });
  }

  // Push deletion vector data spans and row counts to the internal queues
  if (not serialized_roaring_bitmaps.empty()) {
    auto iter = thrust::make_zip_iterator(serialized_roaring_bitmaps.begin(),
                                          deletion_vector_row_counts.begin());
    std::for_each(iter, iter + serialized_roaring_bitmaps.size(), [&](auto const& elem) {
      _deletion_vectors.emplace(cuda::std::get<0>(elem));
      _deletion_vector_row_counts.push(cuda::std::get<1>(elem));
    });
  }
}

/**
 * @copydoc
 * cudf::io::parquet::experimental::chunked_parquet_reader::chunked_parquet_reader
 */
chunked_parquet_reader::chunked_parquet_reader(std::size_t chunk_read_limit,
                                               parquet_reader_options const& options,
                                               deletion_vector_info const& deletion_vector_info,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
  : chunked_parquet_reader(
      chunk_read_limit, std::size_t{0}, options, deletion_vector_info, stream, mr)
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
  if (_deletion_vectors.empty()) {
    return table_with_metadata{std::move(table_with_index), std::move(metadata)};
  }

  // Filter the table using the deletion vectors
  auto row_mask = compute_partial_row_mask_column(table_with_index->get_column(0).view(),
                                                  _deletion_vectors,
                                                  _deletion_vector_row_counts,
                                                  _start_row,
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
                                 deletion_vector_info const& deletion_vector_info,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  auto const& serialized_roaring_bitmaps = deletion_vector_info.serialized_roaring_bitmaps;
  auto const& deletion_vector_row_counts = deletion_vector_info.deletion_vector_row_counts;
  auto const& row_group_offsets          = deletion_vector_info.row_group_offsets;
  auto const& row_group_num_rows         = deletion_vector_info.row_group_num_rows;

  CUDF_EXPECTS(
    row_group_offsets.size() == row_group_num_rows.size(),
    "Encountered a mismatch in the number of row group offsets and row group row counts");
  CUDF_EXPECTS(
    not options.get_filter().has_value(),
    "Encountered a non-empty AST filter expression. Use a roaring64 bitmap deletion vector to "
    "filter the table instead");
  CUDF_EXPECTS(serialized_roaring_bitmaps.empty() or
                 serialized_roaring_bitmaps.size() == deletion_vector_row_counts.size(),
               "Encountered a mismatch in the number of deletion vectors and the number of rows "
               "per deletion vector");

  // Use default mr to read parquet table and build row index column if we will be applying the
  // deletion vector to produce a new table later
  auto const table_mr =
    serialized_roaring_bitmaps.empty() ? mr : rmm::mr::get_current_device_resource_ref();

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
  if (serialized_roaring_bitmaps.empty()) {
    return table_with_metadata{std::move(table_with_index), std::move(metadata)};
  }

  // Construct all deletion vectors and store their references
  auto deletion_vectors      = std::vector<roaring_bitmap_type>{};
  auto deletion_vectors_refs = std::vector<std::reference_wrapper<roaring_bitmap_type>>{};
  deletion_vectors.reserve(serialized_roaring_bitmaps.size());
  deletion_vectors_refs.reserve(serialized_roaring_bitmaps.size());
  std::transform(serialized_roaring_bitmaps.begin(),
                 serialized_roaring_bitmaps.end(),
                 std::back_inserter(deletion_vectors_refs),
                 [&](auto const& serialized_roaring_bitmap) {
                   deletion_vectors.emplace_back(serialized_roaring_bitmap.data(),
                                                 rmm::mr::polymorphic_allocator<char>{},
                                                 stream);
                   return std::ref(deletion_vectors.back());
                 });
  // Compute the row mask column from the deletion vectors
  auto row_mask = compute_row_mask_column(table_with_index->get_column(0).view(),
                                          deletion_vectors_refs,
                                          deletion_vector_row_counts,
                                          stream,
                                          cudf::get_current_device_resource_ref());
  // Filter the table using the deletion vector
  return table_with_metadata{
    // Supply user-provided mr to apply_boolean_mask to allocate output table's memory
    cudf::apply_boolean_mask(table_with_index->view(), row_mask->view(), stream, mr),
    std::move(metadata)};
}

}  // namespace cudf::io::parquet::experimental
