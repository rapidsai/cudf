/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/io/experimental/deletion_vectors.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuco/roaring_bitmap.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <roaring/roaring.h>
#include <roaring/roaring64.h>

#include <utility>

namespace {

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
  CUDF_EXPECTS(not host_data.empty(), "Host data vector must not be empty");

  auto const num_rows = host_data.size();
  rmm::device_buffer buffer{num_rows * sizeof(T), stream, mr};
  cudf::detail::cuda_memcpy_async<T>(
    cudf::device_span<T>{static_cast<T*>(buffer.data()), num_rows}, host_data, stream);
  return std::make_unique<cudf::column>(
    cudf::data_type{data_type}, num_rows, std::move(buffer), rmm::device_buffer{}, 0);
}

/**
 * @brief Builds a host vector of expected row indices from the specified row group offsets and
 * row counts
 *
 * @param row_group_offsets Row group offsets
 * @param row_group_num_rows Number of rows in each row group
 * @param num_rows Total number of table rows
 *
 * @return Host vector of expected row indices
 */
auto build_expected_row_indices(cudf::host_span<size_t const> row_group_offsets,
                                cudf::host_span<cudf::size_type const> row_group_num_rows,
                                cudf::size_type num_rows)
{
  auto const num_row_groups = static_cast<cudf::size_type>(row_group_num_rows.size());

  // Row span offsets
  auto row_group_span_offsets = std::vector<cudf::size_type>(num_row_groups + 1);
  row_group_span_offsets[0]   = 0;
  std::inclusive_scan(
    row_group_num_rows.begin(), row_group_num_rows.end(), row_group_span_offsets.begin() + 1);

  // Expected row indices data
  auto expected_row_indices = std::vector<size_t>(num_rows);
  // Initialize all row indices to 1 so that we can do a segmented inclusive scan
  std::fill(expected_row_indices.begin(), expected_row_indices.end(), 1);

  // Scatter row group offsets to expected row indices
  thrust::scatter(thrust::host,
                  row_group_offsets.begin(),
                  row_group_offsets.end(),
                  row_group_span_offsets.begin(),
                  expected_row_indices.begin());

  // Inclusive scan to compute the rest of the row indices
  std::for_each(
    thrust::counting_iterator(0), thrust::counting_iterator(num_row_groups), [&](auto i) {
      auto start_row_index = row_group_span_offsets[i];
      auto end_row_index   = row_group_span_offsets[i + 1];
      std::inclusive_scan(expected_row_indices.begin() + start_row_index,
                          expected_row_indices.begin() + end_row_index,
                          expected_row_indices.begin() + start_row_index);
    });

  return expected_row_indices;
}

/**
 * @brief Serializes a roaring64 bitmap to a host vector of std::bytes
 *
 * @param deletion_vector Pointer to the roaring64 bitmap to serialize
 *
 * @return Host vector of bytes containing the serialized roaring64 bitmap
 */
auto serialize_deletion_vector(roaring::api::roaring64_bitmap_t const* deletion_vector)
{
  auto const num_bytes = roaring::api::roaring64_bitmap_portable_size_in_bytes(deletion_vector);
  EXPECT_GT(num_bytes, 0);
  auto serialized_bitmap   = std::vector<cuda::std::byte>(num_bytes);
  auto const bytes_written = roaring::api::roaring64_bitmap_portable_serialize(
    deletion_vector, reinterpret_cast<char*>(serialized_bitmap.data()));
  EXPECT_EQ(bytes_written, num_bytes);
  auto num_buckets = uint64_t{0};
  std::memcpy(
    &num_buckets, reinterpret_cast<uint64_t*>(serialized_bitmap.data()), sizeof(uint64_t));
  EXPECT_LT(num_buckets, std::numeric_limits<cudf::size_type>::max());
  return serialized_bitmap;
}

/**
 * @brief Builds a roaring64 deletion vector and a (host) row mask vector based on the specified
 * probability of a row being deleted
 *
 * @param num_rows Number of rows in the table
 * @param deletion_probability The probability of a row being deleted
 * @param row_indices Host vector of row indices
 *
 * @return A pair of a deletion vector and a host row mask vector
 */
auto build_deletion_vector_and_expected_row_mask(cudf::size_type num_rows,
                                                 float deletion_probability,
                                                 cudf::host_span<size_t const> row_indices,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  static constexpr auto seed = 0xbaLL;
  std::mt19937 engine{seed};
  std::bernoulli_distribution dist(deletion_probability);

  CUDF_EXPECTS(std::cmp_equal(row_indices.size(), num_rows),
               "Row indices vector must have the same number of rows as the table");

  auto expected_row_mask = thrust::host_vector<bool>(num_rows);
  std::generate(expected_row_mask.begin(), expected_row_mask.end(), [&]() { return dist(engine); });

  auto deletion_vector = roaring::api::roaring64_bitmap_create();

  // Context for the roaring64 bitmap for faster (bulk) add operations
  auto roaring64_context =
    roaring::api::roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0}, .leaf = nullptr};

  std::for_each(thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator<size_t>(num_rows),
                [&](auto row_idx) {
                  // Insert provided host row index if the row is deleted in the row mask
                  if (not expected_row_mask[row_idx]) {
                    roaring::api::roaring64_bitmap_add_bulk(
                      deletion_vector, &roaring64_context, row_indices[row_idx]);
                  }
                });

  auto serialized_roaring_bitmap = serialize_deletion_vector(deletion_vector);
  roaring::api::roaring64_bitmap_free(deletion_vector);

  return std::make_pair(
    std::move(serialized_roaring_bitmap),
    build_column_from_host_data<bool>(expected_row_mask, cudf::type_id::BOOL8, stream, mr));
}

/**
 * @brief Builds the expected cudf table by prepending the input table columns with the specified
 * row index column and applying the specified row mask column
 *
 * @param input_table_view Input table view
 * @param expected_row_index_column Expected row index column view
 * @param row_mask_column Row mask column view
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory for the returned table
 *
 * @return Unique pointer to the expected table
 */
std::unique_ptr<cudf::table> build_expected_table(
  cudf::table_view const& input_table_view,
  cudf::column_view const& expected_row_index_column,
  cudf::column_view const& row_mask_column,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_rows = input_table_view.num_rows();

  CUDF_EXPECTS(expected_row_index_column.size() == num_rows,
               "Index column and table must have the same number of rows");
  CUDF_EXPECTS(row_mask_column.size() == num_rows,
               "Row mask column and table must have the same number of rows");

  auto index_and_columns = std::vector<cudf::column_view>{};
  index_and_columns.reserve(input_table_view.num_columns() + 1);
  index_and_columns.push_back(expected_row_index_column);
  std::transform(thrust::counting_iterator(0),
                 thrust::counting_iterator(input_table_view.num_columns()),
                 std::back_inserter(index_and_columns),
                 [&](auto col_idx) { return input_table_view.column(col_idx); });
  return cudf::apply_boolean_mask(cudf::table_view{index_and_columns}, row_mask_column, stream, mr);
}

/**
 * @brief Constructs a resultant table by applying the specified deletion vector to the input table
 * and compares it to the table read via the `cudf::io::parquet::experimental::read_parquet` API as
 * well as the `cudf::io::parquet::experimental::chunked_parquet_reader`
 *
 * @param parquet_buffer Span of host buffer containing Parquet data
 * @param deletion_vector_info Information about the deletion vectors and the index column
 * @param input_table_view Input table view
 * @param expected_row_mask Host span of expected row mask
 * @param expected_row_index_column Column view of the expected row index column in the read table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory for the returned table
 */
template <cudf::size_type num_concat = 1>
void test_read_parquet_and_apply_deletion_vector(
  cudf::host_span<char const> parquet_buffer,
  cudf::io::parquet::experimental::deletion_vector_info const& deletion_vector_info,
  cudf::table_view const& input_table_view,
  cudf::column_view const& expected_row_mask_column,
  cudf::column_view const& expected_row_index_column,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  static_assert(std::cmp_greater_equal(num_concat, 1),
                "num_concat must be greater than or equal to 1");

  CUDF_EXPECTS(num_concat == 1 or not deletion_vector_info.serialized_roaring_bitmaps.empty(),
               "serialized_roaring_bitmaps must be non-empty if num_concat is greater than 1");

  auto const num_input_rows = input_table_view.num_rows();

  // Initialize a deletion vector information struct
  auto final_deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{
    .serialized_roaring_bitmaps = deletion_vector_info.serialized_roaring_bitmaps,
    .deletion_vector_row_counts = std::vector(num_concat, input_table_view.num_rows()),
    .row_group_offsets          = {},
    .row_group_num_rows         = {},
  };

  // Vector to hold the Parquet buffer spans
  auto parquet_buffers = std::vector<cudf::host_span<char const>>(num_concat, parquet_buffer);

  // Fill in row index column information
  auto& row_group_row_offsets = final_deletion_vector_info.row_group_offsets;
  auto& row_group_num_counts  = final_deletion_vector_info.row_group_num_rows;
  row_group_row_offsets.reserve(num_concat * deletion_vector_info.row_group_offsets.size());
  row_group_num_counts.reserve(num_concat * deletion_vector_info.row_group_num_rows.size());

  for (auto i = 0; i < num_concat; ++i) {
    row_group_row_offsets.insert(row_group_row_offsets.end(),
                                 deletion_vector_info.row_group_offsets.begin(),
                                 deletion_vector_info.row_group_offsets.end());
    row_group_num_counts.insert(row_group_num_counts.end(),
                                deletion_vector_info.row_group_num_rows.begin(),
                                deletion_vector_info.row_group_num_rows.end());
  }

  // Vector to store the extra serialized roaring bitmap data
  auto serialized_roaring_bitmaps_data = std::vector<std::vector<cuda::std::byte>>{};
  serialized_roaring_bitmaps_data.reserve(num_concat - 1);

  // Build the expected table
  auto expected_table = [&]() {
    auto tables      = std::vector<std::unique_ptr<cudf::table>>{};
    auto table_views = std::vector<cudf::table_view>{};
    tables.reserve(num_concat);
    table_views.reserve(num_concat);
    // Insert the first table
    tables.emplace_back(build_expected_table(
      input_table_view, expected_row_index_column, expected_row_mask_column, stream, mr));
    table_views.emplace_back(tables.back()->view());

    if constexpr (num_concat == 1) {
      return std::move(tables.back());
    } else {
      constexpr auto deletion_probability = 0.45;

      for (auto i = 0; i < num_concat - 1; ++i) {
        // Build deletion vector and the expected row mask column
        auto local_expected_row_indices =
          build_expected_row_indices(deletion_vector_info.row_group_offsets,
                                     deletion_vector_info.row_group_num_rows,
                                     num_input_rows);

        auto local_expected_row_index_column = build_column_from_host_data<size_t>(
          local_expected_row_indices, cudf::type_id::UINT64, stream, mr);

        auto [local_deletion_vector, local_expected_row_mask_column] =
          build_deletion_vector_and_expected_row_mask(
            num_input_rows, deletion_probability, local_expected_row_indices, stream, mr);

        // Insert the expected table, the corresponding deletion vector and its data span
        tables.emplace_back(build_expected_table(input_table_view,
                                                 local_expected_row_index_column->view(),
                                                 local_expected_row_mask_column->view(),
                                                 stream,
                                                 mr));
        table_views.emplace_back(tables.back()->view());
        serialized_roaring_bitmaps_data.emplace_back(std::move(local_deletion_vector));
        final_deletion_vector_info.serialized_roaring_bitmaps.emplace_back(
          serialized_roaring_bitmaps_data.back());
      }

      // Final expected table is the concatenation of all the expected tables
      return cudf::concatenate(table_views, stream, mr);
    }
  }();

  // Read parquet and apply deletion vector
  auto const in_opts =
    cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{cudf::host_span<cudf::host_span<char const>>{parquet_buffers}})
      .build();

  auto const table_with_deletion_vector =
    cudf::io::parquet::experimental::read_parquet(in_opts, final_deletion_vector_info, stream, mr)
      .tbl;

  // Check
  CUDF_TEST_EXPECT_TABLES_EQUAL(table_with_deletion_vector->view(), expected_table->view());

  // Read using the chunked reader
  auto const test_chunked_table_with_deletion_vector = [&](size_t chunk_read_limit,
                                                           size_t pass_read_limit) {
    auto const reader = std::make_unique<cudf::io::parquet::experimental::chunked_parquet_reader>(
      chunk_read_limit, pass_read_limit, in_opts, final_deletion_vector_info, stream, mr);

    auto table_chunks      = std::vector<std::unique_ptr<cudf::table>>{};
    auto table_chunk_views = std::vector<cudf::table_view>{};
    while (reader->has_next()) {
      table_chunks.emplace_back(reader->read_chunk().tbl);
      table_chunk_views.emplace_back(table_chunks.back()->view());
    }
    auto chunked_table_with_deletion_vector = cudf::concatenate(table_chunk_views, stream, mr);
    // Check
    CUDF_TEST_EXPECT_TABLES_EQUAL(chunked_table_with_deletion_vector->view(),
                                  expected_table->view());
  };

  test_chunked_table_with_deletion_vector(512, 2048);
  test_chunked_table_with_deletion_vector(1024, 10240);
  test_chunked_table_with_deletion_vector(10240, 102400);
  test_chunked_table_with_deletion_vector(102400, 0);
  test_chunked_table_with_deletion_vector(0, 0);
}

}  // namespace

// Base test fixture for basic roaring bitmap tests
template <typename T>
struct RoaringBitmapBasicsTest : public cudf::test::BaseFixture {};

using RoaringTypes = cudf::test::Types<cuda::std::uint32_t, cuda::std::uint64_t>;

TYPED_TEST_SUITE(RoaringBitmapBasicsTest, RoaringTypes);

TYPED_TEST(RoaringBitmapBasicsTest, BitmapSerialization)
{
  auto constexpr num_keys = 100'000;
  using Key               = TypeParam;

  auto is_even =
    cudf::detail::make_counting_transform_iterator(0, [](auto const i) { return i % 2 == 0; });

  auto serialized_bitmap = std::vector<cuda::std::byte>{};

  if constexpr (std::is_same_v<Key, cuda::std::uint64_t>) {
    auto roaring64_bitmap = roaring::api::roaring64_bitmap_create();

    // Context for the roaring64 bitmap for faster (bulk) add operations
    auto roaring64_context =
      roaring::api::roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0}, .leaf = nullptr};

    std::for_each(
      thrust::counting_iterator<Key>(0), thrust::counting_iterator<Key>(num_keys), [&](auto key) {
        if (is_even[key]) {
          roaring::api::roaring64_bitmap_add_bulk(roaring64_bitmap, &roaring64_context, key);
        }
      });

    // Serialize and free the bitmap
    auto const serialized_bitmap_size =
      roaring::api::roaring64_bitmap_portable_size_in_bytes(roaring64_bitmap);
    EXPECT_GT(serialized_bitmap_size, 0);

    serialized_bitmap.resize(serialized_bitmap_size);
    std::ignore = roaring::api::roaring64_bitmap_portable_serialize(
      roaring64_bitmap, reinterpret_cast<char*>(serialized_bitmap.data()));

    roaring::api::roaring64_bitmap_free(roaring64_bitmap);
  } else if constexpr (std::is_same_v<Key, cuda::std::uint32_t>) {
    auto roaring_bitmap = roaring::api::roaring_bitmap_create();

    // Context for the roaring64 bitmap for faster (bulk) add operations
    auto roaring_context = roaring::api::roaring_bulk_context_t{};

    std::for_each(
      thrust::counting_iterator<Key>(0), thrust::counting_iterator<Key>(num_keys), [&](auto key) {
        if (is_even[key]) {
          roaring::api::roaring_bitmap_add_bulk(roaring_bitmap, &roaring_context, key);
        }
      });

    // Serialize and free the bitmap
    auto const serialized_bitmap_size =
      roaring::api::roaring_bitmap_portable_size_in_bytes(roaring_bitmap);
    EXPECT_GT(serialized_bitmap_size, 0);

    serialized_bitmap.resize(serialized_bitmap_size);
    std::ignore = roaring::api::roaring_bitmap_portable_serialize(
      roaring_bitmap, reinterpret_cast<char*>(serialized_bitmap.data()));

    roaring::api::roaring_bitmap_free(roaring_bitmap);

  } else {
    CUDF_FAIL("Roaring bitmap key must be either uint32_t or uint64_t");
  }

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // cuCollections roaring bitmap from the serialized roaring64 bitmap bytes
  auto roaring_bitmap =
    cuco::experimental::roaring_bitmap<Key>(serialized_bitmap.data(), {}, stream);

  // Query the roaring bitmap
  auto contained = rmm::device_uvector<bool>(num_keys, stream, mr);
  roaring_bitmap.contains_async(thrust::counting_iterator<Key>(0),
                                thrust::counting_iterator<Key>(num_keys),
                                contained.data(),
                                stream);
  auto results = cudf::detail::make_host_vector_async(contained, stream);

  // Validate
  stream.synchronize();
  EXPECT_TRUE(std::all_of(thrust::counting_iterator<Key>(0),
                          thrust::counting_iterator<Key>(num_keys),
                          [&](auto key) { return results[key] == is_even[key]; }));
}

// Base test fixture for API tests
struct ParquetDeletionVectorsTest : public cudf::test::BaseFixture {};

TEST_F(ParquetDeletionVectorsTest, NoRowIndexColumn)
{
  auto constexpr num_rows             = 50'000;
  auto constexpr num_row_groups       = 5;
  auto constexpr num_columns          = 8;
  auto constexpr include_validity     = false;
  auto constexpr deletion_probability = 0.5;  ///< 50% of the rows are deleted
  auto constexpr rows_per_row_group   = num_rows / num_row_groups;
  static_assert(num_rows % num_row_groups == 0, "num_rows must be a multiple of num_row_groups");

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto input_table = create_random_fixed_table<float>(num_columns, num_rows, include_validity);

  // Write table to parquet buffer
  auto parquet_buffer = std::vector<char>{};
  {
    auto out_opts = cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&parquet_buffer),
                                                              input_table->view())
                      .row_group_size_rows(rows_per_row_group)
                      .build();
    cudf::io::write_parquet(out_opts);
  }

  // Build the expected row index column
  auto expected_row_indices = thrust::host_vector<size_t>(num_rows);
  std::iota(expected_row_indices.begin(), expected_row_indices.end(), size_t{0});
  auto expected_row_index_column =
    build_column_from_host_data<size_t>(expected_row_indices, cudf::type_id::UINT64, stream, mr);

  // Build deletion vector and the expected row mask column
  auto [deletion_vector, expected_row_mask_column] = build_deletion_vector_and_expected_row_mask(
    num_rows, deletion_probability, expected_row_indices, stream, mr);

  // Use num_concat = 1 here since the row index column is simply a sequence and input table
  // concatenation won't properly reset it.
  auto constexpr num_concat = 1;
  cudf::io::parquet::experimental::deletion_vector_info deletion_vector_info{
    .serialized_roaring_bitmaps = {deletion_vector},
    .deletion_vector_row_counts = {input_table->view().num_rows()}};
  test_read_parquet_and_apply_deletion_vector<num_concat>(parquet_buffer,
                                                          deletion_vector_info,
                                                          input_table->view(),
                                                          expected_row_mask_column->view(),
                                                          expected_row_index_column->view(),
                                                          stream,
                                                          mr);
  // Test no row index column and no deletion vector
  {
    // Build expected row mask column containing all true values
    auto expected_row_mask = thrust::host_vector<bool>(num_rows, true);
    auto expected_row_mask_column =
      build_column_from_host_data<bool>(expected_row_mask, cudf::type_id::BOOL8, stream, mr);
    deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{};
    test_read_parquet_and_apply_deletion_vector<num_concat>(parquet_buffer,
                                                            deletion_vector_info,
                                                            input_table->view(),
                                                            expected_row_mask_column->view(),
                                                            expected_row_index_column->view(),
                                                            stream,
                                                            mr);
  }
}

TEST_F(ParquetDeletionVectorsTest, CustomRowIndexColumn)
{
  auto constexpr num_rows             = 25'000;
  auto constexpr num_row_groups       = 5;
  auto constexpr num_columns          = 4;
  auto constexpr include_validity     = false;
  auto constexpr deletion_probability = 0.4;  ///< 40% of the rows are deleted
  auto constexpr rows_per_row_group   = num_rows / num_row_groups;
  static_assert(num_rows % num_row_groups == 0, "num_rows must be a multiple of num_row_groups");

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto input_table = create_random_fixed_table<float>(num_columns, num_rows, include_validity);

  // Write table to parquet buffer
  auto parquet_buffer = std::vector<char>{};
  {
    auto out_opts = cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&parquet_buffer),
                                                              input_table->view())
                      .row_group_size_rows(rows_per_row_group)
                      .build();
    cudf::io::write_parquet(out_opts);
  }

  // Row offsets for each row group - arbitrary, only used to build the UINT64 `index` column
  auto row_group_offsets = std::vector<size_t>(num_row_groups);
  row_group_offsets[0]   = static_cast<size_t>(std::llround(1e9));
  std::transform(
    thrust::counting_iterator(1),
    thrust::counting_iterator(num_row_groups),
    row_group_offsets.begin() + 1,
    [&](auto i) { return static_cast<size_t>(std::llround(row_group_offsets[i - 1] + 0.5e9)); });

  // Split the `num_rows` into `num_row_groups` spans
  auto row_group_splits = std::vector<cudf::size_type>(num_row_groups - 1);
  {
    std::mt19937 engine{0xf00d};
    std::uniform_int_distribution<cudf::size_type> dist{1, num_rows};
    std::generate(row_group_splits.begin(), row_group_splits.end(), [&]() { return dist(engine); });
    std::sort(row_group_splits.begin(), row_group_splits.end());
  }

  // Number of rows in each row group span
  auto row_group_num_rows = std::vector<cudf::size_type>{};
  {
    row_group_num_rows.reserve(num_row_groups);
    auto previous_split = cudf::size_type{0};
    std::transform(row_group_splits.begin(),
                   row_group_splits.end(),
                   std::back_inserter(row_group_num_rows),
                   [&](auto current_split) {
                     auto current_split_size = current_split - previous_split;
                     previous_split          = current_split;
                     return current_split_size;
                   });
    row_group_num_rows.push_back(num_rows - row_group_splits.back());
  }

  // Build the expected row index column
  auto expected_row_indices =
    build_expected_row_indices(row_group_offsets, row_group_num_rows, num_rows);
  auto expected_row_index_column =
    build_column_from_host_data<size_t>(expected_row_indices, cudf::type_id::UINT64, stream, mr);

  // Build deletion vector and the expected row mask column
  auto [deletion_vector, expected_row_mask_column] = build_deletion_vector_and_expected_row_mask(
    num_rows, deletion_probability, expected_row_indices, stream, mr);

  // Don't concatenate the input table and test with single deletion vector
  auto deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{
    .serialized_roaring_bitmaps = {deletion_vector},
    .deletion_vector_row_counts = {input_table->view().num_rows()},
    .row_group_offsets          = row_group_offsets,
    .row_group_num_rows         = row_group_num_rows};
  test_read_parquet_and_apply_deletion_vector<1>(parquet_buffer,
                                                 deletion_vector_info,
                                                 input_table->view(),
                                                 expected_row_mask_column->view(),
                                                 expected_row_index_column->view(),
                                                 stream,
                                                 mr);

  // Concatenate input table and test with multiple deletion vectors
  test_read_parquet_and_apply_deletion_vector<4>(parquet_buffer,
                                                 deletion_vector_info,
                                                 input_table->view(),
                                                 expected_row_mask_column->view(),
                                                 expected_row_index_column->view(),
                                                 stream,
                                                 mr);

  // Concatenate input table and test with many deletion vectors (>= stream fork threshold of 8)
  test_read_parquet_and_apply_deletion_vector<8>(parquet_buffer,
                                                 deletion_vector_info,
                                                 input_table->view(),
                                                 expected_row_mask_column->view(),
                                                 expected_row_index_column->view(),
                                                 stream,
                                                 mr);

  // Test custom row index column and no deletion vector
  {
    // Build expected row mask column containing all true values
    auto expected_row_mask = thrust::host_vector<bool>(num_rows, true);
    auto expected_row_mask_column =
      build_column_from_host_data<bool>(expected_row_mask, cudf::type_id::BOOL8, stream, mr);

    auto deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{
      .row_group_offsets = row_group_offsets, .row_group_num_rows = row_group_num_rows};
    test_read_parquet_and_apply_deletion_vector<1>(parquet_buffer,
                                                   deletion_vector_info,
                                                   input_table->view(),
                                                   expected_row_mask_column->view(),
                                                   expected_row_index_column->view(),
                                                   stream,
                                                   mr);
  }
}
