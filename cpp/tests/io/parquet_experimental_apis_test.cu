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

#include "cudf/detail/iterator.cuh"
#include "parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>

#include <roaring/roaring64.h>

#include <functional>
#include <numeric>

// Base test fixture for tests
struct ParquetExperimentalApisTest : public cudf::test::BaseFixture {};

using namespace roaring;

namespace {

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
  // Total number of row groups
  auto const num_row_groups = static_cast<cudf::size_type>(row_group_num_rows.size());

  // Row span offsets for each row group
  auto row_group_span_offsets = std::vector<cudf::size_type>(num_row_groups + 1);
  row_group_span_offsets[0]   = 0;
  std::inclusive_scan(
    row_group_num_rows.begin(), row_group_num_rows.end(), row_group_span_offsets.begin() + 1);

  // Host vector to store expected row indices data
  auto expected_row_indices = thrust::host_vector<size_t>(num_rows);
  // Initialize all row indices to 1 so that we can do a segmented inclusive scan
  std::fill(expected_row_indices.begin(), expected_row_indices.end(), 1);

  // Scatter row group offsets to their corresponding row group span offsets
  thrust::scatter(thrust::host,
                  row_group_offsets.begin(),
                  row_group_offsets.end(),
                  row_group_span_offsets.begin(),
                  expected_row_indices.begin());

  // Inclusive scan each row group span to compute the rest of the row indices
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
 * @brief Builds a roaring64 deletion vector and a (host) row mask vector based on the specified
 * probability of a row being deleted
 *
 * @param num_rows Number of rows in the table
 * @param deletion_probability The probability of a row being deleted
 *
 * @return A pair of a deletion vector and a host row mask vector
 */
auto build_deletion_vector(cudf::size_type num_rows,
                           float deletion_probability,
                           cudf::host_span<size_t const> row_indices = {})
{
  static constexpr auto seed = 0xbaLL;
  std::mt19937 engine{seed};
  std::bernoulli_distribution dist(deletion_probability);

  auto expected_row_mask = thrust::host_vector<bool>(num_rows);
  std::generate(expected_row_mask.begin(), expected_row_mask.end(), [&]() { return dist(engine); });

  auto* deletion_vector = roaring64_bitmap_create();

  // Context for the roaring64 bitmap for faster (bulk) add operations
  auto roaring64_context =
    roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0}, .leaf = nullptr};

  std::for_each(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator<size_t>(num_rows),
    [&](auto row_idx) {
      // Insert provided host row index (or just the `row_idx` if row_indices is
      // empty) if the row is deleted in the row mask
      auto const insert_row_index = row_indices.empty() ? row_idx : row_indices[row_idx];
      if (not expected_row_mask[row_idx]) {
        roaring64_bitmap_add_bulk(deletion_vector, &roaring64_context, insert_row_index);
      }
    });

  return std::make_pair(deletion_vector, expected_row_mask);
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
  auto const num_rows = host_data.size();
  if (num_rows == 0 or host_data.empty()) {
    return std::make_unique<cudf::column>(
      cudf::data_type{data_type}, num_rows, rmm::device_buffer{}, rmm::device_buffer{}, 0);
  }

  rmm::device_buffer buffer{num_rows * sizeof(T), stream, mr};
  cudf::detail::cuda_memcpy_async<T>(
    cudf::device_span<T>{static_cast<T*>(buffer.data()), num_rows}, host_data, stream);
  return std::make_unique<cudf::column>(
    cudf::data_type{data_type}, num_rows, std::move(buffer), rmm::device_buffer{}, 0);
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
std::unique_ptr<cudf::table> build_expected_table(cudf::table_view const input_table_view,
                                                  cudf::column_view const expected_row_index_column,
                                                  cudf::column_view const row_mask_column,
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

}  // namespace

template <typename T>
struct Roaring64BitmapBasicsTest : public cudf::test::BaseFixture {};

using Roaring64Types = cudf::test::Types<uint32_t, uint64_t>;

TYPED_TEST_SUITE(Roaring64BitmapBasicsTest, Roaring64Types);

TYPED_TEST(Roaring64BitmapBasicsTest, TestRoaring64BitmapBasics)
{
  auto constexpr num_keys = 100'000;
  using Key               = TypeParam;

  // Host vector of keys
  thrust::host_vector<Key> keys(num_keys);
  std::iota(keys.begin(), keys.end(), Key{0});

  // Host vector of booleans to store the result of bitmap queries
  auto contained = thrust::host_vector<bool>(num_keys, false);

  // Create the roaring64 bitmap
  auto roaring64_bitmap = roaring64_bitmap_create();

  // Test empty bitmap
  {
    auto roaring64_context =
      roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0}, .leaf = nullptr};

    // Query all keys and store results
    std::for_each(thrust::counting_iterator(0), thrust::counting_iterator(num_keys), [&](auto key) {
      contained[key] =
        roaring64_bitmap_contains_bulk(roaring64_bitmap, &roaring64_context, static_cast<Key>(key));
    });

    EXPECT_TRUE(std::none_of(contained.begin(), contained.end(), std::identity{}));
  }

  // Insert all keys into the bitmap and check that all keys are contained
  {
    // Context for bulk add operation
    auto roaring64_context =
      roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0}, .leaf = nullptr};

    // Insert all keys into the bitmap
    std::for_each(thrust::counting_iterator(0), thrust::counting_iterator(num_keys), [&](auto key) {
      roaring64_bitmap_add_bulk(roaring64_bitmap, &roaring64_context, static_cast<Key>(key));
    });

    // Reset the context
    roaring64_context = roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0}, .leaf = nullptr};

    // Query all keys and store results
    std::for_each(thrust::counting_iterator(0), thrust::counting_iterator(num_keys), [&](auto key) {
      contained[key] =
        roaring64_bitmap_contains_bulk(roaring64_bitmap, &roaring64_context, static_cast<Key>(key));
    });

    EXPECT_TRUE(std::all_of(contained.begin(), contained.end(), std::identity{}));
  }

  // Insert only even keys into the bitmap and check that they are contained and the rest are not
  // contained
  {
    // Clear the bitmap
    roaring64_bitmap_clear(roaring64_bitmap);

    // Reset the context
    auto roaring64_context =
      roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0}, .leaf = nullptr};

    auto is_even =
      cudf::detail::make_counting_transform_iterator(0, [](auto const i) { return i % 2 == 0; });

    std::for_each(thrust::counting_iterator(0), thrust::counting_iterator(num_keys), [&](auto key) {
      if (is_even[key]) {
        roaring64_bitmap_add_bulk(roaring64_bitmap, &roaring64_context, static_cast<Key>(key));
      }
    });

    // Reset the context
    roaring64_context = roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0}, .leaf = nullptr};

    // Query all keys and store results
    std::for_each(thrust::counting_iterator(0), thrust::counting_iterator(num_keys), [&](auto key) {
      contained[key] =
        roaring64_bitmap_contains_bulk(roaring64_bitmap, &roaring64_context, static_cast<Key>(key));
    });

    // Check that all even keys are contained
    EXPECT_TRUE(std::all_of(thrust::counting_iterator<cudf::size_type>(0),
                            thrust::counting_iterator<cudf::size_type>(num_keys),
                            [&](auto key) { return contained[key] == is_even[key]; }));

    // Check that all other (odd) keys are not contained
    EXPECT_TRUE(std::all_of(thrust::counting_iterator<cudf::size_type>(0),
                            thrust::counting_iterator<cudf::size_type>(num_keys),
                            [&](auto key) { return contained[key] != not is_even[key]; }));
  }
}

TEST_F(ParquetExperimentalApisTest, TestDeletionVectors)
{
  auto constexpr num_rows             = 100'000;
  auto constexpr num_row_groups       = 5;
  auto constexpr num_columns          = 8;
  auto constexpr include_validity     = false;
  auto constexpr deletion_probability = 0.4;  ///< 40% of the rows are deleted
  auto const stream                   = cudf::get_default_stream();
  auto const mr                       = cudf::get_current_device_resource_ref();

  auto input_table      = create_random_fixed_table<float>(num_columns, num_rows, include_validity);
  auto input_table_view = input_table->view();

  // Write parquet to buffer
  auto buffer = std::vector<char>{};
  {
    auto out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&buffer), input_table_view)
        .row_group_size_rows(num_rows / num_row_groups)
        .build();
    cudf::io::write_parquet(out_opts);
  }

  // Parquet source info
  auto const source_info =
    cudf::io::source_info{cudf::host_span<char const>(buffer.data(), buffer.size())};

  auto test_read_parquet_and_apply_deletion_vector =
    [num_columns, &stream, &mr, &source_info, &input_table_view](
      cudf::host_span<bool const> expected_row_mask,
      roaring64_bitmap_t const* deletion_vector,
      cudf::host_span<size_t const> row_group_offsets,
      cudf::host_span<cudf::size_type const> row_group_num_rows,
      cudf::column_view const expected_row_index_column) {
      // Read parquet from buffer and apply deletion vector
      auto in_opts = cudf::io::parquet_reader_options::builder(source_info).build();
      auto [read_table, _] =
        cudf::io::parquet::experimental::read_parquet_and_apply_deletion_vector(
          in_opts, deletion_vector, row_group_offsets, row_group_num_rows, stream, mr);

      EXPECT_EQ(read_table->num_columns(), num_columns + 1);

      // Build the row mask column
      auto expected_row_mask_column =
        build_column_from_host_data<bool>(expected_row_mask, cudf::type_id::BOOL8, stream, mr);

      // Build the expected table
      auto expected_table = build_expected_table(
        input_table_view, expected_row_index_column, expected_row_mask_column->view(), stream, mr);

      CUDF_TEST_EXPECT_TABLES_EQUAL(read_table->view(), expected_table->view());
    };

  // Test read parquet with a simple row index column and apply deletion vector
  {
    auto expected_row_index_column = [&]() {
      rmm::device_buffer index_column_buffer{num_rows * sizeof(size_t), stream, mr};
      thrust::sequence(rmm::exec_policy_nosync(stream),
                       static_cast<size_t*>(index_column_buffer.data()),
                       static_cast<size_t*>(index_column_buffer.data()) + num_rows,
                       0);
      return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT64},
                                            num_rows,
                                            std::move(index_column_buffer),
                                            rmm::device_buffer{},
                                            cudf::size_type{0});
    }();

    // Construct a roaring64 deletion vector and corresponding row mask
    auto [deletion_vector, expected_row_mask] =
      build_deletion_vector(num_rows, deletion_probability);

    test_read_parquet_and_apply_deletion_vector(
      expected_row_mask, deletion_vector, {}, {}, expected_row_index_column->view());
  }

  // Test read parquet with a custom row index column and apply deletion vector
  {
    std::mt19937 engine{0xf00d};

    // Row index offsets for each row group
    auto row_group_offsets = thrust::host_vector<size_t>{static_cast<size_t>(std::llround(2e9)),
                                                         static_cast<size_t>(std::llround(2.5e9)),
                                                         static_cast<size_t>(std::llround(3e9)),
                                                         static_cast<size_t>(std::llround(3.5e9)),
                                                         static_cast<size_t>(std::llround(4e9))};

    // Split the `num_rows` into `num_row_groups` spans
    auto row_group_splits = std::vector<cudf::size_type>(num_row_groups - 1);
    {
      std::uniform_int_distribution<cudf::size_type> dist{1, num_rows};
      std::generate(
        row_group_splits.begin(), row_group_splits.end(), [&]() { return dist(engine); });
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

    // Build the host vector of expected row indices
    auto expected_row_indices =
      build_expected_row_indices(row_group_offsets, row_group_num_rows, num_rows);

    // Build the expected row index column
    auto expected_row_index_column =
      build_column_from_host_data<size_t>(expected_row_indices, cudf::type_id::UINT64, stream, mr);

    // Construct a roaring64 deletion vector and corresponding row mask
    auto [deletion_vector, expected_row_mask] =
      build_deletion_vector(num_rows, deletion_probability, expected_row_indices);

    test_read_parquet_and_apply_deletion_vector(expected_row_mask,
                                                deletion_vector,
                                                row_group_offsets,
                                                row_group_num_rows,
                                                expected_row_index_column->view());
  }
}