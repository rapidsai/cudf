/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/orc_metadata.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <fstream>
#include <type_traits>

namespace {
enum class output_limit : std::size_t {};
enum class input_limit : std::size_t {};
enum class output_row_granularity : cudf::size_type {};

// Global environment for temporary files
auto const temp_env = reinterpret_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

using int32s_col       = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64s_col       = cudf::test::fixed_width_column_wrapper<int64_t>;
using doubles_col      = cudf::test::fixed_width_column_wrapper<double>;
using strings_col      = cudf::test::strings_column_wrapper;
using structs_col      = cudf::test::structs_column_wrapper;
using int32s_lists_col = cudf::test::lists_column_wrapper<int32_t>;

auto write_file(std::vector<std::unique_ptr<cudf::column>>& input_columns,
                std::string const& filename,
                bool nullable                    = false,
                std::size_t stripe_size_bytes    = cudf::io::default_stripe_size_bytes,
                cudf::size_type stripe_size_rows = cudf::io::default_stripe_size_rows)
{
  if (nullable) {
    // Generate deterministic bitmask instead of random bitmask for easy computation of data size.
    auto const valid_iter = cudf::detail::make_counting_transform_iterator(
      0, [](cudf::size_type i) { return i % 4 != 3; });
    cudf::size_type offset{0};
    for (auto& col : input_columns) {
      auto const [null_mask, null_count] =
        cudf::test::detail::make_null_mask(valid_iter + offset, valid_iter + col->size() + offset);
      col = cudf::structs::detail::superimpose_nulls(
        static_cast<cudf::bitmask_type const*>(null_mask.data()),
        null_count,
        std::move(col),
        cudf::get_default_stream(),
        rmm::mr::get_current_device_resource());

      // Shift nulls of the next column by one position, to avoid having all nulls
      // in the same table rows.
      ++offset;
    }
  }

  auto input_table = std::make_unique<cudf::table>(std::move(input_columns));
  auto filepath =
    temp_env->get_temp_filepath(nullable ? filename + "_nullable.orc" : filename + ".orc");

  auto const write_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, *input_table)
      .stripe_size_bytes(stripe_size_bytes)
      .stripe_size_rows(stripe_size_rows)
      .build();
  cudf::io::write_orc(write_opts);

  return std::pair{std::move(input_table), std::move(filepath)};
}

// NOTE: By default, output_row_granularity=10'000 rows.
// This means if the input file has more than 10k rows then the output chunk will never
// have less than 10k rows.
auto chunked_read(std::string const& filepath,
                  output_limit output_limit_bytes,
                  input_limit input_limit_bytes             = input_limit{0},
                  output_row_granularity output_granularity = output_row_granularity{10'000})
{
  auto const read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).build();
  auto reader = cudf::io::chunked_orc_reader(static_cast<std::size_t>(output_limit_bytes),
                                             static_cast<std::size_t>(input_limit_bytes),
                                             static_cast<cudf::size_type>(output_granularity),
                                             read_opts);

  auto num_chunks = 0;
  auto out_tables = std::vector<std::unique_ptr<cudf::table>>{};

  do {
    auto chunk = reader.read_chunk();
    // If the input file is empty, the first call to `read_chunk` will return an empty table.
    // Thus, we only check for non-empty output table from the second call.
    if (num_chunks > 0) {
      CUDF_EXPECTS(chunk.tbl->num_rows() != 0, "Number of rows in the new chunk is zero.");
    }
    ++num_chunks;
    out_tables.emplace_back(std::move(chunk.tbl));
  } while (reader.has_next());

  if (num_chunks > 1) {
    CUDF_EXPECTS(out_tables.front()->num_rows() != 0, "Number of rows in the new chunk is zero.");
  }

  auto out_tviews = std::vector<cudf::table_view>{};
  for (auto const& tbl : out_tables) {
    out_tviews.emplace_back(tbl->view());
  }

  return std::pair(cudf::concatenate(out_tviews), num_chunks);
}

auto chunked_read(std::string const& filepath,
                  output_limit output_limit_bytes,
                  output_row_granularity output_granularity)
{
  return chunked_read(filepath, output_limit_bytes, input_limit{0UL}, output_granularity);
}

}  // namespace

struct OrcChunkedReaderTest : public cudf::test::BaseFixture {};

TEST_F(OrcChunkedReaderTest, TestChunkedReadNoData)
{
  std::vector<std::unique_ptr<cudf::column>> input_columns;
  input_columns.emplace_back(int32s_col{}.release());
  input_columns.emplace_back(int64s_col{}.release());

  auto const [expected, filepath] = write_file(input_columns, "chunked_read_empty");
  auto const [result, num_chunks] = chunked_read(filepath, output_limit{1'000});
  EXPECT_EQ(num_chunks, 1);
  EXPECT_EQ(result->num_rows(), 0);
  EXPECT_EQ(result->num_columns(), 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadSimpleData)
{
  auto constexpr num_rows = 40'000;

  auto const generate_input = [num_rows](bool nullable, std::size_t stripe_rows) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const value_iter = thrust::make_counting_iterator(0);
    input_columns.emplace_back(int32s_col(value_iter, value_iter + num_rows).release());
    input_columns.emplace_back(int64s_col(value_iter, value_iter + num_rows).release());

    return write_file(input_columns,
                      "chunked_read_simple",
                      nullable,
                      cudf::io::default_stripe_size_bytes,
                      stripe_rows);
  };

  {
    auto const [expected, filepath] = generate_input(false, 1'000);
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{245'000});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
  {
    auto const [expected, filepath] = generate_input(false, cudf::io::default_stripe_size_rows);
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{245'000});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  {
    auto const [expected, filepath] = generate_input(true, 1'000);
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{245'000});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
  {
    auto const [expected, filepath] = generate_input(true, cudf::io::default_stripe_size_rows);
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{245'000});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadBoundaryCases)
{
  // Tests some specific boundary conditions in the split calculations.

  auto constexpr num_rows = 40'000;

  auto const [expected, filepath] = [num_rows]() {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const value_iter = thrust::make_counting_iterator(0);
    input_columns.emplace_back(int32s_col(value_iter, value_iter + num_rows).release());
    return write_file(input_columns, "chunked_read_simple_boundary");
  }();

  // Test with zero limit: everything will be read in one chunk.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a very small limit: 1 byte.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{1UL});
    // Number of chunks is 4 because of using default `output_row_granularity = 10k`.
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a very small limit: 1 byte, and small value of `output_row_granularity`.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{1UL}, output_row_granularity{1'000});
    EXPECT_EQ(num_chunks, 40);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a very small limit: 1 byte, and large value of `output_row_granularity`.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{1UL}, output_row_granularity{30'000});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
  // Test with a very large limit
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{2L << 40});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
  // Test with a limit slightly less than one granularity segment of data
  // (output_row_granularity = 10k rows = 40'000 bytes).
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{39'000UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit exactly the size one granularity segment of data
  // (output_row_granularity = 10k rows = 40'000 bytes).
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{40'000UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit slightly more than one granularity segment of data
  // (output_row_granularity = 10k rows = 40'000 bytes).
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{41'000UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit slightly less than two granularity segments of data
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{79'000UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit exactly the size of two granularity segments of data minus 1 byte.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{79'999UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit exactly the size of two granularity segments of data.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{80'000UL});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit slightly more the size two granularity segments of data.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{81'000});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit exactly the size of the input minus 1 byte.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{159'999UL});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a limit exactly the size of the input.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{160'000UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadWithString)
{
  auto constexpr num_rows           = 60'000;
  auto constexpr output_granularity = output_row_granularity{20'000};

  auto const generate_input = [num_rows](bool nullable) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const value_iter = thrust::make_counting_iterator(0);

    // ints                               Granularity Segment  total bytes   cumulative bytes
    // 20000 rows of 4 bytes each               = A0           80000         80000
    // 20000 rows of 4 bytes each               = A1           80000         160000
    // 20000 rows of 4 bytes each               = A2           80000         240000
    input_columns.emplace_back(int32s_col(value_iter, value_iter + num_rows).release());

    // strings                            Granularity Segment  total bytes   cumulative bytes
    // 20000 rows of 1 char each    (20000  + 80004) = B0      100004        100004
    // 20000 rows of 4 chars each   (80000  + 80004) = B1      160004        260008
    // 20000 rows of 16 chars each  (320000 + 80004) = B2      400004        660012
    auto const strings  = std::vector<std::string>{"a", "bbbb", "cccccccccccccccc"};
    auto const str_iter = cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) {
      if (i < 20000) { return strings[0]; }
      if (i < 40000) { return strings[1]; }
      return strings[2];
    });
    input_columns.emplace_back(strings_col(str_iter, str_iter + num_rows).release());

    // Cumulative sizes:
    // A0 + B0 :  180004
    // A1 + B1 :  420008
    // A2 + B2 :  900012
    //                                    skip_rows / num_rows
    // byte_limit==500000  should give 2 chunks: {0, 40000}, {40000, 20000}
    // byte_limit==1000000 should give 1 chunks: {0, 60000},
    return write_file(input_columns, "chunked_read_with_strings", nullable);
  };

  auto const [expected_no_null, filepath_no_null]       = generate_input(false);
  auto const [expected_with_nulls, filepath_with_nulls] = generate_input(true);

  // Test with zero limit: everything will be read in one chunk.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very small limit: 1 byte.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_no_null, output_limit{1UL}, output_granularity);
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_with_nulls, output_limit{1UL}, output_granularity);
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very large limit.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{2L << 40});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{2L << 40});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Other tests:

  {
    auto const [result, num_chunks] =
      chunked_read(filepath_no_null, output_limit{500'000UL}, output_granularity);
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_with_nulls, output_limit{500'000UL}, output_granularity);
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{1'000'000UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{1'000'000UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadWithStructs)
{
  auto constexpr num_rows           = 100'000;
  auto constexpr output_granularity = output_row_granularity{20'000};

  auto const generate_input = [num_rows](bool nullable) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const int_iter = thrust::make_counting_iterator(0);
    input_columns.emplace_back(int32s_col(int_iter, int_iter + num_rows).release());
    input_columns.emplace_back([=] {
      auto child1 = int32s_col(int_iter, int_iter + num_rows);
      auto child2 = int32s_col(int_iter + num_rows, int_iter + num_rows * 2);

      auto const str_iter = cudf::detail::make_counting_transform_iterator(
        0, [&](int32_t i) { return std::to_string(i); });
      auto child3 = strings_col{str_iter, str_iter + num_rows};

      return structs_col{{child1, child2, child3}}.release();
    }());

    return write_file(input_columns, "chunked_read_with_structs", nullable);
  };

  auto const [expected_no_null, filepath_no_null]       = generate_input(false);
  auto const [expected_with_nulls, filepath_with_nulls] = generate_input(true);

  // Test with zero limit: everything will be read in one chunk.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very small limit: 1 byte.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_no_null, output_limit{1UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_with_nulls, output_limit{1UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very large limit.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_no_null, output_limit{2L << 40}, output_granularity);
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_with_nulls, output_limit{2L << 40}, output_granularity);
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Other tests:

  {
    auto const [result, num_chunks] =
      chunked_read(filepath_no_null, output_limit{500'000UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] =
      chunked_read(filepath_with_nulls, output_limit{500'000UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadWithListsNoNulls)
{
  auto constexpr num_rows           = 100'000;
  auto constexpr output_granularity = output_row_granularity{20'000};

  auto const [expected, filepath] = [num_rows]() {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    // 20000 rows in 1 segment consist of:
    //
    // 20001 offsets :   80004  bytes
    // 30000 ints    :   120000 bytes
    // total         :   200004 bytes
    //
    // However, `segmented_row_bit_count` used in chunked reader returns 200000,
    // thus we consider as having only 200000 bytes in total.
    auto const template_lists = int32s_lists_col{
      int32s_lists_col{}, int32s_lists_col{0}, int32s_lists_col{1, 2}, int32s_lists_col{3, 4, 5}};

    auto const gather_iter =
      cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) { return i % 4; });
    auto const gather_map = int32s_col(gather_iter, gather_iter + num_rows);
    input_columns.emplace_back(
      std::move(cudf::gather(cudf::table_view{{template_lists}}, gather_map)->release().front()));

    return write_file(input_columns, "chunked_read_with_lists_no_null");
  }();

  // Test with zero limit: everything will be read in one chunk.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a very small limit: 1 byte.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{1UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a very large limit.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{2L << 40UL}, output_granularity);
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size slightly less than 1 row segment (forcing it to be at least 1 segment per read).
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{199'999UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size exactly 1 row segment.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{200'000UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size == size of 2 segments. Totally have 3 chunks.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{400'000UL}, output_granularity);
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size == size of 2 segment minus one byte: each chunk will be just one segment.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{399'999UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadWithListsHavingNulls)
{
  auto constexpr num_rows           = 100'000;
  auto constexpr output_granularity = output_row_granularity{20'000};

  auto const [expected, filepath] = [num_rows]() {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    // 20000 rows in 1 page consist of:
    //
    // 625 validity words :   2500 bytes   (a null every 4 rows: null at indices [3, 7, 11, ...])
    // 20001 offsets      :   80004  bytes
    // 15000 ints         :   60000 bytes
    // total              :   142504 bytes
    //
    // However, `segmented_row_bit_count` used in chunked reader returns 142500,
    // thus we consider as having only 142500 bytes in total.
    auto const template_lists =
      int32s_lists_col{// these will all be null
                       int32s_lists_col{},
                       int32s_lists_col{0},
                       int32s_lists_col{1, 2},
                       int32s_lists_col{3, 4, 5, 6, 7, 8, 9} /* this list will be nullified out */};
    auto const gather_iter =
      cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) { return i % 4; });
    auto const gather_map = int32s_col(gather_iter, gather_iter + num_rows);
    input_columns.emplace_back(
      std::move(cudf::gather(cudf::table_view{{template_lists}}, gather_map)->release().front()));

    return write_file(input_columns, "chunked_read_with_lists_nulls", true /*nullable*/);
  }();

  // Test with zero limit: everything will be read in one chunk.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a very small limit: 1 byte.
  {
    auto const [result, num_chunks] = chunked_read(filepath, output_limit{1UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Test with a very large limit.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{2L << 40}, output_granularity);
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size slightly less than 1 row segment (forcing it to be at least 1 segment per read).
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{142'499UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size exactly 1 row segment.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{142'500UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size == size of 2 segments. Totally have 3 chunks.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{285'000UL}, output_granularity);
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // Chunk size == size of 2 segment minus one byte: each chunk will be just one segment.
  {
    auto const [result, num_chunks] =
      chunked_read(filepath, output_limit{284'999UL}, output_granularity);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadWithStructsOfLists)
{
  auto constexpr num_rows = 100'000;

  // Size of each segment (10k row by default) is from 537k to 560k bytes (no nulls)
  // and from 456k to 473k (with nulls).
  auto const generate_input = [num_rows](bool nullable) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const int_iter = thrust::make_counting_iterator(0);
    input_columns.emplace_back(int32s_col(int_iter, int_iter + num_rows).release());
    input_columns.emplace_back([=] {
      std::vector<std::unique_ptr<cudf::column>> child_columns;
      child_columns.emplace_back(int32s_col(int_iter, int_iter + num_rows).release());
      child_columns.emplace_back(
        int32s_col(int_iter + num_rows, int_iter + num_rows * 2).release());

      auto const str_iter = cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) {
        return std::to_string(i) + "++++++++++++++++++++" + std::to_string(i);
      });
      child_columns.emplace_back(strings_col{str_iter, str_iter + num_rows}.release());

      auto const template_lists = int32s_lists_col{
        int32s_lists_col{}, int32s_lists_col{0}, int32s_lists_col{0, 1}, int32s_lists_col{0, 1, 2}};
      auto const gather_iter =
        cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) { return i % 4; });
      auto const gather_map = int32s_col(gather_iter, gather_iter + num_rows);
      child_columns.emplace_back(
        std::move(cudf::gather(cudf::table_view{{template_lists}}, gather_map)->release().front()));

      return structs_col(std::move(child_columns)).release();
    }());

    return write_file(input_columns, "chunked_read_with_structs_of_lists", nullable);
  };

  auto const [expected_no_null, filepath_no_null]       = generate_input(false);
  auto const [expected_with_nulls, filepath_with_nulls] = generate_input(true);

  // Test with zero limit: everything will be read in one chunk.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very small limit: 1 byte.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{1UL});
    EXPECT_EQ(num_chunks, 10);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{1UL});
    EXPECT_EQ(num_chunks, 10);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very large limit.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{2L << 40});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{2L << 40});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Other tests:

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{1'000'000UL});
    EXPECT_EQ(num_chunks, 10);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{1'500'000UL});
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{2'000'000UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{5'000'000UL});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{1'000'000UL});
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{1'500'000UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{2'000'000UL});
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{5'000'000UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadWithListsOfStructs)
{
  auto constexpr num_rows = 100'000;

  // Size of each segment (10k row by default) is from 450k to 530k bytes (no nulls)
  // and from 330k to 380k (with nulls).
  auto const generate_input = [num_rows](bool nullable) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const int_iter = thrust::make_counting_iterator(0);
    input_columns.emplace_back(int32s_col(int_iter, int_iter + num_rows).release());

    auto offsets = std::vector<cudf::size_type>{};
    offsets.reserve(num_rows * 2);
    cudf::size_type num_structs = 0;
    for (int i = 0; i < num_rows; ++i) {
      offsets.push_back(num_structs);
      auto const new_list_size = i % 4;
      num_structs += new_list_size;
    }
    offsets.push_back(num_structs);

    auto const make_structs_col = [=] {
      auto child1 = int32s_col(int_iter, int_iter + num_structs);
      auto child2 = int32s_col(int_iter + num_structs, int_iter + num_structs * 2);

      auto const str_iter = cudf::detail::make_counting_transform_iterator(
        0, [&](int32_t i) { return std::to_string(i) + std::to_string(i) + std::to_string(i); });
      auto child3 = strings_col{str_iter, str_iter + num_structs};

      return structs_col{{child1, child2, child3}}.release();
    };

    input_columns.emplace_back(
      cudf::make_lists_column(static_cast<cudf::size_type>(offsets.size() - 1),
                              int32s_col(offsets.begin(), offsets.end()).release(),
                              make_structs_col(),
                              0,
                              rmm::device_buffer{}));

    return write_file(input_columns, "chunked_read_with_lists_of_structs", nullable);
  };

  auto const [expected_no_null, filepath_no_null]       = generate_input(false);
  auto const [expected_with_nulls, filepath_with_nulls] = generate_input(true);

  // Test with zero limit: everything will be read in one chunk.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{0UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very small limit: 1 byte.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{1UL});
    EXPECT_EQ(num_chunks, 10);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{1UL});
    EXPECT_EQ(num_chunks, 10);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Test with a very large limit.
  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{2L << 40});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{2L << 40});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  // Other tests.

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{1'000'000UL});
    EXPECT_EQ(num_chunks, 7);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{1'500'000UL});
    EXPECT_EQ(num_chunks, 4);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{2'000'000UL});
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_no_null, output_limit{5'000'000UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_no_null, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{1'000'000UL});
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{1'500'000UL});
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{2'000'000UL});
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }

  {
    auto const [result, num_chunks] = chunked_read(filepath_with_nulls, output_limit{5'000'000UL});
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_with_nulls, *result);
  }
}

TEST_F(OrcChunkedReaderTest, TestChunkedReadNullCount)
{
  auto constexpr num_rows = 100'000;

  auto const sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return 1; });
  auto const validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 4 != 3; });
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(int32s_col{sequence, sequence + num_rows, validity}.release());
  auto const expected = std::make_unique<cudf::table>(std::move(cols));

  auto const filepath          = temp_env->get_temp_filepath("chunked_reader_null_count.orc");
  auto const stripe_limit_rows = num_rows / 5;
  auto const write_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, *expected)
      .stripe_size_rows(stripe_limit_rows)
      .build();
  cudf::io::write_orc(write_opts);

  auto const byte_limit = stripe_limit_rows * sizeof(int);
  auto const read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).build();
  auto reader =
    cudf::io::chunked_orc_reader(byte_limit, 0UL /*read_limit*/, stripe_limit_rows, read_opts);

  do {
    // Every fourth row is null.
    EXPECT_EQ(reader.read_chunk().tbl->get_column(0).null_count(), stripe_limit_rows / 4UL);
  } while (reader.has_next());
}

namespace {

std::size_t constexpr input_limit_expected_file_count = 3;

std::vector<std::string> input_limit_get_test_names(std::string const& base_filename)
{
  return {base_filename + "_a.orc", base_filename + "_b.orc", base_filename + "_c.orc"};
}

void input_limit_test_write_one(std::string const& filepath,
                                cudf::table_view const& input,
                                cudf::size_type stripe_size_rows,
                                cudf::io::compression_type compression)
{
  auto const out_opts = cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, input)
                          .compression(compression)
                          .stripe_size_rows(stripe_size_rows)
                          .build();
  cudf::io::write_orc(out_opts);
}

void input_limit_test_write(
  std::vector<std::string> const& test_files,
  cudf::table_view const& input,
  cudf::size_type stripe_size_rows = 20'000 /*write relatively small stripes by default*/)
{
  CUDF_EXPECTS(test_files.size() == input_limit_expected_file_count,
               "Unexpected count of test filenames.");

  // ZSTD yields a very small decompression size, can be much smaller than SNAPPY.
  // However, ORC reader typically over-estimates the decompression size of data
  // compressed by ZSTD to be very large, can be much larger than that of SNAPPY.
  // That is because ZSTD may use a lot of scratch space at decode time
  // (2.5x the total decompressed buffer size).
  // As such, we may see smaller output chunks for the input data compressed by ZSTD.
  input_limit_test_write_one(
    test_files[0], input, stripe_size_rows, cudf::io::compression_type::NONE);
  input_limit_test_write_one(
    test_files[1], input, stripe_size_rows, cudf::io::compression_type::ZSTD);
  input_limit_test_write_one(
    test_files[2], input, stripe_size_rows, cudf::io::compression_type::SNAPPY);
}

void input_limit_test_read(int test_location,
                           std::vector<std::string> const& test_files,
                           cudf::table_view const& input,
                           output_limit output_limit_bytes,
                           input_limit input_limit_bytes,
                           int const* expected_chunk_counts)
{
  CUDF_EXPECTS(test_files.size() == input_limit_expected_file_count,
               "Unexpected count of test filenames.");

  for (size_t idx = 0; idx < test_files.size(); ++idx) {
    SCOPED_TRACE("Original line of failure: " + std::to_string(test_location) +
                 ", file idx: " + std::to_string(idx));
    // TODO: remove
    printf("file_idx %d\n", (int)idx);
    auto const [result, num_chunks] =
      chunked_read(test_files[idx], output_limit_bytes, input_limit_bytes);
    EXPECT_EQ(expected_chunk_counts[idx], num_chunks);
    // TODO: equal
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result, input);
  }
}

}  // namespace

struct OrcChunkedReaderInputLimitTest : public cudf::test::BaseFixture {};

TEST_F(OrcChunkedReaderInputLimitTest, SingleFixedWidthColumn)
{
  auto constexpr num_rows = 1'000'000;
  auto const iter1        = thrust::make_constant_iterator(15);
  auto const col1         = doubles_col(iter1, iter1 + num_rows);

  auto const filename   = std::string{"single_col_fixed_width"};
  auto const test_files = input_limit_get_test_names(temp_env->get_temp_filepath(filename));
  auto const input      = cudf::table_view{{col1}};
  input_limit_test_write(test_files, input);

  {
    int constexpr expected[] = {50, 50, 50};
    input_limit_test_read(
      __LINE__, test_files, input, output_limit{0UL}, input_limit{1UL}, expected);
  }

  {
    int constexpr expected[] = {10, 13, 10};
    input_limit_test_read(
      __LINE__, test_files, input, output_limit{0UL}, input_limit{2 * 1024 * 1024UL}, expected);
  }
}

TEST_F(OrcChunkedReaderInputLimitTest, MixedColumns)
{
  auto constexpr num_rows = 1'000'000;

  auto const iter1 = thrust::make_counting_iterator<int>(0);
  auto const col1  = int32s_col(iter1, iter1 + num_rows);

  auto const iter2 = thrust::make_counting_iterator<double>(0);
  auto const col2  = doubles_col(iter2, iter2 + num_rows);

  auto const strings  = std::vector<std::string>{"abc", "de", "fghi"};
  auto const str_iter = cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) {
    if (i < 250000) { return strings[0]; }
    if (i < 750000) { return strings[1]; }
    return strings[2];
  });
  auto const col3     = strings_col(str_iter, str_iter + num_rows);

  auto const filename   = std::string{"mixed_columns"};
  auto const test_files = input_limit_get_test_names(temp_env->get_temp_filepath(filename));
  auto const input      = cudf::table_view{{col1, col2, col3}};
  input_limit_test_write(test_files, input);

  {
    int constexpr expected[] = {50, 50, 50};
    input_limit_test_read(
      __LINE__, test_files, input, output_limit{0UL}, input_limit{1UL}, expected);
  }

  {
    int constexpr expected[] = {10, 50, 15};
    input_limit_test_read(
      __LINE__, test_files, input, output_limit{0UL}, input_limit{2 * 1024 * 1024UL}, expected);
  }
}

namespace {

struct offset_gen {
  int const group_size;
  __device__ int operator()(int i) const { return i * group_size; }
};

template <typename T>
struct value_gen {
  __device__ T operator()(int i) const { return i % 1024; }
};

struct char_values {
  __device__ int8_t operator()(int i) const
  {
    int const index = (i / 2) % 3;
    // Generate repeating 3-runs of 2 values each: "aabbccaabbcc...".
    return index == 0 ? 'a' : (index == 1 ? 'b' : 'c');
  }
};

}  // namespace

TEST_F(OrcChunkedReaderInputLimitTest, ListType)
{
  int constexpr num_rows  = 50'000'000;
  int constexpr list_size = 4;

  auto const stream = cudf::get_default_stream();
  auto const iter   = thrust::make_counting_iterator(0);

  auto offset_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows + 1, cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy(stream),
                    iter,
                    iter + num_rows + 1,
                    offset_col->mutable_view().begin<int>(),
                    offset_gen{list_size});

  int constexpr num_ints = num_rows * list_size;
  auto value_col         = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, num_ints, cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy(stream),
                    iter,
                    iter + num_ints,
                    value_col->mutable_view().begin<int>(),
                    value_gen<int>{});

  auto const lists_col =
    cudf::make_lists_column(num_rows, std::move(offset_col), std::move(value_col), 0, {}, stream);

  auto const filename   = std::string{"list_type"};
  auto const test_files = input_limit_get_test_names(temp_env->get_temp_filepath(filename));
  auto const input      = cudf::table_view{{*lists_col}};

  // Although we set `stripe_size_rows` to be very large, the writer only write
  // 250k rows (top level) per stripe due to having nested type.
  // Thus, we have 200 stripes in total.
  input_limit_test_write(test_files, input, cudf::io::default_stripe_size_rows);

  {
    int constexpr expected[] = {2, 40, 3};
    input_limit_test_read(
      __LINE__, test_files, input, output_limit{0UL}, input_limit{5 * 1024 * 1024UL}, expected);
  }

  {
    int constexpr expected[] = {8, 40, 9};
    input_limit_test_read(__LINE__,
                          test_files,
                          input,
                          output_limit{128 * 1024 * 1024UL},
                          input_limit{5 * 1024 * 1024UL},
                          expected);
  }
}

TEST_F(OrcChunkedReaderInputLimitTest, MixedColumnsHavingList)
{
  int constexpr num_rows  = 1'000'000;
  int constexpr list_size = 4;
  int constexpr str_size  = 3;

  auto const stream = cudf::get_default_stream();
  auto const iter   = thrust::make_counting_iterator(0);

  // list<int>
  auto offset_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows + 1, cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy(stream),
                    iter,
                    iter + num_rows + 1,
                    offset_col->mutable_view().begin<int>(),
                    offset_gen{list_size});

  int constexpr num_ints = num_rows * list_size;
  auto value_col         = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, num_ints, cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy(stream),
                    iter,
                    iter + num_ints,
                    value_col->mutable_view().begin<int>(),
                    value_gen<int>{});

  auto const lists_col =
    cudf::make_lists_column(num_rows, std::move(offset_col), std::move(value_col), 0, {}, stream);

  // strings
  int constexpr num_chars = num_rows * str_size;
  auto str_offset_col     = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows + 1, cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy(stream),
                    iter,
                    iter + num_rows + 1,
                    str_offset_col->mutable_view().begin<int>(),
                    offset_gen{str_size});
  rmm::device_buffer str_chars(num_chars, stream);
  thrust::transform(rmm::exec_policy(stream),
                    iter,
                    iter + num_chars,
                    static_cast<int8_t*>(str_chars.data()),
                    char_values{});
  auto const str_col =
    cudf::make_strings_column(num_rows, std::move(str_offset_col), std::move(str_chars), 0, {});

  // doubles
  auto const double_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::FLOAT64}, num_rows, cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy(stream),
                    iter,
                    iter + num_rows,
                    double_col->mutable_view().begin<double>(),
                    value_gen<double>{});

  auto const filename   = std::string{"mixed_cols_having_list"};
  auto const test_files = input_limit_get_test_names(temp_env->get_temp_filepath(filename));
  auto const input      = cudf::table_view{{*lists_col, *str_col, *double_col}};

  for (int iters = 1; iters <= 100; ++iters) {
    {
      auto const write_opts =
        cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{test_files[0]})
          .stripe_size_rows(cudf::io::default_stripe_size_rows)
          .build();

      auto writer = cudf::io::orc_chunked_writer(write_opts);
      for (int i = 0; i < iters; ++i) {
        writer.write(input);
      }
    }

    // Although we set `stripe_size_rows` to be very large, the writer only write
    // 250k rows (top level) per stripe due to having nested type.
    // Thus, we have 200 stripes in total.
    // input_limit_test_write(test_files, input, cudf::io::default_stripe_size_rows);

    if (0) {
      int constexpr expected[] = {8, 8, 6};
      auto const [result, num_chunks] =
        chunked_read(test_files[0], output_limit{0UL}, input_limit{128 * 1024 * 1024UL});
      EXPECT_EQ(expected[0], num_chunks);
      printf("num_chunks: %d\n", (int)num_chunks);

      // TODO: equal
      // CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result, input);

      // input_limit_test_read(
      //   __LINE__, test_files, input, output_limit{0UL}, input_limit{128 * 1024 * 1024UL},
      //   expected);
    }

    // clang-format off
  /*
complete, 1, peak_memory_usage: 24870400 , MB = 23.7183
complete, 2, peak_memory_usage: 49739984 , MB = 47.4357
complete, 3, peak_memory_usage: 74609600 , MB = 71.1533
complete, 4, peak_memory_usage: 99479184 , MB = 94.8707
complete, 5, peak_memory_usage: 124348528 , MB = 118.588
complete, 6, peak_memory_usage: 149218128 , MB = 142.305
complete, 7, peak_memory_usage: 174087728 , MB = 166.023
complete, 8, peak_memory_usage: 198957312 , MB = 189.74
complete, 9, peak_memory_usage: 223826672 , MB = 213.458
complete, 10, peak_memory_usage: 248696256 , MB = 237.175
complete, 11, peak_memory_usage: 224912432 , MB = 214.493
complete, 12, peak_memory_usage: 225455472 , MB = 215.011
complete, 13, peak_memory_usage: 225998192 , MB = 215.529
complete, 14, peak_memory_usage: 226541072 , MB = 216.046
complete, 15, peak_memory_usage: 227084080 , MB = 216.564
complete, 16, peak_memory_usage: 227626832 , MB = 217.082
complete, 17, peak_memory_usage: 228169712 , MB = 217.6
complete, 18, peak_memory_usage: 228712592 , MB = 218.117
complete, 19, peak_memory_usage: 248696256 , MB = 237.175
complete, 20, peak_memory_usage: 229798352 , MB = 219.153
complete, 21, peak_memory_usage: 230341600 , MB = 219.671
complete, 22, peak_memory_usage: 230884096 , MB = 220.188
complete, 23, peak_memory_usage: 231427152 , MB = 220.706
complete, 24, peak_memory_usage: 231970080 , MB = 221.224
complete, 25, peak_memory_usage: 232513136 , MB = 221.742
complete, 26, peak_memory_usage: 233056016 , MB = 222.26
complete, 27, peak_memory_usage: 233598624 , MB = 222.777
complete, 28, peak_memory_usage: 248696256 , MB = 237.175
complete, 29, peak_memory_usage: 234684480 , MB = 223.813
complete, 30, peak_memory_usage: 235227984 , MB = 224.331
complete, 31, peak_memory_usage: 235770208 , MB = 224.848
complete, 32, peak_memory_usage: 236313040 , MB = 225.366
complete, 33, peak_memory_usage: 236855888 , MB = 225.883
complete, 34, peak_memory_usage: 237399504 , MB = 226.402
complete, 35, peak_memory_usage: 237941776 , MB = 226.919
complete, 36, peak_memory_usage: 238485504 , MB = 227.438
complete, 37, peak_memory_usage: 248696256 , MB = 237.175
complete, 38, peak_memory_usage: 239570400 , MB = 228.472
complete, 39, peak_memory_usage: 240113728 , MB = 228.99
complete, 40, peak_memory_usage: 240656512 , MB = 229.508
complete, 41, peak_memory_usage: 241198848 , MB = 230.025
complete, 42, peak_memory_usage: 241742608 , MB = 230.544
complete, 43, peak_memory_usage: 242285536 , MB = 231.061
complete, 44, peak_memory_usage: 242828368 , MB = 231.579
complete, 45, peak_memory_usage: 243371008 , MB = 232.097
complete, 46, peak_memory_usage: 248696256 , MB = 237.175
complete, 47, peak_memory_usage: 244456448 , MB = 233.132
complete, 48, peak_memory_usage: 245000016 , MB = 233.65
complete, 49, peak_memory_usage: 245542256 , MB = 234.167
complete, 50, peak_memory_usage: 246085472 , MB = 234.685
complete, 51, peak_memory_usage: 246628768 , MB = 235.204
complete, 52, peak_memory_usage: 247171088 , MB = 235.721
complete, 53, peak_memory_usage: 247714240 , MB = 236.239
complete, 54, peak_memory_usage: 248257248 , MB = 236.757
complete, 55, peak_memory_usage: 248799808 , MB = 237.274
complete, 56, peak_memory_usage: 249342880 , MB = 237.792
complete, 57, peak_memory_usage: 249885808 , MB = 238.31
complete, 58, peak_memory_usage: 250428960 , MB = 238.828
complete, 59, peak_memory_usage: 250971984 , MB = 239.346
complete, 60, peak_memory_usage: 251514080 , MB = 239.863
complete, 61, peak_memory_usage: 252057616 , MB = 240.381
complete, 62, peak_memory_usage: 252599968 , MB = 240.898
complete, 63, peak_memory_usage: 253142992 , MB = 241.416
complete, 64, peak_memory_usage: 253686064 , MB = 241.934
complete, 65, peak_memory_usage: 254227872 , MB = 242.451
complete, 66, peak_memory_usage: 254771152 , MB = 242.969
complete, 67, peak_memory_usage: 255313872 , MB = 243.486
complete, 68, peak_memory_usage: 255856912 , MB = 244.004
complete, 69, peak_memory_usage: 256400048 , MB = 244.522
complete, 70, peak_memory_usage: 256943040 , MB = 245.04
complete, 71, peak_memory_usage: 257485520 , MB = 245.557
complete, 72, peak_memory_usage: 258029520 , MB = 246.076
complete, 73, peak_memory_usage: 258572064 , MB = 246.594
complete, 74, peak_memory_usage: 259115328 , MB = 247.112
complete, 75, peak_memory_usage: 259657776 , MB = 247.629
complete, 76, peak_memory_usage: 260200864 , MB = 248.147
complete, 77, peak_memory_usage: 260742832 , MB = 248.664
complete, 78, peak_memory_usage: 261286496 , MB = 249.182
complete, 79, peak_memory_usage: 261828432 , MB = 249.699
complete, 80, peak_memory_usage: 262371920 , MB = 250.217
complete, 81, peak_memory_usage: 262914432 , MB = 250.735
complete, 82, peak_memory_usage: 263458960 , MB = 251.254
complete, 83, peak_memory_usage: 264000816 , MB = 251.771
complete, 84, peak_memory_usage: 264543056 , MB = 252.288
complete, 85, peak_memory_usage: 265085984 , MB = 252.806
complete, 86, peak_memory_usage: 265630256 , MB = 253.325
complete, 87, peak_memory_usage: 266171696 , MB = 253.841
complete, 88, peak_memory_usage: 266714432 , MB = 254.359
complete, 89, peak_memory_usage: 267257392 , MB = 254.877
complete, 90, peak_memory_usage: 267800176 , MB = 255.394
complete, 91, peak_memory_usage: 268343536 , MB = 255.912
complete, 92, peak_memory_usage: 268886256 , MB = 256.43
complete, 93, peak_memory_usage: 269429968 , MB = 256.948
complete, 94, peak_memory_usage: 269971904 , MB = 257.465
complete, 95, peak_memory_usage: 270516528 , MB = 257.985
complete, 96, peak_memory_usage: 271058992 , MB = 258.502
complete, 97, peak_memory_usage: 271601616 , MB = 259.019
complete, 98, peak_memory_usage: 272145536 , MB = 259.538
complete, 99, peak_memory_usage: 272686496 , MB = 260.054
complete, 100, peak_memory_usage: 273230448 , MB = 260.573

num_chunks: 1
num_chunks: 1
num_chunks: 1
num_chunks: 1
num_chunks: 1
num_chunks: 1
num_chunks: 1
num_chunks: 1
num_chunks: 1
num_chunks: 1
num_chunks: 2
num_chunks: 2
num_chunks: 2
num_chunks: 2
num_chunks: 2
num_chunks: 2
num_chunks: 2
num_chunks: 2
num_chunks: 2
num_chunks: 3
num_chunks: 3
num_chunks: 3
num_chunks: 3
num_chunks: 3
num_chunks: 3
num_chunks: 3
num_chunks: 3
num_chunks: 3
num_chunks: 4
num_chunks: 4
num_chunks: 4
num_chunks: 4
num_chunks: 4
num_chunks: 4
num_chunks: 4
num_chunks: 4
num_chunks: 4
num_chunks: 5
num_chunks: 5
num_chunks: 5
num_chunks: 5
num_chunks: 5
num_chunks: 5
num_chunks: 5
num_chunks: 5
num_chunks: 5
num_chunks: 6
num_chunks: 6
num_chunks: 6
num_chunks: 6
num_chunks: 6
num_chunks: 6
num_chunks: 6
num_chunks: 6
num_chunks: 6
num_chunks: 7
num_chunks: 7
num_chunks: 7
num_chunks: 7
num_chunks: 7
num_chunks: 7
num_chunks: 7
num_chunks: 7
num_chunks: 7
num_chunks: 8
num_chunks: 8
num_chunks: 8
num_chunks: 8
num_chunks: 8
num_chunks: 8
num_chunks: 8
num_chunks: 8
num_chunks: 8
num_chunks: 9
num_chunks: 9
num_chunks: 9
num_chunks: 9
num_chunks: 9
num_chunks: 9
num_chunks: 9
num_chunks: 9
num_chunks: 9
num_chunks: 10
num_chunks: 10
num_chunks: 10
num_chunks: 10
num_chunks: 10
num_chunks: 10
num_chunks: 10
num_chunks: 10
num_chunks: 10
num_chunks: 11
num_chunks: 11
num_chunks: 11
num_chunks: 11
num_chunks: 11
num_chunks: 11
num_chunks: 11
num_chunks: 11
num_chunks: 11

   */
    // clang-format on

    printf("\n\n\n\n read full\n");
    fflush(stdout);
    // TODO: remove
    {
      int constexpr expected[] = {1, 1, 1};
      // input_limit_test_read(
      //   __LINE__, test_files, input, output_limit{0UL}, input_limit{0UL}, expected);
      auto const [result, num_chunks] =
        chunked_read(test_files[0], output_limit{0UL}, input_limit{0UL});
      EXPECT_EQ(expected[0], num_chunks);
      printf("num_chunks: %d\n", (int)num_chunks);
      // TODO: equal
      // CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result, input);
    }

    // clang-format off
/*
complete, 1, peak_memory_usage: 24870400 , MB = 23.7183
complete, 2, peak_memory_usage: 49739984 , MB = 47.4357
complete, 3, peak_memory_usage: 74609600 , MB = 71.1533
complete, 4, peak_memory_usage: 99479184 , MB = 94.8707
complete, 5, peak_memory_usage: 124348528 , MB = 118.588
complete, 6, peak_memory_usage: 149218128 , MB = 142.305
complete, 7, peak_memory_usage: 174087728 , MB = 166.023
complete, 8, peak_memory_usage: 198957312 , MB = 189.74
complete, 9, peak_memory_usage: 223826672 , MB = 213.458
complete, 10, peak_memory_usage: 248696256 , MB = 237.175
complete, 11, peak_memory_usage: 273565872 , MB = 260.893
complete, 12, peak_memory_usage: 298435456 , MB = 284.61
complete, 13, peak_memory_usage: 323304800 , MB = 308.327
complete, 14, peak_memory_usage: 348174400 , MB = 332.045
complete, 15, peak_memory_usage: 373044000 , MB = 355.762
complete, 16, peak_memory_usage: 397913584 , MB = 379.48
complete, 17, peak_memory_usage: 422782944 , MB = 403.197
complete, 18, peak_memory_usage: 447652528 , MB = 426.915
complete, 19, peak_memory_usage: 472522144 , MB = 450.632
complete, 20, peak_memory_usage: 497391728 , MB = 474.35
complete, 21, peak_memory_usage: 522261072 , MB = 498.067
complete, 22, peak_memory_usage: 547130672 , MB = 521.784
complete, 23, peak_memory_usage: 572000272 , MB = 545.502
complete, 24, peak_memory_usage: 596869856 , MB = 569.219
complete, 25, peak_memory_usage: 621739216 , MB = 592.937
complete, 26, peak_memory_usage: 646608800 , MB = 616.654
complete, 27, peak_memory_usage: 671478416 , MB = 640.372
complete, 28, peak_memory_usage: 696348000 , MB = 664.089
complete, 29, peak_memory_usage: 721217344 , MB = 687.806
complete, 30, peak_memory_usage: 746086944 , MB = 711.524
complete, 31, peak_memory_usage: 770956544 , MB = 735.241
complete, 32, peak_memory_usage: 795826128 , MB = 758.959
complete, 33, peak_memory_usage: 820695488 , MB = 782.676
complete, 34, peak_memory_usage: 845565072 , MB = 806.394
complete, 35, peak_memory_usage: 870434688 , MB = 830.111
complete, 36, peak_memory_usage: 895304272 , MB = 853.829
complete, 37, peak_memory_usage: 920173616 , MB = 877.546
complete, 38, peak_memory_usage: 945043216 , MB = 901.263
complete, 39, peak_memory_usage: 969912816 , MB = 924.981
complete, 40, peak_memory_usage: 994782400 , MB = 948.698
complete, 41, peak_memory_usage: 1019651760 , MB = 972.416
complete, 42, peak_memory_usage: 1044521344 , MB = 996.133
complete, 43, peak_memory_usage: 1069390960 , MB = 1019.85
complete, 44, peak_memory_usage: 1094260544 , MB = 1043.57
complete, 45, peak_memory_usage: 1119129888 , MB = 1067.29
complete, 46, peak_memory_usage: 1143999488 , MB = 1091
complete, 47, peak_memory_usage: 1168869088 , MB = 1114.72
complete, 48, peak_memory_usage: 1193738672 , MB = 1138.44
complete, 49, peak_memory_usage: 1218608032 , MB = 1162.16
complete, 50, peak_memory_usage: 1243477616 , MB = 1185.87
complete, 51, peak_memory_usage: 1268347232 , MB = 1209.59
complete, 52, peak_memory_usage: 1293216816 , MB = 1233.31
complete, 53, peak_memory_usage: 1318086160 , MB = 1257.02
complete, 54, peak_memory_usage: 1342955760 , MB = 1280.74
complete, 55, peak_memory_usage: 1367825360 , MB = 1304.46
complete, 56, peak_memory_usage: 1392694944 , MB = 1328.18
complete, 57, peak_memory_usage: 1417564560 , MB = 1351.89
complete, 58, peak_memory_usage: 1442433888 , MB = 1375.61
complete, 59, peak_memory_usage: 1467303504 , MB = 1399.33
complete, 60, peak_memory_usage: 1492173088 , MB = 1423.05
complete, 61, peak_memory_usage: 1517042688 , MB = 1446.76
complete, 62, peak_memory_usage: 1541912032 , MB = 1470.48
complete, 63, peak_memory_usage: 1566781632 , MB = 1494.2
complete, 64, peak_memory_usage: 1591651216 , MB = 1517.92
complete, 65, peak_memory_usage: 1616520832 , MB = 1541.63
complete, 66, peak_memory_usage: 1641390160 , MB = 1565.35
complete, 67, peak_memory_usage: 1666259776 , MB = 1589.07
complete, 68, peak_memory_usage: 1691129360 , MB = 1612.79
complete, 69, peak_memory_usage: 1715998960 , MB = 1636.5
complete, 70, peak_memory_usage: 1740868304 , MB = 1660.22
complete, 71, peak_memory_usage: 1765737904 , MB = 1683.94
complete, 72, peak_memory_usage: 1790607488 , MB = 1707.66
complete, 73, peak_memory_usage: 1815477104 , MB = 1731.37
complete, 74, peak_memory_usage: 1840346432 , MB = 1755.09
complete, 75, peak_memory_usage: 1865216048 , MB = 1778.81
complete, 76, peak_memory_usage: 1890085632 , MB = 1802.53
complete, 77, peak_memory_usage: 1914955232 , MB = 1826.24
complete, 78, peak_memory_usage: 1939824576 , MB = 1849.96
complete, 79, peak_memory_usage: 1964694176 , MB = 1873.68
complete, 80, peak_memory_usage: 1989563760 , MB = 1897.4
complete, 81, peak_memory_usage: 2014433376 , MB = 1921.11
complete, 82, peak_memory_usage: 2039302704 , MB = 1944.83
complete, 83, peak_memory_usage: 2064172320 , MB = 1968.55
complete, 84, peak_memory_usage: 2089041904 , MB = 1992.27
complete, 85, peak_memory_usage: 2113911504 , MB = 2015.98
complete, 86, peak_memory_usage: 2138780848 , MB = 2039.7
complete, 87, peak_memory_usage: 2163650448 , MB = 2063.42
complete, 88, peak_memory_usage: 2188520032 , MB = 2087.14
complete, 89, peak_memory_usage: 2213389648 , MB = 2110.85
complete, 90, peak_memory_usage: 2238258976 , MB = 2134.57
complete, 91, peak_memory_usage: 2263128592 , MB = 2158.29
complete, 92, peak_memory_usage: 2287998176 , MB = 2182.01
complete, 93, peak_memory_usage: 2312867776 , MB = 2205.72
complete, 94, peak_memory_usage: 2337737120 , MB = 2229.44
complete, 95, peak_memory_usage: 2362606720 , MB = 2253.16
complete, 96, peak_memory_usage: 2387476304 , MB = 2276.87
complete, 97, peak_memory_usage: 2412345920 , MB = 2300.59
complete, 98, peak_memory_usage: 2437215248 , MB = 2324.31
complete, 99, peak_memory_usage: 2462084864 , MB = 2348.03
complete, 100, peak_memory_usage: 2486954448 , MB = 2371.74
*/
    // clang-format on

  }  // end iters
}

TEST_F(OrcChunkedReaderInputLimitTest, ReadWithRowSelection)
{
  int64_t constexpr num_rows    = 100'000'000l;
  int constexpr rows_per_stripe = 100'000;

  auto const it    = thrust::make_counting_iterator(0);
  auto const col   = int32s_col(it, it + num_rows);
  auto const input = cudf::table_view{{col}};

  auto const filepath = temp_env->get_temp_filepath("chunk_read_with_row_selection.orc");
  auto const write_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, input)
      .stripe_size_rows(rows_per_stripe)
      .build();
  cudf::io::write_orc(write_opts);

  // Verify metadata.
  auto const metadata = cudf::io::read_orc_metadata(cudf::io::source_info{filepath});
  EXPECT_EQ(metadata.num_rows(), num_rows);
  EXPECT_EQ(metadata.num_stripes(), num_rows / rows_per_stripe);

  int constexpr random_val = 123456;

  // Read some random number or rows that is not stripe size.
  int constexpr num_rows_to_read = rows_per_stripe * 5 + random_val;

  // Just shift the read data region back by a random offset.
  const auto num_rows_to_skip = num_rows - num_rows_to_read - random_val;

  const auto sequence_start = num_rows_to_skip % num_rows;
  auto const skipped_col = int32s_col(it + sequence_start, it + sequence_start + num_rows_to_read);
  auto const expected    = cudf::table_view{{skipped_col}};

  auto const read_opts = cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath})
                           .use_index(false)
                           .skip_rows(num_rows_to_skip)
                           .num_rows(num_rows_to_read)
                           .build();

  auto reader = cudf::io::chunked_orc_reader(
    60'000UL * sizeof(int) /*output limit, equal to 60k rows, less than rows in 1 stripe*/,
    rows_per_stripe * sizeof(int) /*input limit, around size of 1 stripe's decoded data*/,
    50'000 /*output granularity, or minimum number of rows for the output chunk*/,
    read_opts);

  auto num_chunks  = 0;
  auto read_tables = std::vector<std::unique_ptr<cudf::table>>{};
  auto tviews      = std::vector<cudf::table_view>{};

  do {
    auto chunk = reader.read_chunk();
    // Each output chunk should have either exactly 50k rows, or num_rows_to_read % 50k.
    EXPECT_TRUE(chunk.tbl->num_rows() == 50000 ||
                chunk.tbl->num_rows() == num_rows_to_read % 50000);

    tviews.emplace_back(chunk.tbl->view());
    read_tables.emplace_back(std::move(chunk.tbl));
    ++num_chunks;
  } while (reader.has_next());

  auto const read_result = cudf::concatenate(tviews);
  EXPECT_EQ(num_chunks, 13);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, read_result->view());
}

#define LOCAL_TEST

// This test is extremely heavy, thus it should be disabled by default.
#ifdef LOCAL_TEST
TEST_F(OrcChunkedReaderInputLimitTest, SizeTypeRowsOverflow)
{
  int64_t constexpr num_rows    = 500'000'000l;
  int constexpr rows_per_stripe = 1'000'000;
  int constexpr num_reps        = 10l;
  int64_t constexpr total_rows  = num_rows * num_reps;
  static_assert(total_rows > std::numeric_limits<cudf::size_type>::max());

  auto const it          = thrust::make_counting_iterator(int64_t{0});
  auto const col         = int64s_col(it, it + num_rows);
  auto const chunk_table = cudf::table_view{{col}};

  std::vector<char> data_buffer;
  {
    auto const write_opts =
      cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info{&data_buffer})
        .stripe_size_rows(rows_per_stripe)
        .build();

    auto writer = cudf::io::orc_chunked_writer(write_opts);
    for (int i = 0; i < num_reps; ++i) {
      writer.write(chunk_table);
    }
  }

  // Verify metadata.
  auto const metadata =
    cudf::io::read_orc_metadata(cudf::io::source_info{data_buffer.data(), data_buffer.size()});
  EXPECT_EQ(metadata.num_rows(), total_rows);
  EXPECT_EQ(metadata.num_stripes(), total_rows / rows_per_stripe);

  int constexpr num_rows_to_read = 5'000'000;
  const auto num_rows_to_skip    = metadata.num_rows() - num_rows_to_read -
                                123456 /*just shift the read data region back by a random offset*/;

  // Check validity of the last 5 million rows.
  const auto sequence_start = num_rows_to_skip % num_rows;
  auto const skipped_col = int64s_col(it + sequence_start, it + sequence_start + num_rows_to_read);
  auto const expected    = cudf::table_view{{skipped_col}};

  auto const read_opts = cudf::io::orc_reader_options::builder(
                           cudf::io::source_info{data_buffer.data(), data_buffer.size()})
                           .use_index(false)
                           .skip_rows(num_rows_to_skip)
                           .num_rows(num_rows_to_read)
                           .build();
  auto reader = cudf::io::chunked_orc_reader(
    600'000UL * sizeof(int64_t) /*output limit, equal to 600k int64_t rows */,
    8'000'000UL /*input limit, around size of 1 stripe's decoded data */,
    500'000 /*output granularity, or minimum number of rows for the output chunk*/,
    read_opts);

  auto num_chunks  = 0;
  auto read_tables = std::vector<std::unique_ptr<cudf::table>>{};
  auto tviews      = std::vector<cudf::table_view>{};

  do {
    auto chunk = reader.read_chunk();
    ++num_chunks;
    tviews.emplace_back(chunk.tbl->view());
    read_tables.emplace_back(std::move(chunk.tbl));
  } while (reader.has_next());

  auto const read_result = cudf::concatenate(tviews);
  EXPECT_EQ(num_chunks, 10);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, read_result->view());
}
#endif
