/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

#include <src/io/parquet/compact_protocol_reader.hpp>
#include <src/io/parquet/parquet.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <rmm/cuda_stream_view.hpp>

#include <fstream>
#include <type_traits>

namespace {
// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64s_col  = cudf::test::fixed_width_column_wrapper<int64_t>;
using strings_col = cudf::test::strings_column_wrapper;
using structs_col = cudf::test::structs_column_wrapper;

auto chunked_read(std::string const& filepath, std::size_t byte_limit)
{
  auto const read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).build();
  auto reader = cudf::io::chunked_parquet_reader(byte_limit, read_opts);

  auto num_chunks = 0;
  auto result     = std::make_unique<cudf::table>();

  while (reader.has_next()) {
    auto chunk = reader.read_chunk();
    if (num_chunks == 0) {
      result = std::move(chunk.tbl);
    } else {
      result = cudf::concatenate(std::vector<cudf::table_view>{result->view(), chunk.tbl->view()});
    }
    ++num_chunks;
  }

  return std::pair(std::move(result), num_chunks);
}

}  // namespace

struct ParquetChunkedReaderTest : public cudf::test::BaseFixture {
};

TEST_F(ParquetChunkedReaderTest, TestChunkedReadSimpleData)
{
  auto constexpr num_rows = 40'000;
  auto const filepath     = temp_env->get_temp_filepath("chunked_read_simple.parquet");

  auto const values = thrust::make_counting_iterator(0);
  auto const a      = int32s_col(values, values + num_rows);
  auto const b      = int64s_col(values, values + num_rows);
  auto const input  = cudf::table_view{{a, b}};

  auto const write_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input).build();
  cudf::io::write_parquet(write_opts);

  auto const [result, num_chunks] = chunked_read(filepath, 240'000);
  EXPECT_EQ(num_chunks, 2);

  CUDF_TEST_EXPECT_TABLES_EQUAL(input, result->view());
}

TEST_F(ParquetChunkedReaderTest, TestChunkedReadWithString)
{
  auto constexpr num_rows = 60'000;
  auto const filepath     = temp_env->get_temp_filepath("chunked_read_with_strings.parquet");
  auto const values       = thrust::make_counting_iterator(0);

  // ints                                            Page    total bytes   cumulative bytes
  // 20000 rows of 4 bytes each                    = A0      80000         80000
  // 20000 rows of 4 bytes each                    = A1      80000         160000
  // 20000 rows of 4 bytes each                    = A2      80000         240000
  auto const a = int32s_col(values, values + num_rows);

  // strings                                         Page    total bytes   cumulative bytes
  // 20000 rows of 1 char each    (20000  + 80004) = B0      100004        100004
  // 20000 rows of 4 chars each   (80000  + 80004) = B1      160004        260008
  // 20000 rows of 16 chars each  (320000 + 80004) = B2      400004        660012
  auto const strings  = std::vector<std::string>{"a", "bbbb", "cccccccccccccccc"};
  auto const str_iter = cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) {
    if (i < 20000) { return strings[0]; }
    if (i < 40000) { return strings[1]; }
    return strings[2];
  });
  auto const b        = strings_col{str_iter, str_iter + num_rows};
  auto const input    = cudf::table_view{{a, b}};

  auto const write_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
      .max_page_size_bytes(512 * 1024)  // 512KB per page
      .max_page_size_rows(20000)        // 20k rows per page
      .build();
  cudf::io::write_parquet(write_opts);

  // Cumulative sizes:
  // A0 + B0 :  180004
  // A1 + B1 :  420008
  // A2 + B2 :  900012
  //                                             skip_rows / num_rows
  // byte_limit==500000  should give 2 chunks: {0, 40000}, {40000, 20000}
  // byte_limit==1000000 should give 1 chunks: {0, 60000},
  {
    auto const [result, num_chunks] = chunked_read(filepath, 500'000);
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, result->view());
  }
  {
    auto const [result, num_chunks] = chunked_read(filepath, 1'000'000);
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, result->view());
  }
}

TEST_F(ParquetChunkedReaderTest, TestChunkedReadSimpleStructs)
{
  auto constexpr num_rows = 100'000;
  auto const filepath     = temp_env->get_temp_filepath("chunked_read_simple_structs.parquet");

  auto const int_iter = thrust::make_counting_iterator(0);
  auto const str_iter =
    cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) { return std::to_string(i); });

  auto const a = int32s_col(int_iter, int_iter + num_rows);
  auto const b = [=] {
    auto child1 = int32s_col(int_iter, int_iter + num_rows);
    auto child2 = int32s_col(int_iter + num_rows, int_iter + num_rows * 2);
    auto child3 = strings_col{str_iter, str_iter + num_rows};
    return structs_col{{child1, child2, child3}};
  }();
  auto const input = cudf::table_view{{a, b}};

  auto const write_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
      .max_page_size_bytes(512 * 1024)  // 512KB per page
      .max_page_size_rows(20000)        // 20k rows per page
      .build();
  cudf::io::write_parquet(write_opts);

  auto const [result, num_chunks] = chunked_read(filepath, 500'000);
  EXPECT_EQ(num_chunks, 5);

  CUDF_TEST_EXPECT_TABLES_EQUAL(input, result->view());
}
