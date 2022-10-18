/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <fstream>
#include <type_traits>

struct ParquetChunkedReaderTest : public cudf::test::BaseFixture {
};

#if 0
TEST_F(ParquetChunkedReaderTest, Test)
{
  std::mt19937 gen(6542);
  std::bernoulli_distribution bn(0.7f);
  auto values = thrust::make_counting_iterator(0);

  constexpr cudf::size_type num_rows = 40000;
  cudf::test::fixed_width_column_wrapper<int> a(values, values + num_rows);
  cudf::test::fixed_width_column_wrapper<int64_t> b(values, values + num_rows);

  cudf::table_view t({a, b});
  cudf::io::parquet_writer_options opts = cudf::io::parquet_writer_options::builder(
    cudf::io::sink_info{"/tmp/chunked_splits.parquet"}, t);
  cudf::io::write_parquet(opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{"/tmp/chunked_splits.parquet"});
  auto result = cudf::io::read_parquet(in_opts);
  printf("\nResult size read all: %d\n\n", result.tbl->num_rows());
}

#else
TEST_F(ParquetChunkedReaderTest, TestChunkedRead)
{
  std::mt19937 gen(6542);
  std::bernoulli_distribution bn(0.7f);
  auto values = thrust::make_counting_iterator(0);

  constexpr cudf::size_type num_rows = 40000;
  cudf::test::fixed_width_column_wrapper<int> a(values, values + num_rows);
  cudf::test::fixed_width_column_wrapper<int64_t> b(values, values + num_rows);

  auto filepath = std::string{"/tmp/chunked_splits.parquet"};
  cudf::table_view t({a, b});
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, t);
  cudf::io::write_parquet(opts);

  //========================================================================================
  {
    cudf::io::parquet_reader_options in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    auto result = cudf::io::read_parquet(in_opts);
    printf("Result size read full: %d\n\n\n\n\n", result.tbl->num_rows());
  }

  cudf::io::chunked_parquet_reader_options in_opts =
    cudf::io::chunked_parquet_reader_options::builder(cudf::io::source_info{filepath});

  cudf::io::chunked_parquet_reader reader(in_opts);

  int count{0};
  while (reader.has_next()) {
    printf("\n\nhas next %d\n\n", count++);

    auto result = reader.read_chunk();
    printf("Result size: %d\n\n\n\n\n", result.tbl->num_rows());
  }
}

TEST_F(ParquetChunkedReaderTest, TestChunkedReadString)
{
  // values the cudf parquet writer uses
  // constexpr size_t default_max_page_size_bytes    = 512 * 1024;   ///< 512KB per page
  // constexpr size_type default_max_page_size_rows  = 20000;        ///< 20k rows per page
  std::mt19937 gen(6542);
  std::bernoulli_distribution bn(0.7f);
  auto values                        = thrust::make_counting_iterator(0);
  constexpr cudf::size_type num_rows = 60000;
  // ints                                            Page    total bytes   cumulative bytes
  // 20000 rows of 4 bytes each                    = A0      80000         80000
  // 20000 rows of 4 bytes each                    = A1      80000         160000
  // 20000 rows of 4 bytes each                    = A2      80000         240000
  cudf::test::fixed_width_column_wrapper<int> a(values, values + num_rows);
  // strings                                         Page    total bytes   cumulative bytes
  // 20000 rows of 1 char each    (20000  + 80004) = B0      100004        100004
  // 20000 rows of 4 chars each   (80000  + 80004) = B1      160004        260008
  // 20000 rows of 16 chars each  (320000 + 80004) = B2      400004        660012
  std::vector<std::string> strings{"a", "bbbb", "cccccccccccccccc"};
  auto const str_iter = cudf::detail::make_counting_transform_iterator(0, [&](int i) {
    if (i < 20000) { return strings[0]; }
    if (i < 40000) { return strings[1]; }
    return strings[2];
  });
  cudf::test::strings_column_wrapper b{str_iter, str_iter + num_rows};
  // cumulative sizes
  // A0 + B0 :  180004
  // A1 + B1 :  420008
  // A2 + B2 :  900012
  //                                                    skip_rows / num_rows
  // chunked_read_size of 500000  should give 2 chunks: {0, 40000},           {40000, 20000}
  // chunked_read_size of 1000000 should give 1 chunks: {0, 60000},
  auto write_tbl = cudf::table_view{{a, b}};
  auto filepath  = std::string{"/tmp/chunked_splits_strings.parquet"};
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, write_tbl);
  cudf::io::write_parquet(out_opts);
  //========================================================================================

  {
    cudf::io::parquet_reader_options in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    auto result = cudf::io::read_parquet(in_opts);
    printf("Result size read full: %d\n\n\n\n\n", result.tbl->num_rows());
  }

  cudf::io::chunked_parquet_reader_options in_opts =
    cudf::io::chunked_parquet_reader_options::builder(cudf::io::source_info{filepath});

  cudf::io::chunked_parquet_reader reader(in_opts);

  int count{0};
  while (reader.has_next()) {
    printf("\n\nhas next %d\n\n", count++);

    auto result = reader.read_chunk();
    printf("Result size: %d\n\n\n\n\n", result.tbl->num_rows());
  }
}
#endif
