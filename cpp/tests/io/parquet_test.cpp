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

TEST_F(ParquetChunkedReaderTest, Test)
{
  std::mt19937 gen(6542);
  std::bernoulli_distribution bn(0.7f);
  // auto valids =
  //    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
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
  printf("result size: %d\n", result.tbl->num_rows());
}
