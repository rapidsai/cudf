/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <cudf/unary.hpp>
#include <cudf/utilities/span.hpp>

#include <src/io/parquet/compact_protocol_reader.hpp>
#include <src/io/parquet/parquet.hpp>
#include <src/io/parquet/parquet_gpu.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>

// Base test fixture for tests
struct ParquetWriterTest : public cudf::test::BaseFixture {};

TEST_F(ParquetWriterTest, NonNullable)
{
  auto in_filepath  = "/home/nghiat/Downloads/batch_0_part_0.parquet";
  auto out_filepath = "/home/nghiat/Downloads/out.parquet";

  {
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{in_filepath});
    auto df = cudf::io::read_parquet(read_opts);
    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{out_filepath}, *df.tbl);
    cudf::io::write_parquet(write_opts);
  }

  {
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{out_filepath});
    auto df = cudf::io::read_parquet(read_opts);
  }
}
