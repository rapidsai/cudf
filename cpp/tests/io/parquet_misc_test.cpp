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

#include "parquet_common.hpp"

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
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <src/io/parquet/compact_protocol_reader.hpp>
#include <src/io/parquet/parquet.hpp>
#include <src/io/parquet/parquet_gpu.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <fstream>
#include <random>
#include <type_traits>

TYPED_TEST_SUITE(ParquetWriterDeltaTest, SupportedDeltaTestTypes);

TYPED_TEST(ParquetWriterDeltaTest, SupportedDeltaTestTypes)
{
  using T   = TypeParam;
  auto col0 = testdata::ascending<T>();
  auto col1 = testdata::unordered<T>();

  auto const expected = table_view{{col0, col1}};

  auto const filepath = temp_env->get_temp_filepath("DeltaBinaryPacked.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(true)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(ParquetWriterDeltaTest, SupportedDeltaTestTypesSliced)
{
  using T                = TypeParam;
  constexpr int num_rows = 4'000;
  auto col0              = testdata::ascending<T>();
  auto col1              = testdata::unordered<T>();

  auto const expected = table_view{{col0, col1}};
  auto expected_slice = cudf::slice(expected, {num_rows, 2 * num_rows});
  ASSERT_EQ(expected_slice[0].num_rows(), num_rows);

  auto const filepath = temp_env->get_temp_filepath("DeltaBinaryPackedSliced.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected_slice)
      .write_v2_headers(true)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_slice, result.tbl->view());
}

TYPED_TEST(ParquetWriterDeltaTest, SupportedDeltaListSliced)
{
  using T = TypeParam;

  constexpr int num_slice = 4'000;
  constexpr int num_rows  = 32 * 1024;

  std::mt19937 gen(6542);
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto values = thrust::make_counting_iterator(0);

  // list<T>
  constexpr int vals_per_row = 4;
  auto c1_offset_iter        = cudf::detail::make_counting_transform_iterator(
    0, [vals_per_row](cudf::size_type idx) { return idx * vals_per_row; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> c1_offsets(c1_offset_iter,
                                                                     c1_offset_iter + num_rows + 1);
  cudf::test::fixed_width_column_wrapper<T> c1_vals(
    values, values + (num_rows * vals_per_row), valids);
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids, valids + num_rows);

  auto _c1 = cudf::make_lists_column(
    num_rows, c1_offsets.release(), c1_vals.release(), null_count, std::move(null_mask));
  auto c1 = cudf::purge_nonempty_nulls(*_c1);

  auto const expected = table_view{{*c1}};
  auto expected_slice = cudf::slice(expected, {num_slice, 2 * num_slice});
  ASSERT_EQ(expected_slice[0].num_rows(), num_slice);

  auto const filepath = temp_env->get_temp_filepath("DeltaBinaryPackedListSliced.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected_slice)
      .write_v2_headers(true)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_slice, result.tbl->view());
}

// test the allowed bit widths for dictionary encoding
INSTANTIATE_TEST_SUITE_P(ParquetDictionaryTest,
                         ParquetSizedTest,
                         testing::Range(1, 25),
                         testing::PrintToStringParamName());

TEST_P(ParquetSizedTest, DictionaryTest)
{
  unsigned int const cardinality = (1 << (GetParam() - 1)) + 1;
  unsigned int const nrows       = std::max(cardinality * 3 / 2, 3'000'000U);

  auto elements       = cudf::detail::make_counting_transform_iterator(0, [cardinality](auto i) {
    return "a unique string value suffixed with " + std::to_string(i % cardinality);
  });
  auto const col0     = cudf::test::strings_column_wrapper(elements, elements + nrows);
  auto const expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("DictionaryTest.parquet");
  // set row group size so that there will be only one row group
  // no compression so we can easily read page data
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::NONE)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
      .row_group_size_rows(nrows)
      .row_group_size_bytes(512 * 1024 * 1024);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());

  // make sure dictionary was used
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);
  auto used_dict = [&fmd]() {
    for (auto enc : fmd.row_groups[0].columns[0].meta_data.encodings) {
      if (enc == cudf::io::parquet::detail::Encoding::PLAIN_DICTIONARY or
          enc == cudf::io::parquet::detail::Encoding::RLE_DICTIONARY) {
        return true;
      }
    }
    return false;
  };
  EXPECT_TRUE(used_dict());

  // and check that the correct number of bits was used
  auto const oi    = read_offset_index(source, fmd.row_groups[0].columns[0]);
  auto const nbits = read_dict_bits(source, oi.page_locations[0]);
  EXPECT_EQ(nbits, GetParam());
}
