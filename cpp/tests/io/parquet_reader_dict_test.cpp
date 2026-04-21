/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <random>
#include <string>
#include <vector>

namespace {

constexpr cudf::size_type num_rows       = 5000;
constexpr cudf::size_type cardinality    = num_rows / 10;
constexpr cudf::size_type row_group_size = 1000;
constexpr unsigned int seed              = 0xcece;
constexpr double null_probability        = 0.1;

cudf::test::strings_column_wrapper make_low_cardinality_strings()
{
  std::mt19937 engine(seed);
  std::uniform_int_distribution<int> value_dist(0, cardinality - 1);
  std::bernoulli_distribution null_dist(null_probability);

  std::vector<std::string> strings(num_rows);
  std::vector<bool> valids(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    strings[i] = "str_" + std::to_string(value_dist(engine));
    valids[i]  = not null_dist(engine);
  }

  return cudf::test::strings_column_wrapper(strings.begin(), strings.end(), valids.begin());
}

void write_parquet(cudf::table_view const& input, std::string const& filepath)
{
  auto const options =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath})
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
      .compression(cudf::io::compression_type::NONE)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .build();

  cudf::io::chunked_parquet_writer writer(options);
  for (auto offset = 0; offset < input.num_rows(); offset += row_group_size) {
    auto const length = std::min(row_group_size, input.num_rows() - offset);
    auto const chunk  = cudf::slice(input, {offset, offset + length});
    writer.write(chunk.front());
  }
  writer.close();
}

cudf::io::table_with_metadata read_parquet_as_dict(std::string const& filepath)
{
  auto const read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .try_output_dict_columns(true)
      .build();
  return cudf::io::read_parquet(read_opts);
}

}  // namespace

struct ParquetReaderDictTest : public cudf::test::BaseFixture {};

// A flat string column that is fully dictionary-encoded in every row group should be returned
// as a DICTIONARY32 column when `try_output_dict_columns` is enabled, and the decoded keys
// should match the original input.
TEST_F(ParquetReaderDictTest, FlatStringDictTranscode)
{
  auto input_col = make_low_cardinality_strings();

  auto const input_tbl = cudf::table_view{{input_col}};
  auto const filepath  = temp_env->get_temp_filepath("FlatStringDictTranscode.parquet");
  write_parquet(input_tbl, filepath);

  auto const dict_input      = cudf::dictionary::encode(input_col);
  auto const dict_input_view = cudf::dictionary_column_view(dict_input->view());
  auto const decoded_input   = cudf::dictionary::decode(dict_input_view);

  auto const read_table = read_parquet_as_dict(filepath).tbl;
  ASSERT_EQ(read_table->num_rows(), num_rows);
  ASSERT_EQ(read_table->num_columns(), 1);

  auto const read_col = read_table->view().column(0);
  ASSERT_EQ(read_col.type().id(), cudf::type_id::DICTIONARY32)
    << "Expected the reader to produce a DICTIONARY32 column when try_output_dict_columns is on";

  cudf::dictionary_column_view dict_read_view(read_col);
  auto const decoded_read = cudf::dictionary::decode(dict_read_view);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input_col, decoded_read->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded_input->view(), decoded_read->view());
}

// When the option is not set, the reader should still produce a plain STRING column, regardless
// of whether the source file is fully dictionary-encoded.
TEST_F(ParquetReaderDictTest, FlatStringNoTranscodeByDefault)
{
  auto input_col = make_low_cardinality_strings();

  auto const input_tbl = cudf::table_view{{input_col}};
  auto const filepath  = temp_env->get_temp_filepath("FlatStringNoTranscodeByDefault.parquet");
  write_parquet(input_tbl, filepath);

  auto const read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).build();
  auto const read_table = cudf::io::read_parquet(read_opts).tbl;

  ASSERT_EQ(read_table->num_columns(), 1);
  auto const read_col = read_table->view().column(0);
  ASSERT_EQ(read_col.type().id(), cudf::type_id::STRING);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input_col, read_col);
}
