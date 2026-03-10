/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table_view.hpp>

#include <random>
#include <vector>

namespace {

constexpr cudf::size_type num_rows       = 5000;
constexpr cudf::size_type cardinality    = num_rows / 10;
constexpr cudf::size_type row_group_size = 1000;
constexpr unsigned int seed              = 0xcece;
constexpr double null_probability        = 0.1;

template <typename T>
cudf::test::fixed_width_column_wrapper<T, int32_t> column_low_cardinality()
  requires(not std::is_same_v<T, cudf::string_view>)
{
  std::mt19937 engine(seed);
  std::uniform_int_distribution<int32_t> value_dist(0, cardinality - 1);
  std::bernoulli_distribution null_dist(null_probability);

  std::vector<int32_t> values(num_rows);
  std::vector<bool> valids(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    values[i] = value_dist(engine);
    valids[i] = not null_dist(engine);
  }

  return cudf::test::fixed_width_column_wrapper<T, int32_t>(
    values.begin(), values.end(), valids.begin());
}

template <typename T>
cudf::test::strings_column_wrapper column_low_cardinality()
  requires(std::is_same_v<T, cudf::string_view>)
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

cudf::io::table_with_metadata read_parquet(std::string const& filepath)
{
  auto const read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).build();
  return cudf::io::read_parquet(read_opts);
}

[[maybe_unused]] cudf::io::table_with_metadata read_parquet_dict_encoded(
  std::string const& filepath)
{
  // Note to Bret: This function should yield a table with an already dictionary encoded column.
  CUDF_FAIL("Not implemented");
}

}  // namespace

// TODO(mh): Temporarily only using integral and string types. Set this to the list of types in
// `hybrid_scan_filters_test.cpp` when we're ready
using DictionaryTestTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::StringTypes>;

template <typename T>
struct ParquetDictDecodeTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ParquetDictDecodeTest, DictionaryTestTypes);

TYPED_TEST(ParquetDictDecodeTest, DictDecodeParquet)
{
  using T = TypeParam;

  // Generate low cardinality data
  auto input_col = column_low_cardinality<T>();

  // Write parquet
  auto const input_tbl = cudf::table_view{{input_col}};
  auto const filepath  = temp_env->get_temp_filepath("DictDecodeParquet.parquet");
  write_parquet(input_tbl, filepath);

  // Dictionary encode and decode input column
  auto const dict_input      = cudf::dictionary::encode(input_col);
  auto const dict_input_view = cudf::dictionary_column_view(dict_input->view());
  auto const decoded_input   = cudf::dictionary::decode(dict_input_view);

  // Test read parquet into standard cudf column
  {
    auto const read_table = read_parquet(filepath).tbl;
    EXPECT_EQ(read_table->num_rows(), num_rows);
    EXPECT_EQ(read_table->num_columns(), 1);

    // Compare input and read columns
    auto const read_col = read_table->view().column(0);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input_col, read_col);

    // Dictionary encode and decode read column
    auto const dict_read = cudf::dictionary::encode(read_col);
    cudf::dictionary_column_view dict_read_view(dict_read->view());
    auto const decoded_read = cudf::dictionary::decode(dict_read_view);

    // Compare
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(read_col, decoded_input->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input_col, decoded_read->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded_input->view(), decoded_read->view());
  }

  // Bret's tests: Test read parquet into dictionary encoded cudf column
#if 0
  {
    auto const read_table =
      read_parquet_dict_encoded(filepath).tbl;  ///< This will CUDF_FAIL for now
    EXPECT_EQ(read_table->num_rows(), num_rows);
    EXPECT_EQ(read_table->num_columns(), 1);

    // Decode the read dictionary column
    cudf::dictionary_column_view dict_read_view(read_table->view().column(0));
    auto const decoded_read = cudf::dictionary::decode(dict_read_view);

    // Compare
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input_col, decoded_read->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded_input->view(), decoded_read->view());
  }
#endif
}
