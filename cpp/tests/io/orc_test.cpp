/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/io/functions.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <type_traits>

namespace cudf_io = cudf::io;

template <typename T>
using column_wrapper = typename std::conditional<std::is_same<T, cudf::string_view>::value,
                                                 cudf::test::strings_column_wrapper,
                                                 cudf::test::fixed_width_column_wrapper<T>>::type;
using column         = cudf::column;
using table          = cudf::table;
using table_view     = cudf::table_view;

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

template <typename T>
std::unique_ptr<cudf::table> create_random_fixed_table(cudf::size_type num_columns,
                                                       cudf::size_type num_rows,
                                                       bool include_validity)
{
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });
  std::vector<cudf::test::fixed_width_column_wrapper<T>> src_cols(num_columns);
  for (int idx = 0; idx < num_columns; idx++) {
    auto rand_elements =
      cudf::test::make_counting_transform_iterator(0, [](T i) { return rand(); });
    if (include_validity) {
      src_cols[idx] =
        cudf::test::fixed_width_column_wrapper<T>(rand_elements, rand_elements + num_rows, valids);
    } else {
      src_cols[idx] =
        cudf::test::fixed_width_column_wrapper<T>(rand_elements, rand_elements + num_rows);
    }
  }
  std::vector<std::unique_ptr<cudf::column>> columns(num_columns);
  std::transform(src_cols.begin(),
                 src_cols.end(),
                 columns.begin(),
                 [](cudf::test::fixed_width_column_wrapper<T>& in) {
                   auto ret = in.release();
                   ret->has_nulls();
                   return ret;
                 });
  return std::make_unique<cudf::table>(std::move(columns));
}

// Base test fixture for tests
struct OrcWriterTest : public cudf::test::BaseFixture {
};

// Typed test fixture for numeric type tests
template <typename T>
struct OrcWriterNumericTypeTest : public OrcWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Typed test fixture for timestamp type tests
template <typename T>
struct OrcWriterTimestampTypeTest : public OrcWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Declare typed test cases
// TODO: Replace with `NumericTypes` when unsigned support is added. Issue #5351
using SupportedTypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, bool, float, double>;
TYPED_TEST_CASE(OrcWriterNumericTypeTest, SupportedTypes);
using SupportedTimestampTypes =
  cudf::test::RemoveIf<cudf::test::ContainedIn<cudf::test::Types<cudf::timestamp_D>>,
                       cudf::test::TimestampTypes>;
TYPED_TEST_CASE(OrcWriterTimestampTypeTest, SupportedTimestampTypes);

// Base test fixture for chunked writer tests
struct OrcChunkedWriterTest : public cudf::test::BaseFixture {
};

// Typed test fixture for numeric type tests
template <typename T>
struct OrcChunkedWriterNumericTypeTest : public OrcChunkedWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Declare typed test cases
TYPED_TEST_CASE(OrcChunkedWriterNumericTypeTest, SupportedTypes);

namespace {
// Generates a vector of uniform random values of type T
template <typename T>
inline auto random_values(size_t size)
{
  std::vector<T> values(size);

  using T1 = T;
  using uniform_distribution =
    typename std::conditional_t<std::is_same<T1, bool>::value,
                                std::bernoulli_distribution,
                                std::conditional_t<std::is_floating_point<T1>::value,
                                                   std::uniform_real_distribution<T1>,
                                                   std::uniform_int_distribution<T1>>>;

  static constexpr auto seed = 0xf00d;
  static std::mt19937 engine{seed};
  static uniform_distribution dist{};
  std::generate_n(values.begin(), size, [&]() { return T{dist(engine)}; });

  return values;
}

// Helper function to compare two tables
void expect_tables_equal(cudf::table_view const& lhs, cudf::table_view const& rhs)
{
  EXPECT_EQ(lhs.num_columns(), rhs.num_columns());
  auto expected = lhs.begin();
  auto result   = rhs.begin();
  while (result != rhs.end()) { cudf::test::expect_columns_equal(*expected++, *result++); }
}

}  // namespace

TYPED_TEST(OrcWriterNumericTypeTest, SingleColumn)
{
  auto sequence =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i); });
  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam> col(sequence, sequence + num_rows, validity);

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(1, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcSingleColumn.orc");
  cudf_io::write_orc_args out_args{cudf_io::sink_info{filepath}, expected->view()};
  cudf_io::write_orc(out_args);

  cudf_io::read_orc_args in_args{cudf_io::source_info{filepath}};
  in_args.use_index = false;
  auto result       = cudf_io::read_orc(in_args);

  expect_tables_equal(expected->view(), result.tbl->view());
}

TYPED_TEST(OrcWriterNumericTypeTest, SingleColumnWithNulls)
{
  auto sequence =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i); });
  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i % 2); });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam> col(sequence, sequence + num_rows, validity);

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(1, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcSingleColumnWithNulls.orc");
  cudf_io::write_orc_args out_args{cudf_io::sink_info{filepath}, expected->view()};
  cudf_io::write_orc(out_args);

  cudf_io::read_orc_args in_args{cudf_io::source_info{filepath}};
  in_args.use_index = false;
  auto result       = cudf_io::read_orc(in_args);

  expect_tables_equal(expected->view(), result.tbl->view());
}

TYPED_TEST(OrcWriterTimestampTypeTest, Timestamps)
{
  auto sequence = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return TypeParam(std::rand() / 10); });
  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam> col(sequence, sequence + num_rows, validity);

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(1, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcTimestamps.orc");
  cudf_io::write_orc_args out_args{cudf_io::sink_info{filepath}, expected->view()};
  cudf_io::write_orc(out_args);

  cudf_io::read_orc_args in_args{cudf_io::source_info{filepath}};
  in_args.use_index      = false;
  in_args.timestamp_type = this->type();
  auto result            = cudf_io::read_orc(in_args);

  expect_tables_equal(expected->view(), result.tbl->view());
}

TYPED_TEST(OrcWriterTimestampTypeTest, TimestampsWithNulls)
{
  auto sequence = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return TypeParam(std::rand() / 10); });
  auto validity =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i > 30) && (i < 60); });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam> col(sequence, sequence + num_rows, validity);

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(1, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcTimestampsWithNulls.orc");
  cudf_io::write_orc_args out_args{cudf_io::sink_info{filepath}, expected->view()};
  cudf_io::write_orc(out_args);

  cudf_io::read_orc_args in_args{cudf_io::source_info{filepath}};
  in_args.use_index      = false;
  in_args.timestamp_type = this->type();
  auto result            = cudf_io::read_orc(in_args);

  expect_tables_equal(expected->view(), result.tbl->view());
}

TEST_F(OrcWriterTest, MultiColumn)
{
  constexpr auto num_rows = 100;

  // auto col0_data = random_values<bool>(num_rows);
  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<float>(num_rows);
  auto col5_data = random_values<double>(num_rows);
  auto validity  = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

  // column_wrapper<bool> col0{
  //    col0_data.begin(), col0_data.end(), validity};
  column_wrapper<int8_t> col1{col1_data.begin(), col1_data.end(), validity};
  column_wrapper<int16_t> col2{col2_data.begin(), col2_data.end(), validity};
  column_wrapper<int32_t> col3{col3_data.begin(), col3_data.end(), validity};
  column_wrapper<float> col4{col4_data.begin(), col4_data.end(), validity};
  column_wrapper<double> col5{col5_data.begin(), col5_data.end(), validity};

  cudf_io::table_metadata expected_metadata;
  // expected_metadata.column_names.emplace_back("bools");
  expected_metadata.column_names.emplace_back("int8s");
  expected_metadata.column_names.emplace_back("int16s");
  expected_metadata.column_names.emplace_back("int32s");
  expected_metadata.column_names.emplace_back("floats");
  expected_metadata.column_names.emplace_back("doubles");

  std::vector<std::unique_ptr<column>> cols;
  // cols.push_back(col0.release());
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  cols.push_back(col3.release());
  cols.push_back(col4.release());
  cols.push_back(col5.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(5, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcMultiColumn.orc");
  cudf_io::write_orc_args out_args{
    cudf_io::sink_info{filepath}, expected->view(), &expected_metadata};
  cudf_io::write_orc(out_args);

  cudf_io::read_orc_args in_args{cudf_io::source_info{filepath}};
  in_args.use_index = false;
  auto result       = cudf_io::read_orc(in_args);

  expect_tables_equal(expected->view(), result.tbl->view());
  EXPECT_EQ(expected_metadata.column_names, result.metadata.column_names);
}

TEST_F(OrcWriterTest, MultiColumnWithNulls)
{
  constexpr auto num_rows = 100;

  // auto col0_data = random_values<bool>(num_rows);
  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<float>(num_rows);
  auto col5_data = random_values<double>(num_rows);
  // auto col0_mask = cudf::test::make_counting_transform_iterator(
  //    0, [](auto i) { return (i % 2); });
  auto col1_mask = cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i < 10); });
  auto col2_mask = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });
  auto col3_mask =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i == (num_rows - 1)); });
  auto col4_mask =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i >= 40 || i <= 60); });
  auto col5_mask = cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i > 80); });

  // column_wrapper<bool> col0{
  //    col0_data.begin(), col0_data.end(), col0_mask};
  column_wrapper<int8_t> col1{col1_data.begin(), col1_data.end(), col1_mask};
  column_wrapper<int16_t> col2{col2_data.begin(), col2_data.end(), col2_mask};
  column_wrapper<int32_t> col3{col3_data.begin(), col3_data.end(), col3_mask};
  column_wrapper<float> col4{col4_data.begin(), col4_data.end(), col4_mask};
  column_wrapper<double> col5{col5_data.begin(), col5_data.end(), col5_mask};

  cudf_io::table_metadata expected_metadata;
  // expected_metadata.column_names.emplace_back("bools");
  expected_metadata.column_names.emplace_back("int8s");
  expected_metadata.column_names.emplace_back("int16s");
  expected_metadata.column_names.emplace_back("int32s");
  expected_metadata.column_names.emplace_back("floats");
  expected_metadata.column_names.emplace_back("doubles");

  std::vector<std::unique_ptr<column>> cols;
  // cols.push_back(col0.release());
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  cols.push_back(col3.release());
  cols.push_back(col4.release());
  cols.push_back(col5.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(5, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcMultiColumnWithNulls.orc");
  cudf_io::write_orc_args out_args{
    cudf_io::sink_info{filepath}, expected->view(), &expected_metadata};
  cudf_io::write_orc(out_args);

  cudf_io::read_orc_args in_args{cudf_io::source_info{filepath}};
  in_args.use_index = false;
  auto result       = cudf_io::read_orc(in_args);

  expect_tables_equal(expected->view(), result.tbl->view());
  EXPECT_EQ(expected_metadata.column_names, result.metadata.column_names);
}

TEST_F(OrcWriterTest, Strings)
{
  std::vector<const char*> strings{
    "Monday", "Monday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  const auto num_rows = strings.size();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);
  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

  column_wrapper<int> col0{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<cudf::string_view> col1{strings.begin(), strings.end()};
  column_wrapper<float> col2{seq_col2.begin(), seq_col2.end(), validity};

  cudf_io::table_metadata expected_metadata;
  expected_metadata.column_names.emplace_back("col_other");
  expected_metadata.column_names.emplace_back("col_string");
  expected_metadata.column_names.emplace_back("col_another");

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col0.release());
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(3, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcStrings.orc");
  cudf_io::write_orc_args out_args{
    cudf_io::sink_info{filepath}, expected->view(), &expected_metadata};
  cudf_io::write_orc(out_args);

  cudf_io::read_orc_args in_args{cudf_io::source_info{filepath}};
  in_args.use_index = false;
  auto result       = cudf_io::read_orc(in_args);

  expect_tables_equal(expected->view(), result.tbl->view());
  EXPECT_EQ(expected_metadata.column_names, result.metadata.column_names);
}

TEST_F(OrcWriterTest, HostBuffer)
{
  constexpr auto num_rows = 100 << 10;
  const auto seq_col      = random_values<int>(num_rows);
  const auto validity =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });
  column_wrapper<int> col{seq_col.begin(), seq_col.end(), validity};

  cudf_io::table_metadata expected_metadata;
  expected_metadata.column_names.emplace_back("col_other");

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col.release());
  const auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(1, expected->num_columns());

  std::vector<char> out_buffer;
  cudf_io::write_orc_args out_args{
    cudf_io::sink_info(&out_buffer), expected->view(), &expected_metadata};
  cudf_io::write_orc(out_args);

  cudf_io::read_orc_args in_args{cudf_io::source_info(out_buffer.data(), out_buffer.size())};
  in_args.use_index = false;
  const auto result = cudf_io::read_orc(in_args);

  expect_tables_equal(expected->view(), result.tbl->view());
  EXPECT_EQ(expected_metadata.column_names, result.metadata.column_names);
}

TEST_F(OrcChunkedWriterTest, SingleTable)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedSingle.orc");
  cudf_io::write_orc_chunked_args args{cudf_io::sink_info{filepath}};
  auto state = cudf_io::write_orc_chunked_begin(args);
  cudf_io::write_orc_chunked(*table1, state);
  cudf_io::write_orc_chunked_end(state);

  cudf_io::read_orc_args read_args{cudf_io::source_info{filepath}};
  auto result = cudf_io::read_orc(read_args);

  expect_tables_equal(*result.tbl, *table1);
}

TEST_F(OrcChunkedWriterTest, SimpleTable)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, true);

  auto full_table = cudf::concatenate({*table1, *table2});

  auto filepath = temp_env->get_temp_filepath("ChunkedSimple.orc");
  cudf_io::write_orc_chunked_args args{cudf_io::sink_info{filepath}};
  auto state = cudf_io::write_orc_chunked_begin(args);
  cudf_io::write_orc_chunked(*table1, state);
  cudf_io::write_orc_chunked(*table2, state);
  cudf_io::write_orc_chunked_end(state);

  cudf_io::read_orc_args read_args{cudf_io::source_info{filepath}};
  auto result = cudf_io::read_orc(read_args);

  expect_tables_equal(*result.tbl, *full_table);
}

TEST_F(OrcChunkedWriterTest, LargeTables)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(512, 4096, true);
  auto table2 = create_random_fixed_table<int>(512, 8192, true);

  auto full_table = cudf::concatenate({*table1, *table2});

  auto filepath = temp_env->get_temp_filepath("ChunkedLarge.orc");
  cudf_io::write_orc_chunked_args args{cudf_io::sink_info{filepath}};
  auto state = cudf_io::write_orc_chunked_begin(args);
  cudf_io::write_orc_chunked(*table1, state);
  cudf_io::write_orc_chunked(*table2, state);
  cudf_io::write_orc_chunked_end(state);

  cudf_io::read_orc_args read_args{cudf_io::source_info{filepath}};
  auto result = cudf_io::read_orc(read_args);

  expect_tables_equal(*result.tbl, *full_table);
}

TEST_F(OrcChunkedWriterTest, ManyTables)
{
  srand(31337);
  std::vector<std::unique_ptr<table>> tables;
  std::vector<table_view> table_views;
  constexpr int num_tables = 96;
  for (int idx = 0; idx < num_tables; idx++) {
    auto tbl = create_random_fixed_table<int>(16, 64, true);
    table_views.push_back(*tbl);
    tables.push_back(std::move(tbl));
  }

  auto expected = cudf::concatenate(table_views);

  auto filepath = temp_env->get_temp_filepath("ChunkedManyTables.orc");
  cudf_io::write_orc_chunked_args args{cudf_io::sink_info{filepath}};
  auto state = cudf_io::write_orc_chunked_begin(args);
  std::for_each(table_views.begin(), table_views.end(), [&state](table_view const& tbl) {
    cudf_io::write_orc_chunked(tbl, state);
  });
  cudf_io::write_orc_chunked_end(state);

  cudf_io::read_orc_args read_args{cudf_io::source_info{filepath}};
  auto result = cudf_io::read_orc(read_args);

  expect_tables_equal(*result.tbl, *expected);
}

TEST_F(OrcChunkedWriterTest, Strings)
{
  std::vector<std::unique_ptr<cudf::column>> cols;

  bool mask1[] = {1, 1, 0, 1, 1, 1, 1};
  std::vector<const char*> h_strings1{"four", "score", "and", "seven", "years", "ago", "abcdefgh"};
  cudf::test::strings_column_wrapper strings1(h_strings1.begin(), h_strings1.end(), mask1);
  cols.push_back(strings1.release());
  cudf::table tbl1(std::move(cols));

  bool mask2[] = {0, 1, 1, 1, 1, 1, 1};
  std::vector<const char*> h_strings2{"ooooo", "ppppppp", "fff", "j", "cccc", "bbb", "zzzzzzzzzzz"};
  cudf::test::strings_column_wrapper strings2(h_strings2.begin(), h_strings2.end(), mask2);
  cols.push_back(strings2.release());
  cudf::table tbl2(std::move(cols));

  auto expected = cudf::concatenate({tbl1, tbl2});

  auto filepath = temp_env->get_temp_filepath("ChunkedStrings.orc");
  cudf_io::write_orc_chunked_args args{cudf_io::sink_info{filepath}};
  auto state = cudf_io::write_orc_chunked_begin(args);
  cudf_io::write_orc_chunked(tbl1, state);
  cudf_io::write_orc_chunked(tbl2, state);
  cudf_io::write_orc_chunked_end(state);

  cudf_io::read_orc_args read_args{cudf_io::source_info{filepath}};
  auto result = cudf_io::read_orc(read_args);

  expect_tables_equal(*result.tbl, *expected);
}

TEST_F(OrcChunkedWriterTest, MismatchedTypes)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(4, 4, true);
  auto table2 = create_random_fixed_table<float>(4, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedMismatchedTypes.orc");
  cudf_io::write_orc_chunked_args args{cudf_io::sink_info{filepath}};
  auto state = cudf_io::write_orc_chunked_begin(args);
  cudf_io::write_orc_chunked(*table1, state);
  EXPECT_THROW(cudf_io::write_orc_chunked(*table2, state), cudf::logic_error);
  cudf_io::write_orc_chunked_end(state);
}

TEST_F(OrcChunkedWriterTest, MismatchedStructure)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(4, 4, true);
  auto table2 = create_random_fixed_table<int>(3, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedMismatchedStructure.orc");
  cudf_io::write_orc_chunked_args args{cudf_io::sink_info{filepath}};
  auto state = cudf_io::write_orc_chunked_begin(args);
  cudf_io::write_orc_chunked(*table1, state);
  EXPECT_THROW(cudf_io::write_orc_chunked(*table2, state), cudf::logic_error);
  cudf_io::write_orc_chunked_end(state);
}

TEST_F(OrcChunkedWriterTest, ReadStripes)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, true);

  auto full_table = cudf::concatenate({*table2, *table1, *table2});

  auto filepath = temp_env->get_temp_filepath("ChunkedStripes.orc");
  cudf_io::write_orc_chunked_args args{cudf_io::sink_info{filepath}};
  auto state = cudf_io::write_orc_chunked_begin(args);
  cudf_io::write_orc_chunked(*table1, state);
  cudf_io::write_orc_chunked(*table2, state);
  cudf_io::write_orc_chunked_end(state);

  cudf_io::read_orc_args read_args{cudf_io::source_info{filepath}};
  read_args.stripe_list = {1, 0, 1};
  auto result           = cudf_io::read_orc(read_args);

  expect_tables_equal(*result.tbl, *full_table);
}

TEST_F(OrcChunkedWriterTest, ReadStripesError)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedStripesError.orc");
  cudf_io::write_orc_chunked_args args{cudf_io::sink_info{filepath}};
  auto state = cudf_io::write_orc_chunked_begin(args);
  cudf_io::write_orc_chunked(*table1, state);
  cudf_io::write_orc_chunked_end(state);

  cudf_io::read_orc_args read_args{cudf_io::source_info{filepath}};
  read_args.stripe_list = {0, 1};
  EXPECT_THROW(cudf_io::read_orc(read_args), cudf::logic_error);
  read_args.stripe_list = {-1};
  EXPECT_THROW(cudf_io::read_orc(read_args), cudf::logic_error);
}

TYPED_TEST(OrcChunkedWriterNumericTypeTest, UnalignedSize)
{
  // write out two 31 row tables and make sure they get
  // read back with all their validity bits in the right place

  using T = TypeParam;

  int num_els = 31;
  std::vector<std::unique_ptr<cudf::column>> cols;

  bool mask[] = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  T c1a[] = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
  T c1b[] = {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
  column_wrapper<T> c1a_w(c1a, c1a + num_els, mask);
  column_wrapper<T> c1b_w(c1b, c1b + num_els, mask);
  cols.push_back(c1a_w.release());
  cols.push_back(c1b_w.release());
  cudf::table tbl1(std::move(cols));

  T c2a[] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
             8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
  T c2b[] = {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
             9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
  column_wrapper<T> c2a_w(c2a, c2a + num_els, mask);
  column_wrapper<T> c2b_w(c2b, c2b + num_els, mask);
  cols.push_back(c2a_w.release());
  cols.push_back(c2b_w.release());
  cudf::table tbl2(std::move(cols));

  auto expected = cudf::concatenate({tbl1, tbl2});

  auto filepath = temp_env->get_temp_filepath("ChunkedUnalignedSize.orc");
  cudf_io::write_orc_chunked_args args{cudf_io::sink_info{filepath}};
  auto state = cudf_io::write_orc_chunked_begin(args);
  cudf_io::write_orc_chunked(tbl1, state);
  cudf_io::write_orc_chunked(tbl2, state);
  cudf_io::write_orc_chunked_end(state);

  cudf_io::read_orc_args read_args{cudf_io::source_info{filepath}};
  auto result = cudf_io::read_orc(read_args);

  expect_tables_equal(*result.tbl, *expected);
}

TYPED_TEST(OrcChunkedWriterNumericTypeTest, UnalignedSize2)
{
  // write out two 33 row tables and make sure they get
  // read back with all their validity bits in the right place

  using T = TypeParam;

  int num_els = 33;
  std::vector<std::unique_ptr<cudf::column>> cols;

  bool mask[] = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  T c1a[] = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
  T c1b[] = {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
  column_wrapper<T> c1a_w(c1a, c1a + num_els, mask);
  column_wrapper<T> c1b_w(c1b, c1b + num_els, mask);
  cols.push_back(c1a_w.release());
  cols.push_back(c1b_w.release());
  cudf::table tbl1(std::move(cols));

  T c2a[] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
             8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
  T c2b[] = {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
             9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
  column_wrapper<T> c2a_w(c2a, c2a + num_els, mask);
  column_wrapper<T> c2b_w(c2b, c2b + num_els, mask);
  cols.push_back(c2a_w.release());
  cols.push_back(c2b_w.release());
  cudf::table tbl2(std::move(cols));

  auto expected = cudf::concatenate({tbl1, tbl2});

  auto filepath = temp_env->get_temp_filepath("ChunkedUnalignedSize2.orc");
  cudf_io::write_orc_chunked_args args{cudf_io::sink_info{filepath}};
  auto state = cudf_io::write_orc_chunked_begin(args);
  cudf_io::write_orc_chunked(tbl1, state);
  cudf_io::write_orc_chunked(tbl2, state);
  cudf_io::write_orc_chunked_end(state);

  cudf_io::read_orc_args read_args{cudf_io::source_info{filepath}};
  auto result = cudf_io::read_orc(read_args);

  expect_tables_equal(*result.tbl, *expected);
}

CUDF_TEST_PROGRAM_MAIN()
