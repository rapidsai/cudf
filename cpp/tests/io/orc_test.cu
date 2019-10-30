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

#include <cudf/io/functions.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <type_traits>

namespace cudf_io = cudf::experimental::io;

template <typename T>
using column_wrapper =
    typename std::conditional<std::is_same<T, cudf::string_view>::value,
                              cudf::test::strings_column_wrapper,
                              cudf::test::fixed_width_column_wrapper<T>>::type;
using column = cudf::column;
using table = cudf::experimental::table;
using table_view = cudf::table_view;

auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(
        new cudf::test::TempDirTestEnvironment));

/**
 * @brief Base test fixture for tests
 **/
struct OrcWriterTest : public cudf::test::BaseFixture {};

/**
 * @brief Typed test fixture for type-parameterized tests
 **/
template <typename T>
struct OrcWriterTypedParamTest : public OrcWriterTest {
  auto data_type() {
    return cudf::data_type{cudf::experimental::type_to_id<T>()};
  }
};

TYPED_TEST_CASE(OrcWriterTypedParamTest, cudf::test::NumericTypes);

namespace {

/**
 * @brief Generates a vector of uniform random values of type T
 **/
template <typename T>
inline auto random_values(size_t size) {
  std::vector<T> values(size);

  using T1 = T;
  using uniform_distribution = typename std::conditional_t<
      std::is_same<T1, bool>::value, std::bernoulli_distribution,
      std::conditional_t<std::is_floating_point<T1>::value,
                         std::uniform_real_distribution<T1>,
                         std::uniform_int_distribution<T1>>>;

  static constexpr auto seed = 0xf00d;
  static std::mt19937 engine{seed};
  static uniform_distribution dist{};
  std::generate_n(values.begin(), size, [&]() { return T{dist(engine)}; });

  return values;
}

/**
 * @brief Helper function to compare two tables
 **/
void expect_tables_equal(cudf::table_view const& lhs,
                         cudf::table_view const& rhs) {
  EXPECT_EQ(lhs.num_columns(), rhs.num_columns());
  auto expected = lhs.begin();
  auto result = rhs.begin();
  while (result != rhs.end()) {
    cudf::test::expect_columns_equal(*expected++, *result++);
  }
}

}  // namespace

TYPED_TEST(OrcWriterTypedParamTest, SingleColumn) {
  auto sequence = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return TypeParam(i); });
  auto validity = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return true; });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam> col(sequence, sequence + num_rows, validity);

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col.release());
  auto expected = table{std::move(cols)};
  EXPECT_EQ(1, expected.num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcSingleColumn.orc");
  cudf_io::write_orc_args out_args{cudf_io::sink_info{filepath},
                                   expected.view()};
  cudf_io::write_orc(out_args);

  cudf_io::read_orc_args in_args{cudf_io::source_info{filepath}};
  in_args.use_index = false;
  auto result = cudf_io::read_orc(in_args);

  expect_tables_equal(expected.view(), result.view());
}

TYPED_TEST(OrcWriterTypedParamTest, SingleColumnWithNulls) {
  auto sequence = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return TypeParam(i); });
  auto validity = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return (i % 2); });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam> col(sequence, sequence + num_rows, validity);

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col.release());
  auto expected = table{std::move(cols)};
  EXPECT_EQ(1, expected.num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcSingleColumnWithNulls.orc");
  cudf_io::write_orc_args out_args{cudf_io::sink_info{filepath},
                                   expected.view()};
  cudf_io::write_orc(out_args);

  cudf_io::read_orc_args in_args{cudf_io::source_info{filepath}};
  in_args.use_index = false;
  auto result = cudf_io::read_orc(in_args);

  expect_tables_equal(expected.view(), result.view());
}

TEST_F(OrcWriterTest, MultiColumn) {
  constexpr auto num_rows = 100;

  auto seq_col0 = random_values<bool>(num_rows);
  auto seq_col1 = random_values<int8_t>(num_rows);
  auto seq_col2 = random_values<int16_t>(num_rows);
  auto seq_col3 = random_values<int32_t>(num_rows);
  auto seq_col4 = random_values<float>(num_rows);
  auto seq_col5 = random_values<double>(num_rows);
  auto validity = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return true; });

  // column_wrapper<bool> col0{
  //    seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<int8_t> col1{seq_col1.begin(), seq_col1.end(), validity};
  column_wrapper<int16_t> col2{seq_col2.begin(), seq_col2.end(), validity};
  column_wrapper<int32_t> col3{seq_col3.begin(), seq_col3.end(), validity};
  column_wrapper<float> col4{seq_col4.begin(), seq_col4.end(), validity};
  column_wrapper<double> col5{seq_col5.begin(), seq_col5.end(), validity};
  // column_set_name(col0.get(), "bools");
  // column_set_name(col1.get(), "int8s");
  // column_set_name(col2.get(), "int16s");
  // column_set_name(col3.get(), "int32s");
  // column_set_name(col4.get(), "floats");
  // column_set_name(col5.get(), "doubles");

  std::vector<std::unique_ptr<column>> cols;
  // cols.push_back(col0.release());
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  cols.push_back(col3.release());
  cols.push_back(col4.release());
  cols.push_back(col5.release());
  auto expected = table{std::move(cols)};
  EXPECT_EQ(5, expected.num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcMultiColumn.orc");
  cudf_io::write_orc_args out_args{cudf_io::sink_info{filepath},
                                   expected.view()};
  cudf_io::write_orc(out_args);

  cudf_io::read_orc_args in_args{cudf_io::source_info{filepath}};
  in_args.use_index = false;
  auto result = cudf_io::read_orc(in_args);

  expect_tables_equal(expected.view(), result.view());
}

TEST_F(OrcWriterTest, Timestamps) {
  constexpr auto num_rows = 100;

  auto sequence = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return cudf::timestamp_ms{std::rand() / 10}; });
  auto validity = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return true; });

  column_wrapper<cudf::timestamp_ms> col{sequence, sequence + num_rows,
                                         validity};

  // column_set_name(col.get(), "col_timestamp");

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col.release());
  auto expected = table{std::move(cols)};
  EXPECT_EQ(1, expected.num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcWriterTimestamps");
  cudf_io::write_orc_args out_args{cudf_io::sink_info{filepath},
                                   expected.view()};
  cudf_io::write_orc(out_args);

  cudf_io::read_orc_args in_args{cudf_io::source_info{filepath}};
  in_args.use_index = false;
  in_args.timestamp_type =
      cudf::data_type{cudf::experimental::type_to_id<cudf::timestamp_ms>()};
  auto result = cudf_io::read_orc(in_args);

  expect_tables_equal(expected.view(), result.view());
}

TEST_F(OrcWriterTest, Strings) {
  std::vector<const char*> strings{"Monday", "Monday", "Friday", "Monday",
                                   "Friday", "Friday", "Friday", "Funday"};
  const auto num_rows = strings.size();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);
  auto validity = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return true; });

  column_wrapper<int> col0{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<cudf::string_view> col1{strings.begin(), strings.end()};
  column_wrapper<float> col2{seq_col2.begin(), seq_col2.end(), validity};

  // column_set_name(col0.get(), "col_other");
  // column_set_name(col1.get(), "col_string");
  // column_set_name(col2.get(), "col_another");

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col0.release());
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  auto expected = table{std::move(cols)};
  EXPECT_EQ(3, expected.num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcStrings.orc");
  cudf_io::write_orc_args out_args{cudf_io::sink_info{filepath},
                                   expected.view()};
  cudf_io::write_orc(out_args);

  cudf_io::read_orc_args in_args{cudf_io::source_info{filepath}};
  in_args.use_index = false;
  auto result = cudf_io::read_orc(in_args);

  expect_tables_equal(expected.view(), result.view());
}
