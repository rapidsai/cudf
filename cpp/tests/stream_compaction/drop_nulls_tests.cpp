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

#include <cudf/copying.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

struct DropNullsTest : public cudf::test::BaseFixture {
};

TEST_F(DropNullsTest, WholeRowIsNull)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1, col2, col3}};
  std::vector<cudf::size_type> keys{0, 1, 2};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{true, false, false, true},
                                                                {1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{10, 40, 5, 2}, {1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{{10, 40, 5, 2}, {1, 1, 1, 1}};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::drop_nulls(input, keys);

  cudf::test::expect_tables_equal(expected, got->view());
}

TEST_F(DropNullsTest, NoNull)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 1, 1, 1, 1}};
  cudf::table_view input{{col1, col2, col3}};
  std::vector<cudf::size_type> keys{0, 1, 2};

  auto got = cudf::drop_nulls(input, keys);

  cudf::test::expect_tables_equal(input, got->view());
}

TEST_F(DropNullsTest, MixedSetOfRows)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 1}};
  cudf::table_view input{{col1, col2, col3}};
  std::vector<cudf::size_type> keys{0, 1, 2};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{true, false, false, true},
                                                                {1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{10, 40, 5, 2}, {1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{{10, 40, 5, 2}, {1, 1, 1, 1}};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::drop_nulls(input, keys);

  cudf::test::expect_tables_equal(expected, got->view());
}

TEST_F(DropNullsTest, MixedSetOfRowsWithThreshold)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 1, 1, 1, 1}};
  cudf::table_view input{{col1, col2, col3}};
  std::vector<cudf::size_type> keys{0, 1, 2};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{true, false, false, true, false},
                                                                {1, 1, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{10, 40, 5, 2, 10},
                                                                {1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{{10, 40, 5, 2, 10}, {1, 1, 1, 1, 1}};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::drop_nulls(input, keys, keys.size() - 1);

  cudf::test::expect_tables_equal(expected, got->view());
}

TEST_F(DropNullsTest, EmptyTable)
{
  cudf::table_view input{std::vector<cudf::column_view>()};
  cudf::table_view expected{std::vector<cudf::column_view>()};
  std::vector<cudf::size_type> keys{};

  auto got = cudf::drop_nulls(input, keys);

  cudf::test::expect_tables_equal(expected, got->view());
}

TEST_F(DropNullsTest, EmptyColumns)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{};
  cudf::test::fixed_width_column_wrapper<double> col3{};
  cudf::table_view input{{col1, col2, col3}};
  std::vector<cudf::size_type> keys{0, 1, 2};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::drop_nulls(input, keys);

  cudf::test::expect_tables_equal(expected, got->view());
}

TEST_F(DropNullsTest, EmptyKeys)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {1, 1, 0, 1, 1, 0}};
  cudf::table_view input{{col1}};
  std::vector<cudf::size_type> keys{};

  auto got = cudf::drop_nulls(input, keys);
  cudf::test::expect_tables_equal(input, got->view());
}

TEST_F(DropNullsTest, StringColWithNull)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{11, 12, 11, 13, 12, 15},
                                                       {1, 1, 0, 1, 0, 1}};
  cudf::test::strings_column_wrapper col2{{"Hi", "Hello", "Hi", "No", "Hello", "Naive"},
                                          {1, 1, 0, 1, 0, 1}};
  cudf::table_view input{{col1, col2}};
  std::vector<cudf::size_type> keys{0, 1};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{11, 12, 13, 15}, {1, 1, 1, 1}};
  cudf::test::strings_column_wrapper col2_expected{{"Hi", "Hello", "No", "Naive"}, {1, 1, 1, 1}};
  cudf::table_view expected{{col1_expected, col2_expected}};

  auto got = cudf::drop_nulls(input, keys);

  cudf::test::expect_tables_equal(expected, got->view());
}

template <typename T>
struct DropNullsTestAll : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(DropNullsTestAll, cudf::test::NumericTypes);

TYPED_TEST(DropNullsTestAll, AllNull)
{
  using T = TypeParam;
  cudf::test::fixed_width_column_wrapper<T> key_col{{true, false, true, false, true, false},
                                                    {0, 0, 0, 0, 0, 0}};
  cudf::test::fixed_width_column_wrapper<T> col{{true, false, true, false, true, false},
                                                {1, 1, 1, 1, 1, 1}};
  cudf::table_view input{{key_col, col}};
  std::vector<cudf::size_type> keys{0};
  cudf::test::fixed_width_column_wrapper<T> expected_col{};
  cudf::column_view view = expected_col;
  cudf::table_view expected{{expected_col, expected_col}};

  auto got = cudf::drop_nulls(input, keys);

  cudf::test::expect_tables_equal(expected, got->view());
}
