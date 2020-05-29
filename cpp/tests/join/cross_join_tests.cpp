/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

struct JoinTest : public cudf::test::BaseFixture {
};

TEST_F(JoinTest, CrossJoin)
{
  auto a_strings  = std::vector<const char*>{"quick", "accénted", "turtlé", "composéd"};
  auto b_strings  = std::vector<const char*>{"result", "", "words"};
  auto e_strings3 = std::vector<const char*>{"quick",
                                             "quick",
                                             "quick",
                                             "accénted",
                                             "accénted",
                                             "accénted",
                                             "turtlé",
                                             "turtlé",
                                             "turtlé",
                                             "composéd",
                                             "composéd",
                                             "composéd"};
  auto e_strings7 = std::vector<const char*>{
    "result", "", "words", "result", "", "words", "result", "", "words", "result", "", "words"};

  auto a_0 = column_wrapper<int32_t>{10, 20, 20, 50};
  auto a_1 = column_wrapper<float>{5.0, .5, .5, .7};
  auto a_2 = column_wrapper<int8_t>{90, 77, 78, 41};
  auto a_3 = cudf::test::strings_column_wrapper(
    a_strings.begin(),
    a_strings.end(),
    thrust::make_transform_iterator(a_strings.begin(), [](auto str) { return str != nullptr; }));

  auto b_0 = column_wrapper<int32_t>{10, 20, 20};
  auto b_1 = column_wrapper<float>{5.0, .7, .7};
  auto b_2 = column_wrapper<int8_t>{90, 75, 62};
  auto b_3 = cudf::test::strings_column_wrapper(
    b_strings.begin(),
    b_strings.end(),
    thrust::make_transform_iterator(b_strings.begin(), [](auto str) { return str != nullptr; }));

  auto expect_0 = column_wrapper<int32_t>{10, 10, 10, 20, 20, 20, 20, 20, 20, 50, 50, 50};
  auto expect_1 = column_wrapper<float>{5.0, 5.0, 5.0, .5, .5, .5, .5, .5, .5, .7, .7, .7};
  auto expect_2 = column_wrapper<int8_t>{90, 90, 90, 77, 77, 77, 78, 78, 78, 41, 41, 41};
  auto expect_3 = cudf::test::strings_column_wrapper(
    e_strings3.begin(),
    e_strings3.end(),
    thrust::make_transform_iterator(e_strings3.begin(), [](auto str) { return str != nullptr; }));
  auto expect_4 = column_wrapper<int32_t>{10, 20, 20, 10, 20, 20, 10, 20, 20, 10, 20, 20};
  auto expect_5 = column_wrapper<float>{5.0, .7, .7, 5.0, .7, .7, 5.0, .7, .7, 5.0, .7, .7};
  auto expect_6 = column_wrapper<int8_t>{90, 75, 62, 90, 75, 62, 90, 75, 62, 90, 75, 62};
  auto expect_7 = cudf::test::strings_column_wrapper(
    e_strings7.begin(),
    e_strings7.end(),
    thrust::make_transform_iterator(e_strings7.begin(), [](auto str) { return str != nullptr; }));

  auto table_a = cudf::table_view{{a_0, a_1, a_2, a_3}};
  auto table_b = cudf::table_view{{b_0, b_1, b_2, b_3}};

  auto join_table = cudf::cross_join(table_a, table_b);

  EXPECT_EQ(join_table->num_columns(), table_a.num_columns() + table_b.num_columns());
  EXPECT_EQ(join_table->num_rows(), table_a.num_rows() * table_b.num_rows());
  expect_columns_equal(join_table->get_column(0), expect_0, true);
  expect_columns_equal(join_table->get_column(1), expect_1, true);
  expect_columns_equal(join_table->get_column(2), expect_2, true);
  expect_columns_equal(join_table->get_column(3), expect_3, true);
  expect_columns_equal(join_table->get_column(4), expect_4, true);
  expect_columns_equal(join_table->get_column(5), expect_5, true);
  expect_columns_equal(join_table->get_column(6), expect_6, true);
  expect_columns_equal(join_table->get_column(7), expect_7, true);
}

TEST_F(JoinTest, CrossJoin_exceptions)
{
  auto b_strings = std::vector<const char*>{"quick", "words", "result", nullptr};

  auto b_0 = column_wrapper<int32_t>{10, 20, 20, 50};
  auto b_1 = column_wrapper<float>{5.0, .7, .7, .7};
  auto b_2 = column_wrapper<int8_t>{90, 75, 62, 41};
  auto b_3 = cudf::test::strings_column_wrapper(
    b_strings.begin(),
    b_strings.end(),
    thrust::make_transform_iterator(b_strings.begin(), [](auto str) { return str != nullptr; }));

  auto column_a = std::vector<std::unique_ptr<cudf::column>>{};
  auto table_a  = cudf::table(std::move(column_a));
  auto table_b  = cudf::table_view{{b_0, b_1, b_2, b_3}};

  //
  //  table_a has no columns, table_b has columns
  //  Let's check different permutations of passing table
  //  with no columns to verify that exceptions are thrown
  //
  EXPECT_THROW(cudf::cross_join(table_a, table_b), cudf::logic_error);
  EXPECT_THROW(cudf::cross_join(table_b, table_a), cudf::logic_error);
}

TEST_F(JoinTest, CrossJoin_empty_result)
{
  auto a_strings = std::vector<const char*>{};
  auto b_strings = std::vector<const char*>{"quick", "words", "result", nullptr};
  auto e_strings = std::vector<const char*>{};

  auto a_0 = column_wrapper<int32_t>{};
  auto a_1 = column_wrapper<float>{};
  auto a_2 = column_wrapper<int8_t>{};
  auto a_3 = cudf::test::strings_column_wrapper(
    a_strings.begin(),
    a_strings.end(),
    thrust::make_transform_iterator(a_strings.begin(), [](auto str) { return str != nullptr; }));

  auto b_0 = column_wrapper<int32_t>{10, 20, 20, 50};
  auto b_1 = column_wrapper<float>{5.0, .7, .7, .7};
  auto b_2 = column_wrapper<int8_t>{90, 75, 62, 41};
  auto b_3 = cudf::test::strings_column_wrapper(
    b_strings.begin(),
    b_strings.end(),
    thrust::make_transform_iterator(b_strings.begin(), [](auto str) { return str != nullptr; }));

  auto expect_0 = column_wrapper<int32_t>{};
  auto expect_1 = column_wrapper<float>{};
  auto expect_2 = column_wrapper<int8_t>{};
  auto expect_3 = cudf::test::strings_column_wrapper(
    e_strings.begin(),
    e_strings.end(),
    thrust::make_transform_iterator(e_strings.begin(), [](auto str) { return str != nullptr; }));

  auto table_a = cudf::table_view{{a_0, a_1, a_2, a_3}};
  auto table_b = cudf::table_view{{b_0, b_1, b_2, b_3}};

  auto join_table = cudf::cross_join(table_a, table_b);

  EXPECT_EQ(join_table->num_columns(), table_a.num_columns() + table_b.num_columns());
  EXPECT_EQ(join_table->num_rows(), 0);
}
