/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

template <typename T, typename SourceT = T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T, SourceT>;

template <typename T>
class CrossJoinTypeTests : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(CrossJoinTypeTests, cudf::test::FixedWidthTypes);

TYPED_TEST(CrossJoinTypeTests, CrossJoin)
{
  auto a_0 = column_wrapper<int32_t>{10, 20, 20, 50};
  auto a_1 = column_wrapper<float>{5.0, .5, .5, .7};
  auto a_2 = column_wrapper<TypeParam, int32_t>{0, 0, 0, 0};
  auto a_3 = cudf::test::strings_column_wrapper({"quick", "accénted", "turtlé", "composéd"});

  auto b_0 = column_wrapper<int32_t>{10, 20, 20};
  auto b_1 = column_wrapper<float>{5.0, .7, .7};
  auto b_2 = column_wrapper<TypeParam, int32_t>{0, 0, 0};
  auto b_3 = cudf::test::strings_column_wrapper({"result", "", "words"});

  auto expect_0 = column_wrapper<int32_t>{10, 10, 10, 20, 20, 20, 20, 20, 20, 50, 50, 50};
  auto expect_1 = column_wrapper<float>{5.0, 5.0, 5.0, .5, .5, .5, .5, .5, .5, .7, .7, .7};
  auto expect_2 = column_wrapper<TypeParam, int32_t>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto expect_3 = cudf::test::strings_column_wrapper({"quick",
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
                                                      "composéd"});
  auto expect_4 = column_wrapper<int32_t>{10, 20, 20, 10, 20, 20, 10, 20, 20, 10, 20, 20};
  auto expect_5 = column_wrapper<float>{5.0, .7, .7, 5.0, .7, .7, 5.0, .7, .7, 5.0, .7, .7};
  auto expect_6 = column_wrapper<TypeParam, int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto expect_7 = cudf::test::strings_column_wrapper(
    {"result", "", "words", "result", "", "words", "result", "", "words", "result", "", "words"});

  auto table_a      = cudf::table_view{{a_0, a_1, a_2, a_3}};
  auto table_b      = cudf::table_view{{b_0, b_1, b_2, b_3}};
  auto table_expect = cudf::table_view{
    {expect_0, expect_1, expect_2, expect_3, expect_4, expect_5, expect_6, expect_7}};

  auto join_table = cudf::cross_join(table_a, table_b);

  EXPECT_EQ(join_table->num_columns(), table_a.num_columns() + table_b.num_columns());
  EXPECT_EQ(join_table->num_rows(), table_a.num_rows() * table_b.num_rows());
  CUDF_TEST_EXPECT_TABLES_EQUAL(join_table->view(), table_expect);
}

class CrossJoinInvalidInputs : public cudf::test::BaseFixture {};

TEST_F(CrossJoinInvalidInputs, EmptyTable)
{
  auto b_0 = column_wrapper<int32_t>{10, 20, 20, 50};
  auto b_1 = column_wrapper<float>{5.0, .7, .7, .7};
  auto b_2 = column_wrapper<int8_t>{90, 75, 62, 41};
  auto b_3 = cudf::test::strings_column_wrapper({"quick", "words", "result", ""});

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

class CrossJoinEmptyResult : public cudf::test::BaseFixture {};

TEST_F(CrossJoinEmptyResult, NoRows)
{
  auto a_0           = column_wrapper<int32_t>{};
  auto a_1           = column_wrapper<float>{};
  auto a_2           = column_wrapper<int8_t>{};
  auto empty_strings = std::vector<std::string>();
  auto a_3 = cudf::test::strings_column_wrapper(empty_strings.begin(), empty_strings.end());

  auto b_0 = column_wrapper<int32_t>{10, 20, 20, 50};
  auto b_1 = column_wrapper<float>{5.0, .7, .7, .7};
  auto b_2 = column_wrapper<int8_t>{90, 75, 62, 41};
  auto b_3 = cudf::test::strings_column_wrapper({"quick", "words", "result", ""});

  auto expect_0 = column_wrapper<int32_t>{};
  auto expect_1 = column_wrapper<float>{};
  auto expect_2 = column_wrapper<int8_t>{};
  auto expect_3 = cudf::test::strings_column_wrapper(empty_strings.begin(), empty_strings.end());
  auto expect_4 = column_wrapper<int32_t>{};
  auto expect_5 = column_wrapper<float>{};
  auto expect_6 = column_wrapper<int8_t>{};
  auto expect_7 = cudf::test::strings_column_wrapper(empty_strings.begin(), empty_strings.end());

  auto table_a      = cudf::table_view{{a_0, a_1, a_2, a_3}};
  auto table_b      = cudf::table_view{{b_0, b_1, b_2, b_3}};
  auto table_expect = cudf::table_view{
    {expect_0, expect_1, expect_2, expect_3, expect_4, expect_5, expect_6, expect_7}};

  auto join_table         = cudf::cross_join(table_a, table_b);
  auto join_table_reverse = cudf::cross_join(table_b, table_a);

  EXPECT_EQ(join_table->num_columns(), table_a.num_columns() + table_b.num_columns());
  EXPECT_EQ(join_table->num_rows(), 0);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_table->view(), table_expect);
  EXPECT_EQ(join_table_reverse->num_columns(), table_a.num_columns() + table_b.num_columns());
  EXPECT_EQ(join_table_reverse->num_rows(), 0);
}
