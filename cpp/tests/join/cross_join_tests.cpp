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
  std::vector<const char*> a_strings{"quick", "accénted", "turtlé", "composéd"};
  std::vector<const char*> b_strings{"result", "", "words"};
  std::vector<const char*> e_strings3{"quick",
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
  std::vector<const char*> e_strings7{
    "result", "", "words", "result", "", "words", "result", "", "words", "result", "", "words"};

  column_wrapper<int32_t> a_0{10, 20, 20, 50};
  column_wrapper<float> a_1{5.0, .5, .5, .7};
  column_wrapper<int8_t> a_2{90, 77, 78, 41};

  cudf::test::strings_column_wrapper a_3(
    a_strings.begin(),
    a_strings.end(),
    thrust::make_transform_iterator(a_strings.begin(), [](auto str) { return str != nullptr; }));

  column_wrapper<int32_t> b_0{10, 20, 20};
  column_wrapper<float> b_1{5.0, .7, .7};
  column_wrapper<int8_t> b_2{90, 75, 62};

  cudf::test::strings_column_wrapper b_3(
    b_strings.begin(),
    b_strings.end(),
    thrust::make_transform_iterator(b_strings.begin(), [](auto str) { return str != nullptr; }));

  column_wrapper<int32_t> expect_0{10, 10, 10, 20, 20, 20, 20, 20, 20, 50, 50, 50};
  column_wrapper<float> expect_1{5.0, 5.0, 5.0, .5, .5, .5, .5, .5, .5, .7, .7, .7};
  column_wrapper<int8_t> expect_2{90, 90, 90, 77, 77, 77, 78, 78, 78, 41, 41, 41};

  cudf::test::strings_column_wrapper expect_3(
    e_strings3.begin(),
    e_strings3.end(),
    thrust::make_transform_iterator(e_strings3.begin(), [](auto str) { return str != nullptr; }));

  column_wrapper<int32_t> expect_4{10, 20, 20, 10, 20, 20, 10, 20, 20, 10, 20, 20};
  column_wrapper<float> expect_5{5.0, .7, .7, 5.0, .7, .7, 5.0, .7, .7, 5.0, .7, .7};
  column_wrapper<int8_t> expect_6{90, 75, 62, 90, 75, 62, 90, 75, 62, 90, 75, 62};

  cudf::test::strings_column_wrapper expect_7(
    e_strings7.begin(),
    e_strings7.end(),
    thrust::make_transform_iterator(e_strings7.begin(), [](auto str) { return str != nullptr; }));

  std::vector<std::unique_ptr<cudf::column>> column_a;
  column_a.push_back(a_0.release());
  column_a.push_back(a_1.release());
  column_a.push_back(a_2.release());
  column_a.push_back(a_3.release());

  std::vector<std::unique_ptr<cudf::column>> column_b;
  column_b.push_back(b_0.release());
  column_b.push_back(b_1.release());
  column_b.push_back(b_2.release());
  column_b.push_back(b_3.release());

  cudf::table table_a(std::move(column_a));
  cudf::table table_b(std::move(column_b));

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
  std::vector<const char*> b_strings{"quick", "words", "result", nullptr};

  column_wrapper<int32_t> b_0{10, 20, 20, 50};
  column_wrapper<float> b_1{5.0, .7, .7, .7};
  column_wrapper<int8_t> b_2{90, 75, 62, 41};

  cudf::test::strings_column_wrapper b_3(
    b_strings.begin(),
    b_strings.end(),
    thrust::make_transform_iterator(b_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<std::unique_ptr<cudf::column>> column_a;

  std::vector<std::unique_ptr<cudf::column>> column_b;
  column_b.push_back(b_0.release());
  column_b.push_back(b_1.release());
  column_b.push_back(b_2.release());
  column_b.push_back(b_3.release());

  cudf::table table_a(std::move(column_a));
  cudf::table table_b(std::move(column_b));

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
  std::vector<const char*> a_strings{};
  std::vector<const char*> b_strings{"quick", "words", "result", nullptr};
  std::vector<const char*> e_strings{};

  column_wrapper<int32_t> a_0{};
  column_wrapper<float> a_1{};
  column_wrapper<int8_t> a_2{};

  cudf::test::strings_column_wrapper a_3(
    a_strings.begin(),
    a_strings.end(),
    thrust::make_transform_iterator(a_strings.begin(), [](auto str) { return str != nullptr; }));

  column_wrapper<int32_t> b_0{10, 20, 20, 50};
  column_wrapper<float> b_1{5.0, .7, .7, .7};
  column_wrapper<int8_t> b_2{90, 75, 62, 41};

  cudf::test::strings_column_wrapper b_3(
    b_strings.begin(),
    b_strings.end(),
    thrust::make_transform_iterator(b_strings.begin(), [](auto str) { return str != nullptr; }));

  column_wrapper<int32_t> expect_0{};
  column_wrapper<float> expect_1{};
  column_wrapper<int8_t> expect_2{};

  cudf::test::strings_column_wrapper expect_3(
    e_strings.begin(),
    e_strings.end(),
    thrust::make_transform_iterator(e_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<std::unique_ptr<cudf::column>> column_a;
  column_a.push_back(a_0.release());
  column_a.push_back(a_1.release());
  column_a.push_back(a_2.release());
  column_a.push_back(a_3.release());

  std::vector<std::unique_ptr<cudf::column>> column_b;
  column_b.push_back(b_0.release());
  column_b.push_back(b_1.release());
  column_b.push_back(b_2.release());
  column_b.push_back(b_3.release());

  cudf::table table_a(std::move(column_a));
  cudf::table table_b(std::move(column_b));

  auto join_table = cudf::cross_join(table_a, table_b);

  EXPECT_EQ(join_table->num_columns(), table_a.num_columns() + table_b.num_columns());
  EXPECT_EQ(join_table->num_rows(), 0);
}
