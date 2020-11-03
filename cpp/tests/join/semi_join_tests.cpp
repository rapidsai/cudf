/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/dictionary/encode.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

struct JoinTest : public cudf::test::BaseFixture {
};

TEST_F(JoinTest, LeftSemiJoin)
{
  std::vector<const char*> a_strings{
    "quick", "accénted", "turtlé", "composéd", "result", "", "words"};
  std::vector<const char*> b_strings{"quick", "words", "result"};
  std::vector<const char*> e_strings{"quick", "composéd", "result", ""};

  column_wrapper<int32_t> a_0{10, 20, 20, 20, 20, 20, 50};
  column_wrapper<float> a_1{5.0, .5, .5, .7, .7, .7, .7};
  column_wrapper<int8_t> a_2{90, 77, 78, 61, 62, 63, 41};

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

  column_wrapper<int32_t> expect_0{10, 20, 20, 20};
  column_wrapper<float> expect_1{5.0, .7, .7, .7};
  column_wrapper<int8_t> expect_2{90, 61, 62, 63};

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

  auto join_table = cudf::left_semi_join(table_a, table_b, {0, 1}, {0, 1}, {0, 1, 2, 3});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(0), expect_0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(1), expect_1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(2), expect_2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(3), expect_3);
}

TEST_F(JoinTest, LeftSemiJoin_with_a_string_key)
{
  std::vector<const char*> a_strings{
    "quick", "accénted", "turtlé", "composéd", "result", "", "words"};
  std::vector<const char*> b_strings{"quick", "words", "result"};
  std::vector<const char*> e_strings{"quick", "result"};

  column_wrapper<int32_t> a_0{10, 20, 20, 20, 20, 20, 50};
  column_wrapper<float> a_1{5.0, .5, .5, .7, .7, .7, .7};
  column_wrapper<int8_t> a_2{90, 77, 78, 61, 62, 63, 41};

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

  column_wrapper<int32_t> expect_0{10, 20};
  column_wrapper<float> expect_1{5.0, .7};
  column_wrapper<int8_t> expect_2{90, 62};

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

  auto join_table = cudf::left_semi_join(table_a, table_b, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(0), expect_0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(1), expect_1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(2), expect_2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(3), expect_3);
}

TEST_F(JoinTest, LeftSemiJoin_with_null)
{
  std::vector<const char*> a_strings{
    "quick", "accénted", "turtlé", "composéd", "result", "", "words"};
  std::vector<const char*> b_strings{"quick", "words", "result", nullptr};
  std::vector<const char*> e_strings{"quick", "result"};

  column_wrapper<int32_t> a_0{10, 20, 20, 20, 20, 20, 50};
  column_wrapper<float> a_1{5.0, .5, .5, .7, .7, .7, .7};
  column_wrapper<int8_t> a_2{90, 77, 78, 61, 62, 63, 41};

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

  column_wrapper<int32_t> expect_0{10, 20};
  column_wrapper<float> expect_1{5.0, .7};
  column_wrapper<int8_t> expect_2{90, 62};

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

  auto join_table = cudf::left_semi_join(table_a, table_b, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(0), expect_0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(1), expect_1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(2), expect_2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(3), expect_3);
}

TEST_F(JoinTest, LeftAntiJoin)
{
  std::vector<const char*> a_strings{
    "quick", "accénted", "turtlé", "composéd", "result", "", "words"};
  std::vector<const char*> b_strings{"quick", "words", "result"};
  std::vector<const char*> e_strings{"accénted", "turtlé", "words"};

  column_wrapper<int32_t> a_0{10, 20, 20, 20, 20, 20, 50};
  column_wrapper<float> a_1{5.0, .5, .5, .7, .7, .7, .7};
  column_wrapper<int8_t> a_2{90, 77, 78, 61, 62, 63, 41};

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

  column_wrapper<int32_t> expect_0{20, 20, 50};
  column_wrapper<float> expect_1{.5, .5, .7};
  column_wrapper<int8_t> expect_2{77, 78, 41};

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

  auto join_table = cudf::left_anti_join(table_a, table_b, {0, 1}, {0, 1}, {0, 1, 2, 3});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(0), expect_0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(1), expect_1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(2), expect_2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(3), expect_3);
}

TEST_F(JoinTest, LeftAntiJoin_with_a_string_key)
{
  std::vector<const char*> a_strings{
    "quick", "accénted", "turtlé", "composéd", "result", "", "words"};
  std::vector<const char*> b_strings{"quick", "words", "result"};
  std::vector<const char*> e_strings{"accénted", "turtlé", "composéd", "", "words"};

  column_wrapper<int32_t> a_0{10, 20, 20, 20, 20, 20, 50};
  column_wrapper<float> a_1{5.0, .5, .5, .7, .7, .7, .7};
  column_wrapper<int8_t> a_2{90, 77, 78, 61, 62, 63, 41};

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

  column_wrapper<int32_t> expect_0{20, 20, 20, 20, 50};
  column_wrapper<float> expect_1{.5, .5, .7, .7, .7};
  column_wrapper<int8_t> expect_2{77, 78, 61, 63, 41};

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

  auto join_table = cudf::left_anti_join(table_a, table_b, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(0), expect_0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(1), expect_1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(2), expect_2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(3), expect_3);
}

TEST_F(JoinTest, LeftAntiJoin_with_null)
{
  std::vector<const char*> a_strings{
    "quick", "accénted", "turtlé", "composéd", "result", "", "words"};
  std::vector<const char*> b_strings{"quick", "words", "result", nullptr};
  std::vector<const char*> e_strings{"accénted", "turtlé", "composéd", "", "words"};

  column_wrapper<int32_t> a_0{10, 20, 20, 20, 20, 20, 50};
  column_wrapper<float> a_1{5.0, .5, .5, .7, .7, .7, .7};
  column_wrapper<int8_t> a_2{90, 77, 78, 61, 62, 63, 41};

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

  column_wrapper<int32_t> expect_0{20, 20, 20, 20, 50};
  column_wrapper<float> expect_1{.5, .5, .7, .7, .7};
  column_wrapper<int8_t> expect_2{77, 78, 61, 63, 41};

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

  auto join_table = cudf::left_anti_join(table_a, table_b, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(0), expect_0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(1), expect_1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(2), expect_2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(3), expect_3);
}

TEST_F(JoinTest, LeftSemiAntiJoin_exceptions)
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
  EXPECT_THROW(cudf::left_semi_join(table_a, table_b, {}, {}, {}), cudf::logic_error);

  EXPECT_THROW(cudf::left_anti_join(table_a, table_b, {}, {}, {}), cudf::logic_error);

  EXPECT_THROW(cudf::left_semi_join(table_b, table_a, {}, {}, {}), cudf::logic_error);

  EXPECT_THROW(cudf::left_anti_join(table_b, table_a, {}, {}, {}), cudf::logic_error);

  //
  //  table_b has columns, so we'll pass the column checks, but
  //  these should fail the exception check that the number of
  //  join columns must be the same for each table
  //
  EXPECT_THROW(cudf::left_semi_join(table_b, table_b, {0}, {}, {}), cudf::logic_error);

  EXPECT_THROW(cudf::left_anti_join(table_b, table_b, {0}, {}, {}), cudf::logic_error);

  EXPECT_THROW(cudf::left_semi_join(table_b, table_b, {}, {0}, {}), cudf::logic_error);

  EXPECT_THROW(cudf::left_anti_join(table_b, table_b, {}, {0}, {}), cudf::logic_error);
}

TEST_F(JoinTest, LeftSemiJoin_empty_result)
{
  std::vector<const char*> a_strings{
    "quick", "accénted", "turtlé", "composéd", "result", "", "words"};
  std::vector<const char*> b_strings{"quick", "words", "result", nullptr};
  std::vector<const char*> e_strings{};

  column_wrapper<int32_t> a_0{10, 20, 20, 20, 20, 20, 50};
  column_wrapper<float> a_1{5.0, .5, .5, .7, .7, .7, .7};
  column_wrapper<int8_t> a_2{90, 77, 78, 61, 62, 63, 41};

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

  auto join_table = cudf::left_semi_join(table_a, table_b, {0, 1, 3}, {0, 1, 3}, {});

  EXPECT_EQ(join_table->num_columns(), 0);
  EXPECT_EQ(join_table->num_rows(), 0);

  auto join_table2 = cudf::left_semi_join(table_a, table_b, {}, {}, {0, 1, 3});

  EXPECT_EQ(join_table2->num_columns(), 3);
  EXPECT_EQ(join_table2->num_rows(), 0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table2->get_column(0), expect_0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table2->get_column(1), expect_1);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(join_table2->get_column(2), expect_3);
}

TEST_F(JoinTest, LeftAntiJoin_empty_result)
{
  std::vector<const char*> a_strings{
    "quick", "accénted", "turtlé", "composéd", "result", "", "words"};
  std::vector<const char*> b_strings{"quick", "words", "result", nullptr};
  std::vector<const char*> e_strings{};

  column_wrapper<int32_t> a_0{10, 20, 20, 20, 20, 20, 50};
  column_wrapper<float> a_1{5.0, .5, .5, .7, .7, .7, .7};
  column_wrapper<int8_t> a_2{90, 77, 78, 61, 62, 63, 41};

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

  auto join_table = cudf::left_anti_join(table_a, table_b, {0, 1, 3}, {0, 1, 3}, {});

  EXPECT_EQ(join_table->num_columns(), 0);
  EXPECT_EQ(join_table->num_rows(), 0);

  auto join_table2 = cudf::left_anti_join(table_a, table_b, {}, {}, {0, 1, 3});

  EXPECT_EQ(join_table2->num_columns(), 3);
  EXPECT_EQ(join_table2->num_rows(), 0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table2->get_column(0), expect_0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table2->get_column(1), expect_1);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(join_table2->get_column(2), expect_3);
}

TEST_F(JoinTest, LeftSemiAntiJoin_empty_table)
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

  auto join_table = cudf::left_semi_join(table_a, table_b, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(0), expect_0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(1), expect_1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(2), expect_2);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(join_table->get_column(3), expect_3);

  auto join_table2 = cudf::left_semi_join(table_b, table_a, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table2->get_column(0), expect_0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table2->get_column(1), expect_1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table2->get_column(2), expect_2);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(join_table2->get_column(3), expect_3);

  auto join_table3 = cudf::left_anti_join(table_a, table_b, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table3->get_column(0), expect_0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table3->get_column(1), expect_1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table3->get_column(2), expect_2);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(join_table3->get_column(3), expect_3);

  auto join_table4 = cudf::left_anti_join(table_a, table_a, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table4->get_column(0), expect_0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table4->get_column(1), expect_1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table4->get_column(2), expect_2);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(join_table4->get_column(3), expect_3);

  auto join_table5 = cudf::left_anti_join(table_a, table_a, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table5->get_column(0), expect_0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table5->get_column(1), expect_1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table5->get_column(2), expect_2);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(join_table5->get_column(3), expect_3);
}

TEST_F(JoinTest, LeftAntiJoin_empty_right_table)
{
  std::vector<const char*> a_strings{"quick", "words", "result", nullptr};
  std::vector<const char*> b_strings{};
  std::vector<const char*> e_strings{"quick", "words", "result", nullptr};

  column_wrapper<int32_t> a_0{10, 20, 20, 50};
  column_wrapper<float> a_1{5.0, .7, .7, .7};
  column_wrapper<int8_t> a_2{90, 75, 62, 41};

  cudf::test::strings_column_wrapper a_3(
    a_strings.begin(),
    a_strings.end(),
    thrust::make_transform_iterator(a_strings.begin(), [](auto str) { return str != nullptr; }));

  column_wrapper<int32_t> b_0{};
  column_wrapper<float> b_1{};
  column_wrapper<int8_t> b_2{};

  cudf::test::strings_column_wrapper b_3(
    b_strings.begin(),
    b_strings.end(),
    thrust::make_transform_iterator(b_strings.begin(), [](auto str) { return str != nullptr; }));

  column_wrapper<int32_t> expect_0{10, 20, 20, 50};
  column_wrapper<float> expect_1{5.0, .7, .7, .7};
  column_wrapper<int8_t> expect_2{90, 75, 62, 41};

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

  auto join_table = cudf::left_anti_join(table_a, table_b, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(0), expect_0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(1), expect_1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(2), expect_2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(join_table->get_column(3), expect_3);
}

struct JoinDictionaryTest : public cudf::test::BaseFixture {
};

TEST_F(JoinDictionaryTest, LeftSemiJoin)
{
  column_wrapper<int32_t> a_0{10, 20, 20, 20, 20, 20, 50};
  column_wrapper<float> a_1{5.0, .5, .5, .7, .7, .7, .7};
  column_wrapper<int8_t> a_2{90, 77, 78, 61, 62, 63, 41};
  cudf::test::strings_column_wrapper a_3_w(
    {"quick", "accénted", "turtlé", "composéd", "result", "", "words"});
  auto a_3 = cudf::dictionary::encode(a_3_w);

  column_wrapper<int32_t> b_0{10, 20, 20};
  column_wrapper<float> b_1{5.0, .7, .7};
  column_wrapper<int8_t> b_2{90, 75, 62};
  cudf::test::strings_column_wrapper b_3_w({"quick", "words", "result"});
  auto b_3 = cudf::dictionary::encode(b_3_w);

  auto table_a  = cudf::table_view({a_0, a_1, a_2, a_3->view()});
  auto table_b  = cudf::table_view({b_0, b_1, b_2, b_3->view()});
  auto expect_a = cudf::table_view({a_0, a_1, a_2, a_3_w});
  auto expect_b = cudf::table_view({b_0, b_1, b_2, b_3_w});
  {
    auto result      = cudf::left_semi_join(table_a, table_b, {0, 1}, {0, 1}, {0, 1, 2, 3});
    auto result_view = result->view();
    auto decoded3    = cudf::dictionary::decode(result_view.column(3));
    std::vector<cudf::column_view> result_decoded(
      {result_view.column(0), result_view.column(1), result_view.column(2), decoded3->view()});

    auto expected = cudf::left_semi_join(expect_a, expect_b, {0, 1}, {0, 1}, {0, 1, 2, 3});
    CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view(result_decoded), *expected);
  }
  {
    auto result      = cudf::left_semi_join(table_a, table_b, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});
    auto result_view = result->view();
    auto decoded3    = cudf::dictionary::decode(result_view.column(3));
    std::vector<cudf::column_view> result_decoded(
      {result_view.column(0), result_view.column(1), result_view.column(2), decoded3->view()});

    auto expected = cudf::left_semi_join(expect_a, expect_b, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view(result_decoded), *expected);
  }
}

TEST_F(JoinDictionaryTest, LeftSemiJoinWithNulls)
{
  column_wrapper<int32_t> a_0{10, 20, 20, 20, 20, 20, 50};
  column_wrapper<float> a_1{5.0, .5, .5, .7, .7, .7, .7};
  column_wrapper<int8_t> a_2{90, 77, 78, 61, 62, 63, 41};
  cudf::test::strings_column_wrapper a_3_w(
    {"quick", "accénted", "turtlé", "composéd", "result", "", "words"});
  auto a_3 = cudf::dictionary::encode(a_3_w);

  column_wrapper<int32_t> b_0{10, 20, 20, 50};
  column_wrapper<float> b_1{5.0, .7, .7, .7};
  column_wrapper<int8_t> b_2{90, 75, 62, 41};
  cudf::test::strings_column_wrapper b_3_w({"quick", "words", "result", ""}, {1, 1, 1, 0});
  auto b_3 = cudf::dictionary::encode(b_3_w);

  auto table_a = cudf::table_view({a_0, a_1, a_2, a_3->view()});
  auto table_b = cudf::table_view({b_0, b_1, b_2, b_3->view()});

  auto result      = cudf::left_semi_join(table_a, table_b, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});
  auto result_view = result->view();
  auto decoded3    = cudf::dictionary::decode(result_view.column(3));
  std::vector<cudf::column_view> result_decoded(
    {result_view.column(0), result_view.column(1), result_view.column(2), decoded3->view()});

  auto expect_a = cudf::table_view({a_0, a_1, a_2, a_3_w});
  auto expect_b = cudf::table_view({b_0, b_1, b_2, b_3_w});
  auto expected = cudf::left_semi_join(expect_a, expect_b, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view(result_decoded), *expected);
}

TEST_F(JoinDictionaryTest, LeftAntiJoin)
{
  column_wrapper<int32_t> a_0{10, 20, 20, 20, 20, 20, 50};
  column_wrapper<float> a_1{5.0, .5, .5, .7, .7, .7, .7};
  column_wrapper<int8_t> a_2{90, 77, 78, 61, 62, 63, 41};
  cudf::test::strings_column_wrapper a_3_w(
    {"quick", "accénted", "turtlé", "composéd", "result", "", "words"});
  auto a_3 = cudf::dictionary::encode(a_3_w);

  column_wrapper<int32_t> b_0{10, 20, 20};
  column_wrapper<float> b_1{5.0, .7, .7};
  column_wrapper<int8_t> b_2{90, 75, 62};
  cudf::test::strings_column_wrapper b_3_w({"quick", "words", "result"});
  auto b_3 = cudf::dictionary::encode(b_3_w);

  auto table_a  = cudf::table_view({a_0, a_1, a_2, a_3->view()});
  auto table_b  = cudf::table_view({b_0, b_1, b_2, b_3->view()});
  auto expect_a = cudf::table_view({a_0, a_1, a_2, a_3_w});
  auto expect_b = cudf::table_view({b_0, b_1, b_2, b_3_w});
  {
    auto result      = cudf::left_anti_join(table_a, table_b, {0, 1}, {0, 1}, {0, 1, 2, 3});
    auto result_view = result->view();
    auto decoded3    = cudf::dictionary::decode(result_view.column(3));
    std::vector<cudf::column_view> result_decoded(
      {result_view.column(0), result_view.column(1), result_view.column(2), decoded3->view()});

    auto expected = cudf::left_anti_join(expect_a, expect_b, {0, 1}, {0, 1}, {0, 1, 2, 3});
    CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view(result_decoded), *expected);
  }
  {
    auto result      = cudf::left_anti_join(table_a, table_b, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});
    auto result_view = result->view();
    auto decoded3    = cudf::dictionary::decode(result_view.column(3));
    std::vector<cudf::column_view> result_decoded(
      {result_view.column(0), result_view.column(1), result_view.column(2), decoded3->view()});

    auto expected = cudf::left_anti_join(expect_a, expect_b, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view(result_decoded), *expected);
  }
}

TEST_F(JoinDictionaryTest, LeftAntiJoinWithNulls)
{
  column_wrapper<int32_t> a_0{10, 20, 20, 20, 20, 20, 50};
  column_wrapper<float> a_1{5.0, .5, .5, .7, .7, .7, .7};
  column_wrapper<int8_t> a_2{90, 77, 78, 61, 62, 63, 41};
  cudf::test::strings_column_wrapper a_3_w(
    {"quick", "accénted", "turtlé", "composéd", "result", "", "words"});
  auto a_3 = cudf::dictionary::encode(a_3_w);

  column_wrapper<int32_t> b_0{10, 20, 20, 50};
  column_wrapper<float> b_1{5.0, .7, .7, .7};
  column_wrapper<int8_t> b_2{90, 75, 62, 41};
  cudf::test::strings_column_wrapper b_3_w({"quick", "words", "result", ""}, {1, 1, 1, 0});
  auto b_3 = cudf::dictionary::encode(b_3_w);

  auto table_a = cudf::table_view({a_0, a_1, a_2, a_3->view()});
  auto table_b = cudf::table_view({b_0, b_1, b_2, b_3->view()});

  auto result      = cudf::left_anti_join(table_a, table_b, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});
  auto result_view = result->view();
  auto decoded3    = cudf::dictionary::decode(result_view.column(3));
  std::vector<cudf::column_view> result_decoded(
    {result_view.column(0), result_view.column(1), result_view.column(2), decoded3->view()});

  auto expect_a = cudf::table_view({a_0, a_1, a_2, a_3_w});
  auto expect_b = cudf::table_view({b_0, b_1, b_2, b_3_w});
  auto expected = cudf::left_anti_join(expect_a, expect_b, {0, 1, 3}, {0, 1, 3}, {0, 1, 2, 3});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view(result_decoded), *expected);
}
