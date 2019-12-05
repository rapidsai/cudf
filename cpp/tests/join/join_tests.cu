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
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/join.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;
using CVector     = std::vector<std::unique_ptr<cudf::column>>;
using Table       = cudf::experimental::table;

struct JoinTest : public cudf::test::BaseFixture {};

//TEST_F(JoinTest, FullJoinNoNulls)
//TEST_F(JoinTest, LeftJoinNoNulls)
//TEST_F(JoinTest, InnerJoinNoNulls)

//TEST_F(JoinTest, FullJoinWithNulls)
//TEST_F(JoinTest, LeftJoinWithNulls)
//TEST_F(JoinTest, InnerJoinWithNulls)

TEST_F(JoinTest, FullJoin)
{
  column_wrapper <int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper           col0_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper <int32_t> col0_2{{0, 1, 2, 4, 1}, {1, 1, 1, 1, 1}};

  column_wrapper <int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper           col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  column_wrapper <int32_t> col1_2{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto join_table = cudf::full_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1,1}});
  for (auto&c : join_table->view()) {
    cudf::test::print(c); std::cout<<"\n";
  }
}

TEST_F(JoinTest, LeftJoin)
{
  column_wrapper <int32_t> col0_0{{3, 1, 2, 0, 3}};
  column_wrapper <int32_t> col0_1{{0, 1, 2, 4, 1}};

  column_wrapper <int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper <int32_t> col1_1{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto join_table = cudf::left_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1,1}});
  cudf::test::print(join_table->get_column(0)); std::cout<<"\n";
  cudf::test::print(join_table->get_column(1)); std::cout<<"\n";
}

TEST_F(JoinTest, InnerJoin)
{
  column_wrapper <int32_t> col0_0{{3, 1, 2, 0, 3}};
  column_wrapper <int32_t> col0_1{{0, 1, 2, 4, 1}};

  column_wrapper <int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper <int32_t> col1_1{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto join_table = cudf::inner_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1,1}});
  cudf::test::print(join_table->get_column(0)); std::cout<<"\n";
  cudf::test::print(join_table->get_column(1)); std::cout<<"\n";
}

TEST_F(JoinTest, LeftSemiJoin) {
  std::vector<const char*> a_strings{ "quick", "accénted", "turtlé", "composéd", "result", "", "words" };
  std::vector<const char*> b_strings { "quick", "words", "result" };
  std::vector<const char*> e_strings { "quick", "composéd", "result", "" };

  column_wrapper<int32_t> a_0 {  10,  20,  20,  20,  20,  20,  50 };
  column_wrapper<float>   a_1 { 5.0,  .5,  .5,  .7,  .7,  .7,  .7 };
  column_wrapper<int8_t>  a_2 {  90,  77,  78,  61,  62,  63,  41 };

  cudf::test::strings_column_wrapper a_3(a_strings.begin(),
                                         a_strings.end(),
                                         thrust::make_transform_iterator( a_strings.begin(), [] (auto str) { return str!=nullptr; }));

  column_wrapper<int32_t> b_0 {  10, 20, 20 };
  column_wrapper<float>   b_1 { 5.0, .7, .7 };
  column_wrapper<int8_t>  b_2 {  90, 75, 62 };

  cudf::test::strings_column_wrapper b_3(b_strings.begin(),
                                         b_strings.end(),
                                         thrust::make_transform_iterator( b_strings.begin(), [] (auto str) { return str!=nullptr; }));


  column_wrapper<int32_t> expect_0 {  10,  20,  20,  20 };
  column_wrapper<float>   expect_1 { 5.0,  .7,  .7,  .7 };
  column_wrapper<int8_t>  expect_2 {  90,  61,  62,  63 };

  cudf::test::strings_column_wrapper expect_3(e_strings.begin(),
                                              e_strings.end(),
                                              thrust::make_transform_iterator( e_strings.begin(), [] (auto str) { return str!=nullptr; }));


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

  cudf::experimental::table table_a(std::move(column_a));
  cudf::experimental::table table_b(std::move(column_b));

  auto join_table = cudf::left_semi_join(table_a, table_b,
                                         {0, 1},
                                         {0, 1},
                                         {0, 1, 2, 3});

  expect_columns_equal(join_table->get_column(0), expect_0);
  expect_columns_equal(join_table->get_column(1), expect_1);
  expect_columns_equal(join_table->get_column(2), expect_2);
  expect_columns_equal(join_table->get_column(3), expect_3);
}

TEST_F(JoinTest, LeftSemiJoin_with_a_string_key) {
  std::vector<const char*> a_strings{ "quick", "accénted", "turtlé", "composéd", "result", "", "words" };
  std::vector<const char*> b_strings { "quick", "words", "result" };
  std::vector<const char*> e_strings { "quick", "result" };

  column_wrapper<int32_t> a_0 {  10,  20,  20,  20,  20,  20,  50 };
  column_wrapper<float>   a_1 { 5.0,  .5,  .5,  .7,  .7,  .7,  .7 };
  column_wrapper<int8_t>  a_2 {  90,  77,  78,  61,  62,  63,  41 };

  cudf::test::strings_column_wrapper a_3(a_strings.begin(),
                                         a_strings.end(),
                                         thrust::make_transform_iterator( a_strings.begin(), [] (auto str) { return str!=nullptr; }));

  column_wrapper<int32_t> b_0 {  10, 20, 20 };
  column_wrapper<float>   b_1 { 5.0, .7, .7 };
  column_wrapper<int8_t>  b_2 {  90, 75, 62 };

  cudf::test::strings_column_wrapper b_3(b_strings.begin(),
                                         b_strings.end(),
                                         thrust::make_transform_iterator( b_strings.begin(), [] (auto str) { return str!=nullptr; }));


  column_wrapper<int32_t> expect_0 {  10,  20 };
  column_wrapper<float>   expect_1 { 5.0,  .7 };
  column_wrapper<int8_t>  expect_2 {  90,  62 };

  cudf::test::strings_column_wrapper expect_3(e_strings.begin(),
                                              e_strings.end(),
                                              thrust::make_transform_iterator( e_strings.begin(), [] (auto str) { return str!=nullptr; }));


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

  cudf::experimental::table table_a(std::move(column_a));
  cudf::experimental::table table_b(std::move(column_b));

  auto join_table = cudf::left_semi_join(table_a, table_b,
                                         {0, 1, 3},
                                         {0, 1, 3},
                                         {0, 1, 2, 3});

  expect_columns_equal(join_table->get_column(0), expect_0);
  expect_columns_equal(join_table->get_column(1), expect_1);
  expect_columns_equal(join_table->get_column(2), expect_2);
  expect_columns_equal(join_table->get_column(3), expect_3);
}

TEST_F(JoinTest, LeftSemiJoin_with_null) {
  std::vector<const char*> a_strings{ "quick", "accénted", "turtlé", "composéd", "result", "", "words" };
  std::vector<const char*> b_strings { "quick", "words", "result", nullptr };
  std::vector<const char*> e_strings { "quick", "result" };

  column_wrapper<int32_t> a_0 {  10,  20,  20,  20,  20,  20,  50 };
  column_wrapper<float>   a_1 { 5.0,  .5,  .5,  .7,  .7,  .7,  .7 };
  column_wrapper<int8_t>  a_2 {  90,  77,  78,  61,  62,  63,  41 };

  cudf::test::strings_column_wrapper a_3(a_strings.begin(),
                                         a_strings.end(),
                                         thrust::make_transform_iterator( a_strings.begin(), [] (auto str) { return str!=nullptr; }));

  column_wrapper<int32_t> b_0 {  10, 20, 20, 50 };
  column_wrapper<float>   b_1 { 5.0, .7, .7, .7 };
  column_wrapper<int8_t>  b_2 {  90, 75, 62, 41 };

  cudf::test::strings_column_wrapper b_3(b_strings.begin(),
                                         b_strings.end(),
                                         thrust::make_transform_iterator( b_strings.begin(), [] (auto str) { return str!=nullptr; }));


  column_wrapper<int32_t> expect_0 {  10,  20 };
  column_wrapper<float>   expect_1 { 5.0,  .7 };
  column_wrapper<int8_t>  expect_2 {  90,  62 };

  cudf::test::strings_column_wrapper expect_3(e_strings.begin(),
                                              e_strings.end(),
                                              thrust::make_transform_iterator( e_strings.begin(), [] (auto str) { return str!=nullptr; }));


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

  cudf::experimental::table table_a(std::move(column_a));
  cudf::experimental::table table_b(std::move(column_b));

  auto join_table = cudf::left_semi_join(table_a, table_b,
                                         {0, 1, 3},
                                         {0, 1, 3},
                                         {0, 1, 2, 3});

  expect_columns_equal(join_table->get_column(0), expect_0);
  expect_columns_equal(join_table->get_column(1), expect_1);
  expect_columns_equal(join_table->get_column(2), expect_2);
  expect_columns_equal(join_table->get_column(3), expect_3);
}

TEST_F(JoinTest, LeftAntiJoin) {
  std::vector<const char*> a_strings { "quick", "accénted", "turtlé", "composéd", "result", "", "words" };
  std::vector<const char*> b_strings { "quick", "words", "result" };
  std::vector<const char*> e_strings { "accénted", "turtlé", "words" };

  column_wrapper<int32_t> a_0 {  10,  20,  20,  20,  20,  20,  50 };
  column_wrapper<float>   a_1 { 5.0,  .5,  .5,  .7,  .7,  .7,  .7 };
  column_wrapper<int8_t>  a_2 {  90,  77,  78,  61,  62,  63,  41 };

  cudf::test::strings_column_wrapper a_3(a_strings.begin(),
                                         a_strings.end(),
                                         thrust::make_transform_iterator( a_strings.begin(), [] (auto str) { return str!=nullptr; }));

  column_wrapper<int32_t> b_0 {  10, 20, 20 };
  column_wrapper<float>   b_1 { 5.0, .7, .7 };
  column_wrapper<int8_t>  b_2 {  90, 75, 62 };

  cudf::test::strings_column_wrapper b_3(b_strings.begin(),
                                         b_strings.end(),
                                         thrust::make_transform_iterator( b_strings.begin(), [] (auto str) { return str!=nullptr; }));


  column_wrapper<int32_t> expect_0 { 20,  20, 50 };
  column_wrapper<float>   expect_1 { .5,  .5, .7 };
  column_wrapper<int8_t>  expect_2 { 77,  78, 41 };

  cudf::test::strings_column_wrapper expect_3(e_strings.begin(),
                                              e_strings.end(),
                                              thrust::make_transform_iterator( e_strings.begin(), [] (auto str) { return str!=nullptr; }));


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

  cudf::experimental::table table_a(std::move(column_a));
  cudf::experimental::table table_b(std::move(column_b));

  auto join_table = cudf::left_anti_join(table_a, table_b,
                                         {0, 1},
                                         {0, 1},
                                         {0, 1, 2, 3});

  expect_columns_equal(join_table->get_column(0), expect_0);
  expect_columns_equal(join_table->get_column(1), expect_1);
  expect_columns_equal(join_table->get_column(2), expect_2);
  expect_columns_equal(join_table->get_column(3), expect_3);
}

TEST_F(JoinTest, LeftAntiJoin_with_a_string_key) {
  std::vector<const char*> a_strings { "quick", "accénted", "turtlé", "composéd", "result", "", "words" };
  std::vector<const char*> b_strings { "quick", "words", "result" };
  std::vector<const char*> e_strings { "accénted", "turtlé", "composéd", "", "words" };

  column_wrapper<int32_t> a_0 {  10,  20,  20,  20,  20,  20,  50 };
  column_wrapper<float>   a_1 { 5.0,  .5,  .5,  .7,  .7,  .7,  .7 };
  column_wrapper<int8_t>  a_2 {  90,  77,  78,  61,  62,  63,  41 };

  cudf::test::strings_column_wrapper a_3(a_strings.begin(),
                                         a_strings.end(),
                                         thrust::make_transform_iterator( a_strings.begin(), [] (auto str) { return str!=nullptr; }));

  column_wrapper<int32_t> b_0 {  10, 20, 20 };
  column_wrapper<float>   b_1 { 5.0, .7, .7 };
  column_wrapper<int8_t>  b_2 {  90, 75, 62 };

  cudf::test::strings_column_wrapper b_3(b_strings.begin(),
                                         b_strings.end(),
                                         thrust::make_transform_iterator( b_strings.begin(), [] (auto str) { return str!=nullptr; }));


  column_wrapper<int32_t> expect_0 { 20,  20,  20,  20,  50 };
  column_wrapper<float>   expect_1 { .5,  .5,  .7,  .7,  .7 };
  column_wrapper<int8_t>  expect_2 { 77,  78,  61,  63,  41 };

  cudf::test::strings_column_wrapper expect_3(e_strings.begin(),
                                              e_strings.end(),
                                              thrust::make_transform_iterator( e_strings.begin(), [] (auto str) { return str!=nullptr; }));


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

  cudf::experimental::table table_a(std::move(column_a));
  cudf::experimental::table table_b(std::move(column_b));

  auto join_table = cudf::left_anti_join(table_a, table_b,
                                         {0, 1, 3},
                                         {0, 1, 3},
                                         {0, 1, 2, 3});

  expect_columns_equal(join_table->get_column(0), expect_0);
  expect_columns_equal(join_table->get_column(1), expect_1);
  expect_columns_equal(join_table->get_column(2), expect_2);
  expect_columns_equal(join_table->get_column(3), expect_3);
}

TEST_F(JoinTest, LeftAntiJoin_with_null) {
  std::vector<const char*> a_strings { "quick", "accénted", "turtlé", "composéd", "result", "", "words" };
  std::vector<const char*> b_strings { "quick", "words", "result", nullptr };
  std::vector<const char*> e_strings { "accénted", "turtlé", "composéd", "", "words" };

  column_wrapper<int32_t> a_0 {  10,  20,  20,  20,  20,  20,  50 };
  column_wrapper<float>   a_1 { 5.0,  .5,  .5,  .7,  .7,  .7,  .7 };
  column_wrapper<int8_t>  a_2 {  90,  77,  78,  61,  62,  63,  41 };

  cudf::test::strings_column_wrapper a_3(a_strings.begin(),
                                         a_strings.end(),
                                         thrust::make_transform_iterator( a_strings.begin(), [] (auto str) { return str!=nullptr; }));

  column_wrapper<int32_t> b_0 {  10, 20, 20, 50 };
  column_wrapper<float>   b_1 { 5.0, .7, .7, .7 };
  column_wrapper<int8_t>  b_2 {  90, 75, 62, 41 };

  cudf::test::strings_column_wrapper b_3(b_strings.begin(),
                                         b_strings.end(),
                                         thrust::make_transform_iterator( b_strings.begin(), [] (auto str) { return str!=nullptr; }));


  column_wrapper<int32_t> expect_0 { 20,  20,  20,  20,  50 };
  column_wrapper<float>   expect_1 { .5,  .5,  .7,  .7,  .7 };
  column_wrapper<int8_t>  expect_2 { 77,  78,  61,  63,  41 };

  cudf::test::strings_column_wrapper expect_3(e_strings.begin(),
                                              e_strings.end(),
                                              thrust::make_transform_iterator( e_strings.begin(), [] (auto str) { return str!=nullptr; }));


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

  cudf::experimental::table table_a(std::move(column_a));
  cudf::experimental::table table_b(std::move(column_b));

  auto join_table = cudf::left_anti_join(table_a, table_b,
                                         {0, 1, 3},
                                         {0, 1, 3},
                                         {0, 1, 2, 3});

  expect_columns_equal(join_table->get_column(0), expect_0);
  expect_columns_equal(join_table->get_column(1), expect_1);
  expect_columns_equal(join_table->get_column(2), expect_2);
  expect_columns_equal(join_table->get_column(3), expect_3);
}

