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

#include <cudf/ast/nodes.hpp>
#include <cudf/ast/operators.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <utility>
#include <vector>

template <typename T>
struct ConditionalJoinTest : public cudf::test::BaseFixture {
  void test(std::vector<std::vector<T>> left_data,
            std::vector<std::vector<T>> right_data,
            cudf::ast::expression predicate,
            std::vector<std::pair<cudf::size_type, cudf::size_type>> expected_outputs)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    std::vector<cudf::test::fixed_width_column_wrapper<T>> left_wrappers;
    std::vector<cudf::test::fixed_width_column_wrapper<T>> right_wrappers;

    std::vector<cudf::column_view> left_columns;
    std::vector<cudf::column_view> right_columns;

    for (auto v : left_data) {
      left_wrappers.push_back(cudf::test::fixed_width_column_wrapper<T>(v.begin(), v.end()));
      left_columns.push_back(left_wrappers.back());
    }

    for (auto v : right_data) {
      right_wrappers.push_back(cudf::test::fixed_width_column_wrapper<T>(v.begin(), v.end()));
      right_columns.push_back(right_wrappers.back());
    }

    cudf::table_view left(left_columns);
    cudf::table_view right(right_columns);

    auto result = cudf::predicate_join(left, right, predicate);

    std::vector<std::pair<cudf::size_type, cudf::size_type>> resulting_pairs;
    for (size_t i = 0; i < result.first->size(); ++i) {
      // Note: Not trying to be terribly efficient here since these tests are
      // small, otherwise a batch copy to host before constructing the tuples
      // would be important.
      resulting_pairs.push_back({result.first->element(i, rmm::cuda_stream_default),
                                 result.second->element(i, rmm::cuda_stream_default)});
    }
    std::sort(resulting_pairs.begin(), resulting_pairs.end());
    std::sort(expected_outputs.begin(), expected_outputs.end());
    EXPECT_TRUE(
      std::equal(resulting_pairs.begin(), resulting_pairs.end(), expected_outputs.begin()));
  }
};

template <typename T>
struct ConditionalJoinFirstColumnEqualityTest : public ConditionalJoinTest<T> {
  void test(std::vector<std::vector<T>> left_data,
            std::vector<std::vector<T>> right_data,
            std::vector<std::pair<cudf::size_type, cudf::size_type>> expected_outputs)
  {
    auto col_ref_0 = cudf::ast::column_reference(0);
    auto col_ref_1 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
    auto predicate = cudf::ast::expression(cudf::ast::ast_operator::EQUAL, col_ref_0, col_ref_1);

    ConditionalJoinTest<T>::test(left_data, right_data, predicate, expected_outputs);
  }
};

// TYPED_TEST_CASE(ConditionalJoinTest, cudf::test::IntegralTypesNotBool);
TYPED_TEST_CASE(ConditionalJoinFirstColumnEqualityTest, cudf::test::Types<int32_t>);

TYPED_TEST(ConditionalJoinFirstColumnEqualityTest, TestOneColumnOneRowAllEqual)
{
  this->test({{0}}, {{0}}, {{0, 0}});
};

TYPED_TEST(ConditionalJoinFirstColumnEqualityTest, TestOneColumnTwoRowAllEqual)
{
  this->test({{0, 1}}, {{0, 0}}, {{0, 0}, {0, 1}});
};

TYPED_TEST(ConditionalJoinFirstColumnEqualityTest, TestTwoColumnOneRowAllEqual)
{
  this->test({{0}, {0}}, {{0}, {0}}, {{0, 0}});
};

TYPED_TEST(ConditionalJoinFirstColumnEqualityTest, TestTwoColumnThreeRowAllEqual)
{
  this->test({{0, 1, 2}, {10, 20, 30}}, {{0, 1, 2}, {30, 40, 50}}, {{0, 0}, {1, 1}, {2, 2}});
};

TYPED_TEST(ConditionalJoinFirstColumnEqualityTest, TestTwoColumnThreeRowSomeEqual)
{
  this->test({{0, 1, 2}, {10, 20, 30}}, {{0, 1, 3}, {30, 40, 50}}, {{0, 0}, {1, 1}});
};

TYPED_TEST_CASE(ConditionalJoinTest, cudf::test::Types<int32_t>);

TYPED_TEST(ConditionalJoinTest, TestNotComparison)
{
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto expression = cudf::ast::expression(cudf::ast::ast_operator::NOT, col_ref_0);

  this->test({{0, 1, 2}}, {{3, 4, 5}}, expression, {{0, 0}, {0, 1}, {0, 2}});
};

TYPED_TEST(ConditionalJoinTest, TestGreaterComparison)
{
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto expression = cudf::ast::expression(cudf::ast::ast_operator::GREATER, col_ref_0, col_ref_1);

  this->test({{0, 1, 2}}, {{1, 0, 0}}, expression, {{1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
};

TYPED_TEST(ConditionalJoinTest, TestComplexConditionMultipleColumns)
{
  // LEFT is implicit, but specifying explicitly to validate that it works.
  auto col_ref_0      = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto literal_value  = cudf::numeric_scalar<int32_t>(1);
  auto literal_0      = cudf::ast::literal(literal_value);
  auto literal_filter = cudf::ast::expression(cudf::ast::ast_operator::EQUAL, col_ref_0, literal_0);

  auto col_ref_1 = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto comparison_filter =
    cudf::ast::expression(cudf::ast::ast_operator::LESS, col_ref_1, col_ref_0);

  auto expression =
    cudf::ast::expression(cudf::ast::ast_operator::LOGICAL_AND, literal_filter, comparison_filter);

  this->test({{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
             {{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2},
              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
              {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
             expression,
             {{4, 0}, {5, 0}, {6, 0}, {7, 0}});
};
