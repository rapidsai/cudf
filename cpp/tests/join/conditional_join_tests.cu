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

#include <thrust/execution_policy.h>
#include <thrust/pair.h>
#include <thrust/sort.h>

#include <algorithm>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

// Defining expressions for AST evaluation is currently a bit tedious, so we
// define some standard nodes here that can be easily reused elsewhere.
namespace {
constexpr cudf::size_type JoinNoneValue =
  std::numeric_limits<cudf::size_type>::min();  // TODO: how to test if this isn't public?

// Common column references.
const auto col_ref_left_0  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
const auto col_ref_right_0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
const auto col_ref_left_1  = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
const auto col_ref_right_1 = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);

// Common expressions.
auto left_zero_eq_right_zero =
  cudf::ast::expression(cudf::ast::ast_operator::EQUAL, col_ref_left_0, col_ref_right_0);
}  // namespace

/**
 * The principal fixture for all conditional joins.
 */
template <typename T>
struct ConditionalJoinTest : public cudf::test::BaseFixture {
  /**
   * Convenience utility for parsing initializer lists of input data into
   * suitable inputs for tables.
   */
  std::tuple<std::vector<cudf::test::fixed_width_column_wrapper<T>>,
             std::vector<cudf::test::fixed_width_column_wrapper<T>>,
             std::vector<cudf::column_view>,
             std::vector<cudf::column_view>,
             cudf::table_view,
             cudf::table_view>
  parse_input(std::vector<std::vector<T>> left_data, std::vector<std::vector<T>> right_data)
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

    return std::make_tuple(std::move(left_wrappers),
                           std::move(right_wrappers),
                           std::move(left_columns),
                           std::move(right_columns),
                           cudf::table_view(left_columns),
                           cudf::table_view(right_columns));
  }

  std::tuple<std::vector<cudf::test::fixed_width_column_wrapper<T>>,
             std::vector<cudf::test::fixed_width_column_wrapper<T>>,
             std::vector<cudf::column_view>,
             std::vector<cudf::column_view>,
             cudf::table_view,
             cudf::table_view>
  parse_input(std::vector<std::pair<std::vector<T>, std::vector<bool>>> left_data,
              std::vector<std::pair<std::vector<T>, std::vector<bool>>> right_data)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    std::vector<cudf::test::fixed_width_column_wrapper<T>> left_wrappers;
    std::vector<cudf::test::fixed_width_column_wrapper<T>> right_wrappers;

    std::vector<cudf::column_view> left_columns;
    std::vector<cudf::column_view> right_columns;

    for (auto v : left_data) {
      left_wrappers.push_back(cudf::test::fixed_width_column_wrapper<T>(
        v.first.begin(), v.first.end(), v.second.begin()));
      left_columns.push_back(left_wrappers.back());
    }

    for (auto v : right_data) {
      right_wrappers.push_back(cudf::test::fixed_width_column_wrapper<T>(
        v.first.begin(), v.first.end(), v.second.begin()));
      right_columns.push_back(right_wrappers.back());
    }

    return std::make_tuple(std::move(left_wrappers),
                           std::move(right_wrappers),
                           std::move(left_columns),
                           std::move(right_columns),
                           cudf::table_view(left_columns),
                           cudf::table_view(right_columns));
  }
};

/**
 * Fixture for join types that return both left and right indices (inner, left,
 * and full joins).
 */
template <typename T>
struct ConditionalJoinPairReturnTest : public ConditionalJoinTest<T> {
  /*
   * Perform a join of tables constructed from two input data sets according to
   * the provided predicate and verify that the outputs match the expected
   * outputs (up to order).
   */
  void test(std::vector<std::vector<T>> left_data,
            std::vector<std::vector<T>> right_data,
            cudf::ast::expression predicate,
            std::vector<std::pair<cudf::size_type, cudf::size_type>> expected_outputs)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    auto [left_wrappers, right_wrappers, left_columns, right_columns, left, right] =
      this->parse_input(left_data, right_data);
    auto result = this->join(left, right, predicate);

    std::vector<std::pair<cudf::size_type, cudf::size_type>> result_pairs;
    for (size_t i = 0; i < result.first->size(); ++i) {
      // Note: Not trying to be terribly efficient here since these tests are
      // small, otherwise a batch copy to host before constructing the tuples
      // would be important.
      result_pairs.push_back({result.first->element(i, rmm::cuda_stream_default),
                              result.second->element(i, rmm::cuda_stream_default)});
    }
    std::sort(result_pairs.begin(), result_pairs.end());
    std::sort(expected_outputs.begin(), expected_outputs.end());

    EXPECT_TRUE(std::equal(result_pairs.begin(), result_pairs.end(), expected_outputs.begin()));
  }

  void test_nulls(std::vector<std::pair<std::vector<T>, std::vector<bool>>> left_data,
                  std::vector<std::pair<std::vector<T>, std::vector<bool>>> right_data,
                  cudf::ast::expression predicate,
                  std::vector<std::pair<cudf::size_type, cudf::size_type>> expected_outputs)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    auto [left_wrappers, right_wrappers, left_columns, right_columns, left, right] =
      this->parse_input(left_data, right_data);
    auto result = this->join(left, right, predicate);

    std::vector<std::pair<cudf::size_type, cudf::size_type>> result_pairs;
    for (size_t i = 0; i < result.first->size(); ++i) {
      // Note: Not trying to be terribly efficient here since these tests are
      // small, otherwise a batch copy to host before constructing the tuples
      // would be important.
      result_pairs.push_back({result.first->element(i, rmm::cuda_stream_default),
                              result.second->element(i, rmm::cuda_stream_default)});
    }
    std::sort(result_pairs.begin(), result_pairs.end());
    std::sort(expected_outputs.begin(), expected_outputs.end());

    EXPECT_TRUE(std::equal(result_pairs.begin(), result_pairs.end(), expected_outputs.begin()));
  }

  /*
   * Perform a join of tables constructed from two input data sets according to
   * an equality predicate on all corresponding columns and verify that the outputs match the
   * expected outputs (up to order).
   */
  void compare_to_hash_join(std::vector<std::vector<T>> left_data,
                            std::vector<std::vector<T>> right_data)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    auto [left_wrappers, right_wrappers, left_columns, right_columns, left, right] =
      this->parse_input(left_data, right_data);
    // TODO: Generalize this to support multiple columns by automatically
    // constructing the appropriate expression.
    auto result    = this->join(left, right, left_zero_eq_right_zero);
    auto reference = this->reference_join(left, right);

    thrust::device_vector<thrust::pair<cudf::size_type, cudf::size_type>> result_pairs(
      result.first->size());
    thrust::device_vector<thrust::pair<cudf::size_type, cudf::size_type>> reference_pairs(
      reference.first->size());

    thrust::transform(thrust::device,
                      result.first->begin(),
                      result.first->end(),
                      result.second->begin(),
                      result_pairs.begin(),
                      [] __device__(cudf::size_type first, cudf::size_type second) {
                        return thrust::make_pair(first, second);
                      });
    thrust::transform(thrust::device,
                      reference.first->begin(),
                      reference.first->end(),
                      reference.second->begin(),
                      reference_pairs.begin(),
                      [] __device__(cudf::size_type first, cudf::size_type second) {
                        return thrust::make_pair(first, second);
                      });

    thrust::sort(thrust::device, result_pairs.begin(), result_pairs.end());
    thrust::sort(thrust::device, reference_pairs.begin(), reference_pairs.end());

    EXPECT_TRUE(thrust::equal(
      thrust::device, result_pairs.begin(), result_pairs.end(), reference_pairs.begin()));
  }

  /**
   * This method must be implemented by subclasses for specific types of joins.
   * It should be a simply forwarding of arguments to the appropriate cudf
   * conditional join API.
   */
  virtual std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                    std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  join(cudf::table_view left, cudf::table_view right, cudf::ast::expression predicate) = 0;

  /**
   * This method must be implemented by subclasses for specific types of joins.
   * It should be a simply forwarding of arguments to the appropriate cudf
   * hash join API for comparison with conditional joins.
   */
  virtual std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                    std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  reference_join(cudf::table_view left, cudf::table_view right) = 0;
};

/**
 * Tests of inner joins.
 */
template <typename T>
struct ConditionalInnerJoinTest : public ConditionalJoinPairReturnTest<T> {
  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  join(cudf::table_view left, cudf::table_view right, cudf::ast::expression predicate) override
  {
    return cudf::conditional_inner_join(left, right, predicate);
  }

  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  reference_join(cudf::table_view left, cudf::table_view right) override
  {
    return cudf::inner_join(left, right);
  }
};

TYPED_TEST_CASE(ConditionalInnerJoinTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(ConditionalInnerJoinTest, TestOneColumnOneRowAllEqual)
{
  this->test({{0}}, {{0}}, left_zero_eq_right_zero, {{0, 0}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestOneColumnTwoRowAllEqual)
{
  this->test({{0, 1}}, {{0, 0}}, left_zero_eq_right_zero, {{0, 0}, {0, 1}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestTwoColumnOneRowAllEqual)
{
  this->test({{0}, {0}}, {{0}, {0}}, left_zero_eq_right_zero, {{0, 0}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestTwoColumnThreeRowAllEqual)
{
  this->test({{0, 1, 2}, {10, 20, 30}},
             {{0, 1, 2}, {30, 40, 50}},
             left_zero_eq_right_zero,
             {{0, 0}, {1, 1}, {2, 2}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestTwoColumnThreeRowSomeEqual)
{
  this->test({{0, 1, 2}, {10, 20, 30}},
             {{0, 1, 3}, {30, 40, 50}},
             left_zero_eq_right_zero,
             {{0, 0}, {1, 1}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestNotComparison)
{
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto expression = cudf::ast::expression(cudf::ast::ast_operator::NOT, col_ref_0);

  this->test({{0, 1, 2}}, {{3, 4, 5}}, expression, {{0, 0}, {0, 1}, {0, 2}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestGreaterComparison)
{
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto expression = cudf::ast::expression(cudf::ast::ast_operator::GREATER, col_ref_0, col_ref_1);

  this->test({{0, 1, 2}}, {{1, 0, 0}}, expression, {{1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestGreaterTwoColumnComparison)
{
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto expression = cudf::ast::expression(cudf::ast::ast_operator::GREATER, col_ref_0, col_ref_1);

  this->test({{0, 1, 2}, {0, 0, 0}},
             {{0, 0, 0}, {1, 0, 0}},
             expression,
             {{1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestGreaterDifferentNumberColumnComparison)
{
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto expression = cudf::ast::expression(cudf::ast::ast_operator::GREATER, col_ref_0, col_ref_1);

  this->test(
    {{0, 1, 2}}, {{0, 0, 0}, {1, 0, 0}}, expression, {{1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestGreaterDifferentNumberColumnDifferentSizeComparison)
{
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto expression = cudf::ast::expression(cudf::ast::ast_operator::GREATER, col_ref_0, col_ref_1);

  this->test({{0, 1}}, {{0, 0, 0}, {1, 0, 0}}, expression, {{1, 1}, {1, 2}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestComplexConditionMultipleColumns)
{
  // LEFT is implicit, but specifying explicitly to validate that it works.
  auto col_ref_0      = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto scalar_1       = cudf::numeric_scalar<TypeParam>(1);
  auto literal_1      = cudf::ast::literal(scalar_1);
  auto left_0_equal_1 = cudf::ast::expression(cudf::ast::ast_operator::EQUAL, col_ref_0, literal_1);

  auto col_ref_1 = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto comparison_filter =
    cudf::ast::expression(cudf::ast::ast_operator::LESS, col_ref_1, col_ref_0);

  auto expression =
    cudf::ast::expression(cudf::ast::ast_operator::LOGICAL_AND, left_0_equal_1, comparison_filter);

  this->test({{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
             {{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2},
              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
              {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
             expression,
             {{4, 0}, {5, 0}, {6, 0}, {7, 0}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestCompareRandomToHash)
{
  // Generate columns of 10 repeats of the integer range [0, 10), then merge
  // a shuffled version and compare to hash join.
  unsigned int N           = 10000;
  unsigned int num_repeats = 10;
  unsigned int num_unique  = N / num_repeats;

  std::vector<TypeParam> left(N);
  std::vector<TypeParam> right(N);

  for (unsigned int i = 0; i < num_repeats; ++i) {
    std::iota(
      std::next(left.begin(), num_unique * i), std::next(left.begin(), num_unique * (i + 1)), 0);
    std::iota(
      std::next(right.begin(), num_unique * i), std::next(right.begin(), num_unique * (i + 1)), 0);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(left.begin(), left.end(), gen);
  std::shuffle(right.begin(), right.end(), gen);

  this->compare_to_hash_join({left}, {right});
};

TYPED_TEST(ConditionalInnerJoinTest, TestOneColumnTwoNullsRowAllEqual)
{
  this->test_nulls(
    {{{0, 1}, {1, 0}}}, {{{0, 0}, {1, 1}}}, left_zero_eq_right_zero, {{0, 0}, {0, 1}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestOneColumnTwoNullsNoOutputRowAllEqual)
{
  this->test_nulls({{{0, 1}, {0, 1}}}, {{{0, 0}, {1, 1}}}, left_zero_eq_right_zero, {{}, {}});
};

/**
 * Tests of left joins.
 */
template <typename T>
struct ConditionalLeftJoinTest : public ConditionalJoinPairReturnTest<T> {
  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  join(cudf::table_view left, cudf::table_view right, cudf::ast::expression predicate) override
  {
    return cudf::conditional_left_join(left, right, predicate);
  }

  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  reference_join(cudf::table_view left, cudf::table_view right) override
  {
    return cudf::left_join(left, right);
  }
};

TYPED_TEST_CASE(ConditionalLeftJoinTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(ConditionalLeftJoinTest, TestTwoColumnThreeRowSomeEqual)
{
  this->test({{0, 1, 2}, {10, 20, 30}},
             {{0, 1, 3}, {30, 40, 50}},
             left_zero_eq_right_zero,
             {{0, 0}, {1, 1}, {2, JoinNoneValue}});
};

TYPED_TEST(ConditionalLeftJoinTest, TestCompareRandomToHash)
{
  // Generate columns of 10 repeats of the integer range [0, 10), then merge
  // a shuffled version and compare to hash join.
  unsigned int N           = 10000;
  unsigned int num_repeats = 10;
  unsigned int num_unique  = N / num_repeats;

  std::vector<TypeParam> left(N);
  std::vector<TypeParam> right(N);

  for (unsigned int i = 0; i < num_repeats; ++i) {
    std::iota(
      std::next(left.begin(), num_unique * i), std::next(left.begin(), num_unique * (i + 1)), 0);
    std::iota(
      std::next(right.begin(), num_unique * i), std::next(right.begin(), num_unique * (i + 1)), 0);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(left.begin(), left.end(), gen);
  std::shuffle(right.begin(), right.end(), gen);

  this->compare_to_hash_join({left}, {right});
};

/**
 * Tests of full joins.
 */
template <typename T>
struct ConditionalFullJoinTest : public ConditionalJoinPairReturnTest<T> {
  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  join(cudf::table_view left, cudf::table_view right, cudf::ast::expression predicate) override
  {
    return cudf::conditional_full_join(left, right, predicate);
  }

  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
  reference_join(cudf::table_view left, cudf::table_view right) override
  {
    return cudf::full_join(left, right);
  }
};

TYPED_TEST_CASE(ConditionalFullJoinTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(ConditionalFullJoinTest, TestTwoColumnThreeRowSomeEqual)
{
  this->test({{0, 1, 2}, {10, 20, 30}},
             {{0, 1, 3}, {30, 40, 50}},
             left_zero_eq_right_zero,
             {{0, 0}, {1, 1}, {2, JoinNoneValue}, {JoinNoneValue, 2}});
};

TYPED_TEST(ConditionalFullJoinTest, TestCompareRandomToHash)
{
  // Generate columns of 10 repeats of the integer range [0, 10), then merge
  // a shuffled version and compare to hash join.
  unsigned int N           = 10000;
  unsigned int num_repeats = 10;
  unsigned int num_unique  = N / num_repeats;

  std::vector<TypeParam> left(N);
  std::vector<TypeParam> right(N);

  for (unsigned int i = 0; i < num_repeats; ++i) {
    std::iota(
      std::next(left.begin(), num_unique * i), std::next(left.begin(), num_unique * (i + 1)), 0);
    std::iota(
      std::next(right.begin(), num_unique * i), std::next(right.begin(), num_unique * (i + 1)), 0);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(left.begin(), left.end(), gen);
  std::shuffle(right.begin(), right.end(), gen);

  this->compare_to_hash_join({left}, {right});
};

/**
 * Fixture for join types that return both only left indices (left semi and
 * left anti).
 */
template <typename T>
struct ConditionalJoinSingleReturnTest : public ConditionalJoinTest<T> {
  /*
   * Perform a join of tables constructed from two input data sets according to
   * the provided predicate and verify that the outputs match the expected
   * outputs (up to order).
   */
  void test(std::vector<std::vector<T>> left_data,
            std::vector<std::vector<T>> right_data,
            cudf::ast::expression predicate,
            std::vector<cudf::size_type> expected_outputs)
  {
    auto [left_wrappers, right_wrappers, left_columns, right_columns, left, right] =
      this->parse_input(left_data, right_data);
    auto result = this->join(left, right, predicate);

    std::vector<cudf::size_type> resulting_indices;
    for (size_t i = 0; i < result->size(); ++i) {
      // Note: Not trying to be terribly efficient here since these tests are
      // small, otherwise a batch copy to host before constructing the tuples
      // would be important.
      resulting_indices.push_back(result->element(i, rmm::cuda_stream_default));
    }
    std::sort(resulting_indices.begin(), resulting_indices.end());
    std::sort(expected_outputs.begin(), expected_outputs.end());
    EXPECT_TRUE(
      std::equal(resulting_indices.begin(), resulting_indices.end(), expected_outputs.begin()));
  }

  /*
   * Perform a join of tables constructed from two input data sets according to
   * an equality predicate on all corresponding columns and verify that the outputs match the
   * expected outputs (up to order).
   */
  void compare_to_hash_join(std::vector<std::vector<T>> left_data,
                            std::vector<std::vector<T>> right_data)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    auto [left_wrappers, right_wrappers, left_columns, right_columns, left, right] =
      this->parse_input(left_data, right_data);
    // TODO: Generalize this to support multiple columns by automatically
    // constructing the appropriate expression.
    auto result    = this->join(left, right, left_zero_eq_right_zero);
    auto reference = this->reference_join(left, right);

    thrust::sort(thrust::device, result->begin(), result->end());
    thrust::sort(thrust::device, reference->begin(), reference->end());

    EXPECT_TRUE(thrust::equal(thrust::device, result->begin(), result->end(), reference->begin()));
  }

  /**
   * This method must be implemented by subclasses for specific types of joins.
   * It should be a simply forwarding of arguments to the appropriate cudf
   * conditional join API.
   */
  virtual std::unique_ptr<rmm::device_uvector<cudf::size_type>> join(
    cudf::table_view left, cudf::table_view right, cudf::ast::expression predicate) = 0;

  /**
   * This method must be implemented by subclasses for specific types of joins.
   * It should be a simply forwarding of arguments to the appropriate cudf
   * hash join API for comparison with conditional joins.
   */
  virtual std::unique_ptr<rmm::device_uvector<cudf::size_type>> reference_join(
    cudf::table_view left, cudf::table_view right) = 0;
};

/**
 * Tests of left semi joins.
 */
template <typename T>
struct ConditionalLeftSemiJoinTest : public ConditionalJoinSingleReturnTest<T> {
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> join(
    cudf::table_view left, cudf::table_view right, cudf::ast::expression predicate) override
  {
    return cudf::conditional_left_semi_join(left, right, predicate);
  }

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> reference_join(
    cudf::table_view left, cudf::table_view right) override
  {
    return cudf::left_semi_join(left, right);
  }
};

TYPED_TEST_CASE(ConditionalLeftSemiJoinTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(ConditionalLeftSemiJoinTest, TestTwoColumnThreeRowSomeEqual)
{
  this->test({{0, 1, 2}, {10, 20, 30}}, {{0, 1, 3}, {30, 40, 50}}, left_zero_eq_right_zero, {0, 1});
};

TYPED_TEST(ConditionalLeftSemiJoinTest, TestCompareRandomToHash)
{
  // Generate columns of 10 repeats of the integer range [0, 10), then merge
  // a shuffled version and compare to hash join.
  unsigned int N           = 10000;
  unsigned int num_repeats = 10;
  unsigned int num_unique  = N / num_repeats;

  std::vector<TypeParam> left(N);
  std::vector<TypeParam> right(N);

  for (unsigned int i = 0; i < num_repeats; ++i) {
    std::iota(
      std::next(left.begin(), num_unique * i), std::next(left.begin(), num_unique * (i + 1)), 0);
    std::iota(
      std::next(right.begin(), num_unique * i), std::next(right.begin(), num_unique * (i + 1)), 0);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(left.begin(), left.end(), gen);
  std::shuffle(right.begin(), right.end(), gen);

  this->compare_to_hash_join({left}, {right});
};

/**
 * Tests of left anti joins.
 */
template <typename T>
struct ConditionalLeftAntiJoinTest : public ConditionalJoinSingleReturnTest<T> {
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> join(
    cudf::table_view left, cudf::table_view right, cudf::ast::expression predicate) override
  {
    return cudf::conditional_left_anti_join(left, right, predicate);
  }

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> reference_join(
    cudf::table_view left, cudf::table_view right) override
  {
    return cudf::left_anti_join(left, right);
  }
};

TYPED_TEST_CASE(ConditionalLeftAntiJoinTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(ConditionalLeftAntiJoinTest, TestTwoColumnThreeRowSomeEqual)
{
  this->test({{0, 1, 2}, {10, 20, 30}}, {{0, 1, 3}, {30, 40, 50}}, left_zero_eq_right_zero, {2});
};

TYPED_TEST(ConditionalLeftAntiJoinTest, TestCompareRandomToHash)
{
  // Generate columns of 10 repeats of the integer range [0, 10), then merge
  // a shuffled version and compare to hash join.
  unsigned int N           = 10000;
  unsigned int num_repeats = 10;
  unsigned int num_unique  = N / num_repeats;

  std::vector<TypeParam> left(N);
  std::vector<TypeParam> right(N);

  for (unsigned int i = 0; i < num_repeats; ++i) {
    std::iota(
      std::next(left.begin(), num_unique * i), std::next(left.begin(), num_unique * (i + 1)), 0);
    std::iota(
      std::next(right.begin(), num_unique * i), std::next(right.begin(), num_unique * (i + 1)), 0);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(left.begin(), left.end(), gen);
  std::shuffle(right.begin(), right.end(), gen);

  this->compare_to_hash_join({left}, {right});
};
