/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <cudf_test/type_lists.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>
#include <thrust/sort.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using PairJoinReturn   = std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                                 std::unique_ptr<rmm::device_uvector<cudf::size_type>>>;
using SingleJoinReturn = std::unique_ptr<rmm::device_uvector<cudf::size_type>>;
using NullMaskVector   = std::vector<bool>;

template <typename T>
using ColumnVector = std::vector<std::vector<T>>;

template <typename T>
using NullableColumnVector = std::vector<std::pair<std::vector<T>, NullMaskVector>>;

constexpr cudf::size_type JoinNoneValue =
  std::numeric_limits<cudf::size_type>::min();  // TODO: how to test if this isn't public?

// Common column references.
auto const col_ref_left_0  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
auto const col_ref_right_0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);

// Common expressions.
auto left_zero_eq_right_zero =
  cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_left_0, col_ref_right_0);

// Generate a single pair of left/right non-nullable columns of random data
// suitable for testing a join against a reference join implementation.
template <typename T>
std::pair<std::vector<T>, std::vector<T>> gen_random_repeated_columns(
  unsigned int N_left            = 10000,
  unsigned int num_repeats_left  = 10,
  unsigned int N_right           = 10000,
  unsigned int num_repeats_right = 10)
{
  // Generate columns of num_repeats repeats of the integer range [0, num_unique),
  // then merge a shuffled version and compare to hash join.
  unsigned int num_unique_left  = N_left / num_repeats_left;
  unsigned int num_unique_right = N_right / num_repeats_right;

  std::vector<T> left(N_left);
  std::vector<T> right(N_right);

  for (unsigned int i = 0; i < num_repeats_left; ++i) {
    std::iota(std::next(left.begin(), num_unique_left * i),
              std::next(left.begin(), num_unique_left * (i + 1)),
              0);
  }
  for (unsigned int i = 0; i < num_repeats_right; ++i) {
    std::iota(std::next(right.begin(), num_unique_right * i),
              std::next(right.begin(), num_unique_right * (i + 1)),
              0);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(left.begin(), left.end(), gen);
  std::shuffle(right.begin(), right.end(), gen);
  return std::pair(std::move(left), std::move(right));
}

// Generate a single pair of left/right nullable columns of random data
// suitable for testing a join against a reference join implementation.
template <typename T>
std::pair<std::pair<std::vector<T>, std::vector<bool>>,
          std::pair<std::vector<T>, std::vector<bool>>>
gen_random_nullable_repeated_columns(unsigned int N = 10000, unsigned int num_repeats = 10)
{
  auto [left, right] = gen_random_repeated_columns<T>(N, num_repeats);

  std::vector<bool> left_nulls(N);
  std::vector<bool> right_nulls(N);

  // Seed with a real random value, if available
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> uniform_dist(0, 1);

  std::generate(left_nulls.begin(), left_nulls.end(), [&uniform_dist, &gen]() {
    return uniform_dist(gen) > 0.5;
  });
  std::generate(right_nulls.begin(), right_nulls.end(), [&uniform_dist, &gen]() {
    return uniform_dist(gen) > 0.5;
  });

  return std::pair(std::pair(std::move(left), std::move(left_nulls)),
                   std::pair(std::move(right), std::move(right_nulls)));
}

}  // namespace

/**
 * Fixture for all mixed hash + conditional joins.
 */
template <typename T>
struct MixedJoinTest : public cudf::test::BaseFixture {
  /**
   * Convenience utility for parsing initializer lists of input data into
   * suitable inputs for tables.
   */
  template <typename U>
  std::tuple<std::vector<cudf::test::fixed_width_column_wrapper<T>>,
             std::vector<cudf::test::fixed_width_column_wrapper<T>>,
             std::vector<cudf::column_view>,
             std::vector<cudf::column_view>,
             cudf::table_view,
             cudf::table_view,
             cudf::table_view,
             cudf::table_view>
  parse_input(std::vector<U> left_data,
              std::vector<U> right_data,
              std::vector<cudf::size_type> equality_columns,
              std::vector<cudf::size_type> conditional_columns)
  {
    auto wrapper_generator = [](U& v) {
      if constexpr (std::is_same_v<U, std::vector<T>>) {
        return cudf::test::fixed_width_column_wrapper<T>(v.begin(), v.end());
      } else if constexpr (std::is_same_v<U, std::pair<std::vector<T>, std::vector<bool>>>) {
        return cudf::test::fixed_width_column_wrapper<T>(
          v.first.begin(), v.first.end(), v.second.begin());
      }
      throw std::runtime_error("Invalid input to parse_input.");
      return cudf::test::fixed_width_column_wrapper<T>();
    };

    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    std::vector<cudf::test::fixed_width_column_wrapper<T>> left_wrappers;
    std::vector<cudf::column_view> left_columns;
    for (auto v : left_data) {
      left_wrappers.push_back(wrapper_generator(v));
      left_columns.push_back(left_wrappers.back());
    }

    std::vector<cudf::test::fixed_width_column_wrapper<T>> right_wrappers;
    std::vector<cudf::column_view> right_columns;
    for (auto v : right_data) {
      right_wrappers.push_back(wrapper_generator(v));
      right_columns.push_back(right_wrappers.back());
    }

    auto left  = cudf::table_view(left_columns);
    auto right = cudf::table_view(right_columns);

    return std::make_tuple(std::move(left_wrappers),
                           std::move(right_wrappers),
                           std::move(left_columns),
                           std::move(right_columns),
                           left.select(equality_columns),
                           right.select(equality_columns),
                           left.select(conditional_columns),
                           right.select(conditional_columns));
  }
};

/**
 * Fixture for join types that return both left and right indices (inner, left,
 * and full joins).
 */
template <typename T>
struct MixedJoinPairReturnTest : public MixedJoinTest<T> {
  /*
   * Perform a join of tables constructed from two input data sets according to
   * verify that the outputs match the expected outputs (up to order).
   */
  virtual void _test(cudf::table_view left_equality,
                     cudf::table_view right_equality,
                     cudf::table_view left_conditional,
                     cudf::table_view right_conditional,
                     cudf::ast::operation predicate,
                     std::vector<cudf::size_type> expected_counts,
                     std::vector<std::pair<cudf::size_type, cudf::size_type>> expected_outputs,
                     cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
  {
    auto [result_size, actual_counts] = this->join_size(
      left_equality, right_equality, left_conditional, right_conditional, predicate, compare_nulls);
    EXPECT_TRUE(result_size == expected_outputs.size());

    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_counts_cw(
      expected_counts.begin(), expected_counts.end());
    auto const actual_counts_view =
      cudf::column_view(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                        actual_counts->size(),
                        actual_counts->data(),
                        nullptr,
                        0);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_counts_cw, actual_counts_view);

    auto result = this->join(
      left_equality, right_equality, left_conditional, right_conditional, predicate, compare_nulls);
    std::vector<std::pair<cudf::size_type, cudf::size_type>> result_pairs;
    for (size_t i = 0; i < result.first->size(); ++i) {
      // Note: Not trying to be terribly efficient here since these tests are
      // small, otherwise a batch copy to host before constructing the tuples
      // would be important.
      result_pairs.push_back({result.first->element(i, cudf::get_default_stream()),
                              result.second->element(i, cudf::get_default_stream())});
    }
    std::sort(result_pairs.begin(), result_pairs.end());
    std::sort(expected_outputs.begin(), expected_outputs.end());

    EXPECT_TRUE(std::equal(expected_outputs.begin(), expected_outputs.end(), result_pairs.begin()));
  }

  /*
   * Perform a join of tables constructed from two input data sets according to
   * the provided predicate and verify that the outputs match the expected
   * outputs (up to order).
   */
  void test(ColumnVector<T> left_data,
            ColumnVector<T> right_data,
            std::vector<cudf::size_type> equality_columns,
            std::vector<cudf::size_type> conditional_columns,
            cudf::ast::operation predicate,
            std::vector<cudf::size_type> expected_counts,
            std::vector<std::pair<cudf::size_type, cudf::size_type>> expected_outputs)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    auto [left_wrappers,
          right_wrappers,
          left_columns,
          right_columns,
          left_equality,
          right_equality,
          left_conditional,
          right_conditional] =
      this->parse_input(left_data, right_data, equality_columns, conditional_columns);
    this->_test(left_equality,
                right_equality,
                left_conditional,
                right_conditional,
                predicate,
                expected_counts,
                expected_outputs);
  }

  /*
   * Perform a join of tables constructed from two input data sets according to
   * the provided predicate and verify that the outputs match the expected
   * outputs (up to order).
   */
  void test_nulls(NullableColumnVector<T> left_data,
                  NullableColumnVector<T> right_data,
                  std::vector<cudf::size_type> equality_columns,
                  std::vector<cudf::size_type> conditional_columns,
                  cudf::ast::operation predicate,
                  std::vector<cudf::size_type> expected_counts,
                  std::vector<std::pair<cudf::size_type, cudf::size_type>> expected_outputs,
                  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    auto [left_wrappers,
          right_wrappers,
          left_columns,
          right_columns,
          left_equality,
          right_equality,
          left_conditional,
          right_conditional] =
      this->parse_input(left_data, right_data, equality_columns, conditional_columns);
    this->_test(left_equality,
                right_equality,
                left_conditional,
                right_conditional,
                predicate,
                expected_counts,
                expected_outputs,
                compare_nulls);
  }

  /**
   * This method must be implemented by subclasses for specific types of joins.
   * It should be a simply forwarding of arguments to the appropriate cudf
   * mixed join API.
   */
  virtual PairJoinReturn join(cudf::table_view left_equality,
                              cudf::table_view right_equality,
                              cudf::table_view left_conditional,
                              cudf::table_view right_conditional,
                              cudf::ast::operation predicate,
                              cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) = 0;

  /**
   * This method must be implemented by subclasses for specific types of joins.
   * It should be a simply forwarding of arguments to the appropriate cudf
   * mixed join size computation API.
   */
  virtual std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<cudf::size_type>>> join_size(
    cudf::table_view left_equality,
    cudf::table_view right_equality,
    cudf::table_view left_conditional,
    cudf::table_view right_conditional,
    cudf::ast::operation predicate,
    cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) = 0;
};

/**
 * Tests of mixed inner joins.
 */
template <typename T>
struct MixedInnerJoinTest : public MixedJoinPairReturnTest<T> {
  PairJoinReturn join(cudf::table_view left_equality,
                      cudf::table_view right_equality,
                      cudf::table_view left_conditional,
                      cudf::table_view right_conditional,
                      cudf::ast::operation predicate,
                      cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) override
  {
    return cudf::mixed_inner_join(
      left_equality, right_equality, left_conditional, right_conditional, predicate, compare_nulls);
  }

  std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<cudf::size_type>>> join_size(
    cudf::table_view left_equality,
    cudf::table_view right_equality,
    cudf::table_view left_conditional,
    cudf::table_view right_conditional,
    cudf::ast::operation predicate,
    cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) override
  {
    return cudf::mixed_inner_join_size(
      left_equality, right_equality, left_conditional, right_conditional, predicate, compare_nulls);
  }
};

TYPED_TEST_SUITE(MixedInnerJoinTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(MixedInnerJoinTest, Empty)
{
  this->test({}, {}, {}, {}, left_zero_eq_right_zero, {}, {});
}

TYPED_TEST(MixedInnerJoinTest, BasicEquality)
{
  this->test({{0, 1, 2}, {3, 4, 5}, {10, 20, 30}},
             {{0, 1, 3}, {5, 4, 5}, {30, 40, 50}},
             {0},
             {1, 2},
             left_zero_eq_right_zero,
             {0, 1, 0},
             {{1, 1}});
}

TYPED_TEST(MixedInnerJoinTest, BasicNullEqualityEqual)
{
  this->test_nulls({{{0, 1, 2}, {1, 1, 0}}, {{3, 4, 5}, {1, 1, 1}}, {{10, 20, 30}, {1, 1, 1}}},
                   {{{0, 1, 3}, {1, 1, 0}}, {{5, 4, 5}, {1, 1, 1}}, {{30, 40, 30}, {1, 1, 1}}},
                   {0},
                   {1, 2},
                   left_zero_eq_right_zero,
                   {0, 1, 1},
                   {{1, 1}, {2, 2}},
                   cudf::null_equality::EQUAL);
};

TYPED_TEST(MixedInnerJoinTest, BasicNullEqualityUnequal)
{
  this->test_nulls({{{0, 1, 2}, {1, 1, 0}}, {{3, 4, 5}, {1, 1, 1}}, {{10, 20, 30}, {1, 1, 1}}},
                   {{{0, 1, 3}, {1, 1, 0}}, {{5, 4, 5}, {1, 1, 1}}, {{30, 40, 30}, {1, 1, 1}}},
                   {0},
                   {1, 2},
                   left_zero_eq_right_zero,
                   {0, 1, 0},
                   {{1, 1}},
                   cudf::null_equality::UNEQUAL);
};

TYPED_TEST(MixedInnerJoinTest, AsymmetricEquality)
{
  this->test({{0, 2, 1}, {3, 5, 4}, {10, 30, 20}},
             {{0, 1, 3}, {5, 4, 5}, {30, 40, 50}},
             {0},
             {1, 2},
             left_zero_eq_right_zero,
             {0, 0, 1},
             {{2, 1}});
}

TYPED_TEST(MixedInnerJoinTest, AsymmetricLeftLargerEquality)
{
  this->test({{0, 2, 1, 4}, {3, 5, 4, 10}, {10, 30, 20, 100}},
             {{0, 1, 3}, {5, 4, 5}, {30, 40, 50}},
             {0},
             {1, 2},
             left_zero_eq_right_zero,
             {0, 0, 1, 0},
             {{2, 1}});
}

TYPED_TEST(MixedInnerJoinTest, AsymmetricLeftLargerGreater)
{
  auto col_ref_left_1  = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
  auto col_ref_right_1 = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto condition =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_left_1, col_ref_right_1);
  this->test({{2, 3, 9, 0, 1, 7, 4, 6, 5, 8}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0}},
             {{6, 5, 9, 8, 10, 32}, {0, 1, 2, 3, 4, 5}, {7, 8, 9, 0, 1, 2}},
             {0},
             {0, 1},
             condition,
             {0, 0, 1, 0, 0, 0, 0, 1, 1, 0},
             {{2, 2}, {7, 0}, {8, 1}});
}

TYPED_TEST(MixedInnerJoinTest, AsymmetricRightLargerEquality)
{
  this->test({{0, 1, 3}, {5, 4, 5}, {30, 40, 50}},
             {{0, 2, 1, 4}, {3, 5, 4, 10}, {10, 30, 20, 100}},
             {0},
             {1, 2},
             left_zero_eq_right_zero,
             {0, 0, 1, 0},
             {{1, 2}});
}

TYPED_TEST(MixedInnerJoinTest, BasicInequality)
{
  auto const col_ref_left_1  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_1 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto const col_ref_left_2  = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_2 = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);

  auto scalar_1        = cudf::numeric_scalar<TypeParam>(35);
  auto const literal_1 = cudf::ast::literal(scalar_1);

  auto const op1 =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_left_1, col_ref_right_1);
  auto const op2 = cudf::ast::operation(cudf::ast::ast_operator::LESS, literal_1, col_ref_right_2);

  auto const predicate = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, op1, op2);

  this->test({{0, 1, 2, 4}, {3, 4, 5, 6}, {10, 20, 30, 40}},
             {{0, 1, 3, 4}, {5, 4, 5, 7}, {30, 40, 50, 60}},
             {0},
             {1, 2},
             predicate,
             {0, 0, 0, 1},
             {{3, 3}});
}

/**
 * Tests of mixed left joins.
 */
template <typename T>
struct MixedLeftJoinTest : public MixedJoinPairReturnTest<T> {
  PairJoinReturn join(cudf::table_view left_equality,
                      cudf::table_view right_equality,
                      cudf::table_view left_conditional,
                      cudf::table_view right_conditional,
                      cudf::ast::operation predicate,
                      cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) override
  {
    return cudf::mixed_left_join(
      left_equality, right_equality, left_conditional, right_conditional, predicate, compare_nulls);
  }

  std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<cudf::size_type>>> join_size(
    cudf::table_view left_equality,
    cudf::table_view right_equality,
    cudf::table_view left_conditional,
    cudf::table_view right_conditional,
    cudf::ast::operation predicate,
    cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) override
  {
    return cudf::mixed_left_join_size(
      left_equality, right_equality, left_conditional, right_conditional, predicate, compare_nulls);
  }
};

TYPED_TEST_SUITE(MixedLeftJoinTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(MixedLeftJoinTest, Basic)
{
  this->test({{0, 1, 2}, {3, 4, 5}, {10, 20, 30}},
             {{0, 1, 3}, {5, 4, 5}, {30, 40, 50}},
             {0},
             {1, 2},
             left_zero_eq_right_zero,
             {1, 1, 1},
             {{0, JoinNoneValue}, {1, 1}, {2, JoinNoneValue}});
}

TYPED_TEST(MixedLeftJoinTest, Basic2)
{
  auto const col_ref_left_1  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_1 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto const col_ref_left_2  = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_2 = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);

  auto scalar_1        = cudf::numeric_scalar<TypeParam>(35);
  auto const literal_1 = cudf::ast::literal(scalar_1);

  auto const op1 =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_left_1, col_ref_right_1);
  auto const op2 = cudf::ast::operation(cudf::ast::ast_operator::LESS, literal_1, col_ref_right_2);

  auto const predicate = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, op1, op2);

  this->test({{0, 1, 2, 4}, {3, 4, 5, 6}, {10, 20, 30, 40}},
             {{0, 1, 3, 4}, {5, 4, 5, 7}, {30, 40, 50, 60}},
             {0},
             {1, 2},
             predicate,
             {1, 1, 1, 1},
             {{0, JoinNoneValue}, {1, JoinNoneValue}, {2, JoinNoneValue}, {3, 3}});
}

/**
 * Tests of mixed full joins.
 */
template <typename T>
struct MixedFullJoinTest : public MixedJoinPairReturnTest<T> {
  PairJoinReturn join(cudf::table_view left_equality,
                      cudf::table_view right_equality,
                      cudf::table_view left_conditional,
                      cudf::table_view right_conditional,
                      cudf::ast::operation predicate,
                      cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) override
  {
    return cudf::mixed_full_join(
      left_equality, right_equality, left_conditional, right_conditional, predicate, compare_nulls);
  }

  std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<cudf::size_type>>> join_size(
    cudf::table_view left_equality,
    cudf::table_view right_equality,
    cudf::table_view left_conditional,
    cudf::table_view right_conditional,
    cudf::ast::operation predicate,
    cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) override
  {
    // Full joins don't actually support size calculations, and there's no easy way to spoof it.
    CUDF_FAIL("Size calculation not supported for full joins.");
  }

  /*
   * Override method to remove size calculation testing since it's not possible for full joins.
   */
  void _test(cudf::table_view left_equality,
             cudf::table_view right_equality,
             cudf::table_view left_conditional,
             cudf::table_view right_conditional,
             cudf::ast::operation predicate,
             std::vector<cudf::size_type> expected_counts,
             std::vector<std::pair<cudf::size_type, cudf::size_type>> expected_outputs,
             cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) override
  {
    auto result = this->join(
      left_equality, right_equality, left_conditional, right_conditional, predicate, compare_nulls);
    std::vector<std::pair<cudf::size_type, cudf::size_type>> result_pairs;
    for (size_t i = 0; i < result.first->size(); ++i) {
      result_pairs.push_back({result.first->element(i, cudf::get_default_stream()),
                              result.second->element(i, cudf::get_default_stream())});
    }
    std::sort(result_pairs.begin(), result_pairs.end());
    std::sort(expected_outputs.begin(), expected_outputs.end());

    EXPECT_TRUE(std::equal(expected_outputs.begin(), expected_outputs.end(), result_pairs.begin()));
  }
};

TYPED_TEST_SUITE(MixedFullJoinTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(MixedFullJoinTest, Basic)
{
  this->test(
    {{0, 1, 2}, {3, 4, 5}, {10, 20, 30}},
    {{0, 1, 3}, {5, 4, 5}, {30, 40, 50}},
    {0},
    {1, 2},
    left_zero_eq_right_zero,
    {1, 1, 1},
    {{0, JoinNoneValue}, {1, 1}, {2, JoinNoneValue}, {JoinNoneValue, 0}, {JoinNoneValue, 2}});
}

TYPED_TEST(MixedFullJoinTest, Basic2)
{
  auto const col_ref_left_1  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_1 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto const col_ref_left_2  = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_2 = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);

  auto scalar_1        = cudf::numeric_scalar<TypeParam>(35);
  auto const literal_1 = cudf::ast::literal(scalar_1);

  auto const op1 =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_left_1, col_ref_right_1);
  auto const op2 = cudf::ast::operation(cudf::ast::ast_operator::LESS, literal_1, col_ref_right_2);

  auto const predicate = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, op1, op2);

  this->test({{0, 1, 2, 4}, {3, 4, 5, 6}, {10, 20, 30, 40}},
             {{0, 1, 3, 4}, {5, 4, 5, 7}, {30, 40, 50, 60}},
             {0},
             {1, 2},
             predicate,
             {1, 1, 1, 1},
             {{0, JoinNoneValue},
              {1, JoinNoneValue},
              {2, JoinNoneValue},
              {3, 3},
              {JoinNoneValue, 0},
              {JoinNoneValue, 1},
              {JoinNoneValue, 2}});
}

template <typename T>
struct MixedJoinSingleReturnTest : public MixedJoinTest<T> {
  /*
   * Perform a join of tables constructed from two input data sets according to
   * verify that the outputs match the expected outputs (up to order).
   */
  virtual void _test(cudf::table_view left_equality,
                     cudf::table_view right_equality,
                     cudf::table_view left_conditional,
                     cudf::table_view right_conditional,
                     cudf::ast::operation predicate,
                     std::vector<cudf::size_type> expected_outputs,
                     cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
  {
    auto result = this->join(
      left_equality, right_equality, left_conditional, right_conditional, predicate, compare_nulls);
    std::vector<cudf::size_type> resulting_indices;
    for (size_t i = 0; i < result->size(); ++i) {
      // Note: Not trying to be terribly efficient here since these tests are
      // small, otherwise a batch copy to host before constructing the tuples
      // would be important.
      resulting_indices.push_back(result->element(i, cudf::get_default_stream()));
    }
    std::sort(resulting_indices.begin(), resulting_indices.end());
    std::sort(expected_outputs.begin(), expected_outputs.end());
    EXPECT_TRUE(
      std::equal(resulting_indices.begin(), resulting_indices.end(), expected_outputs.begin()));
  }

  /*
   * Perform a join of tables constructed from two input data sets according to
   * the provided predicate and verify that the outputs match the expected
   * outputs (up to order).
   */
  void test(ColumnVector<T> left_data,
            ColumnVector<T> right_data,
            std::vector<cudf::size_type> equality_columns,
            std::vector<cudf::size_type> conditional_columns,
            cudf::ast::operation predicate,
            std::vector<cudf::size_type> expected_outputs)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    auto [left_wrappers,
          right_wrappers,
          left_columns,
          right_columns,
          left_equality,
          right_equality,
          left_conditional,
          right_conditional] =
      this->parse_input(left_data, right_data, equality_columns, conditional_columns);
    this->_test(left_equality,
                right_equality,
                left_conditional,
                right_conditional,
                predicate,
                expected_outputs);
  }

  /*
   * Perform a join of tables constructed from two input data sets according to
   * the provided predicate and verify that the outputs match the expected
   * outputs (up to order).
   */
  void test_nulls(NullableColumnVector<T> left_data,
                  NullableColumnVector<T> right_data,
                  std::vector<cudf::size_type> equality_columns,
                  std::vector<cudf::size_type> conditional_columns,
                  cudf::ast::operation predicate,
                  std::vector<cudf::size_type> expected_outputs,
                  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    auto [left_wrappers,
          right_wrappers,
          left_columns,
          right_columns,
          left_equality,
          right_equality,
          left_conditional,
          right_conditional] =
      this->parse_input(left_data, right_data, equality_columns, conditional_columns);
    this->_test(left_equality,
                right_equality,
                left_conditional,
                right_conditional,
                predicate,
                expected_outputs,
                compare_nulls);
  }

  /**
   * This method must be implemented by subclasses for specific types of joins.
   * It should be a simply forwarding of arguments to the appropriate cudf
   * mixed join API.
   */
  virtual SingleJoinReturn join(cudf::table_view left_equality,
                                cudf::table_view right_equality,
                                cudf::table_view left_conditional,
                                cudf::table_view right_conditional,
                                cudf::ast::operation predicate,
                                cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) = 0;
};

/**
 * Tests of mixed left semi joins.
 */
template <typename T>
struct MixedLeftSemiJoinTest : public MixedJoinSingleReturnTest<T> {
  SingleJoinReturn join(cudf::table_view left_equality,
                        cudf::table_view right_equality,
                        cudf::table_view left_conditional,
                        cudf::table_view right_conditional,
                        cudf::ast::operation predicate,
                        cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) override
  {
    return cudf::mixed_left_semi_join(
      left_equality, right_equality, left_conditional, right_conditional, predicate, compare_nulls);
  }
};

TYPED_TEST_SUITE(MixedLeftSemiJoinTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(MixedLeftSemiJoinTest, BasicEquality)
{
  this->test({{0, 1, 2}, {3, 4, 5}, {10, 20, 30}},
             {{0, 1, 3}, {5, 4, 5}, {30, 40, 50}},
             {0},
             {1, 2},
             left_zero_eq_right_zero,
             {1});
}

TYPED_TEST(MixedLeftSemiJoinTest, BasicEqualityDuplicates)
{
  this->test({{0, 1, 2, 1}, {3, 4, 5, 6}, {10, 20, 30, 40}},
             {{0, 1, 3, 1}, {5, 4, 5, 6}, {30, 40, 50, 40}},
             {0},
             {1, 2},
             left_zero_eq_right_zero,
             {1, 3});
}

TYPED_TEST(MixedLeftSemiJoinTest, BasicNullEqualityEqual)
{
  this->test_nulls({{{0, 1, 2}, {1, 1, 0}}, {{3, 4, 5}, {1, 1, 1}}, {{10, 20, 30}, {1, 1, 1}}},
                   {{{0, 1, 3}, {1, 1, 0}}, {{5, 4, 5}, {1, 1, 1}}, {{30, 40, 30}, {1, 1, 1}}},
                   {0},
                   {1, 2},
                   left_zero_eq_right_zero,
                   {1, 2},
                   cudf::null_equality::EQUAL);
};

TYPED_TEST(MixedLeftSemiJoinTest, BasicNullEqualityUnequal)
{
  this->test_nulls({{{0, 1, 2}, {1, 1, 0}}, {{3, 4, 5}, {1, 1, 1}}, {{10, 20, 30}, {1, 1, 1}}},
                   {{{0, 1, 3}, {1, 1, 0}}, {{5, 4, 5}, {1, 1, 1}}, {{30, 40, 30}, {1, 1, 1}}},
                   {0},
                   {1, 2},
                   left_zero_eq_right_zero,
                   {1},
                   cudf::null_equality::UNEQUAL);
};

TYPED_TEST(MixedLeftSemiJoinTest, AsymmetricEquality)
{
  this->test({{0, 2, 1}, {3, 5, 4}, {10, 30, 20}},
             {{0, 1, 3}, {5, 4, 5}, {30, 40, 50}},
             {0},
             {1, 2},
             left_zero_eq_right_zero,
             {2});
}

TYPED_TEST(MixedLeftSemiJoinTest, AsymmetricLeftLargerEquality)
{
  this->test({{0, 2, 1, 4}, {3, 5, 4, 10}, {10, 30, 20, 100}},
             {{0, 1, 3}, {5, 4, 5}, {30, 40, 50}},
             {0},
             {1, 2},
             left_zero_eq_right_zero,
             {2});
}

/**
 * Tests of mixed left semi joins.
 */
template <typename T>
struct MixedLeftAntiJoinTest : public MixedJoinSingleReturnTest<T> {
  SingleJoinReturn join(cudf::table_view left_equality,
                        cudf::table_view right_equality,
                        cudf::table_view left_conditional,
                        cudf::table_view right_conditional,
                        cudf::ast::operation predicate,
                        cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) override
  {
    return cudf::mixed_left_anti_join(
      left_equality, right_equality, left_conditional, right_conditional, predicate, compare_nulls);
  }
};

TYPED_TEST_SUITE(MixedLeftAntiJoinTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(MixedLeftAntiJoinTest, BasicEquality)
{
  this->test({{0, 1, 2}, {3, 4, 5}, {10, 20, 30}},
             {{0, 1, 3}, {5, 4, 5}, {30, 40, 50}},
             {0},
             {1, 2},
             left_zero_eq_right_zero,
             {0, 2});
}

TYPED_TEST(MixedLeftAntiJoinTest, BasicNullEqualityEqual)
{
  this->test_nulls({{{0, 1, 2}, {1, 1, 0}}, {{3, 4, 5}, {1, 1, 1}}, {{10, 20, 30}, {1, 1, 1}}},
                   {{{0, 1, 3}, {1, 1, 0}}, {{5, 4, 5}, {1, 1, 1}}, {{30, 40, 30}, {1, 1, 1}}},
                   {0},
                   {1, 2},
                   left_zero_eq_right_zero,
                   {0},
                   cudf::null_equality::EQUAL);
};

TYPED_TEST(MixedLeftAntiJoinTest, BasicNullEqualityUnequal)
{
  this->test_nulls({{{0, 1, 2}, {1, 1, 0}}, {{3, 4, 5}, {1, 1, 1}}, {{10, 20, 30}, {1, 1, 1}}},
                   {{{0, 1, 3}, {1, 1, 0}}, {{5, 4, 5}, {1, 1, 1}}, {{30, 40, 30}, {1, 1, 1}}},
                   {0},
                   {1, 2},
                   left_zero_eq_right_zero,
                   {0, 2},
                   cudf::null_equality::UNEQUAL);
};

TYPED_TEST(MixedLeftAntiJoinTest, AsymmetricEquality)
{
  this->test({{0, 2, 1}, {3, 5, 4}, {10, 30, 20}},
             {{0, 1, 3}, {5, 4, 5}, {30, 40, 50}},
             {0},
             {1, 2},
             left_zero_eq_right_zero,
             {0, 1});
}

TYPED_TEST(MixedLeftAntiJoinTest, AsymmetricLeftLargerEquality)
{
  this->test({{0, 2, 1, 4}, {3, 5, 4, 10}, {10, 30, 20, 100}},
             {{0, 1, 3}, {5, 4, 5}, {30, 40, 50}},
             {0},
             {1, 2},
             left_zero_eq_right_zero,
             {0, 1, 3});
}
