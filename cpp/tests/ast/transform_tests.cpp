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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <limits>
#include <list>
#include <random>
#include <vector>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

struct TransformTest : public cudf::test::BaseFixture {};

TEST_F(TransformTest, ColumnReference)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  auto const& expected = c_0;
  auto result          = cudf::compute_column(table, col_ref_0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, BasicAdditionDoubleCast)
{
  auto c_0 = column_wrapper<double>{3, 20, 1, 50};
  std::vector<__int128_t> data1{10, 7, 20, 0};
  auto c_1 = cudf::test::fixed_point_column_wrapper<__int128_t>(
    data1.begin(), data1.end(), numeric::scale_type{0});
  auto table      = cudf::table_view{{c_0, c_1}};
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto cast       = cudf::ast::operation(cudf::ast::ast_operator::CAST_TO_FLOAT64, col_ref_1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, cast);
  auto expected   = column_wrapper<double>{13, 27, 21, 50};
  auto result     = cudf::compute_column(table, expression);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, Literal)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto literal_value = cudf::numeric_scalar<int32_t>(42);
  auto literal       = cudf::ast::literal(literal_value);

  auto expected = column_wrapper<int32_t>{42, 42, 42, 42};
  auto result   = cudf::compute_column(table, literal);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, NullLiteral)
{
  auto c_0   = column_wrapper<int32_t>{0, 0, 0, 0};
  auto table = cudf::table_view{{c_0}};

  auto literal_value = cudf::numeric_scalar<int32_t>(-123);
  literal_value.set_valid_async(false);
  auto literal = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, literal);

  auto result   = cudf::compute_column(table, expression);
  auto expected = column_wrapper<int32_t>({-123, -123, -123, -123}, {0, 0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, IsNull)
{
  auto c_0   = column_wrapper<int32_t>{{0, 1, 2, 0}, {0, 1, 1, 0}};
  auto table = cudf::table_view{{c_0}};

  // result of IS_NULL on literal, will be a column of table size, with all values set to
  // !literal.is_valid(). The table values are irrelevant.
  auto literal_value = cudf::numeric_scalar<int32_t>(-123);
  auto literal       = cudf::ast::literal(literal_value);
  auto expression    = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, literal);

  auto result    = cudf::compute_column(table, expression);
  auto expected1 = column_wrapper<bool>({0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, result->view(), verbosity);

  literal_value.set_valid_async(false);
  result         = cudf::compute_column(table, expression);
  auto expected2 = column_wrapper<bool>({1, 1, 1, 1}, cudf::test::iterators::no_nulls());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, result->view(), verbosity);

  auto col_ref_0   = cudf::ast::column_reference(0);
  auto expression2 = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, col_ref_0);
  result           = cudf::compute_column(table, expression2);
  auto expected3   = column_wrapper<bool>({1, 0, 0, 1}, cudf::test::iterators::no_nulls());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, result->view(), verbosity);
}

TEST_F(TransformTest, BasicAddition)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expected = column_wrapper<int32_t>{13, 27, 21, 50};
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, BasicAdditionEmptyTable)
{
  auto c_0   = column_wrapper<int32_t>{};
  auto c_1   = column_wrapper<int32_t>{};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expected = column_wrapper<int32_t>{};
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, BasicAdditionCast)
{
  auto c_0   = column_wrapper<int64_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int8_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto cast       = cudf::ast::operation(cudf::ast::ast_operator::CAST_TO_INT64, col_ref_1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, cast);

  auto expected = column_wrapper<int64_t>{13, 27, 21, 50};
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, BasicEquality)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{3, 7, 1, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{true, false, true, false};
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, BasicAdditionLarge)
{
  auto a     = thrust::make_counting_iterator(0);
  auto col   = column_wrapper<int32_t>(a, a + 2000);
  auto table = cudf::table_view{{col, col}};

  auto col_ref    = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref, col_ref);

  auto b        = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  auto expected = column_wrapper<int32_t>(b, b + 2000);
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, LessComparator)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{true, false, true, false};
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, LessComparatorLarge)
{
  auto a     = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  auto b     = thrust::make_counting_iterator(500);
  auto c_0   = column_wrapper<int32_t>(a, a + 2000);
  auto c_1   = column_wrapper<int32_t>(b, b + 2000);
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_1);

  auto c        = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i < 500; });
  auto expected = column_wrapper<bool>(c, c + 2000);
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, MultiLevelTreeArithmetic)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto c_2   = column_wrapper<int32_t>{-3, 66, 2, -99};
  auto table = cudf::table_view{{c_0, c_1, c_2}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto col_ref_2 = cudf::ast::column_reference(2);

  auto expression_left_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expression_right_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::SUB, col_ref_2, col_ref_0);

  auto expression_tree = cudf::ast::operation(
    cudf::ast::ast_operator::ADD, expression_left_subtree, expression_right_subtree);

  auto result   = cudf::compute_column(table, expression_tree);
  auto expected = column_wrapper<int32_t>{7, 73, 22, -99};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, MultiLevelTreeArithmeticLarge)
{
  auto a     = thrust::make_counting_iterator(0);
  auto b     = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i + 1; });
  auto c     = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  auto c_0   = column_wrapper<int32_t>(a, a + 2000);
  auto c_1   = column_wrapper<int32_t>(b, b + 2000);
  auto c_2   = column_wrapper<int32_t>(c, c + 2000);
  auto table = cudf::table_view{{c_0, c_1, c_2}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto col_ref_2 = cudf::ast::column_reference(2);

  auto expr_left_subtree = cudf::ast::operation(cudf::ast::ast_operator::MUL, col_ref_0, col_ref_1);
  auto expr_right_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_2, col_ref_0);
  auto expr_tree =
    cudf::ast::operation(cudf::ast::ast_operator::SUB, expr_left_subtree, expr_right_subtree);

  auto result = cudf::compute_column(table, expr_tree);
  auto calc   = [](auto i) { return (i * (i + 1)) - (i + (i * 2)); };
  auto d      = cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return calc(i); });
  auto expected = column_wrapper<int32_t>(d, d + 2000);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, ImbalancedTreeArithmetic)
{
  auto c_0   = column_wrapper<double>{0.15, 0.37, 4.2, 21.3};
  auto c_1   = column_wrapper<double>{0.0, -42.0, 1.0, 98.6};
  auto c_2   = column_wrapper<double>{0.6, std::numeric_limits<double>::infinity(), 0.999, 1.0};
  auto table = cudf::table_view{{c_0, c_1, c_2}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto col_ref_2 = cudf::ast::column_reference(2);

  auto expression_right_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::MUL, col_ref_0, col_ref_1);

  auto expression_tree =
    cudf::ast::operation(cudf::ast::ast_operator::SUB, col_ref_2, expression_right_subtree);

  auto result = cudf::compute_column(table, expression_tree);
  auto expected =
    column_wrapper<double>{0.6, std::numeric_limits<double>::infinity(), -3.201, -2099.18};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, ImbalancedTreeArithmeticDeep)
{
  auto c_0   = column_wrapper<int64_t>{4, 5, 6};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  // expression: (c0 < c0) == (c0 < (c0 + c0))
  //              {false, false, false} == (c0 < {8, 10, 12})
  //              {false, false, false} == {true, true, true}
  //              {false, false, false}
  auto expression_left_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_0);
  auto expression_right_inner_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_0);
  auto expression_right_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, expression_right_inner_subtree);

  auto expression_tree = cudf::ast::operation(
    cudf::ast::ast_operator::EQUAL, expression_left_subtree, expression_right_subtree);

  auto result   = cudf::compute_column(table, expression_tree);
  auto expected = column_wrapper<bool>{false, false, false};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, DeeplyNestedArithmeticLogicalExpression)
{
  // Test logic for deeply nested arithmetic and logical expressions.
  constexpr int64_t left_depth_level  = 100;
  constexpr int64_t right_depth_level = 75;

  auto generate_ast_expr = [](int64_t depth_level,
                              cudf::ast::column_reference const& col_ref,
                              cudf::ast::ast_operator root_operator,
                              cudf::ast::ast_operator arithmetic_operator,
                              bool nested_left_tree) {
    // Note that a std::list is required here because of its guarantees against reference
    // invalidation when items are added or removed. References to items in a std::vector are not
    // safe if the vector must re-allocate.
    auto expressions = std::list<cudf::ast::operation>();

    auto op = arithmetic_operator;
    expressions.emplace_back(op, col_ref, col_ref);

    for (int64_t i = 0; i < depth_level - 1; i++) {
      if (i == depth_level - 2) {
        op = root_operator;
      } else {
        op = arithmetic_operator;
      }
      if (nested_left_tree) {
        expressions.emplace_back(op, expressions.back(), col_ref);
      } else {
        expressions.emplace_back(op, col_ref, expressions.back());
      }
    }
    return expressions;
  };

  auto c_0   = column_wrapper<int64_t>{0, 0, 0};
  auto c_1   = column_wrapper<int32_t>{0, 0, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);

  auto left_expression  = generate_ast_expr(left_depth_level,
                                           col_ref_0,
                                           cudf::ast::ast_operator::LESS,
                                           cudf::ast::ast_operator::ADD,
                                           false);
  auto right_expression = generate_ast_expr(right_depth_level,
                                            col_ref_1,
                                            cudf::ast::ast_operator::EQUAL,
                                            cudf::ast::ast_operator::SUB,
                                            true);

  auto expression_tree = cudf::ast::operation(
    cudf::ast::ast_operator::LOGICAL_OR, left_expression.back(), right_expression.back());

  // Expression:
  // OR(<(+(+(+(+($0, $0), $0), $0), $0), $0), ==($1, -($1, -($1, -($1, -($1, $1))))))
  // ...
  // OR(<($L, $0), ==($1, $R))
  // true
  //
  // Breakdown:
  // - Left Operand ($L): (+(+(+(+($0, $0), $0), $0), $0), $0)
  // - Right Operand ($R): -($1, -($1, -($1, -($1, $1))))
  // Explanation:
  // If all $1 values and $R values are zeros, the result is true because of the equality check
  // combined with the OR operator in OR(<($L, $0), ==($1, $R)).

  auto result   = cudf::compute_column(table, expression_tree);
  auto expected = column_wrapper<bool>{true, true, true};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, MultiLevelTreeComparator)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto c_2   = column_wrapper<int32_t>{-3, 66, 2, -99};
  auto table = cudf::table_view{{c_0, c_1, c_2}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto col_ref_2 = cudf::ast::column_reference(2);

  auto expression_left_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, col_ref_1);

  auto expression_right_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_2, col_ref_0);

  auto expression_tree = cudf::ast::operation(
    cudf::ast::ast_operator::LOGICAL_AND, expression_left_subtree, expression_right_subtree);

  auto result   = cudf::compute_column(table, expression_tree);
  auto expected = column_wrapper<bool>{false, true, false, false};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, MultiTypeOperationFailure)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<double>{0.15, 0.77, 4.2, 21.3};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);

  auto expression_0_plus_1 =
    cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);
  auto expression_1_plus_0 =
    cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_1, col_ref_0);

  // Operations on different types are not allowed
  EXPECT_THROW(cudf::compute_column(table, expression_0_plus_1), cudf::logic_error);
  EXPECT_THROW(cudf::compute_column(table, expression_1_plus_0), cudf::logic_error);
}

TEST_F(TransformTest, LiteralComparison)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<int32_t>(41);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_0, literal);

  auto result   = cudf::compute_column(table, expression);
  auto expected = column_wrapper<bool>{false, false, false, true};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, UnaryNot)
{
  auto c_0   = column_wrapper<int32_t>{3, 0, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::NOT, col_ref_0);

  auto result   = cudf::compute_column(table, expression);
  auto expected = column_wrapper<bool>{false, true, false, false};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, UnaryTrigonometry)
{
  auto c_0   = column_wrapper<double>{0.0, M_PI / 4, M_PI / 3};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  auto expected_sin   = column_wrapper<double>{0.0, std::sqrt(2) / 2, std::sqrt(3.0) / 2.0};
  auto expression_sin = cudf::ast::operation(cudf::ast::ast_operator::SIN, col_ref_0);
  auto result_sin     = cudf::compute_column(table, expression_sin);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_sin, result_sin->view(), verbosity);

  auto expected_cos   = column_wrapper<double>{1.0, std::sqrt(2) / 2, 0.5};
  auto expression_cos = cudf::ast::operation(cudf::ast::ast_operator::COS, col_ref_0);
  auto result_cos     = cudf::compute_column(table, expression_cos);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_cos, result_cos->view(), verbosity);

  auto expected_tan   = column_wrapper<double>{0.0, 1.0, std::sqrt(3.0)};
  auto expression_tan = cudf::ast::operation(cudf::ast::ast_operator::TAN, col_ref_0);
  auto result_tan     = cudf::compute_column(table, expression_tan);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_tan, result_tan->view(), verbosity);
}

TEST_F(TransformTest, ArityCheckFailure)
{
  auto col_ref_0 = cudf::ast::column_reference(0);
  EXPECT_THROW(cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0),
               std::invalid_argument);
  EXPECT_THROW(cudf::ast::operation(cudf::ast::ast_operator::ABS, col_ref_0, col_ref_0),
               std::invalid_argument);
}

TEST_F(TransformTest, StringComparison)
{
  auto c_0   = cudf::test::strings_column_wrapper({"a", "bb", "ccc", "dddd"});
  auto c_1   = cudf::test::strings_column_wrapper({"aa", "b", "cccc", "ddd"});
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{true, false, true, false};
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, StringScalarComparison)
{
  auto c_0 =
    cudf::test::strings_column_wrapper({"1", "12", "123", "23"}, {true, true, false, true});
  auto table = cudf::table_view{{c_0}};

  auto literal_value = cudf::string_scalar("2");
  auto literal       = cudf::ast::literal(literal_value);

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto expected = column_wrapper<bool>{{true, true, true, false}, {true, true, false, true}};
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  // compare with null literal
  literal_value.set_valid_async(false);
  auto expected2 = column_wrapper<bool>{{false, false, false, false}, {false, false, false, false}};
  auto result2   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, NumericScalarComparison)
{
  auto c_0   = column_wrapper<int32_t>{1, 12, 123, 23};
  auto table = cudf::table_view{{c_0}};

  auto literal_value = cudf::numeric_scalar<int32_t>(2);
  auto literal       = cudf::ast::literal(literal_value);

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto expected = column_wrapper<bool>{true, false, false, false};
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, CopyColumn)
{
  auto c_0   = column_wrapper<int32_t>{3, 0, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, col_ref_0);

  auto result   = cudf::compute_column(table, expression);
  auto expected = column_wrapper<int32_t>{3, 0, 1, 50};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, CopyLiteral)
{
  auto c_0   = column_wrapper<int32_t>{0, 0, 0, 0};
  auto table = cudf::table_view{{c_0}};

  auto literal_value = cudf::numeric_scalar<int32_t>(-123);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, literal);

  auto result   = cudf::compute_column(table, expression);
  auto expected = column_wrapper<int32_t>{-123, -123, -123, -123};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, TrueDiv)
{
  auto c_0   = column_wrapper<int32_t>{3, 0, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<int32_t>(2);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::TRUE_DIV, col_ref_0, literal);

  auto result   = cudf::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.5, 0.0, 0.5, 25.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, FloorDiv)
{
  auto c_0   = column_wrapper<double>{3.0, 0.0, 1.0, 50.0};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<double>(2.0);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::FLOOR_DIV, col_ref_0, literal);

  auto result   = cudf::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.0, 0.0, 0.0, 25.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, Mod)
{
  auto c_0   = column_wrapper<double>{3.0, 0.0, -1.0, -50.0};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<double>(2.0);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::MOD, col_ref_0, literal);

  auto result   = cudf::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.0, 0.0, -1.0, 0.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, PyMod)
{
  auto c_0   = column_wrapper<double>{3.0, 0.0, -1.0, -50.0};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<double>(2.0);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::PYMOD, col_ref_0, literal);

  auto result   = cudf::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.0, 0.0, 1.0, 0.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, BasicEqualityNullEqualNoNulls)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{3, 7, 1, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::NULL_EQUAL, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{true, false, true, false};
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, BasicEqualityNormalEqualWithNulls)
{
  auto c_0   = column_wrapper<int32_t>{{3, 20, 1, 50}, {1, 1, 0, 0}};
  auto c_1   = column_wrapper<int32_t>{{3, 7, 1, 0}, {1, 1, 0, 0}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{{true, false, true, true}, {1, 1, 0, 0}};
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, BasicEqualityNulls)
{
  auto c_0   = column_wrapper<int32_t>{{3, 20, 1, 2, 50}, {1, 1, 0, 1, 0}};
  auto c_1   = column_wrapper<int32_t>{{3, 7, 1, 2, 0}, {1, 1, 1, 0, 0}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::NULL_EQUAL, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{{true, false, false, false, true}, {1, 1, 1, 1, 1}};
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, UnaryNotNulls)
{
  auto c_0   = column_wrapper<int32_t>{{3, 0, 0, 50}, {0, 0, 1, 1}};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::NOT, col_ref_0);

  auto result   = cudf::compute_column(table, expression);
  auto expected = column_wrapper<bool>{{false, true, true, false}, {0, 0, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, BasicAdditionNulls)
{
  auto c_0   = column_wrapper<int32_t>{{3, 20, 1, 50}, {0, 0, 1, 1}};
  auto c_1   = column_wrapper<int32_t>{{10, 7, 20, 0}, {0, 1, 0, 1}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expected = column_wrapper<int32_t>{{0, 0, 0, 50}, {0, 0, 0, 1}};
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, BasicAdditionLargeNulls)
{
  auto N = 2000;
  auto a = thrust::make_counting_iterator(0);

  auto validities = std::vector<int32_t>(N);
  std::fill(validities.begin(), validities.begin() + N / 2, 0);
  std::fill(validities.begin() + (N / 2), validities.end(), 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(validities.begin(), validities.end(), gen);

  auto col   = column_wrapper<int32_t>(a, a + N, validities.begin());
  auto table = cudf::table_view{{col}};

  auto col_ref    = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref, col_ref);

  auto b        = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  auto expected = column_wrapper<int32_t>(b, b + N, validities.begin());
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, NullLogicalAnd)
{
  auto c_0   = column_wrapper<bool>{{false, false, true, true, false, false, true, true},
                                    {1, 1, 1, 1, 1, 0, 0, 0}};
  auto c_1   = column_wrapper<bool>{{false, true, false, true, true, true, false, true},
                                    {1, 1, 1, 1, 0, 1, 1, 0}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto expression =
    cudf::ast::operation(cudf::ast::ast_operator::NULL_LOGICAL_AND, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{{false, false, false, true, false, false, false, true},
                                       {1, 1, 1, 1, 1, 0, 1, 0}};
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(TransformTest, NullLogicalOr)
{
  auto c_0   = column_wrapper<bool>{{false, false, true, true, false, false, true, true},
                                    {1, 1, 1, 1, 1, 0, 1, 0}};
  auto c_1   = column_wrapper<bool>{{false, true, false, true, true, true, false, true},
                                    {1, 1, 1, 1, 0, 1, 0, 0}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto expression =
    cudf::ast::operation(cudf::ast::ast_operator::NULL_LOGICAL_OR, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{{false, true, true, true, false, true, true, true},
                                       {1, 1, 1, 1, 0, 1, 1, 0}};
  auto result   = cudf::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

CUDF_TEST_PROGRAM_MAIN()
