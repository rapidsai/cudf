/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

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

template <typename T>
struct TransformTest : public cudf::test::BaseFixture {};

struct ComputeColumnTest : public cudf::test::BaseFixture {};

struct executor_ast {
  static std::unique_ptr<cudf::column> compute_column(
    cudf::table_view const& table,
    cudf::ast::expression const& expr,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
  {
    return cudf::compute_column(table, expr, stream, mr);
  }
};

struct executor_jit {
  static std::unique_ptr<cudf::column> compute_column(
    cudf::table_view const& table,
    cudf::ast::expression const& expr,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
  {
    return cudf::compute_column_jit(table, expr, stream, mr);
  }
};

using Executors = cudf::test::Types<executor_ast, executor_jit>;

using AstTransformTest = TransformTest<executor_ast>;

TYPED_TEST_SUITE(TransformTest, Executors);

TYPED_TEST(TransformTest, ColumnReference)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  auto const& expected = c_0;
  auto result          = Executor::compute_column(table, col_ref_0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicAdditionDoubleCast)
{
  using Executor = TypeParam;

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
  auto result     = Executor::compute_column(table, expression);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, Literal)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto literal_value = cudf::numeric_scalar<int32_t>(42);
  auto literal       = cudf::ast::literal(literal_value);

  auto expected = column_wrapper<int32_t>{42, 42, 42, 42};
  auto result   = Executor::compute_column(table, literal);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, NullLiteral)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{0, 0, 0, 0};
  auto table = cudf::table_view{{c_0}};

  auto literal_value = cudf::numeric_scalar<int32_t>(-123);
  literal_value.set_valid_async(false);
  auto literal = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, literal);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<int32_t>({-123, -123, -123, -123}, {0, 0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(AstTransformTest, IsNull)
{
  auto c_0   = column_wrapper<int32_t>{{0, 1, 2, 0}, {0, 1, 1, 0}};
  auto table = cudf::table_view{{c_0}};

  // result of IS_NULL on literal, will be a column of table size, with all values set to
  // !literal.is_valid(). The table values are irrelevant.
  auto literal_value = cudf::numeric_scalar<int32_t>(-123);
  auto literal       = cudf::ast::literal(literal_value);
  auto expression    = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, literal);

  auto result    = executor_ast::compute_column(table, expression);
  auto expected1 = column_wrapper<bool>({0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, result->view(), verbosity);

  literal_value.set_valid_async(false);
  result         = executor_ast::compute_column(table, expression);
  auto expected2 = column_wrapper<bool>({1, 1, 1, 1}, cudf::test::iterators::no_nulls());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected2, result->view(), verbosity);

  auto col_ref_0   = cudf::ast::column_reference(0);
  auto expression2 = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, col_ref_0);
  result           = executor_ast::compute_column(table, expression2);
  auto expected3   = column_wrapper<bool>({1, 0, 0, 1}, cudf::test::iterators::no_nulls());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected3, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicAddition)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expected = column_wrapper<int32_t>{13, 27, 21, 50};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicAdditionEmptyTable)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{};
  auto c_1   = column_wrapper<int32_t>{};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expected = column_wrapper<int32_t>{};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicAdditionCast)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int64_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int8_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto cast       = cudf::ast::operation(cudf::ast::ast_operator::CAST_TO_INT64, col_ref_1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, cast);

  auto expected = column_wrapper<int64_t>{13, 27, 21, 50};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicEquality)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{3, 7, 1, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{true, false, true, false};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicAdditionLarge)
{
  using Executor = TypeParam;

  auto a     = thrust::make_counting_iterator(0);
  auto col   = column_wrapper<int32_t>(a, a + 2000);
  auto table = cudf::table_view{{col, col}};

  auto col_ref    = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref, col_ref);

  auto b        = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  auto expected = column_wrapper<int32_t>(b, b + 2000);
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, LessComparator)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{true, false, true, false};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, LessComparatorLarge)
{
  auto a         = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  auto b         = thrust::make_counting_iterator(500);
  using Executor = TypeParam;
  auto c_0       = column_wrapper<int32_t>(a, a + 2000);
  auto c_1       = column_wrapper<int32_t>(b, b + 2000);
  auto table     = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_1);

  auto c        = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i < 500; });
  auto expected = column_wrapper<bool>(c, c + 2000);
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, MultiLevelTreeArithmetic)
{
  using Executor = TypeParam;

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

  auto result   = Executor::compute_column(table, expression_tree);
  auto expected = column_wrapper<int32_t>{7, 73, 22, -99};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, MultiLevelTreeArithmeticLarge)
{
  auto a         = thrust::make_counting_iterator(0);
  auto b         = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i + 1; });
  auto c         = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  using Executor = TypeParam;
  auto c_0       = column_wrapper<int32_t>(a, a + 2000);
  auto c_1       = column_wrapper<int32_t>(b, b + 2000);
  auto c_2       = column_wrapper<int32_t>(c, c + 2000);
  auto table     = cudf::table_view{{c_0, c_1, c_2}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto col_ref_2 = cudf::ast::column_reference(2);

  auto expr_left_subtree = cudf::ast::operation(cudf::ast::ast_operator::MUL, col_ref_0, col_ref_1);
  auto expr_right_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_2, col_ref_0);
  auto expr_tree =
    cudf::ast::operation(cudf::ast::ast_operator::SUB, expr_left_subtree, expr_right_subtree);

  auto result = Executor::compute_column(table, expr_tree);
  auto calc   = [](auto i) { return (i * (i + 1)) - (i + (i * 2)); };
  auto d      = cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return calc(i); });
  auto expected = column_wrapper<int32_t>(d, d + 2000);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, ImbalancedTreeArithmetic)
{
  using Executor = TypeParam;

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

  auto result = Executor::compute_column(table, expression_tree);
  auto expected =
    column_wrapper<double>{0.6, std::numeric_limits<double>::infinity(), -3.201, -2099.18};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, ImbalancedTreeArithmeticDeep)
{
  using Executor = TypeParam;

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

  auto result   = Executor::compute_column(table, expression_tree);
  auto expected = column_wrapper<bool>{false, false, false};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, DeeplyNestedArithmeticLogicalExpression)
{
  using Executor = TypeParam;

  // Test logic for deeply nested arithmetic and logical expressions.
  constexpr int64_t left_depth_level  = 100;
  constexpr int64_t right_depth_level = 75;

  cudf::ast::tree tree;

  auto generate_ast_expr = [&](int64_t depth_level,
                               cudf::ast::column_reference const& col_ref,
                               cudf::ast::ast_operator root_operator,
                               cudf::ast::ast_operator arithmetic_operator,
                               bool nested_left_tree) -> cudf::ast::expression const& {
    auto op = arithmetic_operator;
    tree.push(cudf::ast::operation{op, col_ref, col_ref});

    for (int64_t i = 0; i < depth_level - 1; i++) {
      if (i == depth_level - 2) {
        op = root_operator;
      } else {
        op = arithmetic_operator;
      }
      if (nested_left_tree) {
        tree.push(cudf::ast::operation{op, tree.back(), col_ref});
      } else {
        tree.push(cudf::ast::operation{op, col_ref, tree.back()});
      }
    }

    return tree.back();
  };

  auto c_0   = column_wrapper<int64_t>{0, 0, 0};
  auto c_1   = column_wrapper<int32_t>{0, 0, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto const& col_ref_0 = tree.push(cudf::ast::column_reference(0));
  auto const& col_ref_1 = tree.push(cudf::ast::column_reference(1));

  auto const& left_expression  = generate_ast_expr(left_depth_level,
                                                  col_ref_0,
                                                  cudf::ast::ast_operator::LESS,
                                                  cudf::ast::ast_operator::ADD,
                                                  false);
  auto const& right_expression = generate_ast_expr(right_depth_level,
                                                   col_ref_1,
                                                   cudf::ast::ast_operator::EQUAL,
                                                   cudf::ast::ast_operator::SUB,
                                                   true);

  auto const& expression = tree.push(
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, left_expression, right_expression));

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

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<bool>{true, true, true};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, MultiLevelTreeComparator)
{
  using Executor = TypeParam;

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

  auto result   = Executor::compute_column(table, expression_tree);
  auto expected = column_wrapper<bool>{false, true, false, false};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(ComputeColumnTest, MultiTypeOperationFailure)
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

TYPED_TEST(TransformTest, LiteralComparison)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<int32_t>(41);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_0, literal);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<bool>{false, false, false, true};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, UnaryNot)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 0, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::NOT, col_ref_0);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<bool>{false, true, false, false};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, UnaryTrigonometry)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<double>{0.0, M_PI / 4, M_PI / 3};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  auto expected_sin   = column_wrapper<double>{0.0, std::sqrt(2) / 2, std::sqrt(3.0) / 2.0};
  auto expression_sin = cudf::ast::operation(cudf::ast::ast_operator::SIN, col_ref_0);
  auto result_sin     = Executor::compute_column(table, expression_sin);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_sin, result_sin->view(), verbosity);

  auto expected_cos   = column_wrapper<double>{1.0, std::sqrt(2) / 2, 0.5};
  auto expression_cos = cudf::ast::operation(cudf::ast::ast_operator::COS, col_ref_0);
  auto result_cos     = Executor::compute_column(table, expression_cos);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_cos, result_cos->view(), verbosity);

  auto expected_tan   = column_wrapper<double>{0.0, 1.0, std::sqrt(3.0)};
  auto expression_tan = cudf::ast::operation(cudf::ast::ast_operator::TAN, col_ref_0);
  auto result_tan     = Executor::compute_column(table, expression_tan);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_tan, result_tan->view(), verbosity);
}

TYPED_TEST(TransformTest, ArityCheckFailure)
{
  auto col_ref_0 = cudf::ast::column_reference(0);
  EXPECT_THROW(cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0),
               std::invalid_argument);
  EXPECT_THROW(cudf::ast::operation(cudf::ast::ast_operator::ABS, col_ref_0, col_ref_0),
               std::invalid_argument);
}

TYPED_TEST(TransformTest, StringComparison)
{
  using Executor = TypeParam;

  auto c_0   = cudf::test::strings_column_wrapper({"a", "bb", "ccc", "dddd"});
  auto c_1   = cudf::test::strings_column_wrapper({"aa", "b", "cccc", "ddd"});
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{true, false, true, false};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, StringScalarComparison)
{
  using Executor = TypeParam;

  auto c_0 =
    cudf::test::strings_column_wrapper({"1", "12", "123", "23"}, {true, true, false, true});
  auto table = cudf::table_view{{c_0}};

  auto literal_value = cudf::string_scalar("2");
  auto literal       = cudf::ast::literal(literal_value);

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto expected = column_wrapper<bool>{{true, true, true, false}, {true, true, false, true}};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  // compare with null literal
  literal_value.set_valid_async(false);
  auto expected2 = column_wrapper<bool>{{false, false, false, false}, {false, false, false, false}};
  auto result2   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, NumericScalarComparison)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{1, 12, 123, 23};
  auto table = cudf::table_view{{c_0}};

  auto literal_value = cudf::numeric_scalar<int32_t>(2);
  auto literal       = cudf::ast::literal(literal_value);

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto expected = column_wrapper<bool>{true, false, false, false};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, CopyColumn)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 0, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, col_ref_0);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<int32_t>{3, 0, 1, 50};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, CopyLiteral)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{0, 0, 0, 0};
  auto table = cudf::table_view{{c_0}};

  auto literal_value = cudf::numeric_scalar<int32_t>(-123);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, literal);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<int32_t>{-123, -123, -123, -123};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, TrueDiv)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 0, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<int32_t>(2);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::TRUE_DIV, col_ref_0, literal);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.5, 0.0, 0.5, 25.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, FloorDiv)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<double>{3.0, 0.0, 1.0, 50.0};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<double>(2.0);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::FLOOR_DIV, col_ref_0, literal);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.0, 0.0, 0.0, 25.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, Mod)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<double>{3.0, 0.0, -1.0, -50.0};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<double>(2.0);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::MOD, col_ref_0, literal);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.0, 0.0, -1.0, 0.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, PyMod)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<double>{3.0, 0.0, -1.0, -50.0};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<double>(2.0);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::PYMOD, col_ref_0, literal);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.0, 0.0, 1.0, 0.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(AstTransformTest, BasicEqualityNullEqualNoNulls)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{3, 7, 1, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::NULL_EQUAL, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{true, false, true, false};
  auto result   = executor_ast::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicEqualityNormalEqualWithNulls)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{{3, 20, 1, 50}, {1, 1, 0, 0}};
  auto c_1   = column_wrapper<int32_t>{{3, 7, 1, 0}, {1, 1, 0, 0}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{{true, false, true, true}, {1, 1, 0, 0}};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(AstTransformTest, BasicEqualityNulls)
{
  auto c_0   = column_wrapper<int32_t>{{3, 20, 1, 2, 50}, {1, 1, 0, 1, 0}};
  auto c_1   = column_wrapper<int32_t>{{3, 7, 1, 2, 0}, {1, 1, 1, 0, 0}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::NULL_EQUAL, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{{true, false, false, false, true}, {1, 1, 1, 1, 1}};
  auto result   = executor_ast::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, UnaryNotNulls)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{{3, 0, 0, 50}, {0, 0, 1, 1}};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::NOT, col_ref_0);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<bool>{{false, true, true, false}, {0, 0, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicAdditionNulls)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{{3, 20, 1, 50}, {0, 0, 1, 1}};
  auto c_1   = column_wrapper<int32_t>{{10, 7, 20, 0}, {0, 1, 0, 1}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expected = column_wrapper<int32_t>{{0, 0, 0, 50}, {0, 0, 0, 1}};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicAdditionLargeNulls)
{
  using Executor = TypeParam;

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
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(AstTransformTest, NullLogicalAnd)
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
  auto result   = executor_ast::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(AstTransformTest, NullLogicalOr)
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
  auto result   = executor_ast::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, ScalarOnly)
{
  using Executor = TypeParam;

  auto column = column_wrapper<int>{1, 2, 3, 4, 5};
  auto table  = cudf::table_view{{column}};

  cudf::ast::tree tree{};

  auto first_value  = cudf::numeric_scalar(0);
  auto second_value = cudf::numeric_scalar(1);
  auto first        = cudf::ast::literal(first_value);
  auto second       = cudf::ast::literal(second_value);

  auto const& neq =
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::NOT_EQUAL, first, second});

  auto result = Executor::compute_column(table, neq);

  auto expected = column_wrapper<bool>{true, true, true, true, true};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(TransformTest, ComplexScalarOnly)
{
  using Executor = TypeParam;

  auto column = column_wrapper<int>{1, 2, 3, 4, 5};
  auto table  = cudf::table_view{{column}};

  cudf::ast::tree tree{};

  auto first_value  = cudf::string_scalar("first");
  auto second_value = cudf::string_scalar("second");
  auto first        = cudf::ast::literal(first_value);
  auto second       = cudf::ast::literal(second_value);

  auto const& neq =
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::NOT_EQUAL, first, second});

  auto result = Executor::compute_column(table, neq);

  auto expected = column_wrapper<bool>{true, true, true, true, true};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

CUDF_TEST_PROGRAM_MAIN()
