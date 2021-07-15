/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/ast/operators.hpp>
#include <cudf/ast/transform.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

struct TransformTest : public cudf::test::BaseFixture {
};

TEST_F(TransformTest, BasicAddition)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::expression(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expected = column_wrapper<int32_t>{13, 27, 21, 50};
  auto result   = cudf::ast::compute_column(table, expression);

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(TransformTest, BasicAdditionLarge)
{
  auto a     = thrust::make_counting_iterator(0);
  auto col   = column_wrapper<int32_t>(a, a + 2000);
  auto table = cudf::table_view{{col, col}};

  auto col_ref    = cudf::ast::column_reference(0);
  auto expression = cudf::ast::expression(cudf::ast::ast_operator::ADD, col_ref, col_ref);

  auto b        = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  auto expected = column_wrapper<int32_t>(b, b + 2000);
  auto result   = cudf::ast::compute_column(table, expression);

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(TransformTest, LessComparator)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::expression(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{true, false, true, false};
  auto result   = cudf::ast::compute_column(table, expression);

  cudf::test::expect_columns_equal(expected, result->view(), true);
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
  auto expression = cudf::ast::expression(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_1);

  auto c        = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i < 500; });
  auto expected = column_wrapper<bool>(c, c + 2000);
  auto result   = cudf::ast::compute_column(table, expression);

  cudf::test::expect_columns_equal(expected, result->view(), true);
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
    cudf::ast::expression(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expression_right_subtree =
    cudf::ast::expression(cudf::ast::ast_operator::SUB, col_ref_2, col_ref_0);

  auto expression_tree = cudf::ast::expression(
    cudf::ast::ast_operator::ADD, expression_left_subtree, expression_right_subtree);

  auto result   = cudf::ast::compute_column(table, expression_tree);
  auto expected = column_wrapper<int32_t>{7, 73, 22, -99};

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(TransformTest, MultiLevelTreeArithmeticLarge)
{
  using namespace cudf::ast;

  auto a     = thrust::make_counting_iterator(0);
  auto b     = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i + 1; });
  auto c     = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  auto c_0   = column_wrapper<int32_t>(a, a + 2000);
  auto c_1   = column_wrapper<int32_t>(b, b + 2000);
  auto c_2   = column_wrapper<int32_t>(c, c + 2000);
  auto table = cudf::table_view{{c_0, c_1, c_2}};

  auto col_ref_0 = column_reference(0);
  auto col_ref_1 = column_reference(1);
  auto col_ref_2 = column_reference(2);

  auto expr_left_subtree  = expression(cudf::ast::ast_operator::MUL, col_ref_0, col_ref_1);
  auto expr_right_subtree = expression(cudf::ast::ast_operator::ADD, col_ref_2, col_ref_0);
  auto expr_tree          = expression(ast_operator::SUB, expr_left_subtree, expr_right_subtree);

  auto result = cudf::ast::compute_column(table, expr_tree);
  auto calc   = [](auto i) { return (i * (i + 1)) - (i + (i * 2)); };
  auto d      = cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return calc(i); });
  auto expected = column_wrapper<int32_t>(d, d + 2000);

  cudf::test::expect_columns_equal(expected, result->view(), true);
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
    cudf::ast::expression(cudf::ast::ast_operator::MUL, col_ref_0, col_ref_1);

  auto expression_tree =
    cudf::ast::expression(cudf::ast::ast_operator::SUB, col_ref_2, expression_right_subtree);

  auto result = cudf::ast::compute_column(table, expression_tree);
  auto expected =
    column_wrapper<double>{0.6, std::numeric_limits<double>::infinity(), -3.201, -2099.18};

  cudf::test::expect_columns_equal(expected, result->view(), true);
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
    cudf::ast::expression(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, col_ref_1);

  auto expression_right_subtree =
    cudf::ast::expression(cudf::ast::ast_operator::GREATER, col_ref_2, col_ref_0);

  auto expression_tree = cudf::ast::expression(
    cudf::ast::ast_operator::LOGICAL_AND, expression_left_subtree, expression_right_subtree);

  auto result   = cudf::ast::compute_column(table, expression_tree);
  auto expected = column_wrapper<bool>{false, true, false, false};

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(TransformTest, MultiTypeOperationFailure)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<double>{0.15, 0.77, 4.2, 21.3};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);

  auto expression_0_plus_1 =
    cudf::ast::expression(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);
  auto expression_1_plus_0 =
    cudf::ast::expression(cudf::ast::ast_operator::ADD, col_ref_1, col_ref_0);

  // Operations on different types are not allowed
  EXPECT_THROW(cudf::ast::compute_column(table, expression_0_plus_1), cudf::logic_error);
  EXPECT_THROW(cudf::ast::compute_column(table, expression_1_plus_0), cudf::logic_error);
}

TEST_F(TransformTest, LiteralComparison)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<int32_t>(41);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::expression(cudf::ast::ast_operator::GREATER, col_ref_0, literal);

  auto result   = cudf::ast::compute_column(table, expression);
  auto expected = column_wrapper<bool>{false, false, false, true};

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(TransformTest, UnaryNot)
{
  auto c_0   = column_wrapper<int32_t>{3, 0, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  auto expression = cudf::ast::expression(cudf::ast::ast_operator::NOT, col_ref_0);

  auto result   = cudf::ast::compute_column(table, expression);
  auto expected = column_wrapper<bool>{false, true, false, false};

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(TransformTest, UnaryTrigonometry)
{
  auto c_0   = column_wrapper<double>{0.0, M_PI / 4, M_PI / 3};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  auto expected_sin   = column_wrapper<double>{0.0, std::sqrt(2) / 2, std::sqrt(3.0) / 2.0};
  auto expression_sin = cudf::ast::expression(cudf::ast::ast_operator::SIN, col_ref_0);
  auto result_sin     = cudf::ast::compute_column(table, expression_sin);
  cudf::test::expect_columns_equivalent(expected_sin, result_sin->view(), true);

  auto expected_cos   = column_wrapper<double>{1.0, std::sqrt(2) / 2, 0.5};
  auto expression_cos = cudf::ast::expression(cudf::ast::ast_operator::COS, col_ref_0);
  auto result_cos     = cudf::ast::compute_column(table, expression_cos);
  cudf::test::expect_columns_equivalent(expected_cos, result_cos->view(), true);

  auto expected_tan   = column_wrapper<double>{0.0, 1.0, std::sqrt(3.0)};
  auto expression_tan = cudf::ast::expression(cudf::ast::ast_operator::TAN, col_ref_0);
  auto result_tan     = cudf::ast::compute_column(table, expression_tan);
  cudf::test::expect_columns_equivalent(expected_tan, result_tan->view(), true);
}

TEST_F(TransformTest, ArityCheckFailure)
{
  auto col_ref_0 = cudf::ast::column_reference(0);
  EXPECT_THROW(cudf::ast::expression(cudf::ast::ast_operator::ADD, col_ref_0), cudf::logic_error);
  EXPECT_THROW(cudf::ast::expression(cudf::ast::ast_operator::ABS, col_ref_0, col_ref_0),
               cudf::logic_error);
}

TEST_F(TransformTest, StringComparison)
{
  auto c_0   = cudf::test::strings_column_wrapper({"a", "bb", "ccc", "dddd"});
  auto c_1   = cudf::test::strings_column_wrapper({"aa", "b", "cccc", "ddd"});
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::expression(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{true, false, true, false};
  auto result   = cudf::ast::compute_column(table, expression);

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(TransformTest, CopyColumn)
{
  auto c_0   = column_wrapper<int32_t>{3, 0, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto expression = cudf::ast::expression(cudf::ast::ast_operator::IDENTITY, col_ref_0);

  auto result   = cudf::ast::compute_column(table, expression);
  auto expected = column_wrapper<int32_t>{3, 0, 1, 50};

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(TransformTest, CopyLiteral)
{
  auto c_0   = column_wrapper<int32_t>{0, 0, 0, 0};
  auto table = cudf::table_view{{c_0}};

  auto literal_value = cudf::numeric_scalar<int32_t>(-123);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::expression(cudf::ast::ast_operator::IDENTITY, literal);

  auto result   = cudf::ast::compute_column(table, expression);
  auto expected = column_wrapper<int32_t>{-123, -123, -123, -123};

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(TransformTest, TrueDiv)
{
  auto c_0   = column_wrapper<int32_t>{3, 0, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<int32_t>(2);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::expression(cudf::ast::ast_operator::TRUE_DIV, col_ref_0, literal);

  auto result   = cudf::ast::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.5, 0.0, 0.5, 25.0};

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(TransformTest, FloorDiv)
{
  auto c_0   = column_wrapper<double>{3.0, 0.0, 1.0, 50.0};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<double>(2.0);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::expression(cudf::ast::ast_operator::FLOOR_DIV, col_ref_0, literal);

  auto result   = cudf::ast::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.0, 0.0, 0.0, 25.0};

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(TransformTest, Mod)
{
  auto c_0   = column_wrapper<double>{3.0, 0.0, -1.0, -50.0};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<double>(2.0);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::expression(cudf::ast::ast_operator::MOD, col_ref_0, literal);

  auto result   = cudf::ast::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.0, 0.0, -1.0, 0.0};

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(TransformTest, PyMod)
{
  auto c_0   = column_wrapper<double>{3.0, 0.0, -1.0, -50.0};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<double>(2.0);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::expression(cudf::ast::ast_operator::PYMOD, col_ref_0, literal);

  auto result   = cudf::ast::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.0, 0.0, 1.0, 0.0};

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(TransformTest, BasicAdditionNulls)
{
  auto c_0   = column_wrapper<int32_t>{{3, 20, 1, 50}, {0, 0, 1, 1}};
  auto c_1   = column_wrapper<int32_t>{{10, 7, 20, 0}, {0, 1, 0, 1}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::expression(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expected = column_wrapper<int32_t>{{0, 0, 0, 50}, {0, 0, 0, 1}};
  auto result   = cudf::ast::compute_column(table, expression);

  cudf::test::expect_columns_equal(expected, result->view(), true);
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
  auto expression = cudf::ast::expression(cudf::ast::ast_operator::ADD, col_ref, col_ref);

  auto b        = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  auto expected = column_wrapper<int32_t>(b, b + N, validities.begin());
  auto result   = cudf::ast::compute_column(table, expression);

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

CUDF_TEST_PROGRAM_MAIN()
