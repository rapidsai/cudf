/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

struct TransformTest : public cudf::test::BaseFixture {
};

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
  EXPECT_THROW(cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0), cudf::logic_error);
  EXPECT_THROW(cudf::ast::operation(cudf::ast::ast_operator::ABS, col_ref_0, col_ref_0),
               cudf::logic_error);
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
  auto c_0   = cudf::test::strings_column_wrapper({"1", "12", "123", "23"});
  auto table = cudf::table_view{{c_0}};

  auto literal_value = cudf::string_scalar("2");
  auto literal       = cudf::ast::literal(literal_value);

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto expected = column_wrapper<bool>{true, true, true, false};
  auto result   = cudf::compute_column(table, expression);

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

struct literal_converter {
  template <typename T>
  static constexpr bool is_supported()
  {
    return std::is_same_v<T, cudf::string_view> ||
           (cudf::is_fixed_width<T>() && !cudf::is_fixed_point<T>());
  }

  template <typename T, std::enable_if_t<is_supported<T>()>* = nullptr>
  cudf::ast::literal operator()(cudf::scalar& _value)
  {
    using scalar_type       = cudf::scalar_type_t<T>;
    auto& low_literal_value = static_cast<scalar_type&>(_value);
    return cudf::ast::literal(low_literal_value);
  }

  template <typename T, std::enable_if_t<!is_supported<T>()>* = nullptr>
  cudf::ast::literal operator()(cudf::scalar& _value)
  {
    CUDF_FAIL("Unsupported type for literal");
  }
};

std::unique_ptr<cudf::table> filter_table_by_range(cudf::table_view const& input,
                                                   cudf::column_view const& sort_col,
                                                   cudf::scalar& low,
                                                   cudf::scalar& high,
                                                   rmm::mr::device_memory_resource* mr)
{
  // return low.compare(elem) <= 0 && high.compare(elem) > 0;
  auto col_ref_0   = cudf::ast::column_reference(0);
  auto low_literal = cudf::type_dispatcher(low.type(), literal_converter{}, low);
  auto expr_1 = cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref_0, low_literal);
  auto high_literal = cudf::type_dispatcher(high.type(), literal_converter{}, high);
  auto expr_2 = cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_0, high_literal);
  auto expr_3 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);
  auto result = cudf::compute_column(input, expr_3);
  return cudf::detail::apply_boolean_mask(input, result->view(), rmm::cuda_stream_default, mr);
}

TEST_F(TransformTest, RangeFilterString)
{
  auto c_0   = cudf::test::strings_column_wrapper({"1", "12", "123", "2"});
  auto table = cudf::table_view{{c_0}};
  auto mr    = rmm::mr::get_current_device_resource();

  auto lower  = cudf::string_scalar("2", true, rmm::cuda_stream_default, mr);
  auto higher = cudf::string_scalar("12", true, rmm::cuda_stream_default, mr);
  // elem > "12" && elem <= "2"  -> "123", "2"
  auto c_1      = cudf::test::strings_column_wrapper({"123", "2"});
  auto expected = cudf::table_view{{c_1}};
  auto result   = filter_table_by_range(table, c_0, lower, higher, mr);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected.column(0), result->view().column(0), verbosity);
}

TEST_F(TransformTest, RangeFilterNumeric)
{
  auto c_0   = column_wrapper<int32_t>{{1, 12, 5, 2}};
  auto table = cudf::table_view{{c_0}};
  auto mr    = rmm::mr::get_current_device_resource();

  auto higher = cudf::numeric_scalar<int32_t>(2, true, rmm::cuda_stream_default, mr);
  auto lower  = cudf::numeric_scalar<int32_t>(12, true, rmm::cuda_stream_default, mr);
  // elem > 2 && elem <= 12  -> 12, 5
  auto c_1      = column_wrapper<int32_t>{{12, 5}};
  auto expected = cudf::table_view{{c_1}};
  auto result   = filter_table_by_range(table, c_0, lower, higher, mr);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected.column(0), result->view().column(0), verbosity);
}

CUDF_TEST_PROGRAM_MAIN()
