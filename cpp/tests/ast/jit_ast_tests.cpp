
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/ast/jit_expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda/iterator>

#include <limits>
#include <vector>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

struct JITExpressionTest : public cudf::test::BaseFixture {};

TEST_F(JITExpressionTest, NullifyIf)
{
  auto a             = column_wrapper<int32_t>{3, 20, 1, 50, 0, 20};
  auto condition     = column_wrapper<bool>{false, true, false, true, false, true};
  auto expected      = column_wrapper<int32_t>{{3, 0, 1, 0, 0, 0}, {1, 0, 1, 0, 1, 0}};
  auto table         = cudf::table_view{{a, condition}};
  auto tree          = cudf::ast::tree{};
  auto a_ref         = cudf::ast::column_reference(0);
  auto condition_ref = cudf::ast::column_reference(1);
  auto& nullify_if   = cudf::ast::jit::nullify_if(tree, a_ref, condition_ref);
  auto result        = cudf::compute_column_jit(table, nullify_if);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(JITExpressionTest, Coalesce)
{
  auto a         = column_wrapper<int32_t>{{1, 3, 5, 7, 9, 11}, {1, 0, 0, 1, 0, 0}};
  auto b         = column_wrapper<int32_t>{{2, 4, 6, 8, 10, 12}, {1, 1, 1, 0, 1, 0}};
  auto expected  = column_wrapper<int32_t>{{1, 4, 6, 7, 10, 0}, {1, 1, 1, 1, 1, 0}};
  auto table     = cudf::table_view{{a, b}};
  auto tree      = cudf::ast::tree{};
  auto a_ref     = cudf::ast::column_reference(0);
  auto b_ref     = cudf::ast::column_reference(1);
  auto& coalesce = cudf::ast::jit::coalesce(tree, a_ref, b_ref);
  auto result    = cudf::compute_column_jit(table, coalesce);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

// TODO: parameterize on decimal types and float types

TEST_F(JITExpressionTest, AnsiAdd_Int)
{
  constexpr auto I32_MAX = std::numeric_limits<int32_t>::max();
  auto a                 = column_wrapper<int32_t>{3, 20, 1, 50};
  auto b                 = column_wrapper<int32_t>{10, 7, 20, 0};
  auto b_fail            = column_wrapper<int32_t>{10, I32_MAX, 20, 0};
  auto expected          = column_wrapper<int32_t>{13, 27, 21, 50};
  auto expected_fail     = column_wrapper<int32_t>{{13, 0, 21, 50}, {1, 0, 1, 1}};
  auto table             = cudf::table_view{{a, b, b_fail}};
  auto tree              = cudf::ast::tree{};
  auto a_ref             = cudf::ast::column_reference(0);
  auto b_ref             = cudf::ast::column_reference(1);
  auto b_fail_ref        = cudf::ast::column_reference(2);
  auto& add              = cudf::ast::jit::ansi_add(tree, a_ref, b_ref);
  auto& add_fail         = cudf::ast::jit::ansi_add(tree, a_ref, b_fail_ref);
  auto& try_add_fail     = cudf::ast::jit::ansi_try_add(tree, a_ref, b_fail_ref);
  auto result            = cudf::compute_column_jit(table, add);
  auto result_fail       = cudf::compute_column_jit(table, try_add_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  EXPECT_THROW(result = cudf::compute_column_jit(table, add_fail), std::overflow_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), verbosity);
}

TEST_F(JITExpressionTest, AnsiAdd_Decimal)
{
  constexpr auto I32_MAX = std::numeric_limits<int32_t>::max();
  auto a = cudf::test::fixed_point_column_wrapper<int32_t>{{3, 20, 1, 50}, numeric::scale_type{0}};
  auto b = cudf::test::fixed_point_column_wrapper<int32_t>{{10, 7, 20, 0}, numeric::scale_type{0}};
  auto b_fail =
    cudf::test::fixed_point_column_wrapper<int32_t>{{10, I32_MAX, 20, 0}, numeric::scale_type{0}};
  auto expected =
    cudf::test::fixed_point_column_wrapper<int32_t>{{13, 27, 21, 50}, numeric::scale_type{0}};
  auto expected_fail = cudf::test::fixed_point_column_wrapper<int32_t>{
    {13, 0, 21, 50}, {1, 0, 1, 1}, numeric::scale_type{0}};
  auto table         = cudf::table_view{{a, b, b_fail}};
  auto tree          = cudf::ast::tree{};
  auto a_ref         = cudf::ast::column_reference(0);
  auto b_ref         = cudf::ast::column_reference(1);
  auto b_fail_ref    = cudf::ast::column_reference(2);
  auto& add          = cudf::ast::jit::ansi_add(tree, a_ref, b_ref);
  auto& add_fail     = cudf::ast::jit::ansi_add(tree, a_ref, b_fail_ref);
  auto& try_add_fail = cudf::ast::jit::ansi_try_add(tree, a_ref, b_fail_ref);
  auto result        = cudf::compute_column_jit(table, add);
  auto result_fail   = cudf::compute_column_jit(table, try_add_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  EXPECT_THROW(result = cudf::compute_column_jit(table, add_fail), std::overflow_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), verbosity);
}

TEST_F(JITExpressionTest, AnsiSub)
{
  constexpr auto I32_MIN = std::numeric_limits<int32_t>::min();
  auto a                 = column_wrapper<int32_t>{3, 20, 1, 50};
  auto b                 = column_wrapper<int32_t>{10, 7, 20, 0};
  auto b_fail            = column_wrapper<int32_t>{10, I32_MIN, 20, 0};
  auto expected          = column_wrapper<int32_t>{-7, 13, -19, 50};
  auto expected_fail     = column_wrapper<int32_t>{{-7, 0, -19, 50}, {1, 0, 1, 1}};
  auto table             = cudf::table_view{{a, b, b_fail}};
  auto tree              = cudf::ast::tree{};
  auto a_ref             = cudf::ast::column_reference(0);
  auto b_ref             = cudf::ast::column_reference(1);
  auto b_fail_ref        = cudf::ast::column_reference(2);
  auto& sub              = cudf::ast::jit::ansi_sub(tree, a_ref, b_ref);
  auto& sub_fail         = cudf::ast::jit::ansi_sub(tree, a_ref, b_fail_ref);
  auto& try_sub_fail     = cudf::ast::jit::ansi_try_sub(tree, a_ref, b_fail_ref);

  auto result      = cudf::compute_column_jit(table, sub);
  auto result_fail = cudf::compute_column_jit(table, try_sub_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  EXPECT_THROW(result = cudf::compute_column_jit(table, sub_fail), std::overflow_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), verbosity);
}

TEST_F(JITExpressionTest, AnsiSub_Decimal)
{
  constexpr auto I32_MIN = std::numeric_limits<int32_t>::min();
  auto a = cudf::test::fixed_point_column_wrapper<int32_t>{{3, 20, 1, 50}, numeric::scale_type{0}};
  auto b = cudf::test::fixed_point_column_wrapper<int32_t>{{10, 7, 20, 0}, numeric::scale_type{0}};
  auto b_fail =
    cudf::test::fixed_point_column_wrapper<int32_t>{{10, I32_MIN, 20, 0}, numeric::scale_type{0}};
  auto expected =
    cudf::test::fixed_point_column_wrapper<int32_t>{{-7, 13, -19, 50}, numeric::scale_type{0}};
  auto expected_fail = cudf::test::fixed_point_column_wrapper<int32_t>{
    {-7, 0, -19, 50}, {1, 0, 1, 1}, numeric::scale_type{0}};
  auto table         = cudf::table_view{{a, b, b_fail}};
  auto tree          = cudf::ast::tree{};
  auto a_ref         = cudf::ast::column_reference(0);
  auto b_ref         = cudf::ast::column_reference(1);
  auto b_fail_ref    = cudf::ast::column_reference(2);
  auto& sub          = cudf::ast::jit::ansi_sub(tree, a_ref, b_ref);
  auto& sub_fail     = cudf::ast::jit::ansi_sub(tree, a_ref, b_fail_ref);
  auto& try_sub_fail = cudf::ast::jit::ansi_try_sub(tree, a_ref, b_fail_ref);
  auto result        = cudf::compute_column_jit(table, sub);
  auto result_fail   = cudf::compute_column_jit(table, try_sub_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  EXPECT_THROW(result = cudf::compute_column_jit(table, sub_fail), std::overflow_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), verbosity);
}

TEST_F(JITExpressionTest, AnsiMul)
{
  constexpr auto I32_MAX = std::numeric_limits<int32_t>::max();
  auto a                 = column_wrapper<int32_t>{3, 20, 2, 50};
  auto b                 = column_wrapper<int32_t>{10, 7, 1, 0};
  auto b_fail            = column_wrapper<int32_t>{10, I32_MAX, 1, 0};
  auto expected          = column_wrapper<int32_t>{30, 140, 2, 0};
  auto expected_fail     = column_wrapper<int32_t>{{30, 0, 2, 0}, {1, 0, 1, 1}};
  auto table             = cudf::table_view{{a, b, b_fail}};
  auto a_ref             = cudf::ast::column_reference(0);
  auto b_ref             = cudf::ast::column_reference(1);
  auto b_fail_ref        = cudf::ast::column_reference(2);
  auto tree              = cudf::ast::tree{};
  auto& mul              = cudf::ast::jit::ansi_mul(tree, a_ref, b_ref);
  auto& mul_fail         = cudf::ast::jit::ansi_mul(tree, a_ref, b_fail_ref);
  auto& try_mul_fail     = cudf::ast::jit::ansi_try_mul(tree, a_ref, b_fail_ref);
  auto result            = cudf::compute_column_jit(table, mul);
  auto result_fail       = cudf::compute_column_jit(table, try_mul_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  EXPECT_THROW(result = cudf::compute_column_jit(table, mul_fail), std::overflow_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), verbosity);
}

TEST_F(JITExpressionTest, AnsiMul_Decimal)
{
  constexpr auto I32_MAX = std::numeric_limits<int32_t>::max();
  auto a = cudf::test::fixed_point_column_wrapper<int32_t>{{3, 20, 2, 50}, numeric::scale_type{0}};
  auto b = cudf::test::fixed_point_column_wrapper<int32_t>{{10, 7, 1, 0}, numeric::scale_type{0}};
  auto b_fail =
    cudf::test::fixed_point_column_wrapper<int32_t>{{10, I32_MAX, 1, 0}, numeric::scale_type{0}};
  auto expected =
    cudf::test::fixed_point_column_wrapper<int32_t>{{30, 140, 2, 0}, numeric::scale_type{0}};
  auto expected_fail = cudf::test::fixed_point_column_wrapper<int32_t>{
    {30, 0, 2, 0}, {1, 0, 1, 1}, numeric::scale_type{0}};
  auto table         = cudf::table_view{{a, b, b_fail}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto b_ref         = cudf::ast::column_reference(1);
  auto b_fail_ref    = cudf::ast::column_reference(2);
  auto tree          = cudf::ast::tree{};
  auto& mul          = cudf::ast::jit::ansi_mul(tree, a_ref, b_ref);
  auto& mul_fail     = cudf::ast::jit::ansi_mul(tree, a_ref, b_fail_ref);
  auto& try_mul_fail = cudf::ast::jit::ansi_try_mul(tree, a_ref, b_fail_ref);
  auto result        = cudf::compute_column_jit(table, mul);
  auto result_fail   = cudf::compute_column_jit(table, try_mul_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  EXPECT_THROW(result = cudf::compute_column_jit(table, mul_fail), std::overflow_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), verbosity);
}

TEST_F(JITExpressionTest, AnsiDiv)
{
  auto a             = column_wrapper<int32_t>{3, 20, 1, 50};
  auto b             = column_wrapper<int32_t>{10, 7, 2, 1};
  auto b_fail        = column_wrapper<int32_t>{10, 1, 20, 0};
  auto expected      = column_wrapper<int32_t>{0, 2, 0, 50};
  auto expected_fail = column_wrapper<int32_t>{{0, 20, 0, 50}, {1, 1, 1, 0}};
  auto table         = cudf::table_view{{a, b, b_fail}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto b_ref         = cudf::ast::column_reference(1);
  auto b_fail_ref    = cudf::ast::column_reference(2);
  auto tree          = cudf::ast::tree{};
  auto& div          = cudf::ast::jit::ansi_div(tree, a_ref, b_ref);
  auto& div_fail     = cudf::ast::jit::ansi_div(tree, a_ref, b_fail_ref);
  auto& try_div_fail = cudf::ast::jit::ansi_try_div(tree, a_ref, b_fail_ref);
  auto result        = cudf::compute_column_jit(table, div);
  auto result_fail   = cudf::compute_column_jit(table, try_div_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  EXPECT_THROW(result = cudf::compute_column_jit(table, div_fail), std::overflow_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), verbosity);
}

TEST_F(JITExpressionTest, AnsiDiv_Decimal)
{
  auto a = cudf::test::fixed_point_column_wrapper<int32_t>{{3, 20, 1, 50}, numeric::scale_type{0}};
  auto b = cudf::test::fixed_point_column_wrapper<int32_t>{{10, 7, 2, 1}, numeric::scale_type{0}};
  auto b_fail =
    cudf::test::fixed_point_column_wrapper<int32_t>{{10, 1, 20, 0}, numeric::scale_type{0}};
  auto expected =
    cudf::test::fixed_point_column_wrapper<int32_t>{{0, 2, 0, 50}, numeric::scale_type{0}};
  auto expected_fail = cudf::test::fixed_point_column_wrapper<int32_t>{
    {0, 20, 0, 50}, {1, 1, 1, 0}, numeric::scale_type{0}};
  auto table         = cudf::table_view{{a, b, b_fail}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto b_ref         = cudf::ast::column_reference(1);
  auto b_fail_ref    = cudf::ast::column_reference(2);
  auto tree          = cudf::ast::tree{};
  auto& div          = cudf::ast::jit::ansi_div(tree, a_ref, b_ref);
  auto& div_fail     = cudf::ast::jit::ansi_div(tree, a_ref, b_fail_ref);
  auto& try_div_fail = cudf::ast::jit::ansi_try_div(tree, a_ref, b_fail_ref);
  auto result        = cudf::compute_column_jit(table, div);
  auto result_fail   = cudf::compute_column_jit(table, try_div_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  EXPECT_THROW(result = cudf::compute_column_jit(table, div_fail), std::overflow_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), verbosity);
}

TEST_F(JITExpressionTest, AnsiMod)
{
  auto a             = column_wrapper<int32_t>{3, 20, 1, 50};
  auto b             = column_wrapper<int32_t>{10, 7, 2, 1};
  auto b_fail        = column_wrapper<int32_t>{10, 1, 20, 0};
  auto expected      = column_wrapper<int32_t>{3, 6, 1, 0};
  auto expected_fail = column_wrapper<int32_t>{{3, 0, 1, 0}, {1, 1, 1, 0}};
  auto table         = cudf::table_view{{a, b, b_fail}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto b_ref         = cudf::ast::column_reference(1);
  auto b_fail_ref    = cudf::ast::column_reference(2);
  auto tree          = cudf::ast::tree{};
  auto& mod          = cudf::ast::jit::ansi_mod(tree, a_ref, b_ref);
  auto& mod_fail     = cudf::ast::jit::ansi_mod(tree, a_ref, b_fail_ref);
  auto& try_mod_fail = cudf::ast::jit::ansi_try_mod(tree, a_ref, b_fail_ref);
  auto result        = cudf::compute_column_jit(table, mod);
  auto result_fail   = cudf::compute_column_jit(table, try_mod_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  EXPECT_THROW(result = cudf::compute_column_jit(table, mod_fail), std::overflow_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), verbosity);
}

TEST_F(JITExpressionTest, AnsiMod_Decimal)
{
  auto a = cudf::test::fixed_point_column_wrapper<int32_t>{{3, 20, 1, 50}, numeric::scale_type{0}};
  auto b = cudf::test::fixed_point_column_wrapper<int32_t>{{10, 7, 2, 1}, numeric::scale_type{0}};
  auto b_fail =
    cudf::test::fixed_point_column_wrapper<int32_t>{{10, 1, 20, 0}, numeric::scale_type{0}};
  auto expected =
    cudf::test::fixed_point_column_wrapper<int32_t>{{3, 6, 1, 0}, numeric::scale_type{0}};
  auto expected_fail = cudf::test::fixed_point_column_wrapper<int32_t>{
    {3, 0, 1, 0}, {1, 1, 1, 0}, numeric::scale_type{0}};
  auto table         = cudf::table_view{{a, b, b_fail}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto b_ref         = cudf::ast::column_reference(1);
  auto b_fail_ref    = cudf::ast::column_reference(2);
  auto tree          = cudf::ast::tree{};
  auto& mod          = cudf::ast::jit::ansi_mod(tree, a_ref, b_ref);
  auto& mod_fail     = cudf::ast::jit::ansi_mod(tree, a_ref, b_fail_ref);
  auto& try_mod_fail = cudf::ast::jit::ansi_try_mod(tree, a_ref, b_fail_ref);
  auto result        = cudf::compute_column_jit(table, mod);
  auto result_fail   = cudf::compute_column_jit(table, try_mod_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  EXPECT_THROW(result = cudf::compute_column_jit(table, mod_fail), std::overflow_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), verbosity);
}

TEST_F(JITExpressionTest, AnsiAbs)
{
  constexpr auto I32_MIN = std::numeric_limits<int32_t>::min();
  constexpr auto I32_MAX = std::numeric_limits<int32_t>::max();
  auto a                 = column_wrapper<int32_t>{3, -20, 1, -50, I32_MAX, I32_MIN + 1, 0};
  auto a_fail            = column_wrapper<int32_t>{3, -20, 1, -50, I32_MIN, 1, 0};
  auto expected          = column_wrapper<int32_t>{3, 20, 1, 50, I32_MAX, std::abs(I32_MIN + 1), 0};
  auto expected_fail     = column_wrapper<int32_t>{{3, 20, 1, 50, 0, 1, 0}, {1, 1, 1, 1, 0, 1, 1}};
  auto table             = cudf::table_view{{a, a_fail}};
  auto a_ref             = cudf::ast::column_reference(0);
  auto a_fail_ref        = cudf::ast::column_reference(1);
  auto tree              = cudf::ast::tree{};
  auto& abs              = cudf::ast::jit::ansi_abs(tree, a_ref);
  auto& abs_fail         = cudf::ast::jit::ansi_abs(tree, a_fail_ref);
  auto& try_abs_fail     = cudf::ast::jit::ansi_try_abs(tree, a_fail_ref);
  auto result            = cudf::compute_column_jit(table, abs);
  auto result_fail       = cudf::compute_column_jit(table, try_abs_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  EXPECT_THROW(result = cudf::compute_column_jit(table, abs_fail), std::overflow_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), verbosity);
}

TEST_F(JITExpressionTest, AnsiAbs_Decimal)
{
  constexpr auto I32_MIN = std::numeric_limits<int32_t>::min();
  constexpr auto I32_MAX = std::numeric_limits<int32_t>::max();
  auto a                 = cudf::test::fixed_point_column_wrapper<int32_t>{
    {3, -20, 1, -50, I32_MAX, I32_MIN + 1, 0}, numeric::scale_type{0}};
  auto a_fail   = cudf::test::fixed_point_column_wrapper<int32_t>{{3, -20, 1, -50, I32_MIN, 1, 0},
                                                                  numeric::scale_type{0}};
  auto expected = cudf::test::fixed_point_column_wrapper<int32_t>{
    {3, 20, 1, 50, I32_MAX, std::abs(I32_MIN + 1), 0}, numeric::scale_type{0}};
  auto expected_fail = cudf::test::fixed_point_column_wrapper<int32_t>{
    {3, 20, 1, 50, 0, 1, 0}, {1, 1, 1, 1, 0, 1, 1}, numeric::scale_type{0}};
  auto table         = cudf::table_view{{a, a_fail}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto a_fail_ref    = cudf::ast::column_reference(1);
  auto tree          = cudf::ast::tree{};
  auto& abs          = cudf::ast::jit::ansi_abs(tree, a_ref);
  auto& abs_fail     = cudf::ast::jit::ansi_abs(tree, a_fail_ref);
  auto& try_abs_fail = cudf::ast::jit::ansi_try_abs(tree, a_fail_ref);
  auto result        = cudf::compute_column_jit(table, abs);
  auto result_fail   = cudf::compute_column_jit(table, try_abs_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  EXPECT_THROW(result = cudf::compute_column_jit(table, abs_fail), std::overflow_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), verbosity);
}

TEST_F(JITExpressionTest, AnsiNeg)
{
  constexpr auto I32_MIN = std::numeric_limits<int32_t>::min();
  constexpr auto I32_MAX = std::numeric_limits<int32_t>::max();
  auto a                 = column_wrapper<int32_t>{3, -20, 1, -50, I32_MAX, -I32_MAX, 0};
  auto a_fail            = column_wrapper<int32_t>{3, -20, 1, -50, I32_MIN, 1, 0};
  auto expected          = column_wrapper<int32_t>{-3, 20, -1, 50, -I32_MAX, I32_MAX, 0};
  auto expected_fail = column_wrapper<int32_t>{{-3, 20, -1, 50, 0, -1, 0}, {1, 1, 1, 1, 0, 1, 1}};
  auto table         = cudf::table_view{{a, a_fail}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto a_fail_ref    = cudf::ast::column_reference(1);
  auto tree          = cudf::ast::tree{};
  auto& neg          = cudf::ast::jit::ansi_neg(tree, a_ref);
  auto& neg_fail     = cudf::ast::jit::ansi_neg(tree, a_fail_ref);
  auto& try_neg_fail = cudf::ast::jit::ansi_try_neg(tree, a_fail_ref);
  auto result        = cudf::compute_column_jit(table, neg);
  auto result_fail   = cudf::compute_column_jit(table, try_neg_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  EXPECT_THROW(result = cudf::compute_column_jit(table, neg_fail), std::overflow_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), verbosity);
}

TEST_F(JITExpressionTest, AnsiNeg_Decimal)
{
  constexpr auto I32_MIN = std::numeric_limits<int32_t>::min();
  constexpr auto I32_MAX = std::numeric_limits<int32_t>::max();
  auto a = cudf::test::fixed_point_column_wrapper<int32_t>{{3, -20, 1, -50, I32_MAX, -I32_MAX, 0},
                                                           numeric::scale_type{0}};
  auto a_fail   = cudf::test::fixed_point_column_wrapper<int32_t>{{3, -20, 1, -50, I32_MIN, 1, 0},
                                                                  numeric::scale_type{0}};
  auto expected = cudf::test::fixed_point_column_wrapper<int32_t>{
    {-3, 20, -1, 50, -I32_MAX, I32_MAX, 0}, numeric::scale_type{0}};
  auto expected_fail = cudf::test::fixed_point_column_wrapper<int32_t>{
    {-3, 20, -1, 50, 0, -1, 0}, {1, 1, 1, 1, 0, 1, 1}, numeric::scale_type{0}};
  auto table         = cudf::table_view{{a, a_fail}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto a_fail_ref    = cudf::ast::column_reference(1);
  auto tree          = cudf::ast::tree{};
  auto& neg          = cudf::ast::jit::ansi_neg(tree, a_ref);
  auto& neg_fail     = cudf::ast::jit::ansi_neg(tree, a_fail_ref);
  auto& try_neg_fail = cudf::ast::jit::ansi_try_neg(tree, a_fail_ref);
  auto result        = cudf::compute_column_jit(table, neg);
  auto result_fail   = cudf::compute_column_jit(table, try_neg_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  EXPECT_THROW(result = cudf::compute_column_jit(table, neg_fail), std::overflow_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), verbosity);
}

TEST_F(JITExpressionTest, AnsiPrecisionCheck)
{
  auto a =
    cudf::test::fixed_point_column_wrapper<int32_t>{{3, 200, 250, 200}, numeric::scale_type{0}};
  auto a_fail =
    cudf::test::fixed_point_column_wrapper<int32_t>{{3, 200, 250, 20000}, numeric::scale_type{0}};
  auto expected =
    cudf::test::fixed_point_column_wrapper<int32_t>{{3, 200, 250, 200}, numeric::scale_type{0}};
  auto expected_fail = cudf::test::fixed_point_column_wrapper<int32_t>{
    {3, 200, 250, 200}, {1, 1, 1, 0}, numeric::scale_type{0}};
  auto max_precision         = cudf::numeric_scalar<int32_t>(3);
  auto table                 = cudf::table_view{{a, a_fail}};
  auto a_ref                 = cudf::ast::column_reference(0);
  auto a_fail_ref            = cudf::ast::column_reference(1);
  auto tree                  = cudf::ast::tree{};
  auto precision             = cudf::ast::literal(max_precision);
  auto& precision_check      = cudf::ast::jit::ansi_precision_check(tree, a_ref, precision);
  auto& precision_check_fail = cudf::ast::jit::ansi_precision_check(tree, a_fail_ref, precision);
  auto& try_precision_check = cudf::ast::jit::ansi_try_precision_check(tree, a_fail_ref, precision);
  auto result               = cudf::compute_column_jit(table, precision_check);
  auto result_fail          = cudf::compute_column_jit(table, try_precision_check);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  EXPECT_THROW(result = cudf::compute_column_jit(table, precision_check_fail), std::overflow_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), verbosity);
}

TEST_F(JITExpressionTest, BitShiftLeft)
{
  auto a = cudf::test::fixed_width_column_wrapper<uint32_t>{0b111111, 0b111110, 0b101111, 0b1100};
  auto expected =
    cudf::test::fixed_width_column_wrapper<uint32_t>{0b11111100, 0b11111000, 0b10111100, 0b110000};
  auto shift         = cudf::numeric_scalar<uint32_t>(2);
  auto table         = cudf::table_view{{a}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto tree          = cudf::ast::tree{};
  auto shift_literal = cudf::ast::literal(shift);
  auto& shift_left   = cudf::ast::jit::bit_shift_left(tree, a_ref, shift_literal);
  auto result        = cudf::compute_column_jit(table, shift_left);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(JITExpressionTest, BitShiftRight)
{
  auto a = cudf::test::fixed_width_column_wrapper<uint32_t>{0b1111, 0b10111, 0b11100, 0b11110011};
  auto expected = cudf::test::fixed_width_column_wrapper<uint32_t>{0b11, 0b101, 0b111, 0b111100};
  auto shift    = cudf::numeric_scalar<uint32_t>(2);
  auto table    = cudf::table_view{{a}};
  auto a_ref    = cudf::ast::column_reference(0);
  auto tree     = cudf::ast::tree{};
  auto shift_literal = cudf::ast::literal(shift);
  auto& shift_right  = cudf::ast::jit::bit_shift_right(tree, a_ref, shift_literal);
  auto result        = cudf::compute_column_jit(table, shift_right);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

template <typename From, typename To>
void test_cast()
{
  auto a        = cudf::test::fixed_width_column_wrapper<From>{0, 1, 2, 3, 4, 5};
  auto expected = cudf::test::fixed_width_column_wrapper<To>{0, 1, 2, 3, 4, 5};
  auto table    = cudf::table_view{{a}};
  auto a_ref    = cudf::ast::column_reference(0);
  auto tree     = cudf::ast::tree{};

  cudf::ast::expression const* cast = nullptr;

  if constexpr (std::is_same_v<To, bool>) {
    cast = &cudf::ast::jit::cast_to_b8(tree, a_ref);
  } else if constexpr (std::is_same_v<To, int8_t>) {
    cast = &cudf::ast::jit::cast_to_i8(tree, a_ref);
  } else if constexpr (std::is_same_v<To, int16_t>) {
    cast = &cudf::ast::jit::cast_to_i16(tree, a_ref);
  } else if constexpr (std::is_same_v<To, int32_t>) {
    cast = &cudf::ast::jit::cast_to_i32(tree, a_ref);
  } else if constexpr (std::is_same_v<To, int64_t>) {
    cast = &cudf::ast::jit::cast_to_i64(tree, a_ref);
  } else if constexpr (std::is_same_v<To, uint8_t>) {
    cast = &cudf::ast::jit::cast_to_u8(tree, a_ref);
  } else if constexpr (std::is_same_v<To, uint16_t>) {
    cast = &cudf::ast::jit::cast_to_u16(tree, a_ref);
  } else if constexpr (std::is_same_v<To, uint32_t>) {
    cast = &cudf::ast::jit::cast_to_u32(tree, a_ref);
  } else if constexpr (std::is_same_v<To, uint64_t>) {
    cast = &cudf::ast::jit::cast_to_u64(tree, a_ref);
  } else if constexpr (std::is_same_v<To, float>) {
    cast = &cudf::ast::jit::cast_to_f32(tree, a_ref);
  } else {
    static_assert(std::is_same_v<To, double>);
    cast = &cudf::ast::jit::cast_to_f64(tree, a_ref);
  }

  auto result = cudf::compute_column_jit(table, *cast);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

template <typename From, typename To>
void test_from_decimal_cast()
{
  auto a        = cudf::test::fixed_point_column_wrapper<typename From::rep>{{0, 1, 2, 3, 4, 5},
                                                                             numeric::scale_type{0}};
  auto expected = cudf::test::fixed_width_column_wrapper<To>{0, 1, 2, 3, 4, 5};
  auto table    = cudf::table_view{{a}};
  auto a_ref    = cudf::ast::column_reference(0);
  auto tree     = cudf::ast::tree{};

  cudf::ast::expression const* cast = nullptr;

  if constexpr (std::is_same_v<To, bool>) {
    cast = &cudf::ast::jit::cast_to_b8(tree, a_ref);
  } else if constexpr (std::is_same_v<To, int8_t>) {
    cast = &cudf::ast::jit::cast_to_i8(tree, a_ref);
  } else if constexpr (std::is_same_v<To, int16_t>) {
    cast = &cudf::ast::jit::cast_to_i16(tree, a_ref);
  } else if constexpr (std::is_same_v<To, int32_t>) {
    cast = &cudf::ast::jit::cast_to_i32(tree, a_ref);
  } else if constexpr (std::is_same_v<To, int64_t>) {
    cast = &cudf::ast::jit::cast_to_i64(tree, a_ref);
  } else if constexpr (std::is_same_v<To, uint8_t>) {
    cast = &cudf::ast::jit::cast_to_u8(tree, a_ref);
  } else if constexpr (std::is_same_v<To, uint16_t>) {
    cast = &cudf::ast::jit::cast_to_u16(tree, a_ref);
  } else if constexpr (std::is_same_v<To, uint32_t>) {
    cast = &cudf::ast::jit::cast_to_u32(tree, a_ref);
  } else if constexpr (std::is_same_v<To, uint64_t>) {
    cast = &cudf::ast::jit::cast_to_u64(tree, a_ref);
  } else if constexpr (std::is_same_v<To, float>) {
    cast = &cudf::ast::jit::cast_to_f32(tree, a_ref);
  } else {
    static_assert(std::is_same_v<To, double>);
    cast = &cudf::ast::jit::cast_to_f64(tree, a_ref);
  }

  auto result = cudf::compute_column_jit(table, *cast);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

template <typename To>
void test_cast_to()
{
  test_cast<uint8_t, To>();
  test_cast<uint16_t, To>();
  test_cast<uint32_t, To>();
  test_cast<uint64_t, To>();
  test_cast<int8_t, To>();
  test_cast<int16_t, To>();
  test_cast<int32_t, To>();
  test_cast<int64_t, To>();
  test_cast<float, To>();
  test_cast<double, To>();
  test_from_decimal_cast<numeric::decimal32, To>();
  test_from_decimal_cast<numeric::decimal64, To>();
  test_from_decimal_cast<numeric::decimal128, To>();
}

TEST_F(JITExpressionTest, Cast)
{
  test_cast_to<bool>();
  test_cast_to<int8_t>();
  test_cast_to<int16_t>();
  test_cast_to<int32_t>();
  test_cast_to<int64_t>();
  test_cast_to<uint8_t>();
  test_cast_to<uint16_t>();
  test_cast_to<uint32_t>();
  test_cast_to<uint64_t>();
  test_cast_to<float>();
  test_cast_to<double>();
}

template <typename From, typename To>
void test_decimal_cast()
{
  auto a        = cudf::test::fixed_point_column_wrapper<typename From::rep>{{0, 1, 2, 3, 4, 5},
                                                                             numeric::scale_type{0}};
  auto expected = cudf::test::fixed_point_column_wrapper<typename To::rep>{{0, 1, 2, 3, 4, 5},
                                                                           numeric::scale_type{0}};
  auto table    = cudf::table_view{{a}};
  auto a_ref    = cudf::ast::column_reference(0);
  auto tree     = cudf::ast::tree{};

  cudf::ast::expression const* cast = nullptr;

  if constexpr (std::is_same_v<To, numeric::decimal32>) {
    cast = &cudf::ast::jit::cast_to_dec32(tree, a_ref);
  } else if constexpr (std::is_same_v<To, numeric::decimal64>) {
    cast = &cudf::ast::jit::cast_to_dec64(tree, a_ref);
  } else if constexpr (std::is_same_v<To, numeric::decimal128>) {
    static_assert(std::is_same_v<To, numeric::decimal128>);
    cast = &cudf::ast::jit::cast_to_dec128(tree, a_ref);
  }

  auto result = cudf::compute_column_jit(table, *cast);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(JITExpressionTest, CastToDec32)
{
  test_decimal_cast<numeric::decimal32, numeric::decimal32>();
  test_decimal_cast<numeric::decimal32, numeric::decimal64>();
  test_decimal_cast<numeric::decimal32, numeric::decimal128>();
}

TEST_F(JITExpressionTest, CastToDec64)
{
  test_decimal_cast<numeric::decimal64, numeric::decimal32>();
  test_decimal_cast<numeric::decimal64, numeric::decimal64>();
  test_decimal_cast<numeric::decimal64, numeric::decimal128>();
}

TEST_F(JITExpressionTest, CastToDec128)
{
  test_decimal_cast<numeric::decimal128, numeric::decimal32>();
  test_decimal_cast<numeric::decimal128, numeric::decimal64>();
  test_decimal_cast<numeric::decimal128, numeric::decimal128>();
}

TEST_F(JITExpressionTest, Rescale)
{
  auto a = cudf::test::fixed_point_column_wrapper<int32_t>{{123, 1234, 12345, 123456, 1234567},
                                                           numeric::scale_type{0}};
  auto expected = cudf::test::fixed_point_column_wrapper<int32_t>{
    {12300, 123400, 1234500, 12345600, 123456700}, numeric::scale_type{-2}};
  auto table     = cudf::table_view{{a}};
  auto a_ref     = cudf::ast::column_reference(0);
  auto tree      = cudf::ast::tree{};
  auto& rescaled = cudf::ast::jit::rescale(tree, a_ref, -2);
  auto result    = cudf::compute_column_jit(table, rescaled);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TEST_F(JITExpressionTest, AnsiFused)
{
  constexpr auto I32_MAX = std::numeric_limits<int32_t>::max();
  auto a                 = column_wrapper<int32_t>{1, 3, 20, 1, 50, 10};
  auto b                 = column_wrapper<int32_t>{1, 10, 7, 20, I32_MAX, 2};
  auto c                 = column_wrapper<int32_t>{1, 5, 4, I32_MAX, 2, 5};
  auto d                 = column_wrapper<int32_t>{0, 1, 0, 0, 1, 5};
  auto expected          = column_wrapper<int32_t>{{0, 65, 0, 0, 0, 12}, {0, 1, 0, 0, 0, 1}};
  auto table             = cudf::table_view{{a, b, c, d}};
  auto tree              = cudf::ast::tree{};
  auto a_ref             = cudf::ast::column_reference(0);
  auto b_ref             = cudf::ast::column_reference(1);
  auto c_ref             = cudf::ast::column_reference(2);
  auto d_ref             = cudf::ast::column_reference(3);
  auto& add              = cudf::ast::jit::ansi_try_add(tree, a_ref, b_ref);
  auto& mul              = cudf::ast::jit::ansi_try_mul(tree, add, c_ref);
  auto& div              = cudf::ast::jit::ansi_try_div(tree, mul, d_ref);
  auto result            = cudf::compute_column_jit(table, div);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

CUDF_TEST_PROGRAM_MAIN()
