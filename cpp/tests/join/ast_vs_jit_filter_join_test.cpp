/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/join/join.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

/**
 * @brief Test fixture for comparing AST and JIT implementations
 * 
 * This test ensures both implementations produce identical results
 * for various join scenarios and predicate types.
 */
class AstVsJitFilterJoinTest : public cudf::test::BaseFixture {};

TEST_F(AstVsJitFilterJoinTest, SimpleGreaterThanComparison)
{
  // Create test tables
  auto left_col0 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4};
  auto left_col1 = cudf::test::fixed_width_column_wrapper<int32_t>{10, 20, 30, 40};
  auto left_table = cudf::table_view{{left_col0, left_col1}};

  auto right_col0 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4};
  auto right_col1 = cudf::test::fixed_width_column_wrapper<int32_t>{15, 15, 25, 35};
  auto right_table = cudf::table_view{{right_col0, right_col1}};

  // Simulate join indices from equality-based join
  auto left_indices_vec = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2, 3};
  auto right_indices_vec = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2, 3};

  cudf::device_span<cudf::size_type const> left_indices{left_indices_vec.data(), left_indices_vec.size()};
  cudf::device_span<cudf::size_type const> right_indices{right_indices_vec.data(), right_indices_vec.size()};

  // Create AST predicate: left.col1 > right.col1
  auto left_col_ref = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
  auto right_col_ref = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto ast_predicate = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_col_ref, right_col_ref);

  // Create JIT predicate equivalent
  std::string jit_predicate = R"(
    __device__ bool predicate(int32_t left_val, int32_t right_val) {
      return left_val > right_val;
    }
  )";

  // Test all supported join types
  std::vector<cudf::join_kind> join_kinds = {
    cudf::join_kind::INNER_JOIN,
    cudf::join_kind::LEFT_JOIN,
    cudf::join_kind::FULL_JOIN
  };

  for (auto join_kind : join_kinds) {
    // Get AST results
    auto [ast_left, ast_right] = cudf::filter_join_indices(
      left_table, right_table, left_indices, right_indices, ast_predicate, join_kind);

    // Get JIT results  
    auto [jit_left, jit_right] = cudf::jit_filter_join_indices(
      left_table, right_table, left_indices, right_indices, jit_predicate, join_kind);

    // Compare results
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ast_left, *jit_left);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ast_right, *jit_right);
    
    EXPECT_EQ(ast_left->size(), jit_left->size()) << "Size mismatch for join_kind: " << static_cast<int>(join_kind);
    EXPECT_EQ(ast_right->size(), jit_right->size()) << "Size mismatch for join_kind: " << static_cast<int>(join_kind);
  }
}

TEST_F(AstVsJitFilterJoinTest, FloatingPointComparison)
{
  // Test with floating point data
  auto left_col0 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto left_col1 = cudf::test::fixed_width_column_wrapper<double>{1.5, 2.7, 3.1};
  auto left_table = cudf::table_view{{left_col0, left_col1}};

  auto right_col0 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto right_col1 = cudf::test::fixed_width_column_wrapper<double>{1.2, 2.8, 3.0};
  auto right_table = cudf::table_view{{right_col0, right_col1}};

  auto left_indices_vec = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2};
  auto right_indices_vec = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2};

  cudf::device_span<cudf::size_type const> left_indices{left_indices_vec.data(), left_indices_vec.size()};
  cudf::device_span<cudf::size_type const> right_indices{right_indices_vec.data(), right_indices_vec.size()};

  // AST predicate: left.col1 >= right.col1
  auto left_col_ref = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
  auto right_col_ref = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto ast_predicate = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, left_col_ref, right_col_ref);

  // JIT predicate equivalent
  std::string jit_predicate = R"(
    __device__ bool predicate(double left_val, double right_val) {
      return left_val >= right_val;
    }
  )";

  // Test INNER_JOIN
  auto [ast_left, ast_right] = cudf::filter_join_indices(
    left_table, right_table, left_indices, right_indices, ast_predicate, cudf::join_kind::INNER_JOIN);

  auto [jit_left, jit_right] = cudf::jit_filter_join_indices(
    left_table, right_table, left_indices, right_indices, jit_predicate, cudf::join_kind::INNER_JOIN);

  // Compare results
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ast_left, *jit_left);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ast_right, *jit_right);
}

TEST_F(AstVsJitFilterJoinTest, ComplexPredicateMultipleColumns)
{
  // Test with predicate involving multiple columns
  auto left_col0 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4};
  auto left_col1 = cudf::test::fixed_width_column_wrapper<int32_t>{10, 20, 30, 40};
  auto left_col2 = cudf::test::fixed_width_column_wrapper<int32_t>{100, 200, 300, 400};
  auto left_table = cudf::table_view{{left_col0, left_col1, left_col2}};

  auto right_col0 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4};
  auto right_col1 = cudf::test::fixed_width_column_wrapper<int32_t>{15, 15, 25, 35};
  auto right_col2 = cudf::test::fixed_width_column_wrapper<int32_t>{150, 150, 250, 350};
  auto right_table = cudf::table_view{{right_col0, right_col1, right_col2}};

  auto left_indices_vec = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2, 3};
  auto right_indices_vec = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2, 3};

  cudf::device_span<cudf::size_type const> left_indices{left_indices_vec.data(), left_indices_vec.size()};
  cudf::device_span<cudf::size_type const> right_indices{right_indices_vec.data(), right_indices_vec.size()};

  // AST predicate: (left.col1 > right.col1) AND (left.col2 > right.col2)
  auto left_col1_ref = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
  auto right_col1_ref = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto left_col2_ref = cudf::ast::column_reference(2, cudf::ast::table_reference::LEFT);
  auto right_col2_ref = cudf::ast::column_reference(2, cudf::ast::table_reference::RIGHT);
  
  auto cmp1 = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_col1_ref, right_col1_ref);
  auto cmp2 = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_col2_ref, right_col2_ref);
  auto ast_predicate = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, cmp1, cmp2);

  // JIT predicate equivalent (columns are passed in table order: left cols then right cols)
  std::string jit_predicate = R"(
    __device__ bool predicate(int32_t left_col0, int32_t left_col1, int32_t left_col2,
                              int32_t right_col0, int32_t right_col1, int32_t right_col2) {
      return (left_col1 > right_col1) && (left_col2 > right_col2);
    }
  )";

  // Test INNER_JOIN
  auto [ast_left, ast_right] = cudf::filter_join_indices(
    left_table, right_table, left_indices, right_indices, ast_predicate, cudf::join_kind::INNER_JOIN);

  auto [jit_left, jit_right] = cudf::jit_filter_join_indices(
    left_table, right_table, left_indices, right_indices, jit_predicate, cudf::join_kind::INNER_JOIN);

  // Compare results
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ast_left, *jit_left);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ast_right, *jit_right);
}

TEST_F(AstVsJitFilterJoinTest, EdgeCaseEmptyResults)
{
  // Test case where predicate filters out all results
  auto left_col0 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto left_col1 = cudf::test::fixed_width_column_wrapper<int32_t>{10, 20, 30};
  auto left_table = cudf::table_view{{left_col0, left_col1}};

  auto right_col0 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto right_col1 = cudf::test::fixed_width_column_wrapper<int32_t>{50, 60, 70}; // All larger than left
  auto right_table = cudf::table_view{{right_col0, right_col1}};

  auto left_indices_vec = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2};
  auto right_indices_vec = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2};

  cudf::device_span<cudf::size_type const> left_indices{left_indices_vec.data(), left_indices_vec.size()};
  cudf::device_span<cudf::size_type const> right_indices{right_indices_vec.data(), right_indices_vec.size()};

  // Predicate: left.col1 > right.col1 (will be false for all)
  auto left_col_ref = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
  auto right_col_ref = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto ast_predicate = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_col_ref, right_col_ref);

  std::string jit_predicate = R"(
    __device__ bool predicate(int32_t left_val, int32_t right_val) {
      return left_val > right_val;
    }
  )";

  // Test INNER_JOIN (should produce empty results)
  auto [ast_left, ast_right] = cudf::filter_join_indices(
    left_table, right_table, left_indices, right_indices, ast_predicate, cudf::join_kind::INNER_JOIN);

  auto [jit_left, jit_right] = cudf::jit_filter_join_indices(
    left_table, right_table, left_indices, right_indices, jit_predicate, cudf::join_kind::INNER_JOIN);

  // Both should be empty
  EXPECT_EQ(ast_left->size(), 0);
  EXPECT_EQ(ast_right->size(), 0);
  EXPECT_EQ(jit_left->size(), 0);
  EXPECT_EQ(jit_right->size(), 0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ast_left, *jit_left);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ast_right, *jit_right);
}
