/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

class JitFilterJoinIndicesTest : public cudf::test::BaseFixture {};

TEST_F(JitFilterJoinIndicesTest, BasicInnerJoinGreaterThan)
{
  // Create simple test tables
  // Left: {0: [1, 2, 3], 1: [10, 20, 30]}
  // Right: {0: [1, 2, 3], 1: [15, 15, 25]}
  auto left_col0  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto left_col1  = cudf::test::fixed_width_column_wrapper<int32_t>{10, 20, 30};
  auto left_table = cudf::table_view{{left_col0, left_col1}};

  auto right_col0  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto right_col1  = cudf::test::fixed_width_column_wrapper<int32_t>{15, 15, 25};
  auto right_table = cudf::table_view{{right_col0, right_col1}};

  // Simulate join indices from equality-based hash join (all pairs match on col0)
  auto left_indices_h  = std::vector<cudf::size_type>{0, 1, 2};
  auto right_indices_h = std::vector<cudf::size_type>{0, 1, 2};
  auto left_indices_d =
    cudf::detail::make_device_uvector_async(left_indices_h, cudf::get_default_stream(),
                                            cudf::get_current_device_resource_ref());
  auto right_indices_d =
    cudf::detail::make_device_uvector_async(right_indices_h, cudf::get_default_stream(),
                                            cudf::get_current_device_resource_ref());

  // JIT predicate: left.col1 > right.col1 (20 > 15, 30 > 25 should pass)
  // Predicate receives all columns from both tables in order: left cols, then right cols
  std::string predicate_code = R"(
    __device__ bool predicate(int32_t left_col0, int32_t left_col1,
                              int32_t right_col0, int32_t right_col1) {
      return left_col1 > right_col1;
    }
  )";

  // Call JIT filter join indices
  auto [filtered_left, filtered_right] = cudf::jit_filter_join_indices(
    left_table,
    right_table,
    cudf::device_span<cudf::size_type const>(left_indices_d),
    cudf::device_span<cudf::size_type const>(right_indices_d),
    predicate_code,
    cudf::join_kind::INNER_JOIN);

  // Verify results
  EXPECT_EQ(filtered_left->size(), 2);
  EXPECT_EQ(filtered_right->size(), 2);
}

TEST_F(JitFilterJoinIndicesTest, BasicLeftJoinGreaterThan)
{
  // Same tables as inner join test
  auto left_col0  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto left_col1  = cudf::test::fixed_width_column_wrapper<int32_t>{10, 20, 30};
  auto left_table = cudf::table_view{{left_col0, left_col1}};

  auto right_col0  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto right_col1  = cudf::test::fixed_width_column_wrapper<int32_t>{15, 15, 25};
  auto right_table = cudf::table_view{{right_col0, right_col1}};

  auto left_indices_h  = std::vector<cudf::size_type>{0, 1, 2};
  auto right_indices_h = std::vector<cudf::size_type>{0, 1, 2};
  auto left_indices_d =
    cudf::detail::make_device_uvector_async(left_indices_h, cudf::get_default_stream(),
                                            cudf::get_current_device_resource_ref());
  auto right_indices_d =
    cudf::detail::make_device_uvector_async(right_indices_h, cudf::get_default_stream(),
                                            cudf::get_current_device_resource_ref());

  // Predicate receives all columns from both tables in order: left cols, then right cols
  std::string predicate_code = R"(
    __device__ bool predicate(int32_t left_col0, int32_t left_col1,
                              int32_t right_col0, int32_t right_col1) {
      return left_col1 > right_col1;
    }
  )";

  // Call JIT filter join indices with LEFT_JOIN
  auto [filtered_left, filtered_right] = cudf::jit_filter_join_indices(
    left_table,
    right_table,
    cudf::device_span<cudf::size_type const>(left_indices_d),
    cudf::device_span<cudf::size_type const>(right_indices_d),
    predicate_code,
    cudf::join_kind::LEFT_JOIN);

  // Expected result: all left rows preserved
  // Row 0: 10 <= 15 -> (0, JoinNoMatch)
  // Row 1: 20 > 15 -> (1, 1)
  // Row 2: 30 > 25 -> (2, 2)
  EXPECT_EQ(filtered_left->size(), 3);
  EXPECT_EQ(filtered_right->size(), 3);
}

TEST_F(JitFilterJoinIndicesTest, EmptyInput)
{
  // Empty tables
  auto left_col    = cudf::test::fixed_width_column_wrapper<int32_t>{};
  auto right_col   = cudf::test::fixed_width_column_wrapper<int32_t>{};
  auto left_table  = cudf::table_view{{left_col}};
  auto right_table = cudf::table_view{{right_col}};

  // Empty indices
  cudf::device_span<cudf::size_type const> left_indices{};
  cudf::device_span<cudf::size_type const> right_indices{};

  std::string predicate_code = R"(
    __device__ bool predicate(int32_t left_val, int32_t right_val) {
      return left_val > right_val;
    }
  )";

  auto [filtered_left, filtered_right] = cudf::jit_filter_join_indices(
    left_table, right_table, left_indices, right_indices, predicate_code,
    cudf::join_kind::INNER_JOIN);

  // Should return empty results
  EXPECT_EQ(filtered_left->size(), 0);
  EXPECT_EQ(filtered_right->size(), 0);
}

TEST_F(JitFilterJoinIndicesTest, InvalidJoinKind)
{
  auto left_col    = cudf::test::fixed_width_column_wrapper<int32_t>{1};
  auto right_col   = cudf::test::fixed_width_column_wrapper<int32_t>{1};
  auto left_table  = cudf::table_view{{left_col}};
  auto right_table = cudf::table_view{{right_col}};

  auto left_indices_h  = std::vector<cudf::size_type>{0};
  auto right_indices_h = std::vector<cudf::size_type>{0};
  auto left_indices_d =
    cudf::detail::make_device_uvector_async(left_indices_h, cudf::get_default_stream(),
                                            cudf::get_current_device_resource_ref());
  auto right_indices_d =
    cudf::detail::make_device_uvector_async(right_indices_h, cudf::get_default_stream(),
                                            cudf::get_current_device_resource_ref());

  std::string predicate_code = R"(
    __device__ bool predicate(int32_t left_val, int32_t right_val) {
      return left_val > right_val;
    }
  )";

  // Should throw for unsupported join kinds
  EXPECT_THROW(
    cudf::jit_filter_join_indices(left_table,
                                  right_table,
                                  cudf::device_span<cudf::size_type const>(left_indices_d),
                                  cudf::device_span<cudf::size_type const>(right_indices_d),
                                  predicate_code,
                                  cudf::join_kind::LEFT_SEMI_JOIN),
    cudf::logic_error);
}
