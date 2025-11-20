/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/detail/join/join.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_uvector.hpp>

struct FilterGatherMapTest : public cudf::test::BaseFixture {
  /**
   * Helper function to convert device memory results to host vectors for comparison.
   * Similar to the pattern used in mixed join tests.
   */
  std::pair<std::vector<cudf::size_type>, std::vector<cudf::size_type>> device_results_to_host(
    const std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                    std::unique_ptr<rmm::device_uvector<cudf::size_type>>>& result)
  {
    std::vector<cudf::size_type> left_result;
    std::vector<cudf::size_type> right_result;

    for (size_t i = 0; i < result.first->size(); ++i) {
      left_result.push_back(result.first->element(i, cudf::get_default_stream()));
      right_result.push_back(result.second->element(i, cudf::get_default_stream()));
    }

    return std::make_pair(std::move(left_result), std::move(right_result));
  }

  /**
   * Common test utility function to perform filter_gather_map and verify results.
   * Similar to the pattern used in mixed join tests.
   */
  void test(cudf::table_view const& left_table,
            cudf::table_view const& right_table,
            cudf::device_span<cudf::size_type const> left_indices,
            cudf::device_span<cudf::size_type const> right_indices,
            cudf::ast::expression const& predicate,
            std::vector<cudf::size_type> const& expected_left_indices,
            std::vector<cudf::size_type> const& expected_right_indices)
  {
    auto result = cudf::filter_gather_map(left_table,
                                          right_table,
                                          left_indices,
                                          right_indices,
                                          predicate,
                                          cudf::detail::join_kind::INNER_JOIN);

    EXPECT_EQ(result.first->size(), expected_left_indices.size());
    EXPECT_EQ(result.second->size(), expected_right_indices.size());

    auto [left_result, right_result] = device_results_to_host(result);

    EXPECT_EQ(left_result, expected_left_indices);
    EXPECT_EQ(right_result, expected_right_indices);
  }
};

TEST_F(FilterGatherMapTest, BasicFilter)
{
  using TypeParam = int32_t;

  // Left Table:  {{0, 1, 2}, {3, 4, 5}}
  // Right Table: {{1, 2, 3}, {4, 6, 7}}
  // Gather Map:  left_indices = {0,1}, right_indices = {0,2}
  // Predicate:   left.col0 + right.col1 > 5
  // Result:      filtered_left_indices = {1}, filtered_right_indices = {2}

  cudf::test::fixed_width_column_wrapper<TypeParam> left_col0({0, 1, 2});
  cudf::test::fixed_width_column_wrapper<TypeParam> left_col1({3, 4, 5});
  cudf::table_view left_table({left_col0, left_col1});

  cudf::test::fixed_width_column_wrapper<TypeParam> right_col0({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<TypeParam> right_col1({4, 6, 7});
  cudf::table_view right_table({right_col0, right_col1});

  std::vector<cudf::size_type> left_host_data  = {0, 1};
  std::vector<cudf::size_type> right_host_data = {0, 2};

  auto left_indices_input = cudf::detail::make_device_uvector(
    left_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto right_indices_input = cudf::detail::make_device_uvector(
    right_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto const col_ref_left_0  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_1 = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  cudf::numeric_scalar<TypeParam> five_scalar{5, true, cudf::get_default_stream()};
  auto const five = cudf::ast::literal(five_scalar);

  auto const add_operation =
    cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_left_0, col_ref_right_1);
  auto const predicate =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, add_operation, five);

  std::vector<cudf::size_type> expected_left_indices  = {1};
  std::vector<cudf::size_type> expected_right_indices = {2};

  test(left_table,
       right_table,
       cudf::device_span<cudf::size_type const>(left_indices_input),
       cudf::device_span<cudf::size_type const>(right_indices_input),
       predicate,
       expected_left_indices,
       expected_right_indices);
}

TEST_F(FilterGatherMapTest, EmptyInput)
{
  using TypeParam = int32_t;

  cudf::test::fixed_width_column_wrapper<TypeParam> left_col0({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<TypeParam> left_col1({4, 5, 6});
  cudf::table_view left_table({left_col0, left_col1});

  cudf::test::fixed_width_column_wrapper<TypeParam> right_col0({7, 8, 9});
  cudf::test::fixed_width_column_wrapper<TypeParam> right_col1({10, 11, 12});
  cudf::table_view right_table({right_col0, right_col1});

  rmm::device_uvector<cudf::size_type> left_indices_input(0, cudf::get_default_stream());
  rmm::device_uvector<cudf::size_type> right_indices_input(0, cudf::get_default_stream());

  auto const col_ref_left_0  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);

  auto const predicate =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_left_0, col_ref_right_0);

  std::vector<cudf::size_type> expected_left_indices  = {};
  std::vector<cudf::size_type> expected_right_indices = {};

  test(left_table,
       right_table,
       cudf::device_span<cudf::size_type const>(left_indices_input),
       cudf::device_span<cudf::size_type const>(right_indices_input),
       predicate,
       expected_left_indices,
       expected_right_indices);
}

TEST_F(FilterGatherMapTest, AllFiltered)
{
  using TypeParam = int32_t;

  cudf::test::fixed_width_column_wrapper<TypeParam> left_col0({1, 2, 3});
  cudf::table_view left_table({left_col0});

  cudf::test::fixed_width_column_wrapper<TypeParam> right_col0({10, 20, 30});
  cudf::table_view right_table({right_col0});

  std::vector<cudf::size_type> left_host_data  = {0, 1, 2};
  std::vector<cudf::size_type> right_host_data = {0, 1, 2};

  auto left_indices_input = cudf::detail::make_device_uvector(
    left_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto right_indices_input = cudf::detail::make_device_uvector(
    right_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto const col_ref_left_0  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);

  // Predicate: left.col0 > right.col0 (always false: 1>10, 2>20, 3>30)
  auto const predicate =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_left_0, col_ref_right_0);

  std::vector<cudf::size_type> expected_left_indices  = {};
  std::vector<cudf::size_type> expected_right_indices = {};

  test(left_table,
       right_table,
       cudf::device_span<cudf::size_type const>(left_indices_input),
       cudf::device_span<cudf::size_type const>(right_indices_input),
       predicate,
       expected_left_indices,
       expected_right_indices);
}

TEST_F(FilterGatherMapTest, NoneFiltered)
{
  using TypeParam = int32_t;

  cudf::test::fixed_width_column_wrapper<TypeParam> left_col0({10, 20, 30});
  cudf::table_view left_table({left_col0});

  cudf::test::fixed_width_column_wrapper<TypeParam> right_col0({1, 2, 3});
  cudf::table_view right_table({right_col0});

  std::vector<cudf::size_type> left_host_data  = {0, 1, 2};
  std::vector<cudf::size_type> right_host_data = {0, 1, 2};

  auto left_indices_input = cudf::detail::make_device_uvector(
    left_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto right_indices_input = cudf::detail::make_device_uvector(
    right_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto const col_ref_left_0  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);

  // Predicate: left.col0 > right.col0 (always true: 10>1, 20>2, 30>3)
  auto const predicate =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_left_0, col_ref_right_0);

  std::vector<cudf::size_type> expected_left_indices  = {0, 1, 2};
  std::vector<cudf::size_type> expected_right_indices = {0, 1, 2};

  test(left_table,
       right_table,
       cudf::device_span<cudf::size_type const>(left_indices_input),
       cudf::device_span<cudf::size_type const>(right_indices_input),
       predicate,
       expected_left_indices,
       expected_right_indices);
}

TEST_F(FilterGatherMapTest, WithNulls)
{
  using TypeParam = int32_t;

  // Left table with nulls
  cudf::test::fixed_width_column_wrapper<TypeParam> left_col0({1, 2, 3, 4}, {1, 0, 1, 1});
  cudf::table_view left_table({left_col0});

  // Right table with nulls
  cudf::test::fixed_width_column_wrapper<TypeParam> right_col0({10, 20, 30, 40}, {1, 1, 0, 1});
  cudf::table_view right_table({right_col0});

  std::vector<cudf::size_type> left_host_data  = {0, 1, 2, 3};
  std::vector<cudf::size_type> right_host_data = {0, 1, 2, 3};

  auto left_indices_input = cudf::detail::make_device_uvector(
    left_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto right_indices_input = cudf::detail::make_device_uvector(
    right_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto const col_ref_left_0  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);

  // Predicate: left.col0 < right.col0
  // Expected: (1 < 10) = true, (null < 20) = null/false, (3 < null) = null/false, (4 < 40) = true
  auto const predicate =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_left_0, col_ref_right_0);

  std::vector<cudf::size_type> expected_left_indices  = {0, 3};
  std::vector<cudf::size_type> expected_right_indices = {0, 3};

  test(left_table,
       right_table,
       cudf::device_span<cudf::size_type const>(left_indices_input),
       cudf::device_span<cudf::size_type const>(right_indices_input),
       predicate,
       expected_left_indices,
       expected_right_indices);
}

TEST_F(FilterGatherMapTest, ComplexPredicate)
{
  using TypeParam = int32_t;

  // Left Table:  {{1, 2, 3}, {10, 20, 30}}
  cudf::test::fixed_width_column_wrapper<TypeParam> left_col0({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<TypeParam> left_col1({10, 20, 30});
  cudf::table_view left_table({left_col0, left_col1});

  // Right Table: {{5, 6, 7}, {15, 25, 35}}
  cudf::test::fixed_width_column_wrapper<TypeParam> right_col0({5, 6, 7});
  cudf::test::fixed_width_column_wrapper<TypeParam> right_col1({15, 25, 35});
  cudf::table_view right_table({right_col0, right_col1});

  std::vector<cudf::size_type> left_host_data  = {0, 1, 2};
  std::vector<cudf::size_type> right_host_data = {0, 1, 2};

  auto left_indices_input = cudf::detail::make_device_uvector(
    left_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto right_indices_input = cudf::detail::make_device_uvector(
    right_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto const col_ref_left_0  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const col_ref_left_1  = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto const col_ref_right_1 = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);

  // Predicate: (left.col0 + left.col1) > (right.col0 + right.col1)
  // (1+10) > (5+15) = 11 > 20 = false
  // (2+20) > (6+25) = 22 > 31 = false
  // (3+30) > (7+35) = 33 > 42 = false
  auto const left_sum =
    cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_left_0, col_ref_left_1);
  auto const right_sum =
    cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_right_0, col_ref_right_1);
  auto const predicate =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_sum, right_sum);

  // All should be filtered out
  std::vector<cudf::size_type> expected_left_indices  = {};
  std::vector<cudf::size_type> expected_right_indices = {};

  test(left_table,
       right_table,
       cudf::device_span<cudf::size_type const>(left_indices_input),
       cudf::device_span<cudf::size_type const>(right_indices_input),
       predicate,
       expected_left_indices,
       expected_right_indices);
}

TEST_F(FilterGatherMapTest, StringColumns)
{
  // Left table with strings
  cudf::test::strings_column_wrapper left_col0({"apple", "banana", "cherry"});
  cudf::test::fixed_width_column_wrapper<int32_t> left_col1({1, 2, 3});
  cudf::table_view left_table({left_col0, left_col1});

  // Right table with strings
  cudf::test::strings_column_wrapper right_col0({"apricot", "blueberry", "coconut"});
  cudf::test::fixed_width_column_wrapper<int32_t> right_col1({10, 20, 30});
  cudf::table_view right_table({right_col0, right_col1});

  std::vector<cudf::size_type> left_host_data  = {0, 1, 2};
  std::vector<cudf::size_type> right_host_data = {0, 1, 2};

  auto left_indices_input = cudf::detail::make_device_uvector(
    left_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto right_indices_input = cudf::detail::make_device_uvector(
    right_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto const col_ref_left_1  = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_1 = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);

  // Predicate: left.col1 * 10 == right.col1
  // (1 * 10) == 10 = true
  // (2 * 10) == 20 = true
  // (3 * 10) == 30 = true
  cudf::numeric_scalar<int32_t> ten_scalar{10, true, cudf::get_default_stream()};
  auto const ten = cudf::ast::literal(ten_scalar);
  auto const left_times_ten =
    cudf::ast::operation(cudf::ast::ast_operator::MUL, col_ref_left_1, ten);
  auto const predicate =
    cudf::ast::operation(cudf::ast::ast_operator::EQUAL, left_times_ten, col_ref_right_1);

  std::vector<cudf::size_type> expected_left_indices  = {0, 1, 2};
  std::vector<cudf::size_type> expected_right_indices = {0, 1, 2};

  test(left_table,
       right_table,
       cudf::device_span<cudf::size_type const>(left_indices_input),
       cudf::device_span<cudf::size_type const>(right_indices_input),
       predicate,
       expected_left_indices,
       expected_right_indices);
}

template <typename T>
struct FilterGatherMapNumericTest : public FilterGatherMapTest {};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_SUITE(FilterGatherMapNumericTest, NumericTypesNotBool);

TYPED_TEST(FilterGatherMapNumericTest, NumericTypes)
{
  using TypeParam = TypeParam;

  cudf::test::fixed_width_column_wrapper<TypeParam> left_col0({1, 2, 3, 4, 5});
  cudf::table_view left_table({left_col0});

  cudf::test::fixed_width_column_wrapper<TypeParam> right_col0({3, 3, 3, 3, 3});
  cudf::table_view right_table({right_col0});

  std::vector<cudf::size_type> left_host_data  = {0, 1, 2, 3, 4};
  std::vector<cudf::size_type> right_host_data = {0, 1, 2, 3, 4};

  auto left_indices_input = cudf::detail::make_device_uvector(
    left_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto right_indices_input = cudf::detail::make_device_uvector(
    right_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto const col_ref_left_0  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);

  // Predicate: left.col0 >= right.col0
  // 1 >= 3 = false, 2 >= 3 = false, 3 >= 3 = true, 4 >= 3 = true, 5 >= 3 = true
  auto const predicate =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_left_0, col_ref_right_0);

  std::vector<cudf::size_type> expected_left_indices  = {2, 3, 4};
  std::vector<cudf::size_type> expected_right_indices = {2, 3, 4};

  this->test(left_table,
             right_table,
             cudf::device_span<cudf::size_type const>(left_indices_input),
             cudf::device_span<cudf::size_type const>(right_indices_input),
             predicate,
             expected_left_indices,
             expected_right_indices);
}

TEST_F(FilterGatherMapTest, MismatchedSizes)
{
  using TypeParam = int32_t;

  cudf::test::fixed_width_column_wrapper<TypeParam> left_col0({1, 2, 3});
  cudf::table_view left_table({left_col0});

  cudf::test::fixed_width_column_wrapper<TypeParam> right_col0({4, 5, 6});
  cudf::table_view right_table({right_col0});

  // Mismatched sizes: left has 2 indices, right has 3 indices
  std::vector<cudf::size_type> left_host_data  = {0, 1};
  std::vector<cudf::size_type> right_host_data = {0, 1, 2};

  auto left_indices_input = cudf::detail::make_device_uvector(
    left_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto right_indices_input = cudf::detail::make_device_uvector(
    right_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto const col_ref_left_0  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);

  auto const predicate =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_left_0, col_ref_right_0);

  // Should throw std::invalid_argument due to mismatched sizes
  EXPECT_THROW(
    cudf::filter_gather_map(left_table,
                            right_table,
                            cudf::device_span<cudf::size_type const>(left_indices_input),
                            cudf::device_span<cudf::size_type const>(right_indices_input),
                            predicate,
                            cudf::detail::join_kind::INNER_JOIN),
    std::invalid_argument);
}

TEST_F(FilterGatherMapTest, NullIndices)
{
  using TypeParam = int32_t;

  cudf::test::fixed_width_column_wrapper<TypeParam> left_col0({1, 2, 3});
  cudf::table_view left_table({left_col0});

  cudf::test::fixed_width_column_wrapper<TypeParam> right_col0({1, 4, 5});
  cudf::table_view right_table({right_col0});

  // Create index arrays with null sentinels (simulating outer join results)
  constexpr cudf::size_type null_sentinel      = std::numeric_limits<cudf::size_type>::min();
  std::vector<cudf::size_type> left_host_data  = {0, 1, null_sentinel};
  std::vector<cudf::size_type> right_host_data = {0, null_sentinel, 1};

  auto left_indices_input = cudf::detail::make_device_uvector(
    left_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto right_indices_input = cudf::detail::make_device_uvector(
    right_host_data, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto const col_ref_left_0  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const col_ref_right_0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);

  // Predicate: left.col0 == right.col0
  auto const predicate =
    cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_left_0, col_ref_right_0);

  auto result =
    cudf::filter_gather_map(left_table,
                            right_table,
                            cudf::device_span<cudf::size_type const>(left_indices_input),
                            cudf::device_span<cudf::size_type const>(right_indices_input),
                            predicate,
                            cudf::detail::join_kind::INNER_JOIN);

  ASSERT_NE(result.first, nullptr) << "Left result is null";
  ASSERT_NE(result.second, nullptr) << "Right result is null";

  // Only the first pair (0,0) should pass: left[0]=1, right[0]=1, 1==1 is true
  // The other pairs have null indices and should be filtered out (can't evaluate predicates on
  // them)
  EXPECT_EQ(result.first->size(), 1);
  EXPECT_EQ(result.second->size(), 1);

  auto [left_result, right_result] = device_results_to_host(result);

  std::vector<cudf::size_type> expected_left_indices  = {0};
  std::vector<cudf::size_type> expected_right_indices = {0};

  EXPECT_EQ(left_result, expected_left_indices);
  EXPECT_EQ(right_result, expected_right_indices);
}
