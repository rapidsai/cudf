/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <vector>

// Compares two tables row by row, if table1 row is less than table2, then corresponding row value
// in `output` would be `true`/1 else `false`/0.
struct TableViewTest : public cudf::test::BaseFixture {};
void row_comparison(cudf::table_view input1,
                    cudf::table_view input2,
                    cudf::mutable_column_view output,
                    std::vector<cudf::order> const& column_order)
{
  rmm::cuda_stream_view stream{cudf::get_default_stream()};

  auto const comparator = cudf::detail::row::lexicographic::two_table_comparator{
    input1, input2, column_order, {}, stream};
  auto const lhs_it = cudf::detail::row::lhs_iterator(0);
  auto const rhs_it = cudf::detail::row::rhs_iterator(0);

  thrust::transform(rmm::exec_policy(stream),
                    lhs_it,
                    lhs_it + input1.num_rows(),
                    rhs_it,
                    output.data<int8_t>(),
                    comparator.less<false>(cudf::nullate::NO{}));
}

TEST_F(TableViewTest, EmptyColumnedTable)
{
  std::vector<cudf::column_view> cols{};

  cudf::table_view input(cols);
  cudf::size_type expected = 0;

  EXPECT_EQ(input.num_columns(), expected);
}

TEST_F(TableViewTest, TestLexicographicalComparatorTwoTableCase)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<int16_t> col2{{0, 1, 4, 3}};
  std::vector<cudf::order> column_order{cudf::order::DESCENDING};

  cudf::table_view input_table_1{{col1}};
  cudf::table_view input_table_2{{col2}};

  auto got = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::INT8), input_table_1.num_rows(), cudf::mask_state::UNALLOCATED);
  cudf::test::fixed_width_column_wrapper<int8_t> expected{{1, 1, 0, 1}};
  row_comparison(input_table_1, input_table_2, got->mutable_view(), column_order);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TEST_F(TableViewTest, TestLexicographicalComparatorSameTable)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{1, 2, 3, 4}};
  std::vector<cudf::order> column_order{cudf::order::DESCENDING};

  cudf::table_view input_table_1{{col1}};

  auto got = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::INT8), input_table_1.num_rows(), cudf::mask_state::UNALLOCATED);
  cudf::test::fixed_width_column_wrapper<int8_t> expected{{0, 0, 0, 0}};
  row_comparison(input_table_1, input_table_1, got->mutable_view(), column_order);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TEST_F(TableViewTest, Select)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<int8_t> col1{{1, 2, 3, 4}};
  fixed_width_column_wrapper<int16_t> col2{{1, 2, 3, 4}};
  fixed_width_column_wrapper<int32_t> col3{{4, 5, 6, 7}};
  fixed_width_column_wrapper<int64_t> col4{{4, 5, 6, 7}};
  cudf::table_view t{{col1, col2, col3, col4}};

  cudf::table_view selected = t.select({2, 3});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(t.column(2), selected.column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(t.column(3), selected.column(1));
}

TEST_F(TableViewTest, SelectOutOfBounds)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<int8_t> col1{{1, 2, 3, 4}};
  fixed_width_column_wrapper<int16_t> col2{{1, 2, 3, 4}};
  fixed_width_column_wrapper<int32_t> col3{{4, 5, 6, 7}};
  fixed_width_column_wrapper<int64_t> col4{{4, 5, 6, 7}};
  cudf::table_view t{{col1, col2}};

  EXPECT_THROW((void)t.select({2, 3, 4}), std::out_of_range);
}

TEST_F(TableViewTest, SelectNoColumns)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<int8_t> col1{{1, 2, 3, 4}};
  fixed_width_column_wrapper<int16_t> col2{{1, 2, 3, 4}};
  fixed_width_column_wrapper<int32_t> col3{{4, 5, 6, 7}};
  fixed_width_column_wrapper<int64_t> col4{{4, 5, 6, 7}};
  cudf::table_view t{{col1, col2, col3, col4}};

  cudf::table_view selected = t.select({});
  EXPECT_EQ(selected.num_columns(), 0);
}
