/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_lists.hpp>

#include <vector>

// Compares two tables row by row, if table1 row is less than table2, then corresponding row value
// in `ouput` would be `true`/1 else `false`/0.
struct TableViewTest : public cudf::test::BaseFixture {
};
void row_comparison(cudf::table_view input1,
                    cudf::table_view input2,
                    cudf::mutable_column_view output,
                    std::vector<cudf::order> const& column_order)
{
  cudaStream_t stream = 0;

  auto device_table_1 = cudf::table_device_view::create(input1, stream);
  auto device_table_2 = cudf::table_device_view::create(input2, stream);
  rmm::device_vector<cudf::order> d_column_order(column_order);

  auto comparator = cudf::row_lexicographic_comparator<false>(
    *device_table_1, *device_table_2, d_column_order.data().get());

  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(input1.num_rows()),
                    thrust::make_counting_iterator(0),
                    output.data<int8_t>(),
                    comparator);
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

  cudf::test::expect_columns_equal(expected, got->view());
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

  cudf::test::expect_columns_equal(expected, got->view());
}

TEST_F(TableViewTest, Select)
{
  using cudf::test::expect_columns_equal;
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<int8_t> col1{{1, 2, 3, 4}};
  fixed_width_column_wrapper<int16_t> col2{{1, 2, 3, 4}};
  fixed_width_column_wrapper<int32_t> col3{{4, 5, 6, 7}};
  fixed_width_column_wrapper<int64_t> col4{{4, 5, 6, 7}};
  cudf::table_view t{{col1, col2, col3, col4}};

  cudf::table_view selected = t.select({2, 3});
  expect_columns_equal(t.column(2), selected.column(0));
  expect_columns_equal(t.column(3), selected.column(1));
}

TEST_F(TableViewTest, SelectOutOfBounds)
{
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<int8_t> col1{{1, 2, 3, 4}};
  fixed_width_column_wrapper<int16_t> col2{{1, 2, 3, 4}};
  fixed_width_column_wrapper<int32_t> col3{{4, 5, 6, 7}};
  fixed_width_column_wrapper<int64_t> col4{{4, 5, 6, 7}};
  cudf::table_view t{{col1, col2}};

  EXPECT_THROW(t.select({2, 3, 4}), std::out_of_range);
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
