/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <vector>

using namespace cudf::test;
using namespace cudf::experimental::row;

// Compares two tables row by row, if table1 row is less than table2, then corresponding row value
// in `output` would be `true`/1 else `false`/0.
template <typename T>
struct TypedTableViewTest : public cudf::test::BaseFixture {
};

using NumericTypesNotBool = Concat<IntegralTypesNotBool, FloatingPointTypes>;
TYPED_TEST_SUITE(TypedTableViewTest, NumericTypesNotBool);

template <typename Comparator>
void row_comparison(cudf::table_view input1,
                    cudf::table_view input2,
                    cudf::mutable_column_view output,
                    std::vector<cudf::order> const& column_order,
                    Comparator c)
{
  rmm::cuda_stream_view stream{};

  auto table_comparator =
    lexicographic::two_table_comparator{input1, input2, column_order, {}, stream};
  auto comparator   = table_comparator.less(cudf::nullate::NO{}, c);
  auto const lhs_it = cudf::experimental::row::lhs_iterator(0);
  auto const rhs_it = cudf::experimental::row::rhs_iterator(0);
  thrust::transform(rmm::exec_policy(stream),
                    lhs_it,
                    lhs_it + input1.num_rows(),
                    rhs_it,
                    output.data<int8_t>(),
                    comparator);
}

template <typename Comparator>
void self_comparison(cudf::table_view input,
                     cudf::mutable_column_view output,
                     std::vector<cudf::order> const& column_order,
                     Comparator c)
{
  rmm::cuda_stream_view stream{};

  auto table_comparator = lexicographic::self_comparator{input, column_order, {}, stream};
  auto comparator       = table_comparator.less(cudf::nullate::NO{}, c);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(input.num_rows()),
                    thrust::make_counting_iterator(0),
                    output.data<int8_t>(),
                    comparator);
}

template <typename Comparator>
void row_equality(cudf::table_view input1,
                  cudf::table_view input2,
                  cudf::mutable_column_view output,
                  std::vector<cudf::order> const& column_order,
                  Comparator c)
{
  rmm::cuda_stream_view stream{};

  auto table_comparator = equality::two_table_comparator{input1, input2, stream};
  auto comparator   = table_comparator.equal_to(cudf::nullate::NO{}, cudf::null_equality::EQUAL, c);
  auto const lhs_it = cudf::experimental::row::lhs_iterator(0);
  auto const rhs_it = cudf::experimental::row::rhs_iterator(0);
  thrust::transform(rmm::exec_policy(stream),
                    lhs_it,
                    lhs_it + input1.num_rows(),
                    rhs_it,
                    output.data<bool>(),
                    comparator);
}

template <typename Comparator>
void self_equality(cudf::table_view input,
                   cudf::mutable_column_view output,
                   std::vector<cudf::order> const& column_order,
                   Comparator c)
{
  rmm::cuda_stream_view stream{};

  auto table_comparator = equality::self_comparator{input, stream};
  auto comparator = table_comparator.equal_to(cudf::nullate::NO{}, cudf::null_equality::EQUAL, c);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(input.num_rows()),
                    thrust::make_counting_iterator(0),
                    output.data<bool>(),
                    comparator);
}

TYPED_TEST(TypedTableViewTest, EmptyColumnedTable)
{
  std::vector<cudf::column_view> cols{};

  cudf::table_view input(cols);
  cudf::size_type expected = 0;

  EXPECT_EQ(input.num_columns(), expected);
}

TYPED_TEST(TypedTableViewTest, TestLexicographicalComparatorTwoTableCase)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> col1{{1, 2, 3, 4}};
  fixed_width_column_wrapper<T> col2{{0, 1, 4, 3}};
  std::vector<cudf::order> column_order{cudf::order::DESCENDING};

  cudf::table_view input_table_1{{col1}};
  cudf::table_view input_table_2{{col2}};

  auto got = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::INT8), input_table_1.num_rows(), cudf::mask_state::UNALLOCATED);
  fixed_width_column_wrapper<int8_t> expected{{1, 1, 0, 1}};

  row_comparison(input_table_1,
                 input_table_2,
                 got->mutable_view(),
                 column_order,
                 lexicographic::physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  row_comparison(input_table_1,
                 input_table_2,
                 got->mutable_view(),
                 column_order,
                 lexicographic::sorting_physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(TypedTableViewTest, TestLexicographicalComparatorSameTable)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> col1{{1, 2, 3, 4}};
  std::vector<cudf::order> column_order{cudf::order::DESCENDING};

  cudf::table_view input_table_1{{col1}};

  auto got = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::INT8), input_table_1.num_rows(), cudf::mask_state::UNALLOCATED);
  fixed_width_column_wrapper<int8_t> expected{{0, 0, 0, 0}};

  self_comparison(
    input_table_1, got->mutable_view(), column_order, lexicographic::physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  self_comparison(input_table_1,
                  got->mutable_view(),
                  column_order,
                  lexicographic::sorting_physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(TypedTableViewTest, Select)
{
  using T = TypeParam;

  fixed_width_column_wrapper<int8_t> col1{{1, 2, 3, 4}};
  fixed_width_column_wrapper<T> col2{{1, 2, 3, 4}};
  fixed_width_column_wrapper<int32_t> col3{{4, 5, 6, 7}};
  fixed_width_column_wrapper<T> col4{{4, 5, 6, 7}};
  cudf::table_view t{{col1, col2, col3, col4}};

  cudf::table_view selected = t.select({2, 3});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(t.column(2), selected.column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(t.column(3), selected.column(1));
}

TYPED_TEST(TypedTableViewTest, SelectOutOfBounds)
{
  using T = TypeParam;

  fixed_width_column_wrapper<int8_t> col1{{1, 2, 3, 4}};
  fixed_width_column_wrapper<T> col2{{1, 2, 3, 4}};
  fixed_width_column_wrapper<int32_t> col3{{4, 5, 6, 7}};
  fixed_width_column_wrapper<T> col4{{4, 5, 6, 7}};
  cudf::table_view t{{col1, col2}};

  EXPECT_THROW((void)t.select({2, 3, 4}), std::out_of_range);
}

TYPED_TEST(TypedTableViewTest, SelectNoColumns)
{
  using T = TypeParam;

  fixed_width_column_wrapper<int8_t> col1{{1, 2, 3, 4}};
  fixed_width_column_wrapper<T> col2{{1, 2, 3, 4}};
  fixed_width_column_wrapper<int32_t> col3{{4, 5, 6, 7}};
  fixed_width_column_wrapper<T> col4{{4, 5, 6, 7}};
  cudf::table_view t{{col1, col2, col3, col4}};

  cudf::table_view selected = t.select({});
  EXPECT_EQ(selected.num_columns(), 0);
}

template <typename T>
struct NaNTableViewTest : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(NaNTableViewTest, FloatingPointTypes);

TYPED_TEST(NaNTableViewTest, TestLexicographicalComparatorTwoTableNaNCase)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col1{{T(NAN), T(NAN), T(1), T(1)}};
  cudf::test::fixed_width_column_wrapper<T> col2{{T(NAN), T(1), T(NAN), T(1)}};
  std::vector<cudf::order> column_order{cudf::order::DESCENDING};

  cudf::table_view input_table_1{{col1}};
  cudf::table_view input_table_2{{col2}};

  auto got = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::INT8), input_table_1.num_rows(), cudf::mask_state::UNALLOCATED);

  cudf::test::fixed_width_column_wrapper<int8_t> expected{{0, 0, 0, 0}};
  row_comparison(input_table_1,
                 input_table_2,
                 got->mutable_view(),
                 column_order,
                 lexicographic::physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  cudf::test::fixed_width_column_wrapper<int8_t> sorting_expected{{0, 1, 0, 0}};
  row_comparison(input_table_1,
                 input_table_2,
                 got->mutable_view(),
                 column_order,
                 lexicographic::sorting_physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorting_expected, got->view());
}

TYPED_TEST(NaNTableViewTest, TestEqualityComparatorTwoTableNaNCase)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col1{{T(NAN), T(NAN), T(1), T(1)}};
  cudf::test::fixed_width_column_wrapper<T> col2{{T(NAN), T(1), T(NAN), T(1)}};
  std::vector<cudf::order> column_order{cudf::order::DESCENDING};

  cudf::table_view input_table_1{{col1}};
  cudf::table_view input_table_2{{col2}};

  auto got = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::INT8), input_table_1.num_rows(), cudf::mask_state::UNALLOCATED);

  cudf::test::fixed_width_column_wrapper<int8_t> expected{{0, 0, 0, 1}};
  row_equality(input_table_1,
               input_table_2,
               got->mutable_view(),
               column_order,
               equality::physical_equality_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  cudf::test::fixed_width_column_wrapper<int8_t> sorting_expected{{1, 0, 0, 1}};
  row_equality(input_table_1,
               input_table_2,
               got->mutable_view(),
               column_order,
               equality::nan_equal_physical_equality_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorting_expected, got->view());
}
