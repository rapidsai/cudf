/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cmath>
#include <vector>

template <typename T>
struct TypedTableViewTest : public cudf::test::BaseFixture {};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(TypedTableViewTest, NumericTypesNotBool);

template <typename PhysicalElementComparator>
auto self_comparison(cudf::table_view input,
                     std::vector<cudf::order> const& column_order,
                     PhysicalElementComparator comparator)
{
  rmm::cuda_stream_view stream{cudf::get_default_stream()};

  auto const table_comparator =
    cudf::experimental::row::lexicographic::self_comparator{input, column_order, {}, stream};

  auto output = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::BOOL8), input.num_rows(), cudf::mask_state::UNALLOCATED);

  if (cudf::detail::has_nested_columns(input)) {
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(input.num_rows()),
                      thrust::make_counting_iterator(0),
                      output->mutable_view().data<bool>(),
                      table_comparator.less<true>(cudf::nullate::NO{}, comparator));
  } else {
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(input.num_rows()),
                      thrust::make_counting_iterator(0),
                      output->mutable_view().data<bool>(),
                      table_comparator.less<false>(cudf::nullate::NO{}, comparator));
  }
  return output;
}

template <typename PhysicalElementComparator>
auto two_table_comparison(cudf::table_view lhs,
                          cudf::table_view rhs,
                          std::vector<cudf::order> const& column_order,
                          PhysicalElementComparator comparator)
{
  rmm::cuda_stream_view stream{cudf::get_default_stream()};

  auto const table_comparator = cudf::experimental::row::lexicographic::two_table_comparator{
    lhs, rhs, column_order, {}, stream};
  auto const lhs_it = cudf::experimental::row::lhs_iterator(0);
  auto const rhs_it = cudf::experimental::row::rhs_iterator(0);

  auto output = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::BOOL8), lhs.num_rows(), cudf::mask_state::UNALLOCATED);

  if (cudf::detail::has_nested_columns(lhs) || cudf::detail::has_nested_columns(rhs)) {
    thrust::transform(rmm::exec_policy(stream),
                      lhs_it,
                      lhs_it + lhs.num_rows(),
                      rhs_it,
                      output->mutable_view().data<bool>(),
                      table_comparator.less<true>(cudf::nullate::NO{}, comparator));
  } else {
    thrust::transform(rmm::exec_policy(stream),
                      lhs_it,
                      lhs_it + lhs.num_rows(),
                      rhs_it,
                      output->mutable_view().data<bool>(),
                      table_comparator.less<false>(cudf::nullate::NO{}, comparator));
  }
  return output;
}

template <typename PhysicalElementComparator>
auto self_equality(cudf::table_view input,
                   std::vector<cudf::order> const& column_order,
                   PhysicalElementComparator comparator)
{
  rmm::cuda_stream_view stream{cudf::get_default_stream()};

  auto const table_comparator = cudf::experimental::row::equality::self_comparator{input, stream};

  auto output = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::BOOL8), input.num_rows(), cudf::mask_state::UNALLOCATED);

  if (cudf::detail::has_nested_columns(input)) {
    auto const equal_comparator =
      table_comparator.equal_to<true>(cudf::nullate::NO{}, cudf::null_equality::EQUAL, comparator);

    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(input.num_rows()),
                      thrust::make_counting_iterator(0),
                      output->mutable_view().data<bool>(),
                      equal_comparator);
  } else {
    auto const equal_comparator =
      table_comparator.equal_to<false>(cudf::nullate::NO{}, cudf::null_equality::EQUAL, comparator);

    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(input.num_rows()),
                      thrust::make_counting_iterator(0),
                      output->mutable_view().data<bool>(),
                      equal_comparator);
  }

  return output;
}

template <typename PhysicalElementComparator>
auto two_table_equality(cudf::table_view lhs,
                        cudf::table_view rhs,
                        std::vector<cudf::order> const& column_order,
                        PhysicalElementComparator comparator)
{
  rmm::cuda_stream_view stream{cudf::get_default_stream()};

  auto const table_comparator =
    cudf::experimental::row::equality::two_table_comparator{lhs, rhs, stream};

  auto const lhs_it = cudf::experimental::row::lhs_iterator(0);
  auto const rhs_it = cudf::experimental::row::rhs_iterator(0);

  auto output = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::BOOL8), lhs.num_rows(), cudf::mask_state::UNALLOCATED);

  if (cudf::detail::has_nested_columns(lhs) or cudf::detail::has_nested_columns(rhs)) {
    auto const equal_comparator =
      table_comparator.equal_to<true>(cudf::nullate::NO{}, cudf::null_equality::EQUAL, comparator);

    thrust::transform(rmm::exec_policy(stream),
                      lhs_it,
                      lhs_it + lhs.num_rows(),
                      rhs_it,
                      output->mutable_view().data<bool>(),
                      equal_comparator);
  } else {
    auto const equal_comparator =
      table_comparator.equal_to<false>(cudf::nullate::NO{}, cudf::null_equality::EQUAL, comparator);

    thrust::transform(rmm::exec_policy(stream),
                      lhs_it,
                      lhs_it + lhs.num_rows(),
                      rhs_it,
                      output->mutable_view().data<bool>(),
                      equal_comparator);
  }
  return output;
}

TYPED_TEST(TypedTableViewTest, TestLexicographicalComparatorTwoTables)
{
  using T = TypeParam;

  auto const col1         = cudf::test::fixed_width_column_wrapper<T>{{1, 2, 3, 4}};
  auto const col2         = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 4, 3}};
  auto const column_order = std::vector{cudf::order::DESCENDING};
  auto const lhs          = cudf::table_view{{col1}};
  auto const rhs          = cudf::table_view{{col2}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{1, 1, 0, 1}};
  auto const got      = two_table_comparison(
    lhs, rhs, column_order, cudf::experimental::row::lexicographic::physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  auto const sorting_got = two_table_comparison(
    lhs,
    rhs,
    column_order,
    cudf::experimental::row::lexicographic::sorting_physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, sorting_got->view());
}

TYPED_TEST(TypedTableViewTest, TestLexicographicalComparatorSameTable)
{
  using T = TypeParam;

  auto const col1         = cudf::test::fixed_width_column_wrapper<T>{{1, 2, 3, 4}};
  auto const column_order = std::vector{cudf::order::DESCENDING};
  auto const input_table  = cudf::table_view{{col1}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{0, 0, 0, 0}};
  auto const got =
    self_comparison(input_table,
                    column_order,
                    cudf::experimental::row::lexicographic::physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  auto const sorting_got =
    self_comparison(input_table,
                    column_order,
                    cudf::experimental::row::lexicographic::sorting_physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, sorting_got->view());
}

template <typename T>
struct NaNTableViewTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(NaNTableViewTest, cudf::test::FloatingPointTypes);

TYPED_TEST(NaNTableViewTest, TestLexicographicalComparatorTwoTableNaNCase)
{
  using T = TypeParam;

  auto const col1         = cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(NAN), T(1), T(1)}};
  auto const col2         = cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(1), T(NAN), T(1)}};
  auto const column_order = std::vector{cudf::order::DESCENDING};

  auto const lhs = cudf::table_view{{col1}};
  auto const rhs = cudf::table_view{{col2}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{0, 0, 0, 0}};
  auto const got      = two_table_comparison(
    lhs, rhs, column_order, cudf::experimental::row::lexicographic::physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  auto const sorting_expected = cudf::test::fixed_width_column_wrapper<bool>{{0, 1, 0, 0}};
  auto const sorting_got      = two_table_comparison(
    lhs,
    rhs,
    column_order,
    cudf::experimental::row::lexicographic::sorting_physical_element_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorting_expected, sorting_got->view());
}

TYPED_TEST(NaNTableViewTest, TestEqualityComparatorTwoTableNaNCase)
{
  using T = TypeParam;

  auto const col1         = cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(NAN), T(1), T(1)}};
  auto const col2         = cudf::test::fixed_width_column_wrapper<T>{{T(NAN), T(1), T(NAN), T(1)}};
  auto const column_order = std::vector{cudf::order::DESCENDING};

  auto const lhs = cudf::table_view{{col1}};
  auto const rhs = cudf::table_view{{col2}};

  auto const expected = cudf::test::fixed_width_column_wrapper<bool>{{0, 0, 0, 1}};
  auto const got      = two_table_equality(
    lhs, rhs, column_order, cudf::experimental::row::equality::physical_equality_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  auto const nan_equal_expected = cudf::test::fixed_width_column_wrapper<bool>{{1, 0, 0, 1}};
  auto const nan_equal_got =
    two_table_equality(lhs,
                       rhs,
                       column_order,
                       cudf::experimental::row::equality::nan_equal_physical_equality_comparator{});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(nan_equal_expected, nan_equal_got->view());
}
