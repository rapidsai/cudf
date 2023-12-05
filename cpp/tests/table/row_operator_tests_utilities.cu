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

#include "row_operator_tests_utilities.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/experimental/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

template <typename PhysicalElementComparator>
void self_comparison_nested(
  cudf::size_type num_rows,
  cudf::experimental::row::lexicographic::self_comparator table_comparator,
  PhysicalElementComparator comparator,
  bool* d_output,
  rmm::cuda_stream_view stream);

template <typename PhysicalElementComparator>
void self_comparison(cudf::size_type num_rows,
                     cudf::experimental::row::lexicographic::self_comparator table_comparator,
                     PhysicalElementComparator comparator,
                     bool* d_output,
                     rmm::cuda_stream_view stream)
{
  auto const itr     = thrust::make_counting_iterator<cudf::size_type>(0);
  auto const less_fn = table_comparator.less<false>(cudf::nullate::NO{}, comparator);
  thrust::transform(rmm::exec_policy(stream), itr, itr + num_rows, itr, d_output, less_fn);
}

template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> self_comparison(cudf::table_view input,
                                              std::vector<cudf::order> const& column_order,
                                              PhysicalElementComparator comparator)
{
  rmm::cuda_stream_view stream{cudf::get_default_stream()};

  auto const table_comparator =
    cudf::experimental::row::lexicographic::self_comparator{input, column_order, {}, stream};

  auto output = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::BOOL8), input.num_rows(), cudf::mask_state::UNALLOCATED);

  if (cudf::detail::has_nested_columns(input)) {
    self_comparison_nested(
      input.num_rows(), table_comparator, comparator, output->mutable_view().data<bool>(), stream);
  } else {
    self_comparison(
      input.num_rows(), table_comparator, comparator, output->mutable_view().data<bool>(), stream);
  }

  return output;
}

using physical_comparator_t = cudf::experimental::row::lexicographic::physical_element_comparator;
using sorting_comparator_t =
  cudf::experimental::row::lexicographic::sorting_physical_element_comparator;

template std::unique_ptr<cudf::column> self_comparison<physical_comparator_t>(
  cudf::table_view input,
  std::vector<cudf::order> const& column_order,
  physical_comparator_t comparator);
template std::unique_ptr<cudf::column> self_comparison<sorting_comparator_t>(
  cudf::table_view input,
  std::vector<cudf::order> const& column_order,
  sorting_comparator_t comparator);

template <typename PhysicalElementComparator>
void two_table_comparison_nested(cudf::size_type num_rows,
                                 cudf::experimental::row::lexicographic::two_table_comparator,
                                 PhysicalElementComparator comparator,
                                 bool* d_output,
                                 rmm::cuda_stream_view stream);

template <typename PhysicalElementComparator>
void two_table_comparison(
  cudf::size_type num_rows,
  cudf::experimental::row::lexicographic::two_table_comparator table_comparator,
  PhysicalElementComparator comparator,
  bool* d_output,
  rmm::cuda_stream_view stream)
{
  auto const lhs_it  = cudf::experimental::row::lhs_iterator(0);
  auto const rhs_it  = cudf::experimental::row::rhs_iterator(0);
  auto const less_fn = table_comparator.less<false>(cudf::nullate::NO{}, comparator);
  thrust::transform(rmm::exec_policy(stream), lhs_it, lhs_it + num_rows, rhs_it, d_output, less_fn);
}

template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> two_table_comparison(cudf::table_view lhs,
                                                   cudf::table_view rhs,
                                                   std::vector<cudf::order> const& column_order,
                                                   PhysicalElementComparator comparator)
{
  rmm::cuda_stream_view stream{cudf::get_default_stream()};

  auto const table_comparator = cudf::experimental::row::lexicographic::two_table_comparator{
    lhs, rhs, column_order, {}, stream};

  auto output = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::BOOL8), lhs.num_rows(), cudf::mask_state::UNALLOCATED);

  if (cudf::detail::has_nested_columns(lhs) || cudf::detail::has_nested_columns(rhs)) {
    two_table_comparison_nested(
      lhs.num_rows(), table_comparator, comparator, output->mutable_view().data<bool>(), stream);
  } else {
    two_table_comparison(
      lhs.num_rows(), table_comparator, comparator, output->mutable_view().data<bool>(), stream);
  }

  return output;
}

template std::unique_ptr<cudf::column> two_table_comparison<physical_comparator_t>(
  cudf::table_view lhs,
  cudf::table_view rhs,
  std::vector<cudf::order> const& column_order,
  physical_comparator_t comparator);
template std::unique_ptr<cudf::column> two_table_comparison<sorting_comparator_t>(
  cudf::table_view lhs,
  cudf::table_view rhs,
  std::vector<cudf::order> const& column_order,
  sorting_comparator_t comparator);

template <typename PhysicalElementComparator>
void sorted_order_nested(cudf::experimental::row::lexicographic::self_comparator table_comparator,
                         cudf::size_type num_rows,
                         PhysicalElementComparator comparator,
                         cudf::size_type* d_output,
                         rmm::cuda_stream_view stream);

template <typename PhysicalElementComparator>
void sorted_order(cudf::experimental::row::lexicographic::self_comparator table_comparator,
                  cudf::size_type num_rows,
                  PhysicalElementComparator comparator,
                  cudf::size_type* d_output,
                  rmm::cuda_stream_view stream)
{
  auto const comp = table_comparator.less<false>(cudf::nullate::NO{}, comparator);
  thrust::stable_sort(rmm::exec_policy(stream), d_output, d_output + num_rows, comp);
}

template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> sorted_order(
  std::shared_ptr<cudf::experimental::row::lexicographic::preprocessed_table> preprocessed_input,
  cudf::size_type num_rows,
  bool has_nested,
  PhysicalElementComparator comparator,
  rmm::cuda_stream_view stream)
{
  auto output = cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<cudf::size_type>()),
                                          num_rows,
                                          cudf::mask_state::UNALLOCATED,
                                          stream);
  auto const out_begin = output->mutable_view().begin<cudf::size_type>();
  thrust::sequence(rmm::exec_policy(stream), out_begin, out_begin + num_rows, 0);

  auto const table_comparator =
    cudf::experimental::row::lexicographic::self_comparator{preprocessed_input};
  if (has_nested) {
    sorted_order_nested(table_comparator, num_rows, comparator, out_begin, stream);
  } else {
    sorted_order(table_comparator, num_rows, comparator, out_begin, stream);
  }

  return output;
}

template std::unique_ptr<cudf::column> sorted_order<physical_comparator_t>(
  std::shared_ptr<cudf::experimental::row::lexicographic::preprocessed_table> preprocessed_input,
  cudf::size_type num_rows,
  bool has_nested,
  physical_comparator_t comparator,
  rmm::cuda_stream_view stream);
template std::unique_ptr<cudf::column> sorted_order<sorting_comparator_t>(
  std::shared_ptr<cudf::experimental::row::lexicographic::preprocessed_table> preprocessed_input,
  cudf::size_type num_rows,
  bool has_nested,
  sorting_comparator_t comparator,
  rmm::cuda_stream_view stream);

template <typename PhysicalElementComparator>
void two_table_equality_nested(
  cudf::size_type num_rows,
  cudf::experimental::row::equality::two_table_comparator table_comparator,
  PhysicalElementComparator comparator,
  bool* d_output,
  rmm::cuda_stream_view stream);

template <typename PhysicalElementComparator>
void two_table_equality(cudf::size_type num_rows,
                        cudf::experimental::row::equality::two_table_comparator table_comparator,
                        PhysicalElementComparator comparator,
                        bool* d_output,
                        rmm::cuda_stream_view stream)
{
  auto const lhs_it = cudf::experimental::row::lhs_iterator(0);
  auto const rhs_it = cudf::experimental::row::rhs_iterator(0);
  auto const equal_comparator =
    table_comparator.equal_to<false>(cudf::nullate::NO{}, cudf::null_equality::EQUAL, comparator);
  thrust::transform(
    rmm::exec_policy(stream), lhs_it, lhs_it + num_rows, rhs_it, d_output, equal_comparator);
}

template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> two_table_equality(cudf::table_view lhs,
                                                 cudf::table_view rhs,
                                                 std::vector<cudf::order> const& column_order,
                                                 PhysicalElementComparator comparator)
{
  rmm::cuda_stream_view stream{cudf::get_default_stream()};

  auto const table_comparator =
    cudf::experimental::row::equality::two_table_comparator{lhs, rhs, stream};

  auto output = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::BOOL8), lhs.num_rows(), cudf::mask_state::UNALLOCATED);

  if (cudf::detail::has_nested_columns(lhs) or cudf::detail::has_nested_columns(rhs)) {
    two_table_equality_nested(
      lhs.num_rows(), table_comparator, comparator, output->mutable_view().data<bool>(), stream);
  } else {
    two_table_equality_nested(
      lhs.num_rows(), table_comparator, comparator, output->mutable_view().data<bool>(), stream);
  }
  return output;
}

using physical_equality_t = cudf::experimental::row::equality::physical_equality_comparator;
using nan_equality_t = cudf::experimental::row::equality::nan_equal_physical_equality_comparator;

template std::unique_ptr<cudf::column> two_table_equality<physical_equality_t>(
  cudf::table_view lhs,
  cudf::table_view rhs,
  std::vector<cudf::order> const& column_order,
  physical_equality_t comparator);
template std::unique_ptr<cudf::column> two_table_equality<nan_equality_t>(
  cudf::table_view lhs,
  cudf::table_view rhs,
  std::vector<cudf::order> const& column_order,
  nan_equality_t comparator);
