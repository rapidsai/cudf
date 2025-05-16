/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> two_table_comparison(cudf::table_view lhs,
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

  if (cudf::has_nested_columns(lhs) || cudf::has_nested_columns(rhs)) {
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
    auto const comp = table_comparator.less<true>(cudf::nullate::NO{}, comparator);
    thrust::stable_sort(rmm::exec_policy(stream), out_begin, out_begin + num_rows, comp);
  } else {
    auto const comp = table_comparator.less<false>(cudf::nullate::NO{}, comparator);
    thrust::stable_sort(rmm::exec_policy(stream), out_begin, out_begin + num_rows, comp);
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
std::unique_ptr<cudf::column> two_table_equality(cudf::table_view lhs,
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

  if (cudf::has_nested_columns(lhs) or cudf::has_nested_columns(rhs)) {
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
