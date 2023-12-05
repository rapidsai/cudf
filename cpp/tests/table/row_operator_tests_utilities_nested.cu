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

#include <cudf/table/experimental/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sort.h>
#include <thrust/transform.h>

template <typename PhysicalElementComparator>
void self_comparison_nested(
  cudf::size_type num_rows,
  cudf::experimental::row::lexicographic::self_comparator table_comparator,
  PhysicalElementComparator comparator,
  bool* d_output,
  rmm::cuda_stream_view stream)
{
  auto const itr     = thrust::make_counting_iterator<cudf::size_type>(0);
  auto const less_fn = table_comparator.less<true>(cudf::nullate::NO{}, comparator);
  thrust::transform(rmm::exec_policy(stream), itr, itr + num_rows, itr, d_output, less_fn);
}

template <typename PhysicalElementComparator>
void two_table_comparison_nested(
  cudf::size_type num_rows,
  cudf::experimental::row::lexicographic::two_table_comparator table_comparator,
  PhysicalElementComparator comparator,
  bool* d_output,
  rmm::cuda_stream_view stream)
{
  auto const lhs_it  = cudf::experimental::row::lhs_iterator(0);
  auto const rhs_it  = cudf::experimental::row::rhs_iterator(0);
  auto const less_fn = table_comparator.less<true>(cudf::nullate::NO{}, comparator);
  thrust::transform(rmm::exec_policy(stream), lhs_it, lhs_it + num_rows, rhs_it, d_output, less_fn);
}

template <typename PhysicalElementComparator>
void sorted_order_nested(cudf::experimental::row::lexicographic::self_comparator table_comparator,
                         cudf::size_type num_rows,
                         PhysicalElementComparator comparator,
                         cudf::size_type* d_output,
                         rmm::cuda_stream_view stream)
{
  auto const comp = table_comparator.less<true>(cudf::nullate::NO{}, comparator);
  thrust::stable_sort(rmm::exec_policy(stream), d_output, d_output + num_rows, comp);
}

template <typename PhysicalElementComparator>
void two_table_equality_nested(
  cudf::size_type num_rows,
  cudf::experimental::row::equality::two_table_comparator table_comparator,
  PhysicalElementComparator comparator,
  bool* d_output,
  rmm::cuda_stream_view stream)
{
  auto const lhs_it = cudf::experimental::row::lhs_iterator(0);
  auto const rhs_it = cudf::experimental::row::rhs_iterator(0);
  auto const equal_comparator =
    table_comparator.equal_to<true>(cudf::nullate::NO{}, cudf::null_equality::EQUAL, comparator);
  thrust::transform(
    rmm::exec_policy(stream), lhs_it, lhs_it + num_rows, rhs_it, d_output, equal_comparator);
}
