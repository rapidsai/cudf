/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

// Including this declaration/defintion in row_operator_tests_utilities.cu causes
// the nvcc compiler to segfault when built with the debug (-g) flag.

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

template std::unique_ptr<cudf::column> self_comparison<physical_comparator_t>(
  cudf::table_view input,
  std::vector<cudf::order> const& column_order,
  physical_comparator_t comparator);
template std::unique_ptr<cudf::column> self_comparison<sorting_comparator_t>(
  cudf::table_view input,
  std::vector<cudf::order> const& column_order,
  sorting_comparator_t comparator);
