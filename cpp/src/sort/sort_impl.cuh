/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#pragma once

#include <sort/sort_column_impl.cuh>

#include <cudf/column/column_factories.hpp>

namespace cudf {
namespace detail {

/**
 * @copydoc
 * sorted_order(table_view&,std::vector<order>,std::vector<null_order>,rmm::mr::device_memory_resource*)
 *
 * @tparam stable Whether to use stable sort
 * @tparam PhysicalElementComparator A relational comparator functor that compares individual
 * values rather than logical elements, defaults to `NaN` aware relational comparator that
 * evaluates `NaN` as greater than all other values.
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <bool stable,
          typename PhysicalElementComparator =
            cudf::experimental::row::lexicographic::sorting_physical_element_comparator>
std::unique_ptr<column> sorted_order(table_view input,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  if (input.num_rows() == 0 or input.num_columns() == 0) {
    return cudf::make_numeric_column(data_type(type_to_id<size_type>()), 0);
  }

  if (not column_order.empty()) {
    CUDF_EXPECTS(static_cast<std::size_t>(input.num_columns()) == column_order.size(),
                 "Mismatch between number of columns and column order.");
  }

  if (not null_precedence.empty()) {
    CUDF_EXPECTS(static_cast<std::size_t>(input.num_columns()) == null_precedence.size(),
                 "Mismatch between number of columns and null_precedence size.");
  }

  // Fast-path for single column sort
  // If the first column is floating-point, only run this path if special NaN handling is
  // required (i.e., `NaN` is always considered as equivalent to other `NaN`s and greater than all
  // non-NaN values, which is equivalent to using `sorting_physical_element_comparator`.).
  if (input.num_columns() == 1 and not cudf::is_nested(input.column(0).type()) and
      (not cudf::is_floating_point(input.column(0).type()) or
       std::is_same_v<
         PhysicalElementComparator,
         cudf::experimental::row::lexicographic::sorting_physical_element_comparator>)) {
    auto const single_col = input.column(0);
    auto const col_order  = column_order.empty() ? order::ASCENDING : column_order.front();
    auto const null_prec  = null_precedence.empty() ? null_order::BEFORE : null_precedence.front();
    return sorted_order<stable>(single_col, col_order, null_prec, stream, mr);
  }

  std::unique_ptr<column> sorted_indices = cudf::make_numeric_column(
    data_type(type_to_id<size_type>()), input.num_rows(), mask_state::UNALLOCATED, stream, mr);
  mutable_column_view mutable_indices_view = sorted_indices->mutable_view();
  thrust::sequence(rmm::exec_policy(stream),
                   mutable_indices_view.begin<size_type>(),
                   mutable_indices_view.end<size_type>(),
                   0);

  auto const do_sort = [&](auto const comparator) {
    if constexpr (stable) {
      thrust::stable_sort(rmm::exec_policy(stream),
                          mutable_indices_view.begin<size_type>(),
                          mutable_indices_view.end<size_type>(),
                          comparator);
    } else {
      thrust::sort(rmm::exec_policy(stream),
                   mutable_indices_view.begin<size_type>(),
                   mutable_indices_view.end<size_type>(),
                   comparator);
    }
  };

  auto const comp = cudf::experimental::row::lexicographic::self_comparator(
    input, column_order, null_precedence, stream);
  if (cudf::detail::has_nested_columns(input)) {
    auto const comparator = comp.less<true, nullate::DYNAMIC, PhysicalElementComparator>(
      nullate::DYNAMIC{has_nested_nulls(input)});
    do_sort(comparator);
  } else {
    auto const comparator = comp.less<false, nullate::DYNAMIC, PhysicalElementComparator>(
      nullate::DYNAMIC{has_nested_nulls(input)});
    do_sort(comparator);
  }

  return sorted_indices;
}

}  // namespace detail
}  // namespace cudf
