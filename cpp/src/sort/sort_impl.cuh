/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/swap.h>

namespace cudf {
namespace detail {

/**
 * @brief Comparator functor needed for single column sort.
 *
 * @tparam Column element type.
 */
template <typename T>
struct simple_comparator {
  __device__ bool operator()(size_type lhs, size_type rhs)
  {
    if (has_nulls) {
      bool lhs_null{d_column.is_null(lhs)};
      bool rhs_null{d_column.is_null(rhs)};
      if (lhs_null || rhs_null) {
        if (!ascending) thrust::swap(lhs_null, rhs_null);
        return (null_precedence == cudf::null_order::BEFORE ? !rhs_null : !lhs_null);
      }
    }
    return relational_compare(d_column.element<T>(lhs), d_column.element<T>(rhs)) ==
           (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
  }
  column_device_view const d_column;
  bool has_nulls;
  bool ascending;
  null_order null_precedence{};
};

/**
 * @brief Sort indices of a single column.
 *
 * @param input Column to sort. The column data is not modified.
 * @param column_order Ascending or descending sort order
 * @param null_precedence How null rows are to be ordered
 * @param stable True if sort should be stable
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Sorted indices for the input column.
 */
template <bool stable>
std::unique_ptr<column> sorted_order(column_view const& input,
                                     order column_order,
                                     null_order null_precedence,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr);

/**
 * @copydoc
 * sorted_order(table_view&,std::vector<order>,std::vector<null_order>,rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <bool stable = false>
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

  // fast-path for single column sort
  if (input.num_columns() == 1 and not cudf::is_nested(input.column(0).type())) {
    auto const single_col = input.column(0);
    auto const col_order  = column_order.empty() ? order::ASCENDING : column_order.front();
    auto const null_prec  = null_precedence.empty() ? null_order::BEFORE : null_precedence.front();
    return stable ? sorted_order<true>(single_col, col_order, null_prec, stream, mr)
                  : sorted_order<false>(single_col, col_order, null_prec, stream, mr);
  }

  std::unique_ptr<column> sorted_indices = cudf::make_numeric_column(
    data_type(type_to_id<size_type>()), input.num_rows(), mask_state::UNALLOCATED, stream, mr);
  mutable_column_view mutable_indices_view = sorted_indices->mutable_view();
  thrust::sequence(rmm::exec_policy(stream),
                   mutable_indices_view.begin<size_type>(),
                   mutable_indices_view.end<size_type>(),
                   0);

  auto comp =
    experimental::row::lexicographic::self_comparator(input, column_order, null_precedence, stream);
  if (cudf::detail::has_nested_columns(input)) {
    auto comparator = comp.less<true>(nullate::DYNAMIC{has_nested_nulls(input)});
    if (stable) {
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
  } else {
    auto comparator = comp.less<false>(nullate::DYNAMIC{has_nested_nulls(input)});
    if (stable) {
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
  }
  // protection for temporary d_column_order and d_null_precedence
  stream.synchronize();

  return sorted_indices;
}

}  // namespace detail
}  // namespace cudf
