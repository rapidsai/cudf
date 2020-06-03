/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/sequence.h>

namespace cudf {
namespace detail {
// Create permuted row indices that would materialize sorted order
template <bool stable = false>
std::unique_ptr<column> sorted_order(table_view input,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
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

  std::unique_ptr<column> sorted_indices = cudf::make_numeric_column(
    data_type(type_to_id<size_type>()), input.num_rows(), mask_state::UNALLOCATED, stream, mr);

  mutable_column_view mutable_indices_view = sorted_indices->mutable_view();

  auto device_table = table_device_view::create(input, stream);

  thrust::sequence(rmm::exec_policy(stream)->on(stream),
                   mutable_indices_view.begin<size_type>(),
                   mutable_indices_view.end<size_type>(),
                   0);

  rmm::device_vector<order> d_column_order(column_order);

  if (has_nulls(input)) {
    rmm::device_vector<null_order> d_null_precedence(null_precedence);
    auto comparator = row_lexicographic_comparator<true>(
      *device_table, *device_table, d_column_order.data().get(), d_null_precedence.data().get());
    if (stable) {
      thrust::stable_sort(rmm::exec_policy(stream)->on(stream),
                          mutable_indices_view.begin<size_type>(),
                          mutable_indices_view.end<size_type>(),
                          comparator);
    } else {
      thrust::sort(rmm::exec_policy(stream)->on(stream),
                   mutable_indices_view.begin<size_type>(),
                   mutable_indices_view.end<size_type>(),
                   comparator);
    }
  } else {
    auto comparator = row_lexicographic_comparator<false>(
      *device_table, *device_table, d_column_order.data().get());
    if (stable) {
      thrust::stable_sort(rmm::exec_policy(stream)->on(stream),
                          mutable_indices_view.begin<size_type>(),
                          mutable_indices_view.end<size_type>(),
                          comparator);
    } else {
      thrust::sort(rmm::exec_policy(stream)->on(stream),
                   mutable_indices_view.begin<size_type>(),
                   mutable_indices_view.end<size_type>(),
                   comparator);
    }
  }

  return sorted_indices;
}

}  // namespace detail
}  // namespace cudf
