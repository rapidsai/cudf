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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <utilities/error_utils.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/sequence.h>

namespace cudf {
namespace exp {

namespace detail {

// Create permuted row indices that would materialize sorted order
std::unique_ptr<column> sorted_order(table_view input,
                                     std::vector<order> const& column_order,
                                     null_order null_precedence,
                                     cudaStream_t stream,
                                     rmm::mr::device_memory_resource* mr) {
  if (input.num_rows() == 0 or input.num_columns() == 0) {
    return cudf::make_numeric_column(data_type{INT32}, 0);
  }

  if (not column_order.empty()) {
    CUDF_EXPECTS(
        static_cast<std::size_t>(input.num_columns()) == column_order.size(),
        "Mismatch between number of columns and column order.");
  }

  std::unique_ptr<column> sorted_indices = cudf::make_numeric_column(
      data_type{INT32}, input.num_rows(), mask_state::UNALLOCATED, stream, mr);

  mutable_column_view mutable_indices_view = sorted_indices->mutable_view();

  auto device_table = table_device_view::create(input, stream);

  thrust::sequence(rmm::exec_policy(stream)->on(stream),
                   mutable_indices_view.begin<int32_t>(),
                   mutable_indices_view.end<int32_t>(), 0);

  rmm::device_vector<order> d_column_order(column_order);

  if (has_nulls(input)) {
    auto comparator = row_lexicographic_comparator<true>(
        *device_table, *device_table, null_precedence,
        d_column_order.data().get());
    thrust::sort(rmm::exec_policy(stream)->on(stream),
                 mutable_indices_view.begin<int32_t>(),
                 mutable_indices_view.end<int32_t>(), comparator);

  } else {
    auto comparator = row_lexicographic_comparator<false>(
        *device_table, *device_table, null_precedence,
        d_column_order.data().get());
    thrust::sort(rmm::exec_policy(stream)->on(stream),
                 mutable_indices_view.begin<int32_t>(),
                 mutable_indices_view.end<int32_t>(), comparator);
  }

  return sorted_indices;
}
}  // namespace detail

std::unique_ptr<column> sorted_order(table_view input,
                                     std::vector<order> const& column_order,
                                     null_order null_precedence,
                                     rmm::mr::device_memory_resource* mr) {
  return detail::sorted_order(input, column_order, null_precedence, 0, mr);
}
}  // namespace exp
}  // namespace cudf