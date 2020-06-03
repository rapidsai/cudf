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

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>

namespace cudf {
namespace detail {
template <bool has_nulls>
auto is_sorted(cudf::table_view const& in,
               std::vector<order> const& column_order,
               std::vector<null_order> const& null_precedence)
{
  cudaStream_t stream = 0;
  auto in_d           = table_device_view::create(in);
  rmm::device_vector<order> d_column_order(column_order);
  rmm::device_vector<null_order> const d_null_precedence =
    (has_nulls) ? rmm::device_vector<null_order>{null_precedence}
                : rmm::device_vector<null_order>{};
  auto ineq_op = row_lexicographic_comparator<has_nulls>(
    *in_d, *in_d, d_column_order.data().get(), d_null_precedence.data().get());

  auto sorted = thrust::is_sorted(rmm::exec_policy(stream)->on(stream),
                                  thrust::make_counting_iterator(0),
                                  thrust::make_counting_iterator(in.num_rows()),
                                  ineq_op);

  return sorted;
}

}  // namespace detail

bool is_sorted(cudf::table_view const& in,
               std::vector<order> const& column_order,
               std::vector<null_order> const& null_precedence)
{
  CUDF_FUNC_RANGE();
  if (in.num_columns() == 0 || in.num_rows() == 0) { return true; }

  if (not column_order.empty()) {
    CUDF_EXPECTS(static_cast<unsigned int>(in.num_columns()) == column_order.size(),
                 "Number of columns in the table doesn't match the vector column_order's size .\n");
  }

  if (not null_precedence.empty()) {
    CUDF_EXPECTS(
      static_cast<unsigned int>(in.num_columns()) == null_precedence.size(),
      "Number of columns in the table doesn't match the vector null_precedence's size .\n");
  }

  if (has_nulls(in)) {
    return detail::is_sorted<true>(in, column_order, null_precedence);
  } else {
    return detail::is_sorted<false>(in, column_order, null_precedence);
  }
}

}  // namespace cudf
