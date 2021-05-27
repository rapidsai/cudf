/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <structs/utilities.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace detail {

template <bool has_nulls>
auto is_sorted(cudf::table_view const& in,
               std::vector<order> const& column_order,
               std::vector<null_order> const& null_precedence,
               rmm::cuda_stream_view stream)
{
  // 0-table_view, 1-column_order, 2-null_precedence, 3-validity_columns
  auto flattened = structs::detail::flatten_nested_columns(in, column_order, null_precedence);

  auto const d_input           = table_device_view::create(std::get<0>(flattened), stream);
  auto const d_column_order    = make_device_uvector_async(std::get<1>(flattened), stream);
  auto const d_null_precedence = has_nulls
                                   ? make_device_uvector_async(std::get<2>(flattened), stream)
                                   : rmm::device_uvector<null_order>(0, stream);

  auto comparator = row_lexicographic_comparator<has_nulls>(
    *d_input, *d_input, d_column_order.data(), d_null_precedence.data());

  auto sorted = thrust::is_sorted(rmm::exec_policy(stream),
                                  thrust::make_counting_iterator(0),
                                  thrust::make_counting_iterator(in.num_rows()),
                                  comparator);

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
    return detail::is_sorted<true>(in, column_order, null_precedence, rmm::cuda_stream_default);
  } else {
    return detail::is_sorted<false>(in, column_order, null_precedence, rmm::cuda_stream_default);
  }
}

}  // namespace cudf
