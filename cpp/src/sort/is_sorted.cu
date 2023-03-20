/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

namespace cudf {
namespace detail {

auto is_sorted(cudf::table_view const& in,
               std::vector<order> const& column_order,
               std::vector<null_order> const& null_precedence,
               rmm::cuda_stream_view stream)
{
  auto const comparator =
    experimental::row::lexicographic::self_comparator{in, column_order, null_precedence, stream};

  if (cudf::detail::has_nested_columns(in)) {
    auto const device_comparator = comparator.less<true>(has_nested_nulls(in));

    return thrust::is_sorted(rmm::exec_policy(stream),
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(in.num_rows()),
                             device_comparator);
  } else {
    auto const device_comparator = comparator.less<false>(has_nested_nulls(in));

    return thrust::is_sorted(rmm::exec_policy(stream),
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(in.num_rows()),
                             device_comparator);
  }
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

  return detail::is_sorted(in, column_order, null_precedence, cudf::get_default_stream());
}

}  // namespace cudf
