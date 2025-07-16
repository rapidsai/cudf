/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "sort_column_impl.cuh"

#include <cudf/column/column.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

std::unique_ptr<column> top_k(column_view const& col,
                              size_type k,
                              order sort_order,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(k <= col.size(),
               "k must be less than or equal to the number of rows in the column",
               std::invalid_argument);

  // code will be specialized for fixed-width types once CUB topk function is available
  auto const nulls     = sort_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
  auto const indices   = sorted_order<sort_method::STABLE>(col, sort_order, nulls, stream, mr);
  auto const k_indices = cudf::detail::split(indices->view(), {k}, stream).front();
  auto result          = cudf::detail::gather(cudf::table_view({col}),
                                     k_indices,
                                     out_of_bounds_policy::DONT_CHECK,
                                     negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);
  return std::move(result->release().front());
}

std::unique_ptr<column> top_k_order(column_view const& col,
                                    size_type k,
                                    order sort_order,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(k <= col.size(),
               "k must be less than or equal to the number of rows in the column",
               std::invalid_argument);

  auto const nulls   = sort_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
  auto const indices = sorted_order<sort_method::STABLE>(col, sort_order, nulls, stream, mr);
  return std::make_unique<column>(cudf::detail::split(indices->view(), {k}, stream).front());
}

}  // namespace detail

std::unique_ptr<column> top_k(column_view const& col,
                              size_type k,
                              order sort_order,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::top_k(col, k, sort_order, stream, mr);
}

std::unique_ptr<column> top_k_order(column_view const& col,
                                    size_type k,
                                    order sort_order,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::top_k_order(col, k, sort_order, stream, mr);
}

}  // namespace cudf
