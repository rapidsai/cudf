/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort_column_impl.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

std::unique_ptr<column> top_k(column_view const& col,
                              size_type k,
                              order topk_order,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(k >= 0, "k must be non-negative", std::invalid_argument);
  if (k == 0 || col.is_empty()) { return empty_like(col); }
  if (k >= col.size()) { return std::make_unique<column>(col, stream, mr); }

  // code will be specialized for fixed-width types once CUB topk function is available
  auto const nulls   = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
  auto const indices = sorted_order<sort_method::STABLE>(
    col, topk_order, nulls, stream, cudf::get_current_device_resource_ref());
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
                                    order topk_order,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(k >= 0, "k must be non-negative", std::invalid_argument);
  if (k == 0 || col.is_empty()) { return make_empty_column(cudf::type_to_id<size_type>()); }
  if (k >= col.size()) {
    return cudf::detail::sequence(
      col.size(), numeric_scalar<size_type>(0, true, stream), stream, mr);
  }

  auto const nulls   = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
  auto const indices = sorted_order<sort_method::STABLE>(
    col, topk_order, nulls, stream, cudf::get_current_device_resource_ref());
  return std::make_unique<column>(
    cudf::detail::split(indices->view(), {k}, stream).front(), stream, mr);
}

}  // namespace detail

std::unique_ptr<column> top_k(column_view const& col,
                              size_type k,
                              order topk_order,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::top_k(col, k, topk_order, stream, mr);
}

std::unique_ptr<column> top_k_order(column_view const& col,
                                    size_type k,
                                    order topk_order,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::top_k_order(col, k, topk_order, stream, mr);
}

}  // namespace cudf
