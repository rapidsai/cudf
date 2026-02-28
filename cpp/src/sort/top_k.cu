/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort.hpp"

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
#include <rmm/exec_policy.hpp>

#include <cub/device/device_topk.cuh>
#include <thrust/sequence.h>

namespace cudf {
namespace detail {
namespace {
bool is_fast_path(column_view const& column)
{
  return !column.has_nulls() && cudf::is_numeric(column.type()) &&
         !cudf::is_floating_point(column.type());
}

template <bool fast_path>
struct dispatch_topk_fn {
  column_view input;
  size_type k;
  order topk_order;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T>
    requires(is_numeric<T>())
  std::unique_ptr<column> operator()()
  {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto env          = cuda::std::execution::env{stream.value(), requirements};
    auto tmp_size     = std::size_t{0};
    auto const size   = input.size();

    auto keys_in  = input.begin<T>();
    auto keys_out = cuda::make_discard_iterator();
    auto indices  = rmm::device_uvector<size_type>(size, stream);
    auto vals_in  = cuda::counting_iterator<size_type>();
    auto vals_out = indices.begin();

    if (topk_order == order::ASCENDING) {
      CUDF_CUDA_TRY(cub::DeviceTopK::MinPairs(
        nullptr, tmp_size, keys_in, keys_out, vals_in, vals_out, size, k, env));
      auto tmp = rmm::device_uvector<char>(tmp_size, stream);
      CUDF_CUDA_TRY(cub::DeviceTopK::MinPairs(
        tmp.data(), tmp_size, keys_in, keys_out, vals_in, vals_out, size, k, env));
    } else {
      CUDF_CUDA_TRY(cub::DeviceTopK::MaxPairs(
        nullptr, tmp_size, keys_in, keys_out, vals_in, vals_out, size, k, env));
      auto tmp = rmm::device_uvector<char>(tmp_size, stream);
      CUDF_CUDA_TRY(cub::DeviceTopK::MaxPairs(
        tmp.data(), tmp_size, keys_in, keys_out, vals_in, vals_out, size, k, env));
    }

    return std::make_unique<column>(std::move(indices), rmm::device_buffer{}, 0);
  }

  template <typename T>
    requires(!is_numeric<T>())
  std::unique_ptr<column> operator()()
  {
    CUDF_FAIL("unexpected type for topk fast path");
  }
};

}  // namespace

std::unique_ptr<column> top_k(column_view const& col,
                              size_type k,
                              order topk_order,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(k >= 0, "k must be non-negative", std::invalid_argument);
  if (k == 0 || col.is_empty()) { return empty_like(col); }
  if (k >= col.size()) { return std::make_unique<column>(col, stream, mr); }

  auto const indices = [&] {
    if (is_fast_path(col)) {
      return type_dispatcher<dispatch_storage_type>(
        col.type(), dispatch_topk_fn<true>{col, k, topk_order, stream, mr});
    }
    auto const nulls = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
    return sorted_order<sort_method::STABLE>(
      col, topk_order, nulls, stream, cudf::get_current_device_resource_ref());
  }();

  // code will be specialized for fixed-width types once CUB topk function is available
  auto const k_indices = cudf::detail::split(indices->view(), {k}, stream).front();

  auto const dont_check  = out_of_bounds_policy::DONT_CHECK;
  auto const not_allowed = negative_index_policy::NOT_ALLOWED;
  auto result =
    cudf::detail::gather(cudf::table_view({col}), k_indices, dont_check, not_allowed, stream, mr);
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

  auto const indices = [&] {
    if (is_fast_path(col)) {
      return type_dispatcher<dispatch_storage_type>(
        col.type(), dispatch_topk_fn<true>{col, k, topk_order, stream, mr});
    }
    auto const nulls = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
    return sorted_order<sort_method::STABLE>(
      col, topk_order, nulls, stream, cudf::get_current_device_resource_ref());
  }();

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
