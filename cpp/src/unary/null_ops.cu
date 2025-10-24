/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace detail {
std::unique_ptr<column> is_null(cudf::column_view const& input,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  auto input_device_view = column_device_view::create(input, stream);
  auto device_view       = *input_device_view;
  auto predicate = [device_view] __device__(auto index) { return (device_view.is_null(index)); };
  return detail::true_if(thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(input.size()),
                         input.size(),
                         predicate,
                         stream,
                         mr);
}

std::unique_ptr<column> is_valid(cudf::column_view const& input,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  auto input_device_view = column_device_view::create(input, stream);
  auto device_view       = *input_device_view;
  auto predicate = [device_view] __device__(auto index) { return device_view.is_valid(index); };
  return detail::true_if(thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(input.size()),
                         input.size(),
                         predicate,
                         stream,
                         mr);
}

}  // namespace detail

std::unique_ptr<column> is_null(cudf::column_view const& input,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_null(input, stream, mr);
}

std::unique_ptr<column> is_valid(cudf::column_view const& input,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_valid(input, stream, mr);
}

}  // namespace cudf
