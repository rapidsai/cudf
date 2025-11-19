/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace detail {
struct dispatch_nan_to_null {
  template <typename T>
  std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> operator()(
    column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
    requires(std::is_floating_point_v<T>)
  {
    auto input_device_view_ptr = column_device_view::create(input, stream);
    auto input_device_view     = *input_device_view_ptr;

    if (input.nullable()) {
      auto pred = [input_device_view] __device__(cudf::size_type idx) {
        return not(std::isnan(input_device_view.element<T>(idx)) ||
                   input_device_view.is_null_nocheck(idx));
      };

      auto mask = detail::valid_if(thrust::make_counting_iterator<cudf::size_type>(0),
                                   thrust::make_counting_iterator<cudf::size_type>(input.size()),
                                   pred,
                                   stream,
                                   mr);

      return std::pair(std::make_unique<rmm::device_buffer>(std::move(mask.first)), mask.second);
    } else {
      auto pred = [input_device_view] __device__(cudf::size_type idx) {
        return not(std::isnan(input_device_view.element<T>(idx)));
      };

      auto mask = detail::valid_if(thrust::make_counting_iterator<cudf::size_type>(0),
                                   thrust::make_counting_iterator<cudf::size_type>(input.size()),
                                   pred,
                                   stream,
                                   mr);

      return std::pair(std::make_unique<rmm::device_buffer>(std::move(mask.first)), mask.second);
    }
  }

  template <typename T>
  std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> operator()(
    column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
    requires(!std::is_floating_point_v<T>)
  {
    CUDF_FAIL("Input column can't be a non-floating type");
  }
};

std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> nans_to_nulls(
  column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return std::pair(std::make_unique<rmm::device_buffer>(), 0); }

  return cudf::type_dispatcher(input.type(), dispatch_nan_to_null{}, input, stream, mr);
}

}  // namespace detail

std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> nans_to_nulls(
  column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::nans_to_nulls(input, stream, mr);
}

}  // namespace cudf
