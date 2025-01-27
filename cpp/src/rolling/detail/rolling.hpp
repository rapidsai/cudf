/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf {
// helper functions - used in the rolling window implementation and tests

namespace detail {

// store functor
template <typename T, bool is_mean = false>
struct rolling_store_output_functor {
  CUDF_HOST_DEVICE inline void operator()(T& out, T& val, size_type count) { out = val; }
};

// Specialization for MEAN
template <typename _T>
struct rolling_store_output_functor<_T, true> {
  // SFINAE for non-bool types
  template <typename T                                                             = _T,
            std::enable_if_t<!(cudf::is_boolean<T>() || cudf::is_timestamp<T>())>* = nullptr>
  CUDF_HOST_DEVICE inline void operator()(T& out, T& val, size_type count)
  {
    out = val / count;
  }

  // SFINAE for bool type
  template <typename T = _T, std::enable_if_t<cudf::is_boolean<T>()>* = nullptr>
  CUDF_HOST_DEVICE inline void operator()(T& out, T& val, size_type count)
  {
    out = static_cast<int32_t>(val) / count;
  }

  // SFINAE for timestamp types
  template <typename T = _T, std::enable_if_t<cudf::is_timestamp<T>()>* = nullptr>
  CUDF_HOST_DEVICE inline void operator()(T& out, T& val, size_type count)
  {
    out = static_cast<T>(val.time_since_epoch() / count);
  }
};

/**
 * @copydoc cudf::rolling_window(column_view const& input,
 *                               column_view const& default_outputs,
 *                               size_type preceding_window,
 *                               size_type following_window,
 *                               size_type min_periods,
 *                               rolling_aggregation const& agg,
 *                               rmm::device_async_resource_ref mr)
 *
 * @param stream CUDA stream to use for device memory operations
 */
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& default_outputs,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::rolling_window(column_view const& input,
 *                               column_view const& preceding_window,
 *                               column_view const& following_window,
 *                               size_type min_periods,
 *                               rolling_aggregation const& agg,
 *                               rmm::device_async_resource_ref mr);
 *
 * @param stream CUDA stream to use for device memory operations
 */
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& preceding_window,
                                       column_view const& following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);
}  // namespace detail

}  // namespace cudf
