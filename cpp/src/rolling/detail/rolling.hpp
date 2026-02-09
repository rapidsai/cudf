/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
  // Don't store the output if count is zero since integral division by zero is
  // undefined behaviour. The caller must ensure that the relevant row is
  // marked as invalid with a null.

  // SFINAE for non-bool, non-timestamp types
  template <typename T = _T>
  CUDF_HOST_DEVICE inline void operator()(T& out, T& val, size_type count)
    requires(!(cudf::is_boolean<T>() || cudf::is_timestamp<T>()))
  {
    if (count > 0) { out = val / count; }
  }

  // SFINAE for timestamp types
  template <typename T = _T>
  CUDF_HOST_DEVICE inline void operator()(T& out, T& val, size_type count)
    requires(cudf::is_timestamp<T>())
  {
    if (count > 0) { out = static_cast<T>(val.time_since_epoch() / count); }
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

bool is_valid_rolling_aggregation(data_type input_type, aggregation::Kind kind);
}  // namespace detail

}  // namespace cudf
