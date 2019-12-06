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

#ifndef ROLLING_DETAIL_HPP
#define ROLLING_DETAIL_HPP

#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/utilities/traits.hpp>

namespace cudf
{

// helper functions - used in the rolling window implementation and tests
namespace detail
{
  // return true if ColumnType is arithmetic type or
  // AggOp is min_op/max_op/count_op for wrapper (non-arithmetic) types
  template <typename ColumnType, class AggOp, bool is_mean>
  static constexpr bool is_supported()
  {
    constexpr bool comparable_countable_op =
      std::is_same<AggOp, DeviceMin>::value ||
      std::is_same<AggOp, DeviceMax>::value ||
      std::is_same<AggOp, DeviceCount>::value;

    constexpr bool timestamp_mean =
      is_mean &&
      std::is_same<AggOp, DeviceSum>::value &&
      cudf::is_timestamp<ColumnType>();

    return !std::is_same<ColumnType, cudf::string_view>::value &&
           (cudf::is_numeric<ColumnType>() ||
            comparable_countable_op ||
            timestamp_mean);
  }

  // store functor
  template <typename T, bool is_mean>
  struct store_output_functor
  {
    CUDA_HOST_DEVICE_CALLABLE void operator()(T &out, T &val, size_type count)
    {
      out = val;  
    }
  };

  // Specialization for MEAN
  template <typename _T>
  struct store_output_functor<_T, true>
  {
    // SFINAE for non-bool types
    template <typename T = _T,
      std::enable_if_t<!(cudf::is_boolean<T>() || cudf::is_timestamp<T>())>* = nullptr>
    CUDA_HOST_DEVICE_CALLABLE void operator()(T &out, T &val, size_type count)
    {
      out = val / count;
    }

    // SFINAE for bool type
    template <typename T = _T, std::enable_if_t<cudf::is_boolean<T>()>* = nullptr>
    CUDA_HOST_DEVICE_CALLABLE void operator()(T &out, T &val, size_type count)
    {
      out = static_cast<int32_t>(val) / count;
    }

    // SFINAE for timestamp types
    template <typename T = _T, std::enable_if_t<cudf::is_timestamp<T>()>* = nullptr>
    CUDA_HOST_DEVICE_CALLABLE void operator()(T &out, T &val, size_type count)
    {
      out = val.time_since_epoch() / count;
    }
  };
}  // namespace cudf::detail

} // namespace cudf

#endif

