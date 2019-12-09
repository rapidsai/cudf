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

#include <utilities/legacy/device_operators.cuh>

namespace cudf
{

// helper functions - used in the rolling window implementation and tests
namespace detail
{
  // return true if ColumnType is arithmetic type or
  // AggOp is min_op/max_op/count_op for wrapper (non-arithmetic) types
  template <typename ColumnType, class AggOp>
  static constexpr bool is_supported()
  {
    return std::is_arithmetic<ColumnType>::value ||
           std::is_same<AggOp, DeviceMin>::value ||
           std::is_same<AggOp, DeviceMax>::value ||
           std::is_same<AggOp, DeviceCount>::value;
  }

  // store functor
  template <typename ColumnType, bool average>
  struct store_output_functor
  {
    CUDA_HOST_DEVICE_CALLABLE void operator()(ColumnType &out, ColumnType &val, cudf::size_type count)
    {
      out = val;
    }
  };

  // partial specialization for AVG
  template <typename ColumnType>
  struct store_output_functor<ColumnType, true>
  {
    CUDA_HOST_DEVICE_CALLABLE void operator()(ColumnType &out, ColumnType &val, cudf::size_type count)
    {
      out = val / count;
    }
  };
}  // namespace cudf::detail

} // namespace cudf

#endif