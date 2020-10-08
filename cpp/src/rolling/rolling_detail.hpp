/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/utilities/traits.hpp>

namespace cudf {
// helper functions - used in the rolling window implementation and tests

namespace detail {
// return true the aggregation is valid for the specified ColumnType
// valid aggregations may still be further specialized (eg, is_string_specialized)
template <typename ColumnType, class AggOp, aggregation::Kind op>
static constexpr bool is_rolling_supported()
{
  if (!cudf::detail::is_valid_aggregation<ColumnType, op>()) {
    return false;
  } else if (cudf::is_numeric<ColumnType>() or cudf::is_duration<ColumnType>()) {
    constexpr bool is_comparable_countable_op = std::is_same<AggOp, DeviceMin>::value or
                                                std::is_same<AggOp, DeviceMax>::value or
                                                std::is_same<AggOp, DeviceCount>::value;

    constexpr bool is_operation_supported =
      (op == aggregation::SUM) or (op == aggregation::MIN) or (op == aggregation::MAX) or
      (op == aggregation::COUNT_VALID) or (op == aggregation::COUNT_ALL) or
      (op == aggregation::MEAN) or (op == aggregation::ROW_NUMBER) or (op == aggregation::LEAD) or
      (op == aggregation::LAG);

    constexpr bool is_valid_numeric_agg =
      (cudf::is_numeric<ColumnType>() or cudf::is_duration<ColumnType>() or
       is_comparable_countable_op) and
      is_operation_supported;

    return is_valid_numeric_agg;

  } else if (cudf::is_timestamp<ColumnType>()) {
    return (op == aggregation::MIN) or (op == aggregation::MAX) or
           (op == aggregation::COUNT_VALID) or (op == aggregation::COUNT_ALL) or
           (op == aggregation::ROW_NUMBER) or (op == aggregation::LEAD) or (op == aggregation::LAG);

  } else if (std::is_same<ColumnType, cudf::string_view>()) {
    return (op == aggregation::MIN) or (op == aggregation::MAX) or
           (op == aggregation::COUNT_VALID) or (op == aggregation::COUNT_ALL) or
           (op == aggregation::ROW_NUMBER);

  } else if (std::is_same<ColumnType, cudf::list_view>()) {
    return (op == aggregation::COUNT_VALID) or (op == aggregation::COUNT_ALL) or
           (op == aggregation::ROW_NUMBER);
  } else
    return false;
}

// return true if this Op is specialized for strings.
template <typename ColumnType, class AggOp, aggregation::Kind Op>
static constexpr bool is_rolling_string_specialization()
{
  return std::is_same<ColumnType, cudf::string_view>::value and
         ((aggregation::MIN == Op and std::is_same<AggOp, DeviceMin>::value) or
          (aggregation::MAX == Op and std::is_same<AggOp, DeviceMax>::value));
}

// store functor
template <typename T, bool is_mean = false>
struct rolling_store_output_functor {
  CUDA_HOST_DEVICE_CALLABLE void operator()(T &out, T &val, size_type count) { out = val; }
};

// Specialization for MEAN
template <typename _T>
struct rolling_store_output_functor<_T, true> {
  // SFINAE for non-bool types
  template <typename T                                                              = _T,
            std::enable_if_t<!(cudf::is_boolean<T>() || cudf::is_timestamp<T>())> * = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(T &out, T &val, size_type count)
  {
    out = val / count;
  }

  // SFINAE for bool type
  template <typename T = _T, std::enable_if_t<cudf::is_boolean<T>()> * = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(T &out, T &val, size_type count)
  {
    out = static_cast<int32_t>(val) / count;
  }

  // SFINAE for timestamp types
  template <typename T = _T, std::enable_if_t<cudf::is_timestamp<T>()> * = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(T &out, T &val, size_type count)
  {
    out = static_cast<T>(val.time_since_epoch() / count);
  }
};
}  // namespace detail

}  // namespace cudf

#endif
