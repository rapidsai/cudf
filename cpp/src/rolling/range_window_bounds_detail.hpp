/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/rolling/range_window_bounds.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/wrappers/durations.hpp>

namespace cudf {
namespace detail {

/// Checks if the specified type is supported in a range_window_bounds.
template <typename RangeType>
constexpr bool is_supported_range_type()
{
  return cudf::is_duration<RangeType>() ||
         (std::is_integral_v<RangeType> && !cudf::is_boolean<RangeType>());
}

/// Checks if the specified type is a supported target type,
/// as an orderby column, for comparisons with a range_window_bounds scalar.
template <typename ColumnType>
constexpr bool is_supported_order_by_column_type()
{
  return cudf::is_timestamp<ColumnType>() ||
         (std::is_integral_v<ColumnType> && !cudf::is_boolean<ColumnType>());
}

/// Range-comparable representation type for an orderby column type.
/// This is the datatype used for range comparisons.
///   1. For integral orderby column types `T`, comparisons are done as `T`.
///      E.g. `range_type_for<int32_t>` == `int32_t`.
///   2. For timestamp orderby columns:
///      a. For `TIMESTAMP_DAYS`, the range-type is `DURATION_DAYS`.
///         Comparisons are done in `int32_t`.
///      b. For all other timestamp types, comparisons are done in `int64_t`.
template <typename ColumnType, typename = void>
struct range_type_impl {
  using type     = void;
  using rep_type = void;
};

template <typename ColumnType>
struct range_type_impl<
  ColumnType,
  std::enable_if_t<std::is_integral_v<ColumnType> && !cudf::is_boolean<ColumnType>(), void>> {
  using type     = ColumnType;
  using rep_type = ColumnType;
};

template <typename TimestampType>
struct range_type_impl<TimestampType, std::enable_if_t<cudf::is_timestamp<TimestampType>(), void>> {
  using type     = typename TimestampType::duration;
  using rep_type = typename type::rep;
};

template <typename ColumnType>
using range_type = typename range_type_impl<ColumnType>::type;

template <typename ColumnType>
using range_rep_type = typename range_type_impl<ColumnType>::rep_type;

namespace {

template <typename T>
void assert_non_negative(T const& value)
{
  (void)value;
  if constexpr (std::numeric_limits<T>::is_signed) {
    CUDF_EXPECTS(value >= T{0}, "Range scalar must be >= 0.");
  }
}

template <
  typename RangeT,
  typename RepT,
  std::enable_if_t<std::is_integral_v<RangeT> && !cudf::is_boolean<RangeT>(), void>* = nullptr>
RepT range_comparable_value_impl(scalar const& range_scalar, rmm::cuda_stream_view stream)
{
  auto val = static_cast<numeric_scalar<RangeT> const&>(range_scalar).value(stream);
  assert_non_negative(val);
  return val;
}

template <typename RangeT,
          typename RepT,
          std::enable_if_t<cudf::is_duration<RangeT>(), void>* = nullptr>
RepT range_comparable_value_impl(scalar const& range_scalar, rmm::cuda_stream_view stream)
{
  auto val = static_cast<duration_scalar<RangeT> const&>(range_scalar).value(stream).count();
  assert_non_negative(val);
  return val;
}

}  // namespace

/**
 * @brief Fetch the value of the range_window_bounds scalar, for comparisons
 *        with an orderby column's rows.
 *
 * @tparam OrderByType The type of the orderby column with which the range value will be compared
 * @param range_bounds The range_window_bounds whose value is to be read
 * @param stream The CUDA stream for device memory operations
 * @return RepType Value of the range scalar
 */
template <typename OrderByType>
range_rep_type<OrderByType> range_comparable_value(
  range_window_bounds const& range_bounds,
  rmm::cuda_stream_view stream = cudf::default_stream_value)
{
  auto const& range_scalar = range_bounds.range_scalar();
  using range_type         = cudf::detail::range_type<OrderByType>;

  CUDF_EXPECTS(range_scalar.type().id() == cudf::type_to_id<range_type>(),
               "Unexpected range type for specified orderby column.");

  using rep_type = cudf::detail::range_rep_type<OrderByType>;
  return range_comparable_value_impl<range_type, rep_type>(range_scalar, stream);
}

}  // namespace detail
}  // namespace cudf
