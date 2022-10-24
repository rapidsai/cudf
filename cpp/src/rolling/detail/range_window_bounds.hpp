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
  return cudf::is_duration<RangeType>() || cudf::is_fixed_point<RangeType>() ||
         (std::is_integral_v<RangeType> && !cudf::is_boolean<RangeType>());
}

/// Checks if the specified type is a supported target type,
/// as an orderby column, for comparisons with a range_window_bounds scalar.
template <typename ColumnType>
constexpr bool is_supported_order_by_column_type()
{
  return cudf::is_timestamp<ColumnType>() || cudf::is_fixed_point<ColumnType>() ||
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
///   3. For decimal types, all comparisons are done with the rep type,
///      after scaling the rep value to the same scale as the order by column:
///      a. For decimal32, the range-type is `int32_t`.
///      b. For decimal64, the range-type is `int64_t`.
///      c. For decimal128, the range-type is `__int128_t`.
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

template <typename FixedPointType>
struct range_type_impl<FixedPointType,
                       std::enable_if_t<cudf::is_fixed_point<FixedPointType>(), void>> {
  using type     = FixedPointType;
  using rep_type = typename type::rep;
};

template <typename ColumnType>
using range_type = typename range_type_impl<ColumnType>::type;

template <typename ColumnType>
using range_rep_type = typename range_type_impl<ColumnType>::rep_type;

template <typename T>
void assert_non_negative([[maybe_unused]] T const& value)
{
  if constexpr (std::numeric_limits<T>::is_signed) {
    CUDF_EXPECTS(value >= T{0}, "Range scalar must be >= 0.");
  }
}

template <typename RangeT,
          typename RepT,
          CUDF_ENABLE_IF(std::is_integral_v<RangeT> && !cudf::is_boolean<RangeT>())>
RepT range_comparable_value_impl(scalar const& range_scalar,
                                 bool,
                                 data_type const&,
                                 rmm::cuda_stream_view stream)
{
  auto val = static_cast<numeric_scalar<RangeT> const&>(range_scalar).value(stream);
  assert_non_negative(val);
  return val;
}

template <typename RangeT, typename RepT, CUDF_ENABLE_IF(cudf::is_duration<RangeT>())>
RepT range_comparable_value_impl(scalar const& range_scalar,
                                 bool,
                                 data_type const&,
                                 rmm::cuda_stream_view stream)
{
  auto val = static_cast<duration_scalar<RangeT> const&>(range_scalar).value(stream).count();
  assert_non_negative(val);
  return val;
}

template <typename RangeT, typename RepT, CUDF_ENABLE_IF(cudf::is_fixed_point<RangeT>())>
RepT range_comparable_value_impl(scalar const& range_scalar,
                                 bool is_unbounded,
                                 data_type const& order_by_data_type,
                                 rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(is_unbounded || range_scalar.type().scale() >= order_by_data_type.scale(),
               "Range bounds scalar must match/exceed the scale of the orderby column.");
  auto const fixed_point_value =
    static_cast<fixed_point_scalar<RangeT> const&>(range_scalar).fixed_point_value(stream);
  auto const value =
    fixed_point_value.rescaled(numeric::scale_type{order_by_data_type.scale()}).value();
  assert_non_negative(value);
  return value;
}

/**
 * @brief Fetch the value of the range_window_bounds scalar, for comparisons
 *        with an orderby column's rows.
 *
 * @tparam OrderByType The type of the orderby column with which the range value will be compared
 * @param range_bounds The range_window_bounds whose value is to be read
 * @param order_by_data_type The data type for the order-by column
 * @param stream The CUDA stream for device memory operations
 * @return RepType Value of the range scalar
 */
template <typename OrderByType>
range_rep_type<OrderByType> range_comparable_value(
  range_window_bounds const& range_bounds,
  data_type const& order_by_data_type = data_type{type_to_id<OrderByType>()},
  rmm::cuda_stream_view stream        = cudf::get_default_stream())
{
  auto const& range_scalar = range_bounds.range_scalar();
  using range_type         = cudf::detail::range_type<OrderByType>;

  CUDF_EXPECTS(range_scalar.type().id() == cudf::type_to_id<range_type>(),
               "Range bounds scalar must match the type of the orderby column.");

  using rep_type = cudf::detail::range_rep_type<OrderByType>;
  return range_comparable_value_impl<range_type, rep_type>(
    range_scalar, range_bounds.is_unbounded(), order_by_data_type, stream);
}

}  // namespace detail
}  // namespace cudf
