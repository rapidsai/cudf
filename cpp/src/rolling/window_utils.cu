/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/rolling.hpp>
#include <cudf/rolling.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/functional.h>

#include <memory>
#include <stdexcept>
#include <utility>

namespace cudf {
namespace detail {

enum class window_type : std::int8_t {
  BOUNDED,      // Window covers a finite interval
  UNBOUNDED,    // Window covers up to first (respectively last) row in the group
  CURRENT_ROW,  // Window covers all values that are the same as the current row
};

enum class window_endpoint : std::int8_t {
  OPEN,    // For BOUNDED windows, the endpoint of the interval is excluded
  CLOSED,  // For BOUNDED windows, the endpoint of the interval is included
};

enum class window_side : std::int8_t {
  PRECEDING,  // Calculating the preceding window
  FOLLOWING,  // Calculating the following window
};

/*
 * Select the appropriate ordering comparator for the window type and if we're
 * computing the preceding or following window.
 */
template <window_side Side, window_endpoint Endpoint>
struct op_impl {
  using type     = void;
  using type_rev = void;
};

template <>
struct op_impl<window_side::PRECEDING, window_endpoint::CLOSED> {
  using type     = thrust::less<>;
  using type_rev = thrust::greater<>;
};
template <>
struct op_impl<window_side::FOLLOWING, window_endpoint::OPEN> {
  using type     = thrust::less<>;
  using type_rev = thrust::greater<>;
};
template <>
struct op_impl<window_side::PRECEDING, window_endpoint::OPEN> {
  using type     = thrust::less_equal<>;
  using type_rev = thrust::greater_equal<>;
};
template <>
struct op_impl<window_side::FOLLOWING, window_endpoint::CLOSED> {
  using type     = thrust::less_equal<>;
  using type_rev = thrust::greater_equal<>;
};

template <window_side Side, window_endpoint Endpoint, order Order>
using op_t = std::conditional_t<Order == order::ASCENDING,
                                typename op_impl<Side, Endpoint>::type,
                                typename op_impl<Side, Endpoint>::type_rev>;

template <typename T, typename V>
[[nodiscard]] constexpr T add_sat(T x, V y) noexcept
{
  if constexpr (cudf::is_timestamp_t<T>()) {
    using RepT = typename T::rep;
    static_assert(cudf::is_duration_t<V>(), "Can only add durations to timestamps");
    static_assert(cuda::std::is_same_v<typename T::duration, V>,
                  "Duration resolution must match timestamp resolution");
    return T{add_sat(x.time_since_epoch(), y)};
  } else {
    static_assert(cuda::std::is_same_v<T, V>, "Cannot add mismatching types");
    if constexpr (cuda::std::is_signed_v<T>) {
      using U  = std::make_unsigned_t<T>;
      U ux     = static_cast<U>(x);
      U uy     = static_cast<U>(y);
      U result = ux + uy;
      ux = (ux >> std::numeric_limits<T>::digits) + static_cast<U>(std::numeric_limits<T>::max());
      // Note: this cast is implementation defined (until C++20) but all
      // the platforms we care about do the twos-complement thing.
      return static_cast<T>((ux ^ uy) | ~(uy ^ result)) >= 0 ? ux : result;
    } else if constexpr (cuda::std::is_unsigned_v<T>) {
      T result = x + y;
      // Only way we can overflow is in the positive direction
      // in which case result will be less than both of x and y.
      // To saturate, we bit-or with (T)-1 in this case
      return result | (-static_cast<T>(result < x));
    } else if constexpr (cudf::is_duration_t<T>()) {
      return T{add_sat(x.count(), y.count())};
    } else {
      static_assert(std::integral_constant<T, false>(),
                    "Saturating addition only for signed, unsigned integers, "
                    "durations, or timestamps.");
    }
  }
}

template <window_type Type, window_side Side, order Order>
struct window_offset_impl {
  template <typename T,
            CUDF_ENABLE_IF(!(cudf::is_timestamp<T>() or
                             (cudf::is_index_type<T>() and !cudf::is_unsigned<T>())))>
  std::unique_ptr<column> operator()(column_view const&,
                                     scalar const& row_delta,
                                     window_endpoint const endpoint,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    CUDF_FAIL("Unsupported rolling window type.", cudf::data_type_error);
  }

  template <typename InputType, typename OffsetType, typename StrictWeakOrdering>
  struct distance_kernel {
    // Delta from current row that defines the interval endpoint.
    // The endpoint is always current_row_value + row_delta, saturated
    // at the datatype bounds.
    // Note that these are always value-wise, so if you have a
    // descending ordered column you often want row_delta to be
    // negative for the following window.
    OffsetType const* row_delta;
    size_type num_rows;
    cudf::column_device_view::const_iterator<InputType> begin;
    cudf::column_device_view::const_iterator<InputType> end;

    __device__ size_type operator()(size_type i)
    {
      if constexpr (Side == window_side::PRECEDING && Type == window_type::UNBOUNDED) {
        return i + 1;
      } else if constexpr (Side == window_side::PRECEDING && Type == window_type::BOUNDED) {
        return 1 + thrust::distance(thrust::lower_bound(thrust::seq,
                                                        begin,
                                                        end,
                                                        add_sat(*(begin + i), *row_delta),
                                                        StrictWeakOrdering{}),
                                    begin + i);
      } else if constexpr (Side == window_side::PRECEDING && Type == window_type::CURRENT_ROW) {
        return 1 +
               thrust::distance(thrust::lower_bound(
                                  thrust::seq, begin, begin + i, *(begin + i), thrust::equal_to{}),
                                begin + i);
      } else if constexpr (Side == window_side::FOLLOWING && Type == window_type::UNBOUNDED) {
        return num_rows - i - 1;
      } else if (Side == window_side::FOLLOWING && Type == window_type::CURRENT_ROW) {
        return thrust::distance(begin + i,
                                thrust::upper_bound(
                                  thrust::seq, begin + i, end, *(begin + i), thrust::equal_to{})) -
               1;
      } else if constexpr (Side == window_side::FOLLOWING && Type == window_type::BOUNDED) {
        if constexpr (Order == order::ASCENDING) {
          return thrust::distance(begin + i,
                                  thrust::lower_bound(thrust::seq,
                                                      begin,
                                                      end,
                                                      add_sat(*(begin + i), *row_delta),
                                                      StrictWeakOrdering{})) -
                 1;
        } else {
          return thrust::distance(begin + i,
                                  thrust::lower_bound(thrust::seq,
                                                      begin,
                                                      end,
                                                      add_sat(*(begin + i), *row_delta),
                                                      StrictWeakOrdering{})) -
                 1;
        }
      } else {
        CUDF_UNREACHABLE("Unhandled combination of window_side and window_type.");
      }
    }
  };

  template <typename T, typename OffsetType, typename ScalarType>
  [[nodiscard]] std::unique_ptr<column> compute_window_bounds(
    column_view const& input,
    scalar const& row_delta,
    window_endpoint const endpoint,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    auto result = cudf::make_numeric_column(
      cudf::data_type(type_to_id<size_type>()), input.size(), mask_state::UNALLOCATED, stream, mr);
    auto const input_device_view = cudf::column_device_view::create(input, stream);
    auto input_begin             = input_device_view->begin<T>();
    auto input_end               = input_device_view->end<T>();
    auto const d_row_delta       = dynamic_cast<ScalarType const&>(row_delta).data();
    auto copy_n                  = [&](auto kernel) {
      thrust::copy_n(rmm::exec_policy_nosync(stream),
                     cudf::detail::make_counting_transform_iterator(0, kernel),
                     input.size(),
                     result->mutable_view().begin<size_type>());
    };
    if (endpoint == window_endpoint::OPEN) {
      copy_n(distance_kernel<T, OffsetType, op_t<Side, window_endpoint::OPEN, Order>>{
        d_row_delta, input.size(), input_begin, input_end});
    } else if (endpoint == window_endpoint::CLOSED) {
      copy_n(distance_kernel<T, OffsetType, op_t<Side, window_endpoint::CLOSED, Order>>{
        d_row_delta, input.size(), input_begin, input_end});
    } else {
      // Unreachable.
      CUDF_FAIL("Unhandled window type.", std::invalid_argument);
    }
    return result;
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_timestamp<T>())>
  [[nodiscard]] std::unique_ptr<column> operator()(column_view const& input,
                                                   scalar const& row_delta,
                                                   window_endpoint const endpoint,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr) const
  {
    using OffsetType = typename T::duration;
    using ScalarType = duration_scalar<OffsetType>;
    CUDF_EXPECTS(cudf::is_duration(row_delta.type()),
                 "Length and offset must be duration types.",
                 cudf::data_type_error);
    CUDF_EXPECTS(row_delta.type().id() == type_to_id<OffsetType>(),
                 "Length must have same the resolution as the input.",
                 cudf::data_type_error);
    return compute_window_bounds<T, OffsetType, ScalarType>(input, row_delta, endpoint, stream, mr);
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_index_type<T>() and !cudf::is_unsigned<T>())>
  [[nodiscard]] std::unique_ptr<column> operator()(column_view const& input,
                                                   scalar const& row_delta,
                                                   window_endpoint const endpoint,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr) const
  {
    using OffsetType = T;
    using ScalarType = numeric_scalar<OffsetType>;
    CUDF_EXPECTS(have_same_types(input, row_delta),
                 "Input column, length, and offset must have the same type.",
                 cudf::data_type_error);
    return compute_window_bounds<T, OffsetType, ScalarType>(input, row_delta, endpoint, stream, mr);
  }
};

template <window_type Type, window_side Side, order Order>
struct grouped_window_offset_impl {
  template <typename T,
            CUDF_ENABLE_IF(!(cudf::is_timestamp<T>() or
                             (cudf::is_index_type<T>() and !cudf::is_unsigned<T>())))>
  std::unique_ptr<column> operator()(column_view const&,
                                     device_span<size_type const> const& group_labels,
                                     device_span<size_type const> const& group_offsets,
                                     scalar const& row_delta,
                                     window_endpoint const endpoint,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    CUDF_FAIL("Unsupported rolling window type.", cudf::data_type_error);
  }

  template <typename InputType, typename OffsetType, typename StrictWeakOrdering>
  struct distance_kernel {
    device_span<size_type const> group_labels;
    device_span<size_type const> group_offsets;
    // Delta from current row that defines the interval endpoint.
    // The endpoint is always current_row_value + row_delta, saturated
    // at the datatype bounds.
    // Note that these are always value-wise, so if you have a
    // descending ordered column you often want row_delta to be
    // negative for the following window.
    OffsetType const* row_delta;
    size_type num_rows;
    cudf::column_device_view::const_iterator<InputType> begin;
    cudf::column_device_view::const_iterator<InputType> end;

    __device__ size_type operator()(size_type i)
    {
      auto const group       = group_labels[i];
      auto const group_start = group_offsets[group];
      auto const group_end   = group_offsets[group + 1];
      if constexpr (Side == window_side::PRECEDING && Type == window_type::UNBOUNDED) {
        return i - group_start + 1;
      } else if constexpr (Side == window_side::PRECEDING && Type == window_type::BOUNDED) {
        return 1 + thrust::distance(thrust::lower_bound(thrust::seq,
                                                        begin + group_start,
                                                        begin + group_end,
                                                        add_sat(*(begin + i), *row_delta),
                                                        StrictWeakOrdering{}),
                                    begin + i);
      } else if constexpr (Side == window_side::PRECEDING && Type == window_type::CURRENT_ROW) {
        return 1 +
               thrust::distance(
                 thrust::lower_bound(
                   thrust::seq, begin + group_start, begin + i, *(begin + i), thrust::equal_to{}),
                 begin + i);
      } else if constexpr (Side == window_side::FOLLOWING && Type == window_type::UNBOUNDED) {
        return group_end - i - 1;
      } else if (Side == window_side::FOLLOWING && Type == window_type::CURRENT_ROW) {
        return thrust::distance(
                 begin + i,
                 thrust::upper_bound(
                   thrust::seq, begin + i, begin + group_end, *(begin + i), thrust::equal_to{})) -
               1;
      } else if constexpr (Side == window_side::FOLLOWING && Type == window_type::BOUNDED) {
        if constexpr (Order == order::ASCENDING) {
          return thrust::distance(begin + i,
                                  thrust::lower_bound(thrust::seq,
                                                      begin + group_start,
                                                      begin + group_end,
                                                      add_sat(*(begin + i), *row_delta),
                                                      StrictWeakOrdering{})) -
                 1;
        } else {
          return thrust::distance(begin + i,
                                  thrust::lower_bound(thrust::seq,
                                                      begin + group_start,
                                                      begin + group_end,
                                                      add_sat(*(begin + i), *row_delta),
                                                      StrictWeakOrdering{})) -
                 1;
        }
      } else {
        CUDF_UNREACHABLE("Unhandled combination of window_side and window_type.");
      }
    }
  };

  template <typename T, typename OffsetType, typename ScalarType>
  [[nodiscard]] std::unique_ptr<column> compute_window_bounds(
    column_view const& input,
    device_span<size_type const> const& group_labels,
    device_span<size_type const> const& group_offsets,
    scalar const& row_delta,
    window_endpoint const endpoint,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    auto result = cudf::make_numeric_column(
      cudf::data_type(type_to_id<size_type>()), input.size(), mask_state::UNALLOCATED, stream, mr);
    auto const input_device_view = cudf::column_device_view::create(input, stream);
    auto input_begin             = input_device_view->begin<T>();
    auto input_end               = input_device_view->end<T>();
    auto const d_row_delta       = dynamic_cast<ScalarType const&>(row_delta).data();
    auto copy_n                  = [&](auto kernel) {
      thrust::copy_n(rmm::exec_policy_nosync(stream),
                     cudf::detail::make_counting_transform_iterator(0, kernel),
                     input.size(),
                     result->mutable_view().begin<size_type>());
    };
    if (endpoint == window_endpoint::OPEN) {
      copy_n(distance_kernel<T, OffsetType, op_t<Side, window_endpoint::OPEN, Order>>{
        group_labels, group_offsets, d_row_delta, input.size(), input_begin, input_end});
    } else if (endpoint == window_endpoint::CLOSED) {
      copy_n(distance_kernel<T, OffsetType, op_t<Side, window_endpoint::CLOSED, Order>>{
        group_labels, group_offsets, d_row_delta, input.size(), input_begin, input_end});
    } else {
      // Unreachable.
      CUDF_FAIL("Unhandled window type.", std::invalid_argument);
    }
    return result;
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_timestamp<T>())>
  [[nodiscard]] std::unique_ptr<column> operator()(
    column_view const& input,
    device_span<size_type const> const& group_labels,
    device_span<size_type const> const& group_offsets,
    scalar const& row_delta,
    window_endpoint const endpoint,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    using OffsetType = typename T::duration;
    using ScalarType = duration_scalar<OffsetType>;
    CUDF_EXPECTS(cudf::is_duration(row_delta.type()),
                 "Length and offset must be duration types.",
                 cudf::data_type_error);
    CUDF_EXPECTS(row_delta.type().id() == type_to_id<OffsetType>(),
                 "Length must have same the resolution as the input.",
                 cudf::data_type_error);
    return compute_window_bounds<T, OffsetType, ScalarType>(
      input, group_labels, group_offsets, row_delta, endpoint, stream, mr);
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_index_type<T>() and !cudf::is_unsigned<T>())>
  [[nodiscard]] std::unique_ptr<column> operator()(
    column_view const& input,
    device_span<size_type const> const& group_labels,
    device_span<size_type const> const& group_offsets,
    scalar const& row_delta,
    window_endpoint const endpoint,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    using OffsetType = T;
    using ScalarType = numeric_scalar<OffsetType>;
    CUDF_EXPECTS(have_same_types(input, row_delta),
                 "Input column, length, and offset must have the same type.",
                 cudf::data_type_error);
    return compute_window_bounds<T, OffsetType, ScalarType>(
      input, group_labels, group_offsets, row_delta, endpoint, stream, mr);
  }
};

template <window_side Side>
std::unique_ptr<column> dispatch_window_offset_impl(column_view const& input,
                                                    order order,
                                                    null_order null_order,
                                                    scalar const& row_delta,
                                                    std::tuple<window_type, window_endpoint> window,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  auto [type, endpoint] = window;
  if (type == window_type::UNBOUNDED && order == order::ASCENDING) {
    return type_dispatcher(input.type(),
                           window_offset_impl<window_type::UNBOUNDED, Side, order::ASCENDING>{},
                           input,
                           row_delta,
                           endpoint,
                           stream,
                           mr);
  } else if (type == window_type::UNBOUNDED && order == order::DESCENDING) {
    return type_dispatcher(input.type(),
                           window_offset_impl<window_type::UNBOUNDED, Side, order::DESCENDING>{},
                           input,
                           row_delta,
                           endpoint,
                           stream,
                           mr);
  } else if (type == window_type::BOUNDED && order == order::ASCENDING) {
    return type_dispatcher(input.type(),
                           window_offset_impl<window_type::BOUNDED, Side, order::ASCENDING>{},
                           input,
                           row_delta,
                           endpoint,
                           stream,
                           mr);
  } else if (type == window_type::BOUNDED && order == order::DESCENDING) {
    return type_dispatcher(input.type(),
                           window_offset_impl<window_type::BOUNDED, Side, order::DESCENDING>{},
                           input,
                           row_delta,
                           endpoint,
                           stream,
                           mr);
  } else if (type == window_type::CURRENT_ROW) {
    // Doesn't matter what order things are sorted in for CURRENT_ROW.
    return type_dispatcher(input.type(),
                           window_offset_impl<window_type::CURRENT_ROW, Side, order::ASCENDING>{},
                           input,
                           row_delta,
                           endpoint,
                           stream,
                           mr);
  } else {
    CUDF_FAIL("Unhandled window_type and order combination");
  }
}
template <window_side Side>
std::unique_ptr<column> grouped_dispatch_window_offset_impl(
  column_view const& input,
  device_span<size_type const> const& group_labels,
  device_span<size_type const> const& group_offsets,
  order order,
  null_order null_order,
  scalar const& row_delta,
  std::tuple<window_type, window_endpoint> window,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto [type, endpoint] = window;
  if (type == window_type::UNBOUNDED && order == order::ASCENDING) {
    return type_dispatcher(
      input.type(),
      grouped_window_offset_impl<window_type::UNBOUNDED, Side, order::ASCENDING>{},
      input,
      group_labels,
      group_offsets,
      row_delta,
      endpoint,
      stream,
      mr);
  } else if (type == window_type::UNBOUNDED && order == order::DESCENDING) {
    return type_dispatcher(
      input.type(),
      grouped_window_offset_impl<window_type::UNBOUNDED, Side, order::DESCENDING>{},
      input,
      group_labels,
      group_offsets,
      row_delta,
      endpoint,
      stream,
      mr);
  } else if (type == window_type::BOUNDED && order == order::ASCENDING) {
    return type_dispatcher(
      input.type(),
      grouped_window_offset_impl<window_type::BOUNDED, Side, order::ASCENDING>{},
      input,
      group_labels,
      group_offsets,
      row_delta,
      endpoint,
      stream,
      mr);
  } else if (type == window_type::BOUNDED && order == order::DESCENDING) {
    return type_dispatcher(
      input.type(),
      grouped_window_offset_impl<window_type::BOUNDED, Side, order::DESCENDING>{},
      input,
      group_labels,
      group_offsets,
      row_delta,
      endpoint,
      stream,
      mr);
  } else if (type == window_type::CURRENT_ROW) {
    // Doesn't matter what order things are sorted in for CURRENT_ROW.
    return type_dispatcher(
      input.type(),
      grouped_window_offset_impl<window_type::CURRENT_ROW, Side, order::ASCENDING>{},
      input,
      group_labels,
      group_offsets,
      row_delta,
      endpoint,
      stream,
      mr);
  } else {
    CUDF_FAIL("Unhandled window_type and order combination");
  }
}

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> windows_from_offset(
  column_view const& input,
  order order,
  null_order null_order,
  scalar const& preceding_delta,
  scalar const& following_delta,
  std::tuple<window_type, window_endpoint> preceding,
  std::optional<std::tuple<window_type, window_endpoint>> following,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!input.has_nulls(), "Input column cannot have nulls.", std::invalid_argument);
  auto preceding_col = dispatch_window_offset_impl<window_side::PRECEDING>(
    input, order, null_order, preceding_delta, preceding, stream, mr);
  if (following.has_value()) {
    auto following_col = dispatch_window_offset_impl<window_side::FOLLOWING>(
      input, order, null_order, following_delta, *following, stream, mr);
    return {std::move(preceding_col), std::move(following_col)};
  } else {
    return {std::move(preceding_col), nullptr};
  }
}

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> grouped_windows_from_offset(
  table_view const& group_keys,
  column_view const& input,
  order order,
  null_order null_order,
  scalar const& preceding_delta,
  scalar const& following_delta,
  std::tuple<window_type, window_endpoint> preceding,
  std::optional<std::tuple<window_type, window_endpoint>> following,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!input.has_nulls(), "Input column cannot have nulls.", std::invalid_argument);
  CUDF_EXPECTS(group_keys.num_columns() > 0,
               "For grouped window calculation, need at least one grouping column",
               std::invalid_argument);
  using sort_helper = cudf::groupby::detail::sort::sort_groupby_helper;

  // TODO: handle null precedence?
  sort_helper helper{group_keys, cudf::null_policy::INCLUDE, cudf::sorted::YES, {}};
  auto preceding_col =
    grouped_dispatch_window_offset_impl<window_side::PRECEDING>(input,
                                                                helper.group_labels(stream),
                                                                helper.group_offsets(stream),
                                                                order,
                                                                null_order,
                                                                preceding_delta,
                                                                preceding,
                                                                stream,
                                                                mr);
  if (following.has_value()) {
    auto following_col =
      grouped_dispatch_window_offset_impl<window_side::FOLLOWING>(input,
                                                                  helper.group_labels(stream),
                                                                  helper.group_offsets(stream),
                                                                  order,
                                                                  null_order,
                                                                  following_delta,
                                                                  *following,
                                                                  stream,
                                                                  mr);
    return {std::move(preceding_col), std::move(following_col)};
  } else {
    return {std::move(preceding_col), nullptr};
  }
}
}  // namespace detail

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> windows_from_offset(
  column_view const& input,
  scalar const& length,
  scalar const& offset,
  window_type const window_type,
  bool only_preceding,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::windows_from_offset(
    input, length, offset, window_type, only_preceding, stream, mr);
}

}  // namespace cudf
