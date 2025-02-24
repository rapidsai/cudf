/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "rolling_utils.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/rolling.hpp>
#include <cudf/rolling.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>

#include <optional>
#include <variant>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace rolling {

/**
 * @brief A group descriptor for an ungrouped rolling window with nulls
 *
 * @param nulls_at_start Are the nulls at the start or end?
 * @param num_rows The number of rows to be rolled over.
 * @param null_count The number of nulls.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped
 * iterator construction.
 */
struct ungrouped_with_nulls {
  bool nulls_at_start;
  cudf::size_type num_rows;
  cudf::size_type null_count;

  static constexpr bool has_nulls{true};
  /**
   * @copydoc ungrouped::row_info
   */
  [[nodiscard]] __device__ constexpr cuda::std::
    tuple<size_type, size_type, size_type, size_type, size_type, size_type, size_type>
    row_info(size_type i) const noexcept
  {
    if (nulls_at_start) {
      return {null_count, 0, num_rows, 0, null_count, null_count, num_rows};
    } else {
      return {null_count,
              num_rows,
              null_count,
              num_rows - null_count,
              num_rows,
              0,
              num_rows - null_count};
    }
  }
};

/**
 * @brief A group descriptor for a grouped rolling window with nulls
 *
 * @param nulls_at_start Are the nulls at the start of each group?
 * @param labels The group labels, mapping from input rows to group.
 * @param offsets The group offsets providing the endpoints of each group.
 * @param null_counts The null counts per group.
 * @param orderby The orderby column, sorted groupwise.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped
 * iterator construction.
 */
struct grouped_with_nulls {
  bool nulls_at_start;
  // Taking raw pointers here to avoid stealing three registers for the sizes which are never
  // needed.
  cudf::size_type const* labels;
  cudf::size_type const* offsets;
  cudf::size_type const* null_counts;

  static constexpr bool has_nulls{true};
  /**
   * @copydoc ungrouped::row_info
   */
  [[nodiscard]] __device__ constexpr cuda::std::
    tuple<size_type, size_type, size_type, size_type, size_type, size_type, size_type>
    row_info(size_type i) const noexcept
  {
    auto const label       = labels[i];
    auto const null_count  = null_counts[label];
    auto const group_start = offsets[label];
    auto const group_end   = offsets[label + 1];
    if (nulls_at_start) {
      return {null_count,
              group_start,
              group_end,
              group_start,
              group_start + null_count,
              group_start + null_count,
              group_end};
    } else {
      return {null_count,
              group_start,
              group_end,
              group_end - null_count,
              group_end,
              group_start,
              group_end - null_count};
    }
  }
};

/*
 * Spark requires that orderby columns with floating point type have a
 * total order on floats where all NaNs compare equal to one-another,
 * and greater than any non-nan value. These structs implement that logic.
 */
template <typename T>
struct less {
  __device__ constexpr bool operator()(T const& x, T const& y) const noexcept
  {
    if constexpr (cuda::std::is_floating_point_v<T>) {
      if (cuda::std::isnan(x)) { return false; }
      return cuda::std::isnan(y) || x < y;
    } else {
      return x < y;
    }
  }
};

template <typename T>
struct less_equal {
  __device__ constexpr bool operator()(T const& x, T const& y) const noexcept
  {
    if constexpr (cuda::std::is_floating_point_v<T>) {
      if (cuda::std::isnan(x)) { return cuda::std::isnan(y); }
      return cuda::std::isnan(y) || x <= y;
    } else {
      return x <= y;
    }
  }
};
template <typename T>
struct greater {
  __device__ constexpr bool operator()(T const& x, T const& y) const noexcept
  {
    if constexpr (cuda::std::is_floating_point_v<T>) {
      if (cuda::std::isnan(x)) { return !cuda::std::isnan(y); }
      return !cuda::std::isnan(y) && x > y;
    } else {
      return x > y;
    }
  }
};

template <typename T>
struct greater_equal {
  __device__ constexpr bool operator()(T const& x, T const& y) const noexcept
  {
    if constexpr (cuda::std::is_floating_point_v<T>) {
      if (cuda::std::isnan(x)) { return true; }
      return !cuda::std::isnan(y) && x >= y;
    } else {
      return x >= y;
    }
  }
};

/**
 * @brief Select the appropriate ordering comparator for the window type.
 *
 * @tparam T The type being compared.
 * @tparam Tag The type of the window.
 */
template <typename T, typename Tag>
struct comparator_impl {
  using op     = void;
  using rev_op = void;
};

template <typename T>
struct comparator_impl<T, bounded_closed> {
  using op     = less<T>;
  using rev_op = greater<T>;
};
template <typename T>
struct comparator_impl<T, bounded_open> {
  using op     = less_equal<T>;
  using rev_op = greater_equal<T>;
};
template <typename T>
struct comparator_impl<T, current_row> {
  using op     = less<T>;
  using rev_op = greater<T>;
};

/**
 * @brief Select the appropriate ordering comparator for the window type.
 *
 * @tparam Tag The type of the window.
 * @tparam Order The sort order of the column used to define the windows.
 */
template <typename T, typename Tag>
struct comparator_t {
  order const order;

  __device__ constexpr bool operator()(T const& x, T const& y) const noexcept
  {
    if (order == order::ASCENDING) {
      using op = typename comparator_impl<T, Tag>::op;
      return op{}(x, y);
    } else {
      using op = typename comparator_impl<T, Tag>::rev_op;
      return op{}(x, y);
    }
  }
};

/**
 * @brief Compute `x + y` saturating at the numeric bounds rather than
 * overflowing.
 *
 * @tparam T the type of the result and left operand.
 * @tparam V the type of the right operand.
 * @param x The left operand.
 * @param y The right operand.
 *
 * @returns Pair of `x + y`, saturated at the numeric limits for the type of
 * `x`, without overflowing or invoking undefined behaviour, and
 * whether overflow occurred.
 *
 * @note If `T` is a numeric type we must have `std::is_same_v<T,
 * V>`. If `T` is a timestamp type, `V` must be a duration type and
 * `std::is_same_v<typename T::duration, V>`. Note in particular that the
 * usual integral promotion rules are not applied. If `T` is a fixed
 * point type, then `V` must be the representation type of `T`, and it
 * is required that `x` and `y` have the same scale. If `T` is a
 * floating point type, then it is required (not checked) that `y` is
 * not inf or nan, otherwise behaviour is undefined, if `x` is finite,
 * then overflow to +-inf is clamped at lowest()/max().
 */
template <typename T, typename V>
[[nodiscard]] __host__ __device__ constexpr inline cuda::std::pair<T, bool> saturating_add(
  T x, V y) noexcept
{
  if constexpr (cudf::is_timestamp_t<T>()) {
    static_assert(cudf::is_duration_t<V>(), "Can only add durations to timestamps");
    static_assert(cuda::std::is_same_v<typename T::duration, V>,
                  "Duration resolution must match timestamp resolution");
    auto const [value, did_overflow] = saturating_add(x.time_since_epoch(), y);
    return {T{value}, did_overflow};
  } else if constexpr (cudf::is_duration_t<T>()) {
    static_assert(cuda::std::is_same_v<T, V>, "Cannot add mismatching types");
    auto const [value, did_overflow] = saturating_add(x.count(), y.count());
    return {T{value}, did_overflow};
  } else if constexpr (cudf::is_fixed_point<T>()) {
    using Rep = typename T::rep;
    // Requirement, not checked, x and y have the same scale.
    static_assert(cuda::std::is_same_v<Rep, V>, "Must add rep type of fixed point to fixed point.");
    auto const [value, did_overflow] = saturating_add(x.value(), y);
    return {T{numeric::scaled_integer<Rep>{value, x.scale()}}, did_overflow};
  } else {
    static_assert(cuda::std::is_same_v<T, V>, "Cannot add mismatching types");

    if constexpr (cuda::std::is_floating_point_v<T>) {
      // Mimicking spark requirements, inf/nan x propagates
      if (cuda::std::isinf(x) || cuda::std::isnan(x)) { return {x, false}; }
      // Requirement, not checked, y is not inf or nan.
      T result = x + y;
      // If the result is outside the range of finite values it can at
      // this point only be +- infinity (we can't generate a nan by
      // adding a non-nan/non-inf y to a non-nan/non-inf x).
      if (result < cuda::std::numeric_limits<T>::lowest()) {
        return {cuda::std::numeric_limits<T>::lowest(), true};
      } else if (result > cuda::std::numeric_limits<T>::max()) {
        return {cuda::std::numeric_limits<T>::max(), true};
      }
      return {result, false};
    } else if constexpr (cuda::std::is_signed_v<T>) {
      using U  = cuda::std::make_unsigned_t<T>;
      U ux     = static_cast<U>(x);
      U uy     = static_cast<U>(y);
      U result = ux + uy;
      ux       = (ux >> cuda::std::numeric_limits<T>::digits) +
           static_cast<U>(cuda::std::numeric_limits<T>::max());
      // Note: this cast is implementation defined (until C++20) but all
      // the platforms we care about do the twos-complement thing.
      auto const did_overflow = static_cast<T>((ux ^ uy) | ~(uy ^ result)) >= 0;
      return {did_overflow ? ux : result, did_overflow};
    } else if constexpr (cuda::std::is_unsigned_v<T>) {
      T result = x + y;
      // Only way we can overflow is in the positive direction
      // in which case result will be less than both of x and y.
      // To saturate, we bit-or with (T)-1 in this case
      auto const did_overflow = result < x;
      return {result | (-static_cast<T>(did_overflow)), did_overflow};
    } else {
      static_assert(cuda::std::integral_constant<T, false>(),
                    "Saturating addition only for signed and unsigned integers, floats, "
                    "durations, fixed point, or timestamps.");
    }
  }
}

/**
 * @brief Compute `x - y` saturating at the numeric bounds rather than
 * overflowing.
 *
 * @tparam T the type of the result and left operand.
 * @tparam V the type of the right operand.
 * @param x The left operand.
 * @param y The right operand.
 *
 * @returns Pair of `x - y`, saturated at the numeric limits for the type of
 * `x`, without overflowing or invoking undefined behaviour, and
 * whether overflow occurred.
 *
 * @note If `T` is a numeric type we must have `std::is_same_v<T,
 * V>`. If `T` is a timestamp type, `V` must be a duration type and
 * `std::is_same_v<typename T::duration, V>`. Note in particular that the
 * usual integral promotion rules are not applied. If `T` is a fixed
 * point type, then `V` must be the representation type of `T`, and it
 * is required that `x` and `y` have the same scale. If `T` is a
 * floating point type, then it is required (not checked) that `y` is
 * not inf or nan, otherwise behaviour is undefined, if `x` is finite,
 * then overflow to +-inf is clamped at lowest()/max().
 */
template <typename T, typename V>
[[nodiscard]] __host__ __device__ constexpr inline cuda::std::pair<T, bool> saturating_sub(
  T x, V y) noexcept
{
  if constexpr (cudf::is_timestamp_t<T>()) {
    static_assert(cudf::is_duration_t<V>(), "Can only add durations to timestamps");
    static_assert(cuda::std::is_same_v<typename T::duration, V>,
                  "Duration resolution must match timestamp resolution");
    auto const [value, did_overflow] = saturating_sub(x.time_since_epoch(), y);
    return {T{value}, did_overflow};
  } else if constexpr (cudf::is_duration_t<T>()) {
    static_assert(cuda::std::is_same_v<T, V>, "Cannot add mismatching types");
    auto const [value, did_overflow] = saturating_sub(x.count(), y.count());
    return {T{value}, did_overflow};
  } else if constexpr (cudf::is_fixed_point<T>()) {
    using Rep = typename T::rep;
    // Requirement, not checked, x and y have the same scale.
    static_assert(cuda::std::is_same_v<Rep, V>, "Must add rep type of fixed point to fixed point.");
    auto const [value, did_overflow] = saturating_sub(x.value(), y);
    return {T{numeric::scaled_integer<Rep>{value, x.scale()}}, did_overflow};
  } else {
    static_assert(cuda::std::is_same_v<T, V>, "Cannot add mismatching types");
    if constexpr (cuda::std::is_floating_point_v<T>) {
      // Mimicking spark requirements, inf/nan x propagates
      if (cuda::std::isinf(x) || cuda::std::isnan(x)) { return {x, false}; }
      // Requirement, not checked, y is not inf or nan.
      T result = x - y;
      // If the result is outside the range of finite values it can at
      // this point only be +- infinity (we can't generate a nan by
      // subtracting a non-nan/non-inf y from a non-nan/non-inf x).
      if (result < cuda::std::numeric_limits<T>::lowest()) {
        return {cuda::std::numeric_limits<T>::lowest(), true};
      } else if (result > cuda::std::numeric_limits<T>::max()) {
        return {cuda::std::numeric_limits<T>::max(), true};
      }
      return {result, false};
    } else if constexpr (cuda::std::is_signed_v<T>) {
      using U  = cuda::std::make_unsigned_t<T>;
      U ux     = static_cast<U>(x);
      U uy     = static_cast<U>(y);
      U result = ux - uy;
      ux       = (ux >> cuda::std::numeric_limits<T>::digits) +
           static_cast<U>(cuda::std::numeric_limits<T>::max());
      // Note: this cast is implementation defined (until C++20) but all
      // the platforms we care about do the twos-complement thing.
      auto const did_overflow = static_cast<T>((ux ^ uy) & (ux ^ result)) < 0;
      return {did_overflow ? ux : result, did_overflow};
    } else if constexpr (cuda::std::is_unsigned_v<T>) {
      T result = x - y;
      // Only way we can overflow is in the negative direction
      // in which case result will be greater than either of x or y.
      // To saturate, we bit-and with (T)0 in this case
      auto const did_overflow = result > x;
      return {result & (-static_cast<T>(!did_overflow)), did_overflow};
    } else {
      static_assert(cuda::std::integral_constant<T, false>(),
                    "Saturating subtraction only for signed and unsigned integers, floats, "
                    "durations, fixed point, or timestamps.");
    }
  }
}
/**
 * @brief Functor to compute distance from a given row to the edge
 * of the window.
 *
 * @tparam Grouping Object defining how the orderby column is
 * grouped.
 * @tparam OrderbyT Type of elements in the orderby columns.
 * @tparam DeltaT Type of the elements in the scalar delta (returned
 * by scalar.data()).
 * @param groups The grouping object.
 * @param row_delta Pointer to row delta on device.
 * @param begin Iterator to begin of orderby column on device.
 * @param end Iterator to end of orderby column on device.
 *
 * @note Let `x` be the value of the current row and `delta` the provided
 * row delta then for bounded windows the endpoints are computed as follows.
 *
 *           | ASCENDING | DESCENDING
 * ----------+-----------+-----------
 * PRECEDING | x - delta | x + delta
 * FOLLOWING | x + delta | x - delta
 *
 * See `saturating_sub` and `saturating_add` for details of the implementation of
 * saturating addition/subtraction.
 */
template <typename WindowTag, typename Grouping, typename OrderbyT, typename DeltaT>
struct distance_functor {
  distance_functor(Grouping groups,
                   DeltaT const* row_delta,
                   direction const direction,
                   order const order,
                   column_device_view::const_iterator<OrderbyT> begin,
                   column_device_view::const_iterator<OrderbyT> end)
    : groups{groups},
      row_delta{row_delta},
      direction{direction},
      order{order},
      begin{begin},
      end{end}
  {
  }
  Grouping groups;  ///< Group information to determine bounds on current row's window
  static_assert(cuda::std::is_same_v<Grouping, ungrouped> ||
                  cuda::std::is_same_v<Grouping, grouped> ||
                  cuda::std::is_same_v<Grouping, ungrouped_with_nulls> ||
                  cuda::std::is_same_v<Grouping, grouped_with_nulls>,
                "Invalid grouping descriptor");
  DeltaT const* row_delta;  ///< Delta from current row that defines the interval endpoint. This
                            ///< pointer is null for UNBOUNDED and CURRENT_ROW windows.
  column_device_view::const_iterator<OrderbyT> begin;  ///< Iterator to beginning of orderby column
  column_device_view::const_iterator<OrderbyT> end;    ///< Iterator to end of orderby column
  direction const direction;
  order const order;

  /**
   * @brief Compute the offset to the end of the window.
   *
   * @param i The current row index.
   * @return Offset to the current row's window endpoint.
   */
  [[nodiscard]] __device__ size_type operator()(size_type i) const
  {
    using Comp = comparator_t<OrderbyT, WindowTag>;
    auto const [null_count, group_start, group_end, null_start, null_end, start, end] =
      groups.row_info(i);
    if (direction == direction::PRECEDING) {
      if constexpr (cuda::std::is_same_v<WindowTag, unbounded>) { return i - group_start + 1; }
      // TODO: If the window is BOUNDED_OPEN, what does it mean for a row to fall in the null
      // group? Not that important because only spark allows nulls in the orderby column, and it
      // doesn't have BOUNDED_OPEN windows.
      if (Grouping::has_nulls && i >= null_start && i < null_end) { return i - null_start + 1; }
      if constexpr (cuda::std::is_same_v<WindowTag, current_row>) {
        return 1 + thrust::distance(thrust::lower_bound(
                                      thrust::seq, begin + start, begin + i, begin[i], Comp{order}),
                                    begin + i);
      } else if constexpr (!cuda::std::is_same_v<WindowTag, unbounded>) {
        // The preceding endpoint is computed via row_value - delta.
        // When delta is positive, this can only overflow towards -infinity.
        // If we did overflow towards -infinity, then the value
        // we're searching for is some min. But -infinity < min, so
        // we must always use a `bounded_closed` window so that
        // orderby = [min, ...] with positive delta picks up that
        // row in the window.
        // Conversely, when delta is negative, we can only overflow
        // towards +infinity. If we did overflow towards +infinity
        // then the value we're searching for is some max. But
        // +infinity > max, so we must use a `bounded_open` window
        // so that orderby = [..., max] with negative delta does not
        // pick up that row in the window.
        // When the orderby column is sorted in descending order the
        // above applies mutatis mutandis with a sign flip.
        OrderbyT value;
        bool did_overflow{false};
        if (order == order::ASCENDING) {
          auto const result = saturating_sub(begin[i], *row_delta);
          value             = cuda::std::get<0>(result);
          did_overflow      = cuda::std::get<1>(result);
        } else {
          auto const result = saturating_add(begin[i], *row_delta);
          value             = cuda::std::get<0>(result);
          did_overflow      = cuda::std::get<1>(result);
        }
        if (did_overflow) {
          if (*row_delta > DeltaT{0}) {
            return 1 + thrust::distance(
                         thrust::lower_bound(thrust::seq,
                                             begin + start,
                                             begin + end,
                                             value,
                                             comparator_t<OrderbyT, bounded_closed>{order}),
                         begin + i);
          } else {
            return 1 + thrust::distance(
                         thrust::lower_bound(thrust::seq,
                                             begin + start,
                                             begin + end,
                                             value,
                                             comparator_t<OrderbyT, bounded_open>{order}),
                         begin + i);
          }
        } else {
          return 1 +
                 thrust::distance(
                   thrust::lower_bound(thrust::seq, begin + start, begin + end, value, Comp{order}),
                   begin + i);
        }
      } else {
        CUDF_UNREACHABLE("Unexpected WindowTag");
      }
    } else {
      if constexpr (cuda::std::is_same_v<WindowTag, unbounded>) { return group_end - i - 1; }
      if (Grouping::has_nulls && i >= null_start && i < null_end) { return null_end - i - 1; }
      if constexpr (cuda::std::is_same_v<WindowTag, current_row>) {
        return thrust::distance(
                 begin + i,
                 thrust::upper_bound(thrust::seq, begin + i, begin + end, begin[i], Comp{order})) -
               1;
      } else if constexpr (!cuda::std::is_same_v<WindowTag, unbounded>) {
        // The following endpoint is computed via row_value + delta.
        // When delta is positive, this can only overflow towards +infinity.
        // If we did overflow towards +infinity, then the value
        // we're searching for is some max. But +infinity > max, so
        // we must always use a `bounded_closed` window so that
        // orderby = [..., max] with positive delta picks up that
        // row in the window.
        // Conversely, when delta is negative, we can only overflow
        // towards -infinity. If we did overflow towards -infinity
        // then the value we're searching for is some min. But
        // -infinity < min, so we must use a `bounded_open` window
        // so that orderby = [min, ...] with negative delta does not
        // pick up that row in the window.
        // When the orderby column is sorted in descending order the
        // above applies mutatis mutandis with a sign flip.
        OrderbyT value;
        bool did_overflow{false};
        if (order == order::ASCENDING) {
          auto const result = saturating_add(begin[i], *row_delta);
          value             = cuda::std::get<0>(result);
          did_overflow      = cuda::std::get<1>(result);
        } else {
          auto const result = saturating_sub(begin[i], *row_delta);
          value             = cuda::std::get<0>(result);
          did_overflow      = cuda::std::get<1>(result);
        }
        if (did_overflow) {
          if (*row_delta > DeltaT{0}) {
            return thrust::distance(
                     begin + i,
                     thrust::upper_bound(thrust::seq,
                                         begin + start,
                                         begin + end,
                                         value,
                                         comparator_t<OrderbyT, bounded_closed>{order})) -
                   1;
          } else {
            return thrust::distance(
                     begin + i,
                     thrust::upper_bound(thrust::seq,
                                         begin + start,
                                         begin + end,
                                         value,
                                         comparator_t<OrderbyT, bounded_open>{order})) -
                   1;
          }
        } else {
          return thrust::distance(begin + i,
                                  thrust::upper_bound(
                                    thrust::seq, begin + start, begin + end, value, Comp{order})) -
                 1;
        }
      } else {
        CUDF_UNREACHABLE("Unexpected WindowTag");
      }
    }
  }
};

/**
 * @brief Functor to dispatch computation of clamped range-based rolling window bounds.
 *
 * @tparam Direction The direction (preceding or following) of the window
 * @tparam WindowTag The tag indicating the type of window being computed
 * @tparam Order The sort order of the orderby column defining the window.
 */
template <typename WindowTag>
struct range_window_clamper {
  direction const direction;
  order const order;
  static_assert(cuda::std::is_same_v<WindowTag, unbounded> ||
                  cuda::std::is_same_v<WindowTag, current_row> ||
                  cuda::std::is_same_v<WindowTag, bounded_closed> ||
                  cuda::std::is_same_v<WindowTag, bounded_open>,
                "Invalid WindowTag descriptor");
  /**
   * @brief Compute the window bounds (possibly grouped) for an orderby column.
   *
   * @tparam OrderbyT element type of the orderby column (dispatched on)
   * @tparam ScalarT Concrete scalar type of the scalar row delta
   * @param orderby Column used to define windows.
   * @param grouping optional pre-processed group information.
   * @param nulls_at_start If the orderby column contains nulls, are they are the start or the end?
   * @param row_delta the delta applied to each row, will be null if the window is of type
   * `UNBOUNDED` or `CURRENT_ROW`, otherwise non-null. If non-null, must be a finite value or
   * behaviour is undefined.
   * @param stream CUDA stream used for kernel launches and memory allocations
   * @param mr Memory resource used for memory allocations.
   *
   * @return A column containing the computed endpoints for the given window description.
   */
  template <typename OrderbyT, typename ScalarT = cudf::scalar_type_t<OrderbyT>>
  [[nodiscard]] std::unique_ptr<column> window_bounds(
    column_view const& orderby,
    std::optional<preprocessed_group_info> const& grouping,
    bool nulls_at_start,
    scalar const* row_delta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    auto result = make_numeric_column(
      data_type(type_to_id<size_type>()), orderby.size(), mask_state::UNALLOCATED, stream, mr);
    auto d_orderby          = column_device_view::create(orderby, stream);
    auto d_begin            = d_orderby->begin<OrderbyT>();
    auto d_end              = d_orderby->end<OrderbyT>();
    auto const* d_row_delta = row_delta ? dynamic_cast<ScalarT const*>(row_delta)->data() : nullptr;
    using DeltaT = cuda::std::remove_cv_t<cuda::std::remove_pointer_t<decltype(d_row_delta)>>;
    auto copy_n  = [&](auto&& grouping) {
      using Grouping = cuda::std::decay_t<decltype(grouping)>;
      thrust::copy_n(rmm::exec_policy_nosync(stream),
                     cudf::detail::make_counting_transform_iterator(
                       0,
                       distance_functor<WindowTag, Grouping, OrderbyT, DeltaT>{
                         grouping, d_row_delta, direction, order, d_begin, d_end}),
                     orderby.size(),
                     result->mutable_view().begin<size_type>());
    };
    if (grouping.has_value()) {
      if (orderby.has_nulls()) {
        copy_n(grouped_with_nulls{nulls_at_start,
                                  grouping->labels.data(),
                                  grouping->offsets.data(),
                                  grouping->nulls_per_group.data()});
      } else {
        copy_n(grouped{grouping->labels.data(), grouping->offsets.data()});
      }
    } else {
      if (orderby.has_nulls()) {
        copy_n(ungrouped_with_nulls{nulls_at_start, orderby.size(), orderby.null_count()});
      } else {
        copy_n(ungrouped{orderby.size()});
      }
    }
    return result;
  }

  /**
   * @brief Is the given type supported as an orderby column.
   *
   * @tparam The type of the elements of the orderby column.
   */
  template <typename OrderbyT>
  static constexpr bool is_supported()
  {
    return (cuda::std::is_same_v<OrderbyT, cudf::string_view> &&
            (cuda::std::is_same_v<WindowTag, current_row> ||
             cuda::std::is_same_v<WindowTag, unbounded>)) ||
           cudf::is_numeric_not_bool<OrderbyT>() || cudf::is_timestamp<OrderbyT>() ||
           cudf::is_fixed_point<OrderbyT>();
  }

  template <typename OrderbyT, CUDF_ENABLE_IF(cudf::is_timestamp<OrderbyT>())>
  [[nodiscard]] std::unique_ptr<column> operator()(
    column_view const& orderby,
    std::optional<preprocessed_group_info> const& grouping,
    bool nulls_at_start,
    scalar const* row_delta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    using ScalarT = cudf::scalar_type_t<typename OrderbyT::duration>;
    CUDF_EXPECTS(!row_delta || cudf::is_duration(row_delta->type()),
                 "Row delta must be a duration type.",
                 cudf::data_type_error);
    CUDF_EXPECTS(!row_delta || row_delta->type().id() == type_to_id<typename OrderbyT::duration>(),
                 "Row delta must have same the resolution as orderby.",
                 cudf::data_type_error);
    return window_bounds<OrderbyT, ScalarT>(
      orderby, grouping, nulls_at_start, row_delta, stream, mr);
  }

  template <typename OrderbyT, CUDF_ENABLE_IF(cudf::is_fixed_point<OrderbyT>())>
  [[nodiscard]] std::unique_ptr<column> operator()(
    column_view const& orderby,
    std::optional<preprocessed_group_info> const& grouping,
    bool nulls_at_start,
    scalar const* row_delta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(!row_delta || (orderby.type().id() == row_delta->type().id()),
                 "Orderby column and row_delta must both be fixed point.",
                 cudf::data_type_error);
    // TODO: Push this requirement onto the caller and just check for
    // equal scales (avoids a kernel launch to rescale)
    CUDF_EXPECTS(!row_delta || row_delta->type().scale() >= orderby.type().scale(),
                 "row_delta must have at least as much scale as orderby column.",
                 cudf::data_type_error);
    if (row_delta && row_delta->type().scale() != orderby.type().scale()) {
      auto const value =
        dynamic_cast<fixed_point_scalar<OrderbyT> const*>(row_delta)->fixed_point_value(stream);
      auto const new_scalar = cudf::fixed_point_scalar<OrderbyT>{
        value.rescaled(numeric::scale_type{orderby.type().scale()}), true, stream};
      return window_bounds<OrderbyT>(orderby, grouping, nulls_at_start, &new_scalar, stream, mr);
    }
    return window_bounds<OrderbyT>(orderby, grouping, nulls_at_start, row_delta, stream, mr);
  }

  template <typename OrderbyT, CUDF_ENABLE_IF(cudf::is_numeric_not_bool<OrderbyT>())>
  [[nodiscard]] std::unique_ptr<column> operator()(
    column_view const& orderby,
    std::optional<preprocessed_group_info> const& grouping,
    bool nulls_at_start,
    scalar const* row_delta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(!row_delta || cudf::have_same_types(orderby, *row_delta),
                 "Orderby column and row_delta must have the same type.",
                 cudf::data_type_error);
    return window_bounds<OrderbyT>(orderby, grouping, nulls_at_start, row_delta, stream, mr);
  }

  template <typename OrderbyT,
            CUDF_ENABLE_IF(cuda::std::is_same_v<OrderbyT, cudf::string_view> &&
                           (cuda::std::is_same_v<WindowTag, current_row> ||
                            cuda::std::is_same_v<WindowTag, unbounded>))>
  [[nodiscard]] std::unique_ptr<column> operator()(
    column_view const& orderby,
    std::optional<preprocessed_group_info> const& grouping,
    bool nulls_at_start,
    scalar const* row_delta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(!row_delta,
                 "Not expecting window range to have value for string-based window calculation");
    return window_bounds<OrderbyT>(orderby, grouping, nulls_at_start, row_delta, stream, mr);
  }

  template <typename OrderbyT, CUDF_ENABLE_IF(!is_supported<OrderbyT>())>
  std::unique_ptr<column> operator()(column_view const&,
                                     std::optional<preprocessed_group_info> const&,
                                     bool,
                                     scalar const*,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref) const
  {
    CUDF_FAIL("Unsupported rolling window type.", cudf::data_type_error);
  }
};
}  // namespace rolling
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
