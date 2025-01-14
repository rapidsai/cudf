/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "detail/optimized_unbounded_window.hpp"
#include "detail/range_comparator_utils.cuh"
#include "detail/range_window_bounds.hpp"
#include "detail/rolling.cuh"
#include "detail/rolling_jit.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/rolling.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/rolling/range_window_bounds.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/partition.h>

namespace cudf {

namespace detail {

/// Preceding window calculation functor.
template <bool preceding_less_than_1>
struct row_based_preceding_calc {
  cudf::size_type const* _group_offsets_begin;
  cudf::size_type const* _group_labels_begin;
  cudf::size_type const _preceding_window;

  row_based_preceding_calc(rmm::device_uvector<cudf::size_type> const& group_offsets,
                           rmm::device_uvector<cudf::size_type> const& group_labels,
                           cudf::size_type const& preceding_window)
    : _group_offsets_begin(group_offsets.data()),
      _group_labels_begin(group_labels.data()),
      _preceding_window(preceding_window)
  {
  }

  __device__ cudf::size_type operator()(cudf::size_type const& idx) const
  {
    auto group_label = _group_labels_begin[idx];
    if constexpr (preceding_less_than_1) {  // where 1 indicates only the current row.
      auto group_end = _group_offsets_begin[group_label + 1];
      return thrust::maximum{}(_preceding_window, -(group_end - 1 - idx));
    } else {
      auto group_start = _group_offsets_begin[group_label];
      return thrust::minimum{}(_preceding_window,
                               idx - group_start + 1);  // Preceding includes current row.
    }
  }
};

/// Helper to materialize preceding-window column, corrected to respect group boundaries.
/// E.g. If preceding window == 5, then,
///   1. For the first row in the group, the preceding is set to 1,
///   2. For the next row in the group, preceding is set to 2, etc.
std::unique_ptr<cudf::column> make_preceding_column(
  rmm::device_uvector<cudf::size_type> const& group_offsets,
  rmm::device_uvector<cudf::size_type> const& group_labels,
  cudf::size_type const& preceding_window,
  cudf::size_type const& num_rows,
  rmm::cuda_stream_view stream)
{
  if (preceding_window < 1) {
    auto const calc = row_based_preceding_calc<true>(group_offsets, group_labels, preceding_window);
    return cudf::detail::expand_to_column(calc, num_rows, stream);
  } else {
    auto const calc =
      row_based_preceding_calc<false>(group_offsets, group_labels, preceding_window);
    return cudf::detail::expand_to_column(calc, num_rows, stream);
  }
}

/// Following window calculation functor.
template <bool following_less_than_0>
struct row_based_following_calc {
  cudf::size_type const* _group_offsets_begin;
  cudf::size_type const* _group_labels_begin;
  cudf::size_type const _following_window;

  row_based_following_calc(rmm::device_uvector<cudf::size_type> const& group_offsets,
                           rmm::device_uvector<cudf::size_type> const& group_labels,
                           cudf::size_type const& following_window)
    : _group_offsets_begin(group_offsets.data()),
      _group_labels_begin(group_labels.data()),
      _following_window(following_window)
  {
  }

  __device__ cudf::size_type operator()(cudf::size_type const& idx) const
  {
    auto group_label = _group_labels_begin[idx];
    if constexpr (following_less_than_0) {
      auto group_start = _group_offsets_begin[group_label];
      return thrust::maximum{}(_following_window, -(idx - group_start) - 1);
    } else {
      auto group_end =
        _group_offsets_begin[group_label + 1];  // Cannot fall off the end, since offsets
                                                // is capped with `input.size()`.
      return thrust::minimum{}(_following_window, (group_end - 1) - idx);
    }
  }
};

/// Helper to materialize following-window column, corrected to respect group boundaries.
/// i.e. If following window == 5, then:
///   1. For the last row in the group, the following is set to 0.
///   2. For the second last row in the group, following is set to 1, etc.
std::unique_ptr<cudf::column> make_following_column(
  rmm::device_uvector<cudf::size_type> const& group_offsets,
  rmm::device_uvector<cudf::size_type> const& group_labels,
  cudf::size_type const& following_window,
  cudf::size_type const& num_rows,
  rmm::cuda_stream_view stream)
{
  if (following_window < 0) {
    auto const calc = row_based_following_calc<true>(group_offsets, group_labels, following_window);
    return cudf::detail::expand_to_column(calc, num_rows, stream);
  } else {
    auto const calc =
      row_based_following_calc<false>(group_offsets, group_labels, following_window);
    return cudf::detail::expand_to_column(calc, num_rows, stream);
  }
}

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               column_view const& default_outputs,
                                               window_bounds preceding_window_bounds,
                                               window_bounds following_window_bounds,
                                               size_type min_periods,
                                               rolling_aggregation const& aggr,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (input.is_empty()) { return cudf::detail::empty_output_for_rolling_aggregation(input, aggr); }

  CUDF_EXPECTS((group_keys.num_columns() == 0 || group_keys.num_rows() == input.size()),
               "Size mismatch between group_keys and input vector.");

  CUDF_EXPECTS((min_periods >= 0), "min_periods must be non-negative");

  CUDF_EXPECTS((default_outputs.is_empty() || default_outputs.size() == input.size()),
               "Defaults column must be either empty or have as many rows as the input column.");

  // Detect and bypass fully UNBOUNDED windows.
  if (can_optimize_unbounded_window(preceding_window_bounds.is_unbounded(),
                                    following_window_bounds.is_unbounded(),
                                    min_periods,
                                    aggr)) {
    return optimized_unbounded_window(group_keys, input, aggr, stream, mr);
  }

  auto const preceding_window = preceding_window_bounds.value();
  auto const following_window = following_window_bounds.value();

  CUDF_EXPECTS(-(preceding_window - 1) <= following_window,
               "Preceding window bounds must precede the following window bounds.");

  if (group_keys.num_columns() == 0) {
    // No Groupby columns specified. Treat as one big group.
    return detail::rolling_window(
      input, default_outputs, preceding_window, following_window, min_periods, aggr, stream, mr);
  }

  using sort_groupby_helper = cudf::groupby::detail::sort::sort_groupby_helper;

  sort_groupby_helper helper{group_keys, cudf::null_policy::INCLUDE, cudf::sorted::YES, {}};
  auto const& group_offsets{helper.group_offsets(stream)};
  auto const& group_labels{helper.group_labels(stream)};

  // `group_offsets` are interpreted in adjacent pairs, each pair representing the offsets
  // of the first, and one past the last elements in a group.
  //
  // If `group_offsets` is not empty, it must contain at least two offsets:
  //   a. 0, indicating the first element in `input`
  //   b. input.size(), indicating one past the last element in `input`.
  //
  // Thus, for an input of 1000 rows,
  //   0. [] indicates a single group, spanning the entire column.
  //   1  [10] is invalid.
  //   2. [0, 1000] indicates a single group, spanning the entire column (thus, equivalent to no
  //   groups.)
  //   3. [0, 500, 1000] indicates two equal-sized groups: [0,500), and [500,1000).

  assert(group_offsets.size() >= 2 && group_offsets.element(0, stream) == 0 &&
         group_offsets.element(group_offsets.size() - 1, stream) == input.size() &&
         "Must have at least one group.");

  if (aggr.kind == aggregation::CUDA || aggr.kind == aggregation::PTX) {
    cudf::detail::preceding_window_wrapper grouped_preceding_window{
      group_offsets.data(), group_labels.data(), preceding_window};

    cudf::detail::following_window_wrapper grouped_following_window{
      group_offsets.data(), group_labels.data(), following_window};

    return cudf::detail::rolling_window_udf(input,
                                            grouped_preceding_window,
                                            "cudf::detail::preceding_window_wrapper",
                                            grouped_following_window,
                                            "cudf::detail::following_window_wrapper",
                                            min_periods,
                                            aggr,
                                            stream,
                                            mr);
  } else {
    auto const preceding_column =
      make_preceding_column(group_offsets, group_labels, preceding_window, input.size(), stream);
    auto const following_column =
      make_following_column(group_offsets, group_labels, following_window, input.size(), stream);
    return cudf::detail::rolling_window(input,
                                        default_outputs,
                                        preceding_column->view().begin<cudf::size_type>(),
                                        following_column->view().begin<cudf::size_type>(),
                                        min_periods,
                                        aggr,
                                        stream,
                                        mr);
  }
}

}  // namespace detail

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               column_view const& default_outputs,
                                               window_bounds preceding_window_bounds,
                                               window_bounds following_window_bounds,
                                               size_type min_periods,
                                               rolling_aggregation const& aggr,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  return detail::grouped_rolling_window(group_keys,
                                        input,
                                        default_outputs,
                                        preceding_window_bounds,
                                        following_window_bounds,
                                        min_periods,
                                        aggr,
                                        stream,
                                        mr);
}

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               size_type preceding_window,
                                               size_type following_window,
                                               size_type min_periods,
                                               rolling_aggregation const& aggr,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  return grouped_rolling_window(group_keys,
                                input,
                                window_bounds::get(preceding_window),
                                window_bounds::get(following_window),
                                min_periods,
                                aggr,
                                stream,
                                mr);
}

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               window_bounds preceding_window,
                                               window_bounds following_window,
                                               size_type min_periods,
                                               rolling_aggregation const& aggr,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  return detail::grouped_rolling_window(group_keys,
                                        input,
                                        empty_like(input)->view(),
                                        preceding_window,
                                        following_window,
                                        min_periods,
                                        aggr,
                                        stream,
                                        mr);
}

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               column_view const& default_outputs,
                                               size_type preceding_window,
                                               size_type following_window,
                                               size_type min_periods,
                                               rolling_aggregation const& aggr,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  return detail::grouped_rolling_window(group_keys,
                                        input,
                                        default_outputs,
                                        window_bounds::get(preceding_window),
                                        window_bounds::get(following_window),
                                        min_periods,
                                        aggr,
                                        stream,
                                        mr);
}

namespace {

/**
 * @brief For a specified idx, find the lowest value of the (sorted) orderby column that
 * participates in a range-window query.
 */
template <typename ElementT, typename ElementIter>
__device__ ElementT compute_lowest_in_window(ElementIter orderby_iter,
                                             size_type idx,
                                             [[maybe_unused]] ElementT delta)
{
  if constexpr (std::is_same_v<ElementT, cudf::string_view>) {
    return orderby_iter[idx];
  } else {
    return cudf::detail::subtract_safe(orderby_iter[idx], delta);
  }
}

/**
 * @brief For a specified idx, find the highest value of the (sorted) orderby column that
 * participates in a range-window query.
 */
template <typename ElementT, typename ElementIter>
__device__ ElementT compute_highest_in_window(ElementIter orderby_iter,
                                              size_type idx,
                                              [[maybe_unused]] ElementT delta)
{
  if constexpr (std::is_same_v<ElementT, cudf::string_view>) {
    return orderby_iter[idx];
  } else {
    return cudf::detail::add_safe(orderby_iter[idx], delta);
  }
}

/**
 * Accessor for values in an order-by column, on the device.
 */
template <typename T>
struct device_value_accessor {
  column_device_view const col;  ///< column view of column in device

  /// Checks that the type used to access device values matches the rep-type
  /// of the order-by column.
  struct is_correct_range_rep {
    template <typename U>  /// Order-by type.
    constexpr bool operator()() const
    {
      return std::is_same_v<T, cudf::detail::range_rep_type<U>>;
    }
  };

  /**
   * @brief constructor
   *
   * @param[in] col_ column device view of cudf column
   */
  explicit __device__ device_value_accessor(column_device_view const& col_) : col{col_}
  {
    // For non-timestamp types, T must match the order-by column's type.
    // For timestamp types, T must match the range rep type for the order-by column.
    cudf_assert((type_id_matches_device_storage_type<T>(col.type().id()) or
                 cudf::type_dispatcher(col.type(), is_correct_range_rep{})) &&
                "data type mismatch when accessing the order-by column");
  }

  /**
   * @brief Returns the value of element at index `i`
   * @param[in] i index of element
   * @return value of element at index `i`
   */
  __device__ T operator()(cudf::size_type i) const { return col.element<T>(i); }
};

template <typename T>
using const_device_iterator =
  thrust::transform_iterator<device_value_accessor<T>, thrust::counting_iterator<size_type>>;

/// This is a stand-in for the `cudf::column_device_view::begin<T>()`, which is `__host__` only.
/// For range window functions, one might need to iterate over the order-by column, per row.
template <typename T, CUDF_ENABLE_IF(cudf::column_device_view::has_element_accessor<T>())>
[[nodiscard]] __device__ const_device_iterator<T> begin(cudf::column_device_view const& col)
{
  return const_device_iterator<T>{thrust::make_counting_iterator<cudf::size_type>(0),
                                  device_value_accessor<T>{col}};
}

/// Given a single, ungrouped order-by column, return the indices corresponding
/// to the first null element, and (one past) the last null timestamp.
/// The input column is sorted, with all null values clustered either
/// at the beginning of the column or at the end.
/// If no null values are founds, null_begin and null_end are 0.
std::tuple<size_type, size_type> get_null_bounds_for_orderby_column(
  column_view const& orderby_column)
{
  auto const num_rows  = orderby_column.size();
  auto const num_nulls = orderby_column.null_count();

  if (num_nulls == num_rows || num_nulls == 0) {
    // Short-circuit: All nulls, or no nulls.
    return std::make_tuple(0, num_nulls);
  }

  auto const first_row_is_null = orderby_column.null_count(0, 1) == 1;

  return first_row_is_null ? std::make_tuple(0, num_nulls)
                           : std::make_tuple(num_rows - num_nulls, num_rows);
}

/// Range window computation, with
///   1. no grouping keys specified
///   2. rows in ASCENDING order.
/// Treat as one single group.
template <typename T>
std::unique_ptr<column> range_window_ASC(column_view const& input,
                                         column_view const& orderby_column,
                                         T preceding_window,
                                         bool preceding_window_is_unbounded,
                                         T following_window,
                                         bool following_window_is_unbounded,
                                         size_type min_periods,
                                         rolling_aggregation const& aggr,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  auto [h_nulls_begin_idx, h_nulls_end_idx] = get_null_bounds_for_orderby_column(orderby_column);
  auto const p_orderby_device_view = cudf::column_device_view::create(orderby_column, stream);

  auto const preceding_calculator = cuda::proclaim_return_type<size_type>(
    [nulls_begin_idx     = h_nulls_begin_idx,
     nulls_end_idx       = h_nulls_end_idx,
     orderby_device_view = *p_orderby_device_view,
     preceding_window,
     preceding_window_is_unbounded] __device__(size_type idx) -> size_type {
      if (preceding_window_is_unbounded) {
        return idx + 1;  // Technically `idx - 0 + 1`,
                         // where 0 == Group start,
                         // and   1 accounts for the current row
      }
      if (idx >= nulls_begin_idx && idx < nulls_end_idx) {
        // Current row is in the null group.
        // Must consider beginning of null-group as window start.
        return idx - nulls_begin_idx + 1;
      }

      auto const d_orderby = begin<T>(orderby_device_view);
      // orderby[idx] not null. Binary search the group, excluding null group.
      // If nulls_begin_idx == 0, either
      //  1. NULLS FIRST ordering: Binary search starts where nulls_end_idx.
      //  2. NO NULLS: Binary search starts at 0 (also nulls_end_idx).
      // Otherwise, NULLS LAST ordering. Start at 0.
      auto const group_start      = nulls_begin_idx == 0 ? nulls_end_idx : 0;
      auto const lowest_in_window = compute_lowest_in_window(d_orderby, idx, preceding_window);

      return ((d_orderby + idx) - thrust::lower_bound(thrust::seq,
                                                      d_orderby + group_start,
                                                      d_orderby + idx,
                                                      lowest_in_window,
                                                      cudf::detail::nan_aware_less{})) +
             1;  // Add 1, for `preceding` to account for current row.
    });

  auto const preceding_column =
    cudf::detail::expand_to_column(preceding_calculator, input.size(), stream);

  auto const following_calculator = cuda::proclaim_return_type<size_type>(
    [nulls_begin_idx     = h_nulls_begin_idx,
     nulls_end_idx       = h_nulls_end_idx,
     num_rows            = input.size(),
     orderby_device_view = *p_orderby_device_view,
     following_window,
     following_window_is_unbounded] __device__(size_type idx) -> size_type {
      if (following_window_is_unbounded) { return num_rows - idx - 1; }
      if (idx >= nulls_begin_idx && idx < nulls_end_idx) {
        // Current row is in the null group.
        // Window ends at the end of the null group.
        return nulls_end_idx - idx - 1;
      }

      auto const d_orderby = begin<T>(orderby_device_view);
      // orderby[idx] not null. Binary search the group, excluding null group.
      // If nulls_begin_idx == 0, either
      //  1. NULLS FIRST ordering: Binary search ends at num_rows.
      //  2. NO NULLS: Binary search also ends at num_rows.
      // Otherwise, NULLS LAST ordering. End at nulls_begin_idx.

      auto const group_end         = nulls_begin_idx == 0 ? num_rows : nulls_begin_idx;
      auto const highest_in_window = compute_highest_in_window(d_orderby, idx, following_window);

      return (thrust::upper_bound(thrust::seq,
                                  d_orderby + idx,
                                  d_orderby + group_end,
                                  highest_in_window,
                                  cudf::detail::nan_aware_less{}) -
              (d_orderby + idx)) -
             1;
    });

  auto const following_column =
    cudf::detail::expand_to_column(following_calculator, input.size(), stream);

  return cudf::detail::rolling_window(
    input, preceding_column->view(), following_column->view(), min_periods, aggr, stream, mr);
}

// Given an orderby column grouped as specified in group_offsets,
// return the following two vectors:
//  1. Vector with one entry per group, indicating the offset in the group
//     where the null values begin.
//  2. Vector with one entry per group, indicating the offset in the group
//     where the null values end. (i.e. 1 past the last null.)
// Each group in the input orderby column must be sorted,
// with null values clustered at either the start or the end of each group.
// If there are no nulls for any given group, (nulls_begin, nulls_end) == (0,0).
std::tuple<rmm::device_uvector<size_type>, rmm::device_uvector<size_type>>
get_null_bounds_for_orderby_column(column_view const& orderby_column,
                                   cudf::device_span<size_type const> group_offsets,
                                   rmm::cuda_stream_view stream)
{
  // For each group, the null values are clustered at the beginning or the end of the group.
  // These nulls cannot participate, except in their own window.

  auto const num_groups = group_offsets.size() - 1;

  if (orderby_column.has_nulls()) {
    auto null_start = rmm::device_uvector<size_type>(num_groups, stream);
    auto null_end   = rmm::device_uvector<size_type>(num_groups, stream);

    auto p_orderby_device_view = column_device_view::create(orderby_column, stream);

    // Null timestamps exist. Find null bounds, per group.
    thrust::for_each(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(static_cast<size_type>(0)),
      thrust::make_counting_iterator(static_cast<size_type>(num_groups)),
      [d_orderby       = *p_orderby_device_view,
       d_group_offsets = group_offsets.data(),
       d_null_start    = null_start.data(),
       d_null_end      = null_end.data()] __device__(auto group_label) {
        auto group_start           = d_group_offsets[group_label];
        auto group_end             = d_group_offsets[group_label + 1];
        auto first_element_is_null = d_orderby.is_null_nocheck(group_start);
        auto last_element_is_null  = d_orderby.is_null_nocheck(group_end - 1);
        if (!first_element_is_null && !last_element_is_null) {
          // Short circuit: No nulls.
          d_null_start[group_label] = group_start;
          d_null_end[group_label]   = group_start;
        } else if (first_element_is_null && last_element_is_null) {
          // Short circuit: All nulls.
          d_null_start[group_label] = group_start;
          d_null_end[group_label]   = group_end;
        } else if (first_element_is_null) {
          // NULLS FIRST.
          d_null_start[group_label] = group_start;
          d_null_end[group_label]   = *thrust::partition_point(
            thrust::seq,
            thrust::make_counting_iterator(group_start),
            thrust::make_counting_iterator(group_end),
            [&d_orderby] __device__(auto i) { return d_orderby.is_null_nocheck(i); });
        } else {
          // NULLS LAST.
          d_null_end[group_label]   = group_end;
          d_null_start[group_label] = *thrust::partition_point(
            thrust::seq,
            thrust::make_counting_iterator(group_start),
            thrust::make_counting_iterator(group_end),
            [&d_orderby] __device__(auto i) { return d_orderby.is_valid_nocheck(i); });
        }
      });

    return std::make_tuple(std::move(null_start), std::move(null_end));
  } else {
    // The returned vectors have num_groups items, but the input offsets have num_groups+1
    // Drop the last element using a span
    auto const group_offsets_span =
      cudf::device_span<cudf::size_type const>(group_offsets.data(), num_groups);

    // When there are no nulls, just copy the input group offsets to the output.
    return std::make_tuple(cudf::detail::make_device_uvector_async(
                             group_offsets_span, stream, cudf::get_current_device_resource_ref()),
                           cudf::detail::make_device_uvector_async(
                             group_offsets_span, stream, cudf::get_current_device_resource_ref()));
  }
}

// Range window computation, for orderby column in ASCENDING order.
template <typename T>
std::unique_ptr<column> range_window_ASC(column_view const& input,
                                         column_view const& orderby_column,
                                         rmm::device_uvector<cudf::size_type> const& group_offsets,
                                         rmm::device_uvector<cudf::size_type> const& group_labels,
                                         T preceding_window,
                                         bool preceding_window_is_unbounded,
                                         T following_window,
                                         bool following_window_is_unbounded,
                                         size_type min_periods,
                                         rolling_aggregation const& aggr,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  auto [null_start, null_end] =
    get_null_bounds_for_orderby_column(orderby_column, group_offsets, stream);
  auto const p_orderby_device_view = cudf::column_device_view::create(orderby_column, stream);

  auto const preceding_calculator = cuda::proclaim_return_type<size_type>(
    [d_group_offsets     = group_offsets.data(),
     d_group_labels      = group_labels.data(),
     orderby_device_view = *p_orderby_device_view,
     d_nulls_begin       = null_start.data(),
     d_nulls_end         = null_end.data(),
     preceding_window,
     preceding_window_is_unbounded] __device__(size_type idx) -> size_type {
      auto const group_label = d_group_labels[idx];
      auto const group_start = d_group_offsets[group_label];
      auto const nulls_begin = d_nulls_begin[group_label];
      auto const nulls_end   = d_nulls_end[group_label];

      if (preceding_window_is_unbounded) { return idx - group_start + 1; }

      // If idx lies in the null-range, the window is the null range.
      if (idx >= nulls_begin && idx < nulls_end) {
        // Current row is in the null group.
        // The window starts at the start of the null group.
        return idx - nulls_begin + 1;
      }

      auto const d_orderby = begin<T>(orderby_device_view);

      // orderby[idx] not null. Search must exclude the null group.
      // If nulls_begin == group_start, either of the following is true:
      //  1. NULLS FIRST ordering: Search must begin at nulls_end.
      //  2. NO NULLS: Search must begin at group_start (which also equals nulls_end.)
      // Otherwise, NULLS LAST ordering. Search must start at nulls group_start.
      auto const search_start     = nulls_begin == group_start ? nulls_end : group_start;
      auto const lowest_in_window = compute_lowest_in_window(d_orderby, idx, preceding_window);

      return ((d_orderby + idx) - thrust::lower_bound(thrust::seq,
                                                      d_orderby + search_start,
                                                      d_orderby + idx,
                                                      lowest_in_window,
                                                      cudf::detail::nan_aware_less{})) +
             1;  // Add 1, for `preceding` to account for current row.
    });

  auto const preceding_column =
    cudf::detail::expand_to_column(preceding_calculator, input.size(), stream);

  auto const following_calculator = cuda::proclaim_return_type<size_type>(
    [d_group_offsets     = group_offsets.data(),
     d_group_labels      = group_labels.data(),
     orderby_device_view = *p_orderby_device_view,
     d_nulls_begin       = null_start.data(),
     d_nulls_end         = null_end.data(),
     following_window,
     following_window_is_unbounded] __device__(size_type idx) -> size_type {
      auto const group_label = d_group_labels[idx];
      auto const group_start = d_group_offsets[group_label];
      auto const group_end =
        d_group_offsets[group_label + 1];  // Cannot fall off the end, since offsets
                                           // is capped with `input.size()`.
      auto const nulls_begin = d_nulls_begin[group_label];
      auto const nulls_end   = d_nulls_end[group_label];

      if (following_window_is_unbounded) { return (group_end - idx) - 1; }

      // If idx lies in the null-range, the window is the null range.
      if (idx >= nulls_begin && idx < nulls_end) {
        // Current row is in the null group.
        // The window ends at the end of the null group.
        return nulls_end - idx - 1;
      }

      auto const d_orderby = begin<T>(orderby_device_view);

      // orderby[idx] not null. Search must exclude the null group.
      // If nulls_begin == group_start, either of the following is true:
      //  1. NULLS FIRST ordering: Search ends at group_end.
      //  2. NO NULLS: Search ends at group_end.
      // Otherwise, NULLS LAST ordering. Search ends at nulls_begin.
      auto const search_end        = nulls_begin == group_start ? group_end : nulls_begin;
      auto const highest_in_window = compute_highest_in_window(d_orderby, idx, following_window);

      return (thrust::upper_bound(thrust::seq,
                                  d_orderby + idx,
                                  d_orderby + search_end,
                                  highest_in_window,
                                  cudf::detail::nan_aware_less{}) -
              (d_orderby + idx)) -
             1;
    });

  auto const following_column =
    cudf::detail::expand_to_column(following_calculator, input.size(), stream);

  return cudf::detail::rolling_window(
    input, preceding_column->view(), following_column->view(), min_periods, aggr, stream, mr);
}

/// Range window computation, with
///   1. no grouping keys specified
///   2. rows in DESCENDING order.
/// Treat as one single group.
template <typename T>
std::unique_ptr<column> range_window_DESC(column_view const& input,
                                          column_view const& orderby_column,
                                          T preceding_window,
                                          bool preceding_window_is_unbounded,
                                          T following_window,
                                          bool following_window_is_unbounded,
                                          size_type min_periods,
                                          rolling_aggregation const& aggr,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  auto [h_nulls_begin_idx, h_nulls_end_idx] = get_null_bounds_for_orderby_column(orderby_column);
  auto const p_orderby_device_view = cudf::column_device_view::create(orderby_column, stream);

  auto const preceding_calculator = cuda::proclaim_return_type<size_type>(
    [nulls_begin_idx     = h_nulls_begin_idx,
     nulls_end_idx       = h_nulls_end_idx,
     orderby_device_view = *p_orderby_device_view,
     preceding_window,
     preceding_window_is_unbounded] __device__(size_type idx) -> size_type {
      if (preceding_window_is_unbounded) {
        return idx + 1;  // Technically `idx - 0 + 1`,
                         // where 0 == Group start,
                         // and   1 accounts for the current row
      }
      if (idx >= nulls_begin_idx && idx < nulls_end_idx) {
        // Current row is in the null group.
        // Must consider beginning of null-group as window start.
        return idx - nulls_begin_idx + 1;
      }

      auto const d_orderby = begin<T>(orderby_device_view);
      // orderby[idx] not null. Binary search the group, excluding null group.
      // If nulls_begin_idx == 0, either
      //  1. NULLS FIRST ordering: Binary search starts where nulls_end_idx.
      //  2. NO NULLS: Binary search starts at 0 (also nulls_end_idx).
      // Otherwise, NULLS LAST ordering. Start at 0.
      auto const group_start       = nulls_begin_idx == 0 ? nulls_end_idx : 0;
      auto const highest_in_window = compute_highest_in_window(d_orderby, idx, preceding_window);

      return ((d_orderby + idx) - thrust::lower_bound(thrust::seq,
                                                      d_orderby + group_start,
                                                      d_orderby + idx,
                                                      highest_in_window,
                                                      cudf::detail::nan_aware_greater{})) +
             1;  // Add 1, for `preceding` to account for current row.
    });

  auto const preceding_column =
    cudf::detail::expand_to_column(preceding_calculator, input.size(), stream);

  auto const following_calculator = cuda::proclaim_return_type<size_type>(
    [nulls_begin_idx     = h_nulls_begin_idx,
     nulls_end_idx       = h_nulls_end_idx,
     num_rows            = input.size(),
     orderby_device_view = *p_orderby_device_view,
     following_window,
     following_window_is_unbounded] __device__(size_type idx) -> size_type {
      if (following_window_is_unbounded) { return (num_rows - idx) - 1; }
      if (idx >= nulls_begin_idx && idx < nulls_end_idx) {
        // Current row is in the null group.
        // Window ends at the end of the null group.
        return nulls_end_idx - idx - 1;
      }

      auto const d_orderby = begin<T>(orderby_device_view);
      // orderby[idx] not null. Search must exclude null group.
      // If nulls_begin_idx = 0, either
      //  1. NULLS FIRST ordering: Search ends at num_rows.
      //  2. NO NULLS: Search also ends at num_rows.
      // Otherwise, NULLS LAST ordering: End at nulls_begin_idx.

      auto const group_end        = nulls_begin_idx == 0 ? num_rows : nulls_begin_idx;
      auto const lowest_in_window = compute_lowest_in_window(d_orderby, idx, following_window);

      return (thrust::upper_bound(thrust::seq,
                                  d_orderby + idx,
                                  d_orderby + group_end,
                                  lowest_in_window,
                                  cudf::detail::nan_aware_greater{}) -
              (d_orderby + idx)) -
             1;
    });

  auto const following_column =
    cudf::detail::expand_to_column(following_calculator, input.size(), stream);

  return cudf::detail::rolling_window(
    input, preceding_column->view(), following_column->view(), min_periods, aggr, stream, mr);
}

// Range window computation, for rows in DESCENDING order.
template <typename T>
std::unique_ptr<column> range_window_DESC(column_view const& input,
                                          column_view const& orderby_column,
                                          rmm::device_uvector<cudf::size_type> const& group_offsets,
                                          rmm::device_uvector<cudf::size_type> const& group_labels,
                                          T preceding_window,
                                          bool preceding_window_is_unbounded,
                                          T following_window,
                                          bool following_window_is_unbounded,
                                          size_type min_periods,
                                          rolling_aggregation const& aggr,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  auto [null_start, null_end] =
    get_null_bounds_for_orderby_column(orderby_column, group_offsets, stream);
  auto const p_orderby_device_view = cudf::column_device_view::create(orderby_column, stream);

  auto const preceding_calculator = cuda::proclaim_return_type<size_type>(
    [d_group_offsets     = group_offsets.data(),
     d_group_labels      = group_labels.data(),
     orderby_device_view = *p_orderby_device_view,
     d_nulls_begin       = null_start.data(),
     d_nulls_end         = null_end.data(),
     preceding_window,
     preceding_window_is_unbounded] __device__(size_type idx) -> size_type {
      auto const group_label = d_group_labels[idx];
      auto const group_start = d_group_offsets[group_label];
      auto const nulls_begin = d_nulls_begin[group_label];
      auto const nulls_end   = d_nulls_end[group_label];

      if (preceding_window_is_unbounded) { return (idx - group_start) + 1; }

      // If idx lies in the null-range, the window is the null range.
      if (idx >= nulls_begin && idx < nulls_end) {
        // Current row is in the null group.
        // The window starts at the start of the null group.
        return idx - nulls_begin + 1;
      }

      auto const d_orderby = begin<T>(orderby_device_view);
      // orderby[idx] not null. Search must exclude the null group.
      // If nulls_begin == group_start, either of the following is true:
      //  1. NULLS FIRST ordering: Search must begin at nulls_end.
      //  2. NO NULLS: Search must begin at group_start (which also equals nulls_end.)
      // Otherwise, NULLS LAST ordering. Search must start at nulls group_start.
      auto const search_start      = nulls_begin == group_start ? nulls_end : group_start;
      auto const highest_in_window = compute_highest_in_window(d_orderby, idx, preceding_window);

      return ((d_orderby + idx) - thrust::lower_bound(thrust::seq,
                                                      d_orderby + search_start,
                                                      d_orderby + idx,
                                                      highest_in_window,
                                                      cudf::detail::nan_aware_greater{})) +
             1;  // Add 1, for `preceding` to account for current row.
    });

  auto const preceding_column =
    cudf::detail::expand_to_column(preceding_calculator, input.size(), stream);

  auto const following_calculator = cuda::proclaim_return_type<size_type>(
    [d_group_offsets     = group_offsets.data(),
     d_group_labels      = group_labels.data(),
     orderby_device_view = *p_orderby_device_view,
     d_nulls_begin       = null_start.data(),
     d_nulls_end         = null_end.data(),
     following_window,
     following_window_is_unbounded] __device__(size_type idx) -> size_type {
      auto const group_label = d_group_labels[idx];
      auto const group_start = d_group_offsets[group_label];
      auto const group_end   = d_group_offsets[group_label + 1];
      auto const nulls_begin = d_nulls_begin[group_label];
      auto const nulls_end   = d_nulls_end[group_label];

      if (following_window_is_unbounded) { return (group_end - idx) - 1; }

      // If idx lies in the null-range, the window is the null range.
      if (idx >= nulls_begin && idx < nulls_end) {
        // Current row is in the null group.
        // The window ends at the end of the null group.
        return nulls_end - idx - 1;
      }

      auto const d_orderby = begin<T>(orderby_device_view);
      // orderby[idx] not null. Search must exclude the null group.
      // If nulls_begin == group_start, either of the following is true:
      //  1. NULLS FIRST ordering: Search ends at group_end.
      //  2. NO NULLS: Search ends at group_end.
      // Otherwise, NULLS LAST ordering. Search ends at nulls_begin.
      auto const search_end       = nulls_begin == group_start ? group_end : nulls_begin;
      auto const lowest_in_window = compute_lowest_in_window(d_orderby, idx, following_window);

      return (thrust::upper_bound(thrust::seq,
                                  d_orderby + idx,
                                  d_orderby + search_end,
                                  lowest_in_window,
                                  cudf::detail::nan_aware_greater{}) -
              (d_orderby + idx)) -
             1;
    });

  auto const following_column =
    cudf::detail::expand_to_column(following_calculator, input.size(), stream);

  if (aggr.kind == aggregation::CUDA || aggr.kind == aggregation::PTX) {
    CUDF_FAIL("Ranged rolling window does NOT (yet) support UDF.");
  } else {
    return cudf::detail::rolling_window(
      input, preceding_column->view(), following_column->view(), min_periods, aggr, stream, mr);
  }
}

template <typename OrderByT>
std::unique_ptr<column> grouped_range_rolling_window_impl(
  column_view const& input,
  column_view const& orderby_column,
  cudf::order const& order_of_orderby_column,
  rmm::device_uvector<cudf::size_type> const& group_offsets,
  rmm::device_uvector<cudf::size_type> const& group_labels,
  range_window_bounds const& preceding_window,
  range_window_bounds const& following_window,
  size_type min_periods,
  rolling_aggregation const& aggr,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto [preceding_value, following_value] = [&] {
    if constexpr (std::is_same_v<OrderByT, cudf::string_view>) {
      CUDF_EXPECTS(
        preceding_window.is_unbounded() || preceding_window.is_current_row(),
        "For STRING order-by column, preceding range has to be either UNBOUNDED or CURRENT ROW.");
      CUDF_EXPECTS(
        following_window.is_unbounded() || following_window.is_current_row(),
        "For STRING order-by column, following range has to be either UNBOUNDED or CURRENT ROW.");
      return std::pair{cudf::string_view{}, cudf::string_view{}};
    } else {
      return std::pair{
        detail::range_comparable_value<OrderByT>(preceding_window, orderby_column.type(), stream),
        detail::range_comparable_value<OrderByT>(following_window, orderby_column.type(), stream)};
    }
  }();

  if (order_of_orderby_column == cudf::order::ASCENDING) {
    return group_offsets.is_empty() ? range_window_ASC(input,
                                                       orderby_column,
                                                       preceding_value,
                                                       preceding_window.is_unbounded(),
                                                       following_value,
                                                       following_window.is_unbounded(),
                                                       min_periods,
                                                       aggr,
                                                       stream,
                                                       mr)
                                    : range_window_ASC(input,
                                                       orderby_column,
                                                       group_offsets,
                                                       group_labels,
                                                       preceding_value,
                                                       preceding_window.is_unbounded(),
                                                       following_value,
                                                       following_window.is_unbounded(),
                                                       min_periods,
                                                       aggr,
                                                       stream,
                                                       mr);
  } else {
    return group_offsets.is_empty() ? range_window_DESC(input,
                                                        orderby_column,
                                                        preceding_value,
                                                        preceding_window.is_unbounded(),
                                                        following_value,
                                                        following_window.is_unbounded(),
                                                        min_periods,
                                                        aggr,
                                                        stream,
                                                        mr)
                                    : range_window_DESC(input,
                                                        orderby_column,
                                                        group_offsets,
                                                        group_labels,
                                                        preceding_value,
                                                        preceding_window.is_unbounded(),
                                                        following_value,
                                                        following_window.is_unbounded(),
                                                        min_periods,
                                                        aggr,
                                                        stream,
                                                        mr);
  }
}

struct dispatch_grouped_range_rolling_window {
  template <typename OrderByColumnType, typename... Args>
  std::enable_if_t<!detail::is_supported_order_by_column_type<OrderByColumnType>(),
                   std::unique_ptr<column>>
  operator()(Args&&...) const
  {
    CUDF_FAIL("Unsupported OrderBy column type.");
  }

  template <typename OrderByColumnType>
  std::enable_if_t<detail::is_supported_order_by_column_type<OrderByColumnType>(),
                   std::unique_ptr<column>>
  operator()(column_view const& input,
             column_view const& orderby_column,
             cudf::order const& order_of_orderby_column,
             rmm::device_uvector<cudf::size_type> const& group_offsets,
             rmm::device_uvector<cudf::size_type> const& group_labels,
             range_window_bounds const& preceding_window,
             range_window_bounds const& following_window,
             size_type min_periods,
             rolling_aggregation const& aggr,
             rmm::cuda_stream_view stream,
             rmm::device_async_resource_ref mr) const
  {
    return grouped_range_rolling_window_impl<OrderByColumnType>(input,
                                                                orderby_column,
                                                                order_of_orderby_column,
                                                                group_offsets,
                                                                group_labels,
                                                                preceding_window,
                                                                following_window,
                                                                min_periods,
                                                                aggr,
                                                                stream,
                                                                mr);
  }
};

/**
 * @brief Functor to convert from size_type (number of days) to appropriate duration type.
 */
struct to_duration_bounds {
  template <typename OrderBy, std::enable_if_t<cudf::is_timestamp<OrderBy>(), void>* = nullptr>
  range_window_bounds operator()(size_type num_days, rmm::cuda_stream_view stream) const
  {
    using DurationT = typename OrderBy::duration;
    return range_window_bounds::get(duration_scalar<DurationT>{duration_D{num_days}, true, stream},
                                    stream);
  }

  template <typename OrderBy, std::enable_if_t<!cudf::is_timestamp<OrderBy>(), void>* = nullptr>
  range_window_bounds operator()(size_type, rmm::cuda_stream_view) const
  {
    CUDF_FAIL("Expected timestamp orderby column.");
  }
};

/**
 * @brief Get duration type corresponding to specified timestamp type.
 */
data_type get_duration_type_for(cudf::data_type timestamp_type)
{
  switch (timestamp_type.id()) {
    case type_id::TIMESTAMP_DAYS: return data_type{type_id::DURATION_DAYS};
    case type_id::TIMESTAMP_SECONDS: return data_type{type_id::DURATION_SECONDS};
    case type_id::TIMESTAMP_MILLISECONDS: return data_type{type_id::DURATION_MILLISECONDS};
    case type_id::TIMESTAMP_MICROSECONDS: return data_type{type_id::DURATION_MICROSECONDS};
    case type_id::TIMESTAMP_NANOSECONDS: return data_type{type_id::DURATION_NANOSECONDS};
    default: CUDF_FAIL("Expected timestamp orderby column.");
  }
}

/**
 * @brief Bridge function to convert from size_type (number of days) to appropriate duration type.
 *
 * This helps adapt the old `grouped_time_range_rolling_window()` functions that took a "number of
 * days" to the new `range_window_bounds` interface.
 *
 * @param num_days Window bounds specified in number of days in `size_type`
 * @param timestamp_type Data-type of the orderby column to which the `num_days` is to be adapted.
 * @return range_window_bounds A `range_window_bounds` to be used with the new API.
 */
range_window_bounds to_range_bounds(cudf::size_type num_days,
                                    cudf::data_type timestamp_type,
                                    rmm::cuda_stream_view stream)
{
  return cudf::type_dispatcher(timestamp_type, to_duration_bounds{}, num_days, stream);
}

/**
 * @brief Bridge function to convert from `window_bounds` (in days) to appropriate duration type.
 *
 * This helps adapt the old `grouped_time_range_rolling_window()` functions that took a
 * `window_bounds` to the new `range_window_bounds` interface.
 *
 * @param days_bounds The static window-width `window_bounds` object
 * @param timestamp_type Data-type of the orderby column to which the `num_days` is to be adapted.
 * @return range_window_bounds A `range_window_bounds` to be used with the new API.
 */
range_window_bounds to_range_bounds(cudf::window_bounds const& days_bounds,
                                    cudf::data_type timestamp_type,
                                    rmm::cuda_stream_view stream)
{
  return days_bounds.is_unbounded()
           ? range_window_bounds::unbounded(get_duration_type_for(timestamp_type), stream)
           : cudf::type_dispatcher(
               timestamp_type, to_duration_bounds{}, days_bounds.value(), stream);
}

}  // namespace

namespace detail {

/**
 * @copydoc  std::unique_ptr<column> grouped_range_rolling_window(
 *               table_view const& group_keys,
 *               column_view const& orderby_column,
 *               cudf::order const& order,
 *               column_view const& input,
 *               range_window_bounds const& preceding,
 *               range_window_bounds const& following,
 *               size_type min_periods,
 *               rolling_aggregation const& aggr,
 *               rmm::device_async_resource_ref mr );
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> grouped_range_rolling_window(table_view const& group_keys,
                                                     column_view const& order_by_column,
                                                     cudf::order const& order,
                                                     column_view const& input,
                                                     range_window_bounds const& preceding,
                                                     range_window_bounds const& following,
                                                     size_type min_periods,
                                                     rolling_aggregation const& aggr,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (input.is_empty()) { return cudf::detail::empty_output_for_rolling_aggregation(input, aggr); }

  CUDF_EXPECTS((group_keys.num_columns() == 0 || group_keys.num_rows() == input.size()),
               "Size mismatch between group_keys and input vector.");

  CUDF_EXPECTS((min_periods > 0), "min_periods must be positive");

  // Detect and bypass fully UNBOUNDED windows.
  if (can_optimize_unbounded_window(
        preceding.is_unbounded(), following.is_unbounded(), min_periods, aggr)) {
    return optimized_unbounded_window(group_keys, input, aggr, stream, mr);
  }

  using sort_groupby_helper = cudf::groupby::detail::sort::sort_groupby_helper;
  using index_vector        = sort_groupby_helper::index_vector;

  index_vector group_offsets(0, stream), group_labels(0, stream);
  if (group_keys.num_columns() > 0) {
    sort_groupby_helper helper{group_keys, cudf::null_policy::INCLUDE, cudf::sorted::YES, {}};
    group_offsets = index_vector(helper.group_offsets(stream), stream);
    group_labels  = index_vector(helper.group_labels(stream), stream);
  }

  return cudf::type_dispatcher(order_by_column.type(),
                               dispatch_grouped_range_rolling_window{},
                               input,
                               order_by_column,
                               order,
                               group_offsets,
                               group_labels,
                               preceding,
                               following,
                               min_periods,
                               aggr,
                               stream,
                               mr);
}

}  // namespace detail

/**
 * @copydoc std::unique_ptr<column> grouped_time_range_rolling_window(
 *              table_view const& group_keys,
 *              column_view const& timestamp_column,
 *              cudf::order const& timestamp_order,
 *              column_view const& input,
 *              size_type preceding_window_in_days,
 *              size_type following_window_in_days,
 *              size_type min_periods,
 *              rolling_aggregation const& aggr,
 *              rmm::device_async_resource_ref mr);
 */
std::unique_ptr<column> grouped_time_range_rolling_window(table_view const& group_keys,
                                                          column_view const& timestamp_column,
                                                          cudf::order const& timestamp_order,
                                                          column_view const& input,
                                                          size_type preceding_window_in_days,
                                                          size_type following_window_in_days,
                                                          size_type min_periods,
                                                          rolling_aggregation const& aggr,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto preceding = to_range_bounds(preceding_window_in_days, timestamp_column.type(), stream);
  auto following = to_range_bounds(following_window_in_days, timestamp_column.type(), stream);

  return detail::grouped_range_rolling_window(group_keys,
                                              timestamp_column,
                                              timestamp_order,
                                              input,
                                              preceding,
                                              following,
                                              min_periods,
                                              aggr,
                                              stream,
                                              mr);
}

/**
 * @copydoc grouped_time_range_rolling_window(
 *            table_view const& group_keys,
 *            column_view const& timestamp_column,
 *            cudf::order const& timestamp_order,
 *            column_view const& input,
 *            window_bounds preceding_window_in_days,
 *            window_bounds following_window_in_days,
 *            size_type min_periods,
 *            rolling_aggregation const& aggr,
 *            rmm::device_async_resource_ref mr);
 */
std::unique_ptr<column> grouped_time_range_rolling_window(table_view const& group_keys,
                                                          column_view const& timestamp_column,
                                                          cudf::order const& timestamp_order,
                                                          column_view const& input,
                                                          window_bounds preceding_window_in_days,
                                                          window_bounds following_window_in_days,
                                                          size_type min_periods,
                                                          rolling_aggregation const& aggr,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  range_window_bounds preceding =
    to_range_bounds(preceding_window_in_days, timestamp_column.type(), stream);
  range_window_bounds following =
    to_range_bounds(following_window_in_days, timestamp_column.type(), stream);

  return detail::grouped_range_rolling_window(group_keys,
                                              timestamp_column,
                                              timestamp_order,
                                              input,
                                              preceding,
                                              following,
                                              min_periods,
                                              aggr,
                                              stream,
                                              mr);
}

/**
 * @copydoc grouped_range_rolling_window(
 *               table_view const& group_keys,
 *               column_view const& orderby_column,
 *               cudf::order const& order,
 *               column_view const& input,
 *               range_window_bounds const& preceding,
 *               range_window_bounds const& following,
 *               size_type min_periods,
 *               rolling_aggregation const& aggr,
 *               rmm::device_async_resource_ref mr );
 */
std::unique_ptr<column> grouped_range_rolling_window(table_view const& group_keys,
                                                     column_view const& timestamp_column,
                                                     cudf::order const& timestamp_order,
                                                     column_view const& input,
                                                     range_window_bounds const& preceding,
                                                     range_window_bounds const& following,
                                                     size_type min_periods,
                                                     rolling_aggregation const& aggr,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::grouped_range_rolling_window(group_keys,
                                              timestamp_column,
                                              timestamp_order,
                                              input,
                                              preceding,
                                              following,
                                              min_periods,
                                              aggr,
                                              stream,
                                              mr);
}

}  // namespace cudf
