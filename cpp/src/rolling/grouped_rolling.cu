/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/unary.hpp>
#include "rolling_detail.cuh"

namespace cudf {

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               size_type preceding_window,
                                               size_type following_window,
                                               size_type min_periods,
                                               std::unique_ptr<aggregation> const& aggr,
                                               rmm::mr::device_memory_resource* mr)
{
  return grouped_rolling_window(group_keys,
                                input,
                                window_bounds::get(preceding_window),
                                window_bounds::get(following_window),
                                min_periods,
                                aggr,
                                mr);
}

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               window_bounds preceding_window,
                                               window_bounds following_window,
                                               size_type min_periods,
                                               std::unique_ptr<aggregation> const& aggr,
                                               rmm::mr::device_memory_resource* mr)
{
  return grouped_rolling_window(group_keys,
                                input,
                                empty_like(input)->view(),
                                preceding_window,
                                following_window,
                                min_periods,
                                aggr,
                                mr);
}

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               column_view const& default_outputs,
                                               size_type preceding_window,
                                               size_type following_window,
                                               size_type min_periods,
                                               std::unique_ptr<aggregation> const& aggr,
                                               rmm::mr::device_memory_resource* mr)
{
  return grouped_rolling_window(group_keys,
                                input,
                                default_outputs,
                                window_bounds::get(preceding_window),
                                window_bounds::get(following_window),
                                min_periods,
                                aggr,
                                mr);
}

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               column_view const& default_outputs,
                                               window_bounds preceding_window_bounds,
                                               window_bounds following_window_bounds,
                                               size_type min_periods,
                                               std::unique_ptr<aggregation> const& aggr,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  if (input.is_empty()) return empty_like(input);

  CUDF_EXPECTS((group_keys.num_columns() == 0 || group_keys.num_rows() == input.size()),
               "Size mismatch between group_keys and input vector.");

  CUDF_EXPECTS((min_periods > 0), "min_periods must be positive");

  CUDF_EXPECTS((default_outputs.is_empty() || default_outputs.size() == input.size()),
               "Defaults column must be either empty or have as many rows as the input column.");

  auto const preceding_window = preceding_window_bounds.value;
  auto const following_window = following_window_bounds.value;

  if (group_keys.num_columns() == 0) {
    // No Groupby columns specified. Treat as one big group.
    return rolling_window(
      input, default_outputs, preceding_window, following_window, min_periods, aggr, mr);
  }

  using sort_groupby_helper = cudf::groupby::detail::sort::sort_groupby_helper;

  sort_groupby_helper helper{group_keys, cudf::null_policy::INCLUDE, cudf::sorted::YES};
  auto group_offsets{helper.group_offsets()};
  auto const& group_labels{helper.group_labels()};

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

  assert(group_offsets.size() >= 2 && group_offsets[0] == 0 &&
         group_offsets[group_offsets.size() - 1] == input.size() &&
         "Must have at least one group.");

  auto preceding_calculator = [d_group_offsets = group_offsets.data().get(),
                               d_group_labels  = group_labels.data().get(),
                               preceding_window] __device__(size_type idx) {
    auto group_label = d_group_labels[idx];
    auto group_start = d_group_offsets[group_label];
    return thrust::minimum<size_type>{}(preceding_window,
                                        idx - group_start + 1);  // Preceding includes current row.
  };

  auto following_calculator = [d_group_offsets = group_offsets.data().get(),
                               d_group_labels  = group_labels.data().get(),
                               following_window] __device__(size_type idx) {
    auto group_label = d_group_labels[idx];
    auto group_end =
      d_group_offsets[group_label +
                      1];  // Cannot fall off the end, since offsets is capped with `input.size()`.
    return thrust::minimum<size_type>{}(following_window, (group_end - 1) - idx);
  };

  if (aggr->kind == aggregation::CUDA || aggr->kind == aggregation::PTX) {
    cudf::detail::preceding_window_wrapper grouped_preceding_window{
      group_offsets.data().get(), group_labels.data().get(), preceding_window};

    cudf::detail::following_window_wrapper grouped_following_window{
      group_offsets.data().get(), group_labels.data().get(), following_window};

    return cudf::detail::rolling_window_udf(input,
                                            grouped_preceding_window,
                                            "cudf::detail::preceding_window_wrapper",
                                            grouped_following_window,
                                            "cudf::detail::following_window_wrapper",
                                            min_periods,
                                            aggr,
                                            rmm::cuda_stream_default,
                                            mr);
  } else {
    return cudf::detail::rolling_window(
      input,
      default_outputs,
      thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                      preceding_calculator),
      thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                      following_calculator),
      min_periods,
      aggr,
      rmm::cuda_stream_default,
      mr);
  }
}

namespace {

bool is_supported_range_frame_unit(cudf::data_type const& data_type)
{
  auto id = data_type.id();
  return id == cudf::type_id::TIMESTAMP_DAYS || id == cudf::type_id::TIMESTAMP_SECONDS ||
         id == cudf::type_id::TIMESTAMP_MILLISECONDS ||
         id == cudf::type_id::TIMESTAMP_MICROSECONDS || id == cudf::type_id::TIMESTAMP_NANOSECONDS;
}

/// Fetches multiplication factor to normalize window sizes, depending on the datatype of the
/// timestamp column. Used for time-based rolling-window operations. E.g. If the timestamp column is
/// in TIMESTAMP_SECONDS, and the window sizes are specified in DAYS, the window size needs to be
/// multiplied by `24*60*60`, before comparisons with the timestamps.
size_t multiplication_factor(cudf::data_type const& data_type)
{
  // Assume timestamps.
  switch (data_type.id()) {
    case cudf::type_id::TIMESTAMP_DAYS: return 1L;
    case cudf::type_id::TIMESTAMP_SECONDS: return 24L * 60 * 60;
    case cudf::type_id::TIMESTAMP_MILLISECONDS: return 24L * 60 * 60 * 1000;
    case cudf::type_id::TIMESTAMP_MICROSECONDS: return 24L * 60 * 60 * 1000 * 1000;
    default:
      CUDF_EXPECTS(data_type.id() == cudf::type_id::TIMESTAMP_NANOSECONDS,
                   "Unexpected data-type for timestamp-based rolling window operation!");
      return 24L * 60 * 60 * 1000 * 1000 * 1000;
  }
}

/// Given a single, ungrouped timestamp column, return the indices corresponding
/// to the first null timestamp, and (one past) the last null timestamp.
/// The input column is sorted, with all null values clustered either
/// at the beginning of the column or at the end.
/// If no null values are founds, null_begin and null_end are 0.
std::tuple<size_type, size_type> get_null_bounds_for_timestamp_column(
  column_view const& timestamp_column)
{
  auto const num_rows  = timestamp_column.size();
  auto const num_nulls = timestamp_column.null_count();

  if (num_nulls == num_rows || num_nulls == 0) {
    // Short-circuit: All nulls, or no nulls.
    return std::make_tuple(0, num_nulls);
  }

  auto const first_row_is_null = timestamp_column.null_count(0, 1) == 1;

  return first_row_is_null ? std::make_tuple(0, num_nulls)
                           : std::make_tuple(num_rows - num_nulls, num_rows);
}

using TimeT = int64_t;  // Timestamp representations normalized to int64_t.

/// Time-range window computation, with
///   1. no grouping keys specified
///   2. timetamps in ASCENDING order.
/// Treat as one single group.
std::unique_ptr<column> time_range_window_ASC(column_view const& input,
                                              column_view const& timestamp_column,
                                              TimeT preceding_window,
                                              bool preceding_window_is_unbounded,
                                              TimeT following_window,
                                              bool following_window_is_unbounded,
                                              size_type min_periods,
                                              std::unique_ptr<aggregation> const& aggr,
                                              rmm::mr::device_memory_resource* mr)
{
  size_type nulls_begin_idx, nulls_end_idx;
  std::tie(nulls_begin_idx, nulls_end_idx) = get_null_bounds_for_timestamp_column(timestamp_column);

  auto preceding_calculator =
    [nulls_begin_idx,
     nulls_end_idx,
     d_timestamps = timestamp_column.data<TimeT>(),
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

    // timestamp[idx] not null. Binary search the group, excluding null group.
    // If nulls_begin_idx == 0, either
    //  1. NULLS FIRST ordering: Binary search starts where nulls_end_idx.
    //  2. NO NULLS: Binary search starts at 0 (also nulls_end_idx).
    // Otherwise, NULLS LAST ordering. Start at 0.
    auto group_start                = nulls_begin_idx == 0 ? nulls_end_idx : 0;
    auto lowest_timestamp_in_window = d_timestamps[idx] - preceding_window;

    return ((d_timestamps + idx) - thrust::lower_bound(thrust::seq,
                                                       d_timestamps + group_start,
                                                       d_timestamps + idx,
                                                       lowest_timestamp_in_window)) +
           1;  // Add 1, for `preceding` to account for current row.
  };

  auto following_calculator =
    [nulls_begin_idx,
     nulls_end_idx,
     num_rows     = input.size(),
     d_timestamps = timestamp_column.data<TimeT>(),
     following_window,
     following_window_is_unbounded] __device__(size_type idx) -> size_type {
    if (following_window_is_unbounded) { return num_rows - idx - 1; }
    if (idx >= nulls_begin_idx && idx < nulls_end_idx) {
      // Current row is in the null group.
      // Window ends at the end of the null group.
      return nulls_end_idx - idx - 1;
    }

    // timestamp[idx] not null. Binary search the group, excluding null group.
    // If nulls_begin_idx == 0, either
    //  1. NULLS FIRST ordering: Binary search ends at num_rows.
    //  2. NO NULLS: Binary search also ends at num_rows.
    // Otherwise, NULLS LAST ordering. End at nulls_begin_idx.

    auto group_end                   = nulls_begin_idx == 0 ? num_rows : nulls_begin_idx;
    auto highest_timestamp_in_window = d_timestamps[idx] + following_window;

    return (thrust::upper_bound(thrust::seq,
                                d_timestamps + idx,
                                d_timestamps + group_end,
                                highest_timestamp_in_window) -
            (d_timestamps + idx)) -
           1;
  };

  return cudf::detail::rolling_window(
    input,
    empty_like(input)->view(),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                    preceding_calculator),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                    following_calculator),
    min_periods,
    aggr,
    rmm::cuda_stream_default,
    mr);
}

/// Given a timestamp column grouped as specified in group_offsets,
/// return the following two vectors:
///  1. Vector with one entry per group, indicating the offset in the group
///     where the null values begin.
///  2. Vector with one entry per group, indicating the offset in the group
///     where the null values end. (i.e. 1 past the last null.)
/// Each group in the input timestamp column must be sorted,
/// with null values clustered at either the start or the end of each group.
/// If there are no nulls for any given group, (nulls_begin, nulls_end) == (0,0).
std::tuple<rmm::device_vector<size_type>, rmm::device_vector<size_type>>
get_null_bounds_for_timestamp_column(column_view const& timestamp_column,
                                     rmm::device_vector<size_type> const& group_offsets)
{
  // For each group, the null values are themselves clustered
  // at the beginning or the end of the group.
  // These nulls cannot participate, except in their own window.

  // If the input has n groups, group_offsets will have n+1 values.
  // null_start and null_end should eventually have 1 entry per group.
  auto null_start = rmm::device_vector<size_type>(group_offsets.begin(), group_offsets.end() - 1);
  auto null_end   = rmm::device_vector<size_type>(group_offsets.begin(), group_offsets.end() - 1);

  if (timestamp_column.has_nulls()) {
    auto p_timestamps_device_view = column_device_view::create(timestamp_column);
    auto num_groups               = group_offsets.size();

    // Null timestamps exist. Find null bounds, per group.
    thrust::for_each(
      thrust::device,
      thrust::make_counting_iterator(static_cast<size_type>(0)),
      thrust::make_counting_iterator(static_cast<size_type>(num_groups)),
      [d_timestamps    = *p_timestamps_device_view,
       d_group_offsets = group_offsets.data().get(),
       d_null_start    = null_start.data(),
       d_null_end      = null_end.data()] __device__(auto group_label) {
        auto group_start           = d_group_offsets[group_label];
        auto group_end             = d_group_offsets[group_label + 1];
        auto first_element_is_null = d_timestamps.is_null_nocheck(group_start);
        auto last_element_is_null  = d_timestamps.is_null_nocheck(group_end - 1);
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
            [&d_timestamps] __device__(auto i) { return d_timestamps.is_null_nocheck(i); });
        } else {
          // NULLS LAST.
          d_null_end[group_label]   = group_end;
          d_null_start[group_label] = *thrust::partition_point(
            thrust::seq,
            thrust::make_counting_iterator(group_start),
            thrust::make_counting_iterator(group_end),
            [&d_timestamps] __device__(auto i) { return d_timestamps.is_valid_nocheck(i); });
        }
      });
  }

  return std::make_tuple(std::move(null_start), std::move(null_end));
}

// Time-range window computation, for timestamps in ASCENDING order.
std::unique_ptr<column> time_range_window_ASC(
  column_view const& input,
  column_view const& timestamp_column,
  rmm::device_vector<cudf::size_type> const& group_offsets,
  rmm::device_vector<cudf::size_type> const& group_labels,
  TimeT preceding_window,
  bool preceding_window_is_unbounded,
  TimeT following_window,
  bool following_window_is_unbounded,
  size_type min_periods,
  std::unique_ptr<aggregation> const& aggr,
  rmm::mr::device_memory_resource* mr)
{
  rmm::device_vector<size_type> null_start, null_end;
  std::tie(null_start, null_end) =
    get_null_bounds_for_timestamp_column(timestamp_column, group_offsets);

  auto preceding_calculator =
    [d_group_offsets = group_offsets.data().get(),
     d_group_labels  = group_labels.data().get(),
     d_timestamps    = timestamp_column.data<TimeT>(),
     d_nulls_begin   = null_start.data().get(),
     d_nulls_end     = null_end.data().get(),
     preceding_window,
     preceding_window_is_unbounded] __device__(size_type idx) -> size_type {
    auto group_label = d_group_labels[idx];
    auto group_start = d_group_offsets[group_label];
    auto nulls_begin = d_nulls_begin[group_label];
    auto nulls_end   = d_nulls_end[group_label];

    if (preceding_window_is_unbounded) { return idx - group_start + 1; }

    // If idx lies in the null-range, the window is the null range.
    if (idx >= nulls_begin && idx < nulls_end) {
      // Current row is in the null group.
      // The window starts at the start of the null group.
      return idx - nulls_begin + 1;
    }

    // timestamp[idx] not null. Search must exclude the null group.
    // If nulls_begin == group_start, either of the following is true:
    //  1. NULLS FIRST ordering: Search must begin at nulls_end.
    //  2. NO NULLS: Search must begin at group_start (which also equals nulls_end.)
    // Otherwise, NULLS LAST ordering. Search must start at nulls group_start.
    auto search_start = nulls_begin == group_start ? nulls_end : group_start;

    auto lowest_timestamp_in_window = d_timestamps[idx] - preceding_window;

    return ((d_timestamps + idx) - thrust::lower_bound(thrust::seq,
                                                       d_timestamps + search_start,
                                                       d_timestamps + idx,
                                                       lowest_timestamp_in_window)) +
           1;  // Add 1, for `preceding` to account for current row.
  };

  auto following_calculator =
    [d_group_offsets = group_offsets.data().get(),
     d_group_labels  = group_labels.data().get(),
     d_timestamps    = timestamp_column.data<TimeT>(),
     d_nulls_begin   = null_start.data().get(),
     d_nulls_end     = null_end.data().get(),
     following_window,
     following_window_is_unbounded] __device__(size_type idx) -> size_type {
    auto group_label = d_group_labels[idx];
    auto group_start = d_group_offsets[group_label];
    auto group_end =
      d_group_offsets[group_label +
                      1];  // Cannot fall off the end, since offsets is capped with `input.size()`.
    auto nulls_begin = d_nulls_begin[group_label];
    auto nulls_end   = d_nulls_end[group_label];

    if (following_window_is_unbounded) { return (group_end - idx) - 1; }

    // If idx lies in the null-range, the window is the null range.
    if (idx >= nulls_begin && idx < nulls_end) {
      // Current row is in the null group.
      // The window ends at the end of the null group.
      return nulls_end - idx - 1;
    }

    // timestamp[idx] not null. Search must exclude the null group.
    // If nulls_begin == group_start, either of the following is true:
    //  1. NULLS FIRST ordering: Search ends at group_end.
    //  2. NO NULLS: Search ends at group_end.
    // Otherwise, NULLS LAST ordering. Search ends at nulls_begin.
    auto search_end = nulls_begin == group_start ? group_end : nulls_begin;

    auto highest_timestamp_in_window = d_timestamps[idx] + following_window;

    return (thrust::upper_bound(thrust::seq,
                                d_timestamps + idx,
                                d_timestamps + search_end,
                                highest_timestamp_in_window) -
            (d_timestamps + idx)) -
           1;
  };

  return cudf::detail::rolling_window(
    input,
    empty_like(input)->view(),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                    preceding_calculator),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                    following_calculator),
    min_periods,
    aggr,
    rmm::cuda_stream_default,
    mr);
}

/// Time-range window computation, with
///   1. no grouping keys specified
///   2. timetamps in DESCENDING order.
/// Treat as one single group.
std::unique_ptr<column> time_range_window_DESC(column_view const& input,
                                               column_view const& timestamp_column,
                                               TimeT preceding_window,
                                               bool preceding_window_is_unbounded,
                                               TimeT following_window,
                                               bool following_window_is_unbounded,
                                               size_type min_periods,
                                               std::unique_ptr<aggregation> const& aggr,
                                               rmm::mr::device_memory_resource* mr)
{
  size_type nulls_begin_idx, nulls_end_idx;
  std::tie(nulls_begin_idx, nulls_end_idx) = get_null_bounds_for_timestamp_column(timestamp_column);

  auto preceding_calculator =
    [nulls_begin_idx,
     nulls_end_idx,
     d_timestamps = timestamp_column.data<TimeT>(),
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

    // timestamp[idx] not null. Binary search the group, excluding null group.
    // If nulls_begin_idx == 0, either
    //  1. NULLS FIRST ordering: Binary search starts where nulls_end_idx.
    //  2. NO NULLS: Binary search starts at 0 (also nulls_end_idx).
    // Otherwise, NULLS LAST ordering. Start at 0.
    auto group_start                 = nulls_begin_idx == 0 ? nulls_end_idx : 0;
    auto highest_timestamp_in_window = d_timestamps[idx] + preceding_window;

    return ((d_timestamps + idx) -
            thrust::lower_bound(thrust::seq,
                                d_timestamps + group_start,
                                d_timestamps + idx,
                                highest_timestamp_in_window,
                                thrust::greater<decltype(highest_timestamp_in_window)>())) +
           1;  // Add 1, for `preceding` to account for current row.
  };

  auto following_calculator =
    [nulls_begin_idx,
     nulls_end_idx,
     num_rows     = input.size(),
     d_timestamps = timestamp_column.data<TimeT>(),
     following_window,
     following_window_is_unbounded] __device__(size_type idx) -> size_type {
    if (following_window_is_unbounded) { return (num_rows - idx) - 1; }
    if (idx >= nulls_begin_idx && idx < nulls_end_idx) {
      // Current row is in the null group.
      // Window ends at the end of the null group.
      return nulls_end_idx - idx - 1;
    }

    // timestamp[idx] not null. Search must exclude null group.
    // If nulls_begin_idx = 0, either
    //  1. NULLS FIRST ordering: Search ends at num_rows.
    //  2. NO NULLS: Search also ends at num_rows.
    // Otherwise, NULLS LAST ordering: End at nulls_begin_idx.

    auto group_end                  = nulls_begin_idx == 0 ? num_rows : nulls_begin_idx;
    auto lowest_timestamp_in_window = d_timestamps[idx] - following_window;

    return (thrust::upper_bound(thrust::seq,
                                d_timestamps + idx,
                                d_timestamps + group_end,
                                lowest_timestamp_in_window,
                                thrust::greater<decltype(lowest_timestamp_in_window)>()) -
            (d_timestamps + idx)) -
           1;
  };

  return cudf::detail::rolling_window(
    input,
    empty_like(input)->view(),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                    preceding_calculator),
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                    following_calculator),
    min_periods,
    aggr,
    rmm::cuda_stream_default,
    mr);
}

// Time-range window computation, for timestamps in DESCENDING order.
std::unique_ptr<column> time_range_window_DESC(
  column_view const& input,
  column_view const& timestamp_column,
  rmm::device_vector<cudf::size_type> const& group_offsets,
  rmm::device_vector<cudf::size_type> const& group_labels,
  TimeT preceding_window,
  bool preceding_window_is_unbounded,
  TimeT following_window,
  bool following_window_is_unbounded,
  size_type min_periods,
  std::unique_ptr<aggregation> const& aggr,
  rmm::mr::device_memory_resource* mr)
{
  rmm::device_vector<size_type> null_start, null_end;
  std::tie(null_start, null_end) =
    get_null_bounds_for_timestamp_column(timestamp_column, group_offsets);

  auto preceding_calculator =
    [d_group_offsets = group_offsets.data().get(),
     d_group_labels  = group_labels.data().get(),
     d_timestamps    = timestamp_column.data<TimeT>(),
     d_nulls_begin   = null_start.data().get(),
     d_nulls_end     = null_end.data().get(),
     preceding_window,
     preceding_window_is_unbounded] __device__(size_type idx) -> size_type {
    auto group_label = d_group_labels[idx];
    auto group_start = d_group_offsets[group_label];
    auto nulls_begin = d_nulls_begin[group_label];
    auto nulls_end   = d_nulls_end[group_label];

    if (preceding_window_is_unbounded) { return (idx - group_start) + 1; }

    // If idx lies in the null-range, the window is the null range.
    if (idx >= nulls_begin && idx < nulls_end) {
      // Current row is in the null group.
      // The window starts at the start of the null group.
      return idx - nulls_begin + 1;
    }

    // timestamp[idx] not null. Search must exclude the null group.
    // If nulls_begin == group_start, either of the following is true:
    //  1. NULLS FIRST ordering: Search must begin at nulls_end.
    //  2. NO NULLS: Search must begin at group_start (which also equals nulls_end.)
    // Otherwise, NULLS LAST ordering. Search must start at nulls group_start.
    auto search_start = nulls_begin == group_start ? nulls_end : group_start;

    auto highest_timestamp_in_window = d_timestamps[idx] + preceding_window;

    return ((d_timestamps + idx) -
            thrust::lower_bound(thrust::seq,
                                d_timestamps + search_start,
                                d_timestamps + idx,
                                highest_timestamp_in_window,
                                thrust::greater<decltype(highest_timestamp_in_window)>())) +
           1;  // Add 1, for `preceding` to account for current row.
  };

  auto following_calculator =
    [d_group_offsets = group_offsets.data().get(),
     d_group_labels  = group_labels.data().get(),
     d_timestamps    = timestamp_column.data<TimeT>(),
     d_nulls_begin   = null_start.data().get(),
     d_nulls_end     = null_end.data().get(),
     following_window,
     following_window_is_unbounded] __device__(size_type idx) -> size_type {
    auto group_label = d_group_labels[idx];
    auto group_start = d_group_offsets[group_label];
    auto group_end   = d_group_offsets[group_label + 1];
    auto nulls_begin = d_nulls_begin[group_label];
    auto nulls_end   = d_nulls_end[group_label];

    if (following_window_is_unbounded) { return (group_end - idx) - 1; }

    // If idx lies in the null-range, the window is the null range.
    if (idx >= nulls_begin && idx < nulls_end) {
      // Current row is in the null group.
      // The window ends at the end of the null group.
      return nulls_end - idx - 1;
    }

    // timestamp[idx] not null. Search must exclude the null group.
    // If nulls_begin == group_start, either of the following is true:
    //  1. NULLS FIRST ordering: Search ends at group_end.
    //  2. NO NULLS: Search ends at group_end.
    // Otherwise, NULLS LAST ordering. Search ends at nulls_begin.
    auto search_end = nulls_begin == group_start ? group_end : nulls_begin;

    auto lowest_timestamp_in_window = d_timestamps[idx] - following_window;

    return (thrust::upper_bound(thrust::seq,
                                d_timestamps + idx,
                                d_timestamps + search_end,
                                lowest_timestamp_in_window,
                                thrust::greater<decltype(lowest_timestamp_in_window)>()) -
            (d_timestamps + idx)) -
           1;
  };

  if (aggr->kind == aggregation::CUDA || aggr->kind == aggregation::PTX) {
    CUDF_FAIL("Time ranged rolling window does NOT (yet) support UDF.");
  } else {
    return cudf::detail::rolling_window(
      input,
      empty_like(input)->view(),
      thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                      preceding_calculator),
      thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                      following_calculator),
      min_periods,
      aggr,
      rmm::cuda_stream_default,
      mr);
  }
}

std::unique_ptr<column> grouped_time_range_rolling_window_impl(
  column_view const& input,
  column_view const& timestamp_column,
  cudf::order const& timestamp_ordering,
  rmm::device_vector<cudf::size_type> const& group_offsets,
  rmm::device_vector<cudf::size_type> const& group_labels,
  window_bounds preceding_window_in_days,  // TODO: Consider taking offset-type as type_id. Assumes
                                           // days for now.
  window_bounds following_window_in_days,
  size_type min_periods,
  std::unique_ptr<aggregation> const& aggr,
  rmm::mr::device_memory_resource* mr)
{
  TimeT mult_factor{static_cast<TimeT>(multiplication_factor(timestamp_column.type()))};

  if (timestamp_ordering == cudf::order::ASCENDING) {
    return group_offsets.empty()
             ? time_range_window_ASC(input,
                                     timestamp_column,
                                     preceding_window_in_days.value * mult_factor,
                                     preceding_window_in_days.is_unbounded,
                                     following_window_in_days.value * mult_factor,
                                     following_window_in_days.is_unbounded,
                                     min_periods,
                                     aggr,
                                     mr)
             : time_range_window_ASC(input,
                                     timestamp_column,
                                     group_offsets,
                                     group_labels,
                                     preceding_window_in_days.value * mult_factor,
                                     preceding_window_in_days.is_unbounded,
                                     following_window_in_days.value * mult_factor,
                                     following_window_in_days.is_unbounded,
                                     min_periods,
                                     aggr,
                                     mr);
  } else {
    return group_offsets.empty()
             ? time_range_window_DESC(input,
                                      timestamp_column,
                                      preceding_window_in_days.value * mult_factor,
                                      preceding_window_in_days.is_unbounded,
                                      following_window_in_days.value * mult_factor,
                                      following_window_in_days.is_unbounded,
                                      min_periods,
                                      aggr,
                                      mr)
             : time_range_window_DESC(input,
                                      timestamp_column,
                                      group_offsets,
                                      group_labels,
                                      preceding_window_in_days.value * mult_factor,
                                      preceding_window_in_days.is_unbounded,
                                      following_window_in_days.value * mult_factor,
                                      following_window_in_days.is_unbounded,
                                      min_periods,
                                      aggr,
                                      mr);
  }
}

}  // namespace

std::unique_ptr<column> grouped_time_range_rolling_window(table_view const& group_keys,
                                                          column_view const& timestamp_column,
                                                          cudf::order const& timestamp_order,
                                                          column_view const& input,
                                                          size_type preceding_window_in_days,
                                                          size_type following_window_in_days,
                                                          size_type min_periods,
                                                          std::unique_ptr<aggregation> const& aggr,
                                                          rmm::mr::device_memory_resource* mr)
{
  return grouped_time_range_rolling_window(group_keys,
                                           timestamp_column,
                                           timestamp_order,
                                           input,
                                           window_bounds::get(preceding_window_in_days),
                                           window_bounds::get(following_window_in_days),
                                           min_periods,
                                           aggr,
                                           mr);
}

std::unique_ptr<column> grouped_time_range_rolling_window(table_view const& group_keys,
                                                          column_view const& timestamp_column,
                                                          cudf::order const& timestamp_order,
                                                          column_view const& input,
                                                          window_bounds preceding_window_in_days,
                                                          window_bounds following_window_in_days,
                                                          size_type min_periods,
                                                          std::unique_ptr<aggregation> const& aggr,
                                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  if (input.is_empty()) return empty_like(input);

  CUDF_EXPECTS((group_keys.num_columns() == 0 || group_keys.num_rows() == input.size()),
               "Size mismatch between group_keys and input vector.");

  CUDF_EXPECTS((min_periods > 0), "min_periods must be positive");

  using sort_groupby_helper = cudf::groupby::detail::sort::sort_groupby_helper;
  using index_vector        = sort_groupby_helper::index_vector;

  index_vector group_offsets, group_labels;
  if (group_keys.num_columns() > 0) {
    sort_groupby_helper helper{group_keys, cudf::null_policy::INCLUDE, cudf::sorted::YES};
    group_offsets = helper.group_offsets();
    group_labels  = helper.group_labels();
  }

  // Assumes that `timestamp_column` is actually of a timestamp type.
  CUDF_EXPECTS(is_supported_range_frame_unit(timestamp_column.type()),
               "Unsupported data-type for `timestamp`-based rolling window operation!");

  auto is_timestamp_in_days = timestamp_column.type().id() == cudf::type_id::TIMESTAMP_DAYS;

  return grouped_time_range_rolling_window_impl(
    input,
    is_timestamp_in_days
      ? cudf::cast(timestamp_column, cudf::data_type(cudf::type_id::TIMESTAMP_SECONDS), mr)->view()
      : timestamp_column,
    timestamp_order,
    group_offsets,
    group_labels,
    preceding_window_in_days,
    following_window_in_days,
    min_periods,
    aggr,
    mr);
}

}  // namespace cudf