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

#pragma once

#include <cudf/types.hpp>

#include <memory>

namespace cudf {
/**
 * @addtogroup aggregation_rolling
 * @{
 * @file
 */

/**
 * @brief  Applies a fixed-size rolling window function to the values in a column.
 *
 * This function aggregates values in a window around each element i of the input column, and
 * invalidates the bit mask for element i if there are not enough observations. The window size is
 * static (the same for each element). This matches Pandas' API for DataFrame.rolling with a few
 * notable differences:
 * - instead of the center flag it uses a two-part window to allow for more flexible windows.
 *   The total window size = `preceding_window + following_window`. Element `i` uses elements
 *   `[i-preceding_window+1, i+following_window]` to do the window computation.
 * - instead of storing NA/NaN for output rows that do not meet the minimum number of observations
 *   this function updates the valid bitmask of the column to indicate which elements are valid.
 *
 * The returned column for count aggregation always has `INT32` type. All other operators return a
 * column of the same type as the input. Therefore it is suggested to convert integer column types
 * (especially low-precision integers) to `FLOAT32` or `FLOAT64` before doing a rolling `MEAN`.
 *
 * @param[in] input_col The input column
 * @param[in] preceding_window The static rolling window size in the backward direction.
 * @param[in] following_window The static rolling window size in the forward direction.
 * @param[in] min_periods Minimum number of observations in window required to have a value,
 *                        otherwise element `i` is null.
 * @param[in] agg The rolling window aggregation type (SUM, MAX, MIN, etc.)
 *
 * @returns   A nullable output column containing the rolling window results
 **/
std::unique_ptr<column> rolling_window(
  column_view const& input,
  size_type preceding_window,
  size_type following_window,
  size_type min_periods,
  std::unique_ptr<aggregation> const& agg,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc std::unique_ptr<column> rolling_window(
 *            column_view const& input,
 *            size_type preceding_window,
 *            size_type following_window,
 *            size_type min_periods,
 *            std::unique_ptr<aggregation> const& agg,
 *            rmm::mr::device_memory_resource* mr)
 *
 * @param default_outputs A column of per-row default values to be returned instead
 *                        of nulls. Used for LEAD()/LAG(), if the row offset crosses
 *                        the boundaries of the column.
 */
std::unique_ptr<column> rolling_window(
  column_view const& input,
  column_view const& default_outputs,
  size_type preceding_window,
  size_type following_window,
  size_type min_periods,
  std::unique_ptr<aggregation> const& agg,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Applies a grouping-aware, fixed-size rolling window function to the values in a column.
 *
 * Like `rolling_window()`, this function aggregates values in a window around each
 * element of a specified `input` column. It differs from `rolling_window()` in that elements of the
 * `input` column are grouped into distinct groups (e.g. the result of a groupby). The window
 * aggregation cannot cross the group boundaries. For a row `i` of `input`, the group is determined
 * from the corresponding (i.e. i-th) values of the columns under `group_keys`.
 *
 * Note: This method requires that the rows are presorted by the `group_key` values.
 *
 * @code{.pseudo}
 * Example: Consider a user-sales dataset, where the rows look as follows:
 * { "user_id", sales_amt, day }
 *
 * The `grouped_rolling_window()` method enables windowing queries such as grouping a dataset by
 * `user_id`, and summing up the `sales_amt` column over a window of 3 rows (2 preceding (including
 * current row), 1 row following).
 *
 * In this example,
 *    1. `group_keys == [ user_id ]`
 *    2. `input == sales_amt`
 * The data are grouped by `user_id`, and ordered by `day`-string. The aggregation
 * (SUM) is then calculated for a window of 3 values around (and including) each row.
 *
 * For the following input:
 *
 *  [ // user,  sales_amt
 *    { "user1",   10      },
 *    { "user2",   20      },
 *    { "user1",   20      },
 *    { "user1",   10      },
 *    { "user2",   30      },
 *    { "user2",   80      },
 *    { "user1",   50      },
 *    { "user1",   60      },
 *    { "user2",   40      }
 *  ]
 *
 * Partitioning (grouping) by `user_id` yields the following `sales_amt` vector
 * (with 2 groups, one for each distinct `user_id`):
 *
 *    [ 10,  20,  10,  50,  60,  20,  30,  80,  40 ]
 *      <-------user1-------->|<------user2------->
 *
 * The SUM aggregation is applied with 1 preceding and 1 following
 * row, with a minimum of 1 period. The aggregation window is thus 3 rows wide,
 * yielding the following column:
 *
 *    [ 30, 40,  80, 120, 110,  50, 130, 150, 120 ]
 *
 * Note: The SUMs calculated at the group boundaries (i.e. indices 0, 4, 5, and 8)
 * consider only 2 values each, in spite of the window-size being 3.
 * Each aggregation operation cannot cross group boundaries.
 * @endcode
 *
 * The returned column for `op == COUNT` always has `INT32` type. All other operators return a
 * column of the same type as the input. Therefore it is suggested to convert integer column types
 * (especially low-precision integers) to `FLOAT32` or `FLOAT64` before doing a rolling `MEAN`.
 *
 * @param[in] group_keys The (pre-sorted) grouping columns
 * @param[in] input The input column (to be aggregated)
 * @param[in] preceding_window The static rolling window size in the backward direction.
 * @param[in] following_window The static rolling window size in the forward direction.
 * @param[in] min_periods Minimum number of observations in window required to have a value,
 *                        otherwise element `i` is null.
 * @param[in] aggr The rolling window aggregation type (SUM, MAX, MIN, etc.)
 *
 * @returns   A nullable output column containing the rolling window results
 **/
std::unique_ptr<column> grouped_rolling_window(
  table_view const& group_keys,
  column_view const& input,
  size_type preceding_window,
  size_type following_window,
  size_type min_periods,
  std::unique_ptr<aggregation> const& aggr,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc std::unique_ptr<column> grouped_rolling_window(
 *            table_view const& group_keys,
 *            column_view const& input,
 *            size_type preceding_window,
 *            size_type following_window,
 *            size_type min_periods,
 *            std::unique_ptr<aggregation> const& aggr,
 *            rmm::mr::device_memory_resource* mr)
 *
 * @param default_outputs A column of per-row default values to be returned instead
 *                        of nulls. Used for LEAD()/LAG(), if the row offset crosses
 *                        the boundaries of the column or group.
 */
std::unique_ptr<column> grouped_rolling_window(
  table_view const& group_keys,
  column_view const& input,
  column_view const& default_outputs,
  size_type preceding_window,
  size_type following_window,
  size_type min_periods,
  std::unique_ptr<aggregation> const& aggr,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Applies a grouping-aware, timestamp-based rolling window function to the values in a
 *column.
 *
 * Like `rolling_window()`, this function aggregates values in a window around each
 * element of a specified `input` column. It differs from `rolling_window()` in two respects:
 *   1. The elements of the `input` column are grouped into distinct groups (e.g. the result of a
 *      groupby), determined by the corresponding values of the columns under `group_keys`. The
 *      window-aggregation cannot cross the group boundaries.
 *   2. Within a group, the aggregation window is calculated based on a time interval (e.g. number
 *      of days preceding/following the current row). The timestamps for the input data are
 *      specified by the `timestamp_column` argument.
 *
 * Note: This method requires that the rows are presorted by the group keys and timestamp values.
 *
 * @code{.pseudo}
 * Example: Consider a user-sales dataset, where the rows look as follows:
 *  { "user_id", sales_amt, date }
 *
 * This method enables windowing queries such as grouping a dataset by `user_id`, sorting by
 * increasing `date`, and summing up the `sales_amt` column over a window of 3 days (1 preceding
 *day, the current day, and 1 following day).
 *
 * In this example,
 *    1. `group_keys == [ user_id ]`
 *    2. `timestamp_column == date`
 *    3. `input == sales_amt`
 * The data are grouped by `user_id`, and ordered by `date`. The aggregation
 * (SUM) is then calculated for a window of 3 days around (and including) each row.
 *
 * For the following input:
 *
 *  [ // user,  sales_amt,  YYYYMMDD (date)
 *    { "user1",   10,      20200101    },
 *    { "user2",   20,      20200101    },
 *    { "user1",   20,      20200102    },
 *    { "user1",   10,      20200103    },
 *    { "user2",   30,      20200101    },
 *    { "user2",   80,      20200102    },
 *    { "user1",   50,      20200107    },
 *    { "user1",   60,      20200107    },
 *    { "user2",   40,      20200104    }
 *  ]
 *
 * Partitioning (grouping) by `user_id`, and ordering by `date` yields the following `sales_amt`
 * vector (with 2 groups, one for each distinct `user_id`):
 *
 * Date :(202001-)  [ 01,  02,  03,  07,  07,    01,   01,   02,  04 ]
 * Input:           [ 10,  20,  10,  50,  60,    20,   30,   80,  40 ]
 *                    <-------user1-------->|<---------user2--------->
 *
 * The SUM aggregation is applied, with 1 day preceding, and 1 day following, with a minimum of 1
 * period. The aggregation window is thus 3 *days* wide, yielding the following output column:
 *
 *  Results:        [ 30,  40,  30,  110, 110,  130,  130,  130,  40 ]
 *
 * @endcode
 *
 * Note: The number of rows participating in each window might vary, based on the index within the
 * group, datestamp, and `min_periods`. Apropos:
 *  1. results[0] considers 2 values, because it is at the beginning of its group, and has no
 *     preceding values.
 *  2. results[5] considers 3 values, despite being at the beginning of its group. It must include 2
 *     following values, based on its datestamp.
 *
 * Each aggregation operation cannot cross group boundaries.
 *
 * The returned column for `op == COUNT` always has `INT32` type. All other operators return a
 * column of the same type as the input. Therefore it is suggested to convert integer column types
 * (especially low-precision integers) to `FLOAT32` or `FLOAT64` before doing a rolling `MEAN`.
 *
 * @param[in] group_keys The (pre-sorted) grouping columns
 * @param[in] timestamp_column The (pre-sorted) timestamps for each row
 * @param[in] timestamp_order  The order (ASCENDING/DESCENDING) in which the timestamps are sorted
 * @param[in] input The input column (to be aggregated)
 * @param[in] preceding_window_in_days The rolling window time-interval in the backward direction.
 * @param[in] following_window_in_days The rolling window time-interval in the forward direction.
 * @param[in] min_periods Minimum number of observations in window required to have a value,
 *                        otherwise element `i` is null.
 * @param[in] aggr The rolling window aggregation type (SUM, MAX, MIN, etc.)
 *
 * @returns   A nullable output column containing the rolling window results
 */
std::unique_ptr<column> grouped_time_range_rolling_window(
  table_view const& group_keys,
  column_view const& timestamp_column,
  cudf::order const& timestamp_order,
  column_view const& input,
  size_type preceding_window_in_days,
  size_type following_window_in_days,
  size_type min_periods,
  std::unique_ptr<aggregation> const& aggr,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Applies a variable-size rolling window function to the values in a column.
 *
 * This function aggregates values in a window around each element i of the input column, and
 * invalidates the bit mask for element i if there are not enough observations. The window size is
 * dynamic (varying for each element). This matches Pandas' API for DataFrame.rolling with a few
 * notable differences:
 * - instead of the center flag it uses a two-part window to allow for more flexible windows.
 *   The total window size = `preceding_window + following_window`. Element `i` uses elements
 *   `[i-preceding_window+1, i+following_window]` to do the window computation.
 * - instead of storing NA/NaN for output rows that do not meet the minimum number of observations
 *   this function updates the valid bitmask of the column to indicate which elements are valid.
 * - support for dynamic rolling windows, i.e. window size can be specified for each element using
 *   an additional array.
 *
 * The returned column for count aggregation always has INT32 type. All other operators return a
 * column of the same type as the input. Therefore it is suggested to convert integer column types
 * (especially low-precision integers) to `FLOAT32` or `FLOAT64` before doing a rolling `MEAN`.
 *
 * @throws cudf::logic_error if window column type is not INT32
 *
 * @param[in] input_col The input column
 * @param[in] preceding_window A non-nullable column of INT32 window sizes in the forward direction.
 *                             `preceding_window[i]` specifies preceding window size for
 *                             element `i`.
 * @param[in] following_window A non-nullable column of INT32 window sizes in the backward
 *                             direction. `following_window[i]` specifies following window size
 *                             for element `i`.
 * @param[in] min_periods Minimum number of observations in window required to have a value,
 *                        otherwise element `i` is null.
 * @param[in] agg The rolling window aggregation type (sum, max, min, etc.)
 *
 * @returns   A nullable output column containing the rolling window results
 */
std::unique_ptr<column> rolling_window(
  column_view const& input,
  column_view const& preceding_window,
  column_view const& following_window,
  size_type min_periods,
  std::unique_ptr<aggregation> const& agg,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
