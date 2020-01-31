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

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <utility>
#include <vector>

namespace cudf {
namespace experimental {
namespace groupby {

namespace detail {
namespace sort {

class sort_groupby_helper;
    
} // namespace sort  
} // namespace detail


/**
 * @brief Request for groupby aggregation(s) to perform on a column.
 *
 * The group membership of each `value[i]` is determined by the corresponding
 * row `i` in the original order of `keys` used to construct the
 * `groupby`. I.e., for each `aggregation`, `values[i]` is aggregated with all
 * other `values[j]` where rows `i` and `j` in `keys` are equivalent.
 *
 * `values.size()` column must equal `keys.num_rows()`.
 */
struct aggregation_request {
  column_view values;  ///< The elements to aggregate
  std::vector<std::unique_ptr<aggregation>>
      aggregations;  ///< Desired aggregations
};

/**
 * @brief Struct defining the bounds of a window-based analytical query.
 */
struct window_bounds {

  // Number of rows preceding the current row, to consider in the window
  cudf::size_type preceding;    

  // Number of rows following the current row, to consider in the window
  cudf::size_type following; 

  // The minimum number of observations to consider, for a window to be
  // valid, including: 
  // 1. The current row
  // 2. Upto `preceding` rows appearing prior to the current row
  //    (subject to belonging within a group)
  // 3. Upto `following` rows appearing after the current row
  //    (subject to belonging within a group)
  // 
  // E.g. Consider the following int column-vector:
  //   [0,1,2,3,4,5]
  // If an aggregation operation (E.g. SUM) is specified with
  // the following window parameters:
  //   preceding   = 1
  //   following   = 1
  //   min_periods = 3
  //
  // Then, the resulting column-vector would be as follows:
  //   [INVALID, 3, 6, 9, 12, INVALID]
  // 
  // Specifically, the results at indices `0` and `5` are INVALID, 
  // since there are only two observations (each) at either end,
  // and *three* are needed at minimum, for the result to be valid.
  cudf::size_type min_periods;
};

/**
 * @brief Request for groupby window-based aggregation(s) to perform on a column.
 * 
 * Similarly to `aggregation_request`, a `window_aggregation_request` is applied
 * to all values in the column vector, but only across the elements within the 
 * confines of the associated `window_bounds`.
 * 
 * `values.size()` must equal `keys.num_rows()`.
 */
struct window_aggregation_request {
  column_view values; ///< the elements to aggregate
  std::vector<
    std::pair<window_bounds, std::unique_ptr<aggregation>>> 
    aggregations;  ///< Desired aggregations
};

/**
 * @brief The result(s) of an `aggregation_request`
 *
 * For every `aggregation_request` given to `groupby::aggregate` an
 * `aggregation_result` will be returned. The `aggregation_result` holds the
 * resulting column(s) for each requested aggregation on the `request`s values.
 *
 */
struct aggregation_result {
  /// Columns of results from an `aggregation_request`
  std::vector<std::unique_ptr<column>> results{};
};

/**
 * @brief Groups values by keys and computes aggregations on those groups.
 */
class groupby {
 public:
  groupby() = delete;
  ~groupby();
  groupby(groupby const&) = delete;
  groupby(groupby&&) = delete;
  groupby& operator=(groupby const&) = delete;
  groupby& operator=(groupby&&) = delete;

  /**
   * @brief Construct a groupby object with the specified `keys`
   *
   * If the `keys` are already sorted, better performance may be achieved by
   * passing `keys_are_sorted == true` and indicating the  ascending/descending
   * order of each column and null order in  `column_order` and
   * `null_precedence`, respectively.
   *
   * @note This object does *not* maintain the lifetime of `keys`. It is the
   * user's responsibility to ensure the `groupby` object does not outlive the
   * data viewed by the `keys` `table_view`.
   *
   * @param keys Table whose rows act as the groupby keys
   * @param ignore_null_keys Indicates whether rows in `keys` that contain NULL
   * values should be ignored
   * @param keys_are_sorted Indicates whether rows in `keys` are already sorted
   * @param column_order If `keys_are_sorted == true`, indicates whether each
   * column is ascending/descending. If empty, assumes all  columns are
   * ascending. Ignored if `keys_are_sorted == false`.
   * @param null_precedence If `keys_are_sorted == true`, indicates the ordering
   * of null values in each column. Else, ignored. If empty, assumes all columns
   * use `null_order::BEFORE`. Ignored if `keys_are_sorted == false`.
   */
  explicit groupby(table_view const& keys, bool ignore_null_keys = true,
                   bool keys_are_sorted = false,
                   std::vector<order> const& column_order = {},
                   std::vector<null_order> const& null_precedence = {});

  /**
   * @brief Performs grouped aggregations on the specified values.
   *
   * The values to aggregate and the aggregations to perform are specifed in an
   * `aggregation_request`. Each request contains a `column_view` of values to
   * aggregate and a set of `aggregation`s to perform on those elements.
   *
   * For each `aggregation` in a request, `values[i]` is aggregated with
   * all other `values[j]` where rows `i` and `j` in `keys` are equivalent.
   *
   * The `size()` of the request column must equal `keys.num_rows()`.
   *
   * For every `aggregation_request` an `aggregation_result` will be returned.
   * The `aggregation_result` holds the resulting column(s) for each requested
   * aggregation on the `request`s values. The order of the columns in each
   * result is the same order as was specified in the request.
   *
   * The returned `table` contains the group labels for each group, i.e., the
   * unique rows from `keys`. Element `i` across all aggregation results
   * belongs to the group at row `i` in the group labels table.
   *
   * The order of the rows in the group labels is arbitrary. Furthermore,
   * successive `groupby::aggregate` calls may return results in different
   * orders.
   *
   * @throws cudf::logic_error If `requests[i].values.size() !=
   * keys.num_rows()`.
   *
   * Example:
   * ```
   * Input:
   * keys:     {1 2 1 3 1}
   *           {1 2 1 4 1}
   * request:
   *   values: {3 1 4 9 2}
   *   aggregations: {{SUM}, {MIN}}
   *
   * result:
   *
   * keys:  {3 1 2}
   *        {4 1 2}
   * values:
   *   SUM: {9 9 1}
   *   MIN: {9 2 1}
   * ```
   *
   * @param requests The set of columns to aggregate and the aggregations to
   * perform
   * @param mr Memory resource used to allocate the returned table and columns
   * @return Pair containing the table with each group's unique key and
   * a vector of aggregation_results for each request in the same order as
   * specified in `requests`.
   */
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> aggregate(
      std::vector<aggregation_request> const& requests,
      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  /**
   * @brief Performs grouped window-based aggregations on the specified values.
   * 
   * A `window_aggregation_request` specifies a column-vector of values to
   * aggregate, a list of aggregation operations, and the `window_bounds`
   * that governs each aggregation.
   * 
   * Consider a user-sales dataset, where the rows look as follows:
   *  { "user_id", sales_amt, day }
   *
   * The `windowed_aggregate` method enables windowing queries such as 
   * grouping a dataset by `user_id`, sorting by increasing `day`, and summing up
   * the `sales_amt` column over a window of 3 days (1 day preceding, 
   * 1 day following).
   * 
   * Note: The groupby object must already be grouped and sorted appropriately,
   * before window-based `aggregate()` is invoked.
   * In the example above, the data would be grouped by `user_id`, and ordered
   * by `date`. The aggregation (SUM) would then be calculated for a window of
   * 3 values around (and including) each row.
   * 
   * If the input rows were as follows:
   * 
   *  [ 
   *    { "user1", 10, "Monday" },
   *    { "user2", 20, "Monday" },
   *    { "user1", 20, "Monday" },
   *    { "user1", 10, "Tuesday" },
   *    { "user2", 30, "Tuesday" },
   *    { "user2", 80, "Tuesday" },
   *    { "user1", 50, "Wednesday" },
   *    { "user1", 60, "Wednesday" },
   *    { "user2", 40, "Wednesday" }
   *  ]
   * 
   * Partitioning (grouping) by `user_id`, and ordering by `day` would yield
   * the following `sales_amt` vector (with 2 groups, * one for each distinct 
   * `user_id`):
   * 
   *    [ 10,  20,  10,  50,  60,  20,  30,  80,  40 ]
   *      <-------user1-------->|<------user2------->
   * 
   * The SUM aggregation would then be applied, with 1 preceding, and 1 following
   * row, with a minimum of 1 period. The aggregation window is thus 3 rows wide,
   * thereby yielding the following column:
   * 
   *    [ 30, 40,  80, 120, 110,  50, 130, 150, 120 ]
   * 
   * Note: The SUMs calculated at the group boundaries (i.e. indices 0,4,5, and 8)
   * consider only 2 values each, in spite of the window-size being 3.
   * Each aggregation operation cannot cross group boundaries.
   *
   * @param requests The list of window_aggregation_requests, specifying the columns
   * to aggregate, the aggregation operations per column, and the window-specification
   * per aggregation
   * @param mr Memory resource used to allocate the returned table and columns
   * @return Pair containing the table with each group's unique key and
   * a vector of aggregation_results for each request in the same order as
   * specified in `requests`.
   */
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> windowed_aggregate(
      std::vector<window_aggregation_request> const& requests,
      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

 private:
  table_view _keys;                    ///< Keys that determine grouping
  bool _ignore_null_keys{true};        ///< Ignore rows in keys with NULLs
  bool _keys_are_sorted{false};        ///< Whether or not the keys are sorted
  std::vector<order> _column_order{};  ///< If keys are sorted, indicates
                                       ///< the order of each column
  std::vector<null_order> _null_precedence{};  ///< If keys are sorted,
                                               ///< indicates null order
                                               ///< of each column
  std::unique_ptr<detail::sort::sort_groupby_helper> _helper; ///< Helper object
                                       ///< used by sort based implementation

  /**
   * @brief Get the sort helper object
   * 
   * The object is constructed on first invocation and subsequent invocations
   * of this function return the memoized object.
   */
  detail::sort::sort_groupby_helper& helper();

  /**
   * @brief Dispatches to the appropriate implementation to satisfy the
   * aggregation requests.
   */
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>>
  dispatch_aggregation(std::vector<aggregation_request> const& requests,
                       cudaStream_t stream,
                       rmm::mr::device_memory_resource* mr);

  // Sort-based groupby
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> 
  sort_aggregate(std::vector<aggregation_request> const& requests,
                 cudaStream_t stream, rmm::mr::device_memory_resource* mr);

};
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
