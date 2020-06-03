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
//! `groupby` APIs
namespace groupby {
namespace detail {
namespace sort {
class sort_groupby_helper;

}  // namespace sort
}  // namespace detail

/**
 * @addtogroup aggregation_groupby
 * @{
 */

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
  column_view values;                                      ///< The elements to aggregate
  std::vector<std::unique_ptr<aggregation>> aggregations;  ///< Desired aggregations
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
  groupby(groupby&&)      = delete;
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
   * @param null_handling Indicates whether rows in `keys` that contain
   * NULL values should be included
   * @param keys_are_sorted Indicates whether rows in `keys` are already sorted
   * @param column_order If `keys_are_sorted == YES`, indicates whether each
   * column is ascending/descending. If empty, assumes all  columns are
   * ascending. Ignored if `keys_are_sorted == false`.
   * @param null_precedence If `keys_are_sorted == YES`, indicates the ordering
   * of null values in each column. Else, ignored. If empty, assumes all columns
   * use `null_order::BEFORE`. Ignored if `keys_are_sorted == false`.
   */
  explicit groupby(table_view const& keys,
                   null_policy null_handling                      = null_policy::EXCLUDE,
                   sorted keys_are_sorted                         = sorted::NO,
                   std::vector<order> const& column_order         = {},
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
   * @param mr Device memory resource used to allocate the returned table and columns' device memory
   * @return Pair containing the table with each group's unique key and
   * a vector of aggregation_results for each request in the same order as
   * specified in `requests`.
   */
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> aggregate(
    std::vector<aggregation_request> const& requests,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  /**
   * @brief The grouped data corresponding to a groupby operation on a set of values.
   *
   * A `groups` object holds two tables of identical number of rows:
   * a table of grouped keys and a table of grouped values. In addition, it holds
   * a vector of integer offsets into the rows of the tables, such that
   * `offsets[i+1] - offsets[i]` gives the size of group `i`.
   */
  struct groups {
    std::unique_ptr<table> keys;
    std::vector<size_type> offsets;
    std::unique_ptr<table> values;
  };

  /**
   * @brief Get the grouped keys and values corresponding to a groupby operation on a set of values
   *
   * Returns a `groups` object representing the grouped keys and values.
   * If values is not provided, only a grouping of the keys is performed,
   * and the `values` of the `groups` object will be `nullptr`.
   *
   * @param values Table representing values on which a groupby operation is to be performed
   * @param mr Device memory resource used to allocate the returned tables's device memory in the
   * returned groups
   * @return A `groups` object representing grouped keys and values
   */
  groups get_groups(cudf::table_view values             = {},
                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

 private:
  table_view _keys;                                      ///< Keys that determine grouping
  null_policy _include_null_keys{null_policy::EXCLUDE};  ///< Include rows in keys
                                                         ///< with NULLs
  sorted _keys_are_sorted{sorted::NO};                   ///< Whether or not the keys are sorted
  std::vector<order> _column_order{};                    ///< If keys are sorted, indicates
                                                         ///< the order of each column
  std::vector<null_order> _null_precedence{};            ///< If keys are sorted,
                                                         ///< indicates null order
                                                         ///< of each column
  std::unique_ptr<detail::sort::sort_groupby_helper>
    _helper;  ///< Helper object
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
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> dispatch_aggregation(
    std::vector<aggregation_request> const& requests,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr);

  // Sort-based groupby
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> sort_aggregate(
    std::vector<aggregation_request> const& requests,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr);
};
/** @} */
}  // namespace groupby
}  // namespace cudf
