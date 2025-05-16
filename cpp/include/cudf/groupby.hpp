/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/column/column_view.hpp>
#include <cudf/replace.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
//! `groupby` APIs
namespace groupby {
namespace detail {
namespace sort {
struct sort_groupby_helper;

}  // namespace sort
}  // namespace detail

/**
 * @addtogroup aggregation_groupby
 * @{
 * @file
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
  column_view values;                                              ///< The elements to aggregate
  std::vector<std::unique_ptr<groupby_aggregation>> aggregations;  ///< Desired aggregations
};

/**
 * @brief Request for groupby aggregation(s) for scanning a column.
 *
 * The group membership of each `value[i]` is determined by the corresponding
 * row `i` in the original order of `keys` used to construct the
 * `groupby`. I.e., for each `aggregation`, `values[i]` is aggregated with all
 * other `values[j]` where rows `i` and `j` in `keys` are equivalent.
 *
 * `values.size()` column must equal `keys.num_rows()`.
 */
struct scan_request {
  column_view values;  ///< The elements to aggregate
  std::vector<std::unique_ptr<groupby_scan_aggregation>> aggregations;  ///< Desired aggregations
};

/**
 * @brief The result(s) of an `aggregation_request`
 *
 * For every `aggregation_request` given to `groupby::aggregate` an
 * `aggregation_result` will be returned. The `aggregation_result` holds the
 * resulting column(s) for each requested aggregation on the `request`s values.
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
  groupby(groupby const&)            = delete;
  groupby(groupby&&)                 = delete;
  groupby& operator=(groupby const&) = delete;
  groupby& operator=(groupby&&)      = delete;

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
   * use `null_order::AFTER`. Ignored if `keys_are_sorted == false`.
   */
  explicit groupby(table_view const& keys,
                   null_policy null_handling                      = null_policy::EXCLUDE,
                   sorted keys_are_sorted                         = sorted::NO,
                   std::vector<order> const& column_order         = {},
                   std::vector<null_order> const& null_precedence = {});

  /**
   * @brief Performs grouped aggregations on the specified values.
   *
   * The values to aggregate and the aggregations to perform are specified in an
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
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned table and columns' device memory
   * @return Pair containing the table with each group's unique key and
   * a vector of aggregation_results for each request in the same order as
   * specified in `requests`.
   */
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> aggregate(
    host_span<aggregation_request const> requests,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
  /**
   * @brief Performs grouped scans on the specified values.
   *
   * The values to aggregate and the aggregations to perform are specified in an
   * `aggregation_request`. Each request contains a `column_view` of values to
   * aggregate and a set of `aggregation`s to perform on those elements.
   *
   * For each `aggregation` in a request, `values[i]` is scan aggregated with
   * all previous `values[j]` where rows `i` and `j` in `keys` are equivalent.
   *
   * The `size()` of the request column must equal `keys.num_rows()`.
   *
   * For every `aggregation_request` an `aggregation_result` will be returned.
   * The `aggregation_result` holds the resulting column(s) for each requested
   * aggregation on the `request`s values. The order of the columns in each
   * result is the same order as was specified in the request.
   *
   * The returned `table` contains the group labels for each row, i.e., the
   * `keys` given to groupby object. Element `i` across all aggregation results
   * belongs to the group at row `i` in the group labels table.
   *
   * The order of the rows in the group labels is arbitrary. Furthermore,
   * successive `groupby::scan` calls may return results in different orders.
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
   * keys:  {3 1 1 1 2}
   *        {4 1 1 1 2}
   * values:
   *   SUM: {9 3 7 9 1}
   *   MIN: {9 3 3 2 1}
   * ```
   *
   * @param requests The set of columns to scan and the scans to perform
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned table and columns' device memory
   * @return Pair containing the table with each group's key and
   * a vector of aggregation_results for each request in the same order as
   * specified in `requests`.
   */
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> scan(
    host_span<scan_request const> requests,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Performs grouped shifts for specified values.
   *
   * In `j`th column, for each group, `i`th element is determined by the `i - offsets[j]`th
   * element of the group. If `i - offsets[j] < 0 or >= group_size`, the value is determined by
   * @p fill_values[j].
   *
   * @note The first returned table stores the keys passed to the groupby object. Row `i` of the key
   * table corresponds to the group labels of row `i` in the shifted columns. The key order in
   * each group matches the input order. The order of each group is arbitrary. The group order
   * in successive calls to `groupby::shifts` may be different.
   *
   * Example:
   * @code{.pseudo}
   * keys:    {1 4 1 3 4 4 1}
   *          {1 2 1 3 2 2 1}
   * values:  {3 9 1 4 2 5 7}
   *          {"a" "c" "bb" "ee" "z" "x" "d"}
   * offset:  {2, -1}
   * fill_value: {@, @}
   * result (group order maybe different):
   *    keys:   {3 1 1 1 4 4 4}
   *            {3 1 1 1 2 2 2}
   *    values: {@ @ @ 3 @ @ 9}
   *            {@ "bb" "d" @ "z" "x" @}
   *
   * -------------------------------------------------
   * keys:    {1 4 1 3 4 4 1}
   *          {1 2 1 3 2 2 1}
   * values:  {3 9 1 4 2 5 7}
   *          {"a" "c" "bb" "ee" "z" "x" "d"}
   * offset:  {-2, 1}
   * fill_value: {-1, "42"}
   * result (group order maybe different):
   *    keys:   {3 1 1 1 4 4 4}
   *            {3 1 1 1 2 2 2}
   *    values: {-1 7 -1 -1 5 -1 -1}
   *            {"42" "42" "a" "bb" "42" "c" "z"}
   *
   * @endcode
   *
   * @param values Table whose columns to be shifted
   * @param offsets The offsets by which to shift the input
   * @param fill_values Fill values for indeterminable outputs
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned table and columns' device memory
   * @return Pair containing the tables with each group's key and the columns shifted
   *
   * @throws cudf::logic_error if @p fill_value[i] dtype does not match @p values[i] dtype for
   * `i`th column
   */
  std::pair<std::unique_ptr<table>, std::unique_ptr<table>> shift(
    table_view const& values,
    host_span<size_type const> offsets,
    std::vector<std::reference_wrapper<scalar const>> const& fill_values,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief The grouped data corresponding to a groupby operation on a set of values.
   *
   * A `groups` object holds two tables of identical number of rows:
   * a table of grouped keys and a table of grouped values. In addition, it holds
   * a vector of integer offsets into the rows of the tables, such that
   * `offsets[i+1] - offsets[i]` gives the size of group `i`.
   */
  struct groups {
    std::unique_ptr<table> keys;     ///< Table of grouped keys
    std::vector<size_type> offsets;  ///< Group Offsets
    std::unique_ptr<table> values;   ///< Table of grouped values
  };

  /**
   * @brief Get the grouped keys and values corresponding to a groupby operation on a set of values
   *
   * Returns a `groups` object representing the grouped keys and values.
   * If values is not provided, only a grouping of the keys is performed,
   * and the `values` of the `groups` object will be `nullptr`.
   *
   * @param values Table representing values on which a groupby operation is to be performed
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned tables's device memory in the
   * returned groups
   * @return A `groups` object representing grouped keys and values
   */
  groups get_groups(cudf::table_view values           = {},
                    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Performs grouped replace nulls on @p value
   *
   * For each `value[i] == NULL` in group `j`, `value[i]` is replaced with the first non-null value
   * in group `j` that precedes or follows `value[i]`. If a non-null value is not found in the
   * specified direction, `value[i]` is left NULL.
   *
   * The returned pair contains a column of the sorted keys and the result column. In result column,
   * values of the same group are in contiguous memory. In each group, the order of values maintain
   * their original order. The order of groups are not guaranteed.
   *
   * Example:
   * @code{.pseudo}
   *
   * //Inputs:
   * keys:    {3 3 1 3 1 3 4}
   *          {2 2 1 2 1 2 5}
   * values:  {3 4 7 @ @ @ @}
   *          {@ @ @ "x" "tt" @ @}
   * replace_policies:    {FORWARD, BACKWARD}
   *
   * //Outputs (group orders may be different):
   * keys:    {3 3 3 3 1 1 4}
   *          {2 2 2 2 1 1 5}
   * result:  {3 4 4 4 7 7 @}
   *          {"x" "x" "x" @ "tt" "tt" @}
   * @endcode
   *
   * @param[in] values A table whose column null values will be replaced
   * @param[in] replace_policies Specify the position of replacement values relative to null values,
   * one for each column
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   * @param[in] mr Device memory resource used to allocate device memory of the returned column
   *
   * @return Pair that contains a table with the sorted keys and the result column
   */
  std::pair<std::unique_ptr<table>, std::unique_ptr<table>> replace_nulls(
    table_view const& values,
    host_span<cudf::replace_policy const> replace_policies,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

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
    host_span<aggregation_request const> requests,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  // Sort-based groupby
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> sort_aggregate(
    host_span<aggregation_request const> requests,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> sort_scan(
    host_span<scan_request const> requests,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);
};
/** @} */
}  // namespace groupby
}  // namespace CUDF_EXPORT cudf
