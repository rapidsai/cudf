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

#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/table/table.hpp>

#include <tuple>
#include <vector>

namespace cudf {
namespace experimental {

namespace groupby {

/**---------------------------------------------------------------------------*
 * @brief Top-level options for controlling behavior of the groupby operation.
 *
 * This structure defines all of the shared options between the hash and
 * sort-based groupby algorithms. Implementation specific options should be
 * defined by a new structure that inherits from this one inside the appropriate
 * namespace.
 *
 *---------------------------------------------------------------------------**/
struct Options {
  Options(bool _ignore_null_keys) : ignore_null_keys{_ignore_null_keys} {}

  Options() = default;

  /**---------------------------------------------------------------------------*
   * Determines whether key rows with null values are ignored.
   *
   * If `true`, any row in the `keys` table that contains a NULL value will be
   * ignored. That is, the row will not be present in the output keys, and it's
   * associated row in the `values` table will also be ignored.
   *
   * If `false`, rows in the `keys` table with NULL values will be treated as
   * any other row. Furthermore, a NULL value will be considered equal to
   * another NULL value. For example, two rows `{1, 2, 3, NULL}` and `{1, 2, 3,
   * NULL}` will be considered equal, and their associated rows in the `values`
   * table will be aggregated.
   *
   * @note The behavior for a Pandas groupby operation is `ignore_null_keys ==
   *true`.
   * @note The behavior for a SQL groupby operation is `ignore_null_keys ==
   *false`.
   *
   *---------------------------------------------------------------------------**/
  bool const ignore_null_keys{true};
};

/**---------------------------------------------------------------------------*
 * @brief Supported aggregation operations
 *
 *---------------------------------------------------------------------------**/
enum operators { SUM, MIN, MAX, COUNT, MEAN, MEDIAN, QUANTILE};

namespace hash {

/**---------------------------------------------------------------------------*
 * @brief  Options unique to the hash-based groupby
 *---------------------------------------------------------------------------**/
struct Options : groupby::Options {
  Options(bool _ignore_null_keys = true)
      : groupby::Options(_ignore_null_keys) {}
}; 

/**---------------------------------------------------------------------------*
 * @brief Performs groupby operation(s) via a hash-based implementation
 *
 * Given a table of keys and corresponding table of values, equivalent keys will
 * be grouped together and a reduction operation performed across the associated
 * values (i.e., reduce by key). The reduction operation to be performed is
 * specified by a list of operator enums of equal length to the number of value
 * columns.
 *
 * The output of the operation is the table of key columns that hold all the
 * unique keys from the input key columns and a table of aggregation columns
 * that hold the specified reduction across associated values among all
 * identical keys.
 *
 * @param keys The table of keys
 * @param values The table of aggregation values
 * @param ops The list of aggregation operations
 * @return A tuple whose first member contains the table of output keys, and
 * second member contains the table of reduced output values
 *---------------------------------------------------------------------------**/
std::pair<std::unique_ptr<table>, std::unique_ptr<table>> groupby(table_view const& keys,
                                                                  table_view const& values,
                                                                  std::vector<operators> const& ops,
                                                                  Options options = Options{});

}  // namespace hash

namespace sort {

/**---------------------------------------------------------------------------*
 * @brief  Arguments to be passed to an operator
 *---------------------------------------------------------------------------**/
struct operation_args {};

/**---------------------------------------------------------------------------*
 * @brief  Arguments to be passed to the quantile operator
 *---------------------------------------------------------------------------**/
struct quantile_args : operation_args {
  std::vector<double> quantiles;
  cudf::interpolation interpolation;
  
  quantile_args(const std::vector<double> &_quantiles,  cudf::interpolation _interpolation)
  : operation_args{}, quantiles{_quantiles}, interpolation{_interpolation}
  {}
};

/**---------------------------------------------------------------------------*
 * @brief  pairs an operator with its arguments
 *---------------------------------------------------------------------------**/
struct operation {
  operators                         op_name;
  std::unique_ptr<operation_args>   args;
};

/**---------------------------------------------------------------------------*
 * @brief  Options unique to the sort-based groupby
 * The priority of determining the sort flags:
 * - The `ignore_null_keys` take precedence over the `null_sort_behavior`
 *---------------------------------------------------------------------------**/
struct Options : groupby::Options {
  null_order null_sort_behavior; ///< Indicates how nulls are treated
  bool input_sorted; ///< Indicates if the input data is sorted. 

  Options(bool _ignore_null_keys = true,
          null_order _null_sort_behavior = null_order::AFTER,
          bool _input_sorted = false)
      : groupby::Options(_ignore_null_keys),
        input_sorted(_input_sorted),
        null_sort_behavior(_null_sort_behavior) {}
};

/**---------------------------------------------------------------------------*
 * @brief Performs groupby operation(s) via a sort-based implementation
 *
 * Given a table of keys and corresponding table of values, equivalent keys will
 * be grouped together and a reduction operation performed across the associated
 * values (i.e., reduce by key). The reduction operation to be performed is
 * specified by a list of operator enums of equal length to the number of value
 * columns.
 *
 * The output of the operation is the table of key columns that hold all the
 * unique keys from the input key columns and a table of aggregation columns
 * that hold the specified reduction across associated values among all
 * identical keys.
 *
 * @param keys The table of keys
 * @param values The table of aggregation values
 * @param ops The list of aggregation operations
 * @return A tuple whose first member contains the table of output keys, and
 * second member contains the reduced output values
 *---------------------------------------------------------------------------**/
std::pair<std::unique_ptr<table>, std::unique_ptr<table>> groupby(table_view const& keys,
                                                                  table_view const& values,
                                                                  std::vector<operation> const& ops,
                                                                  Options options = Options{});
}  // namespace sort
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf

/**
 * @brief Returns the first index of each unique row. Assumes the data is
 * already sorted
 *
 * @param[in]  input_table          The input columns whose rows are sorted.
 *
 * @returns A non-nullable column of `GDF_INT32` elements containing the indices of the first occurrences of each unique row.
 */
std::unique_ptr<column> unique_indices(table_view const& input_table);

/**
 * @brief Sorts a set of columns based on specified "key" columns. Returns a
 * column containing the offset to the start of each set of unique keys.
 *
 * @param[in]  input_table        The input columns whose rows will be grouped.
 * @param[in]  key_col_indices    Vector ordering the key columns by which
 *                                data will be grouped.
 * @param[in]  null_precedence    The size of a NULL value in comparison to all
 *                                other values
 * @param[in]  ignore_null_keys   If true, ignore groups that have nulls in
 *                                their keys (Pandas style).  If false, null keys
 *                                compare as equal (SQL style).
 * @param[in]  mr                 Device memory resource to use for device memory
 *                                allocation
 *
 * @returns A tuple containing:
 *          - A cudf::table containing a set of columns sorted by the key columns.
 *          - A non-nullable column of `INT32` elements containing the indices of
 *            the first occurrences of each unique row.
 */
std::pair<std::unique_ptr<table>, std::unique_ptr<column>>
  gdf_group_by_without_aggregations(table_view const& input_table,
                                    std::vector<size_type> const& key_col_indices,
                                    null_order null_precedence = null_order::AFTER,
                                    bool ignore_null_keys = true,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
