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

#ifndef GROUPBY_HPP
#define GROUPBY_HPP

#include "cudf.h"
#include "types.hpp"
#include <cudf/quantiles.hpp>

#include <tuple>
#include <vector>

// Not possible to forward declare rmm::device_vector because it's a type alias
// to a type with a default template arg. Therefore, this is the best we can do
namespace thrust {
template <typename T, typename A>
class device_vector;
}
template <typename T>
class rmm_allocator;

namespace cudf {

class table;

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
std::pair<cudf::table, cudf::table> groupby(cudf::table const& keys,
                                            cudf::table const& values,
                                            std::vector<operators> const& ops,
                                            Options options = Options{});

}  // namespace hash

namespace sort {

struct operation_args {};

struct quantile_args : operation_args {
  std::vector<double> quantiles;
  cudf::interpolation interpolation;
  
  quantile_args(const std::vector<double> &_quantiles,  cudf::interpolation _interpolation)
  : operation_args{}, quantiles{_quantiles}, interpolation{_interpolation}
  {}
};

struct operation {
  operators                         op_name;
  std::unique_ptr<operation_args>   args;
};

enum class null_order : bool { AFTER, BEFORE }; 

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
std::pair<cudf::table, std::vector<gdf_column*>> groupby(cudf::table const& keys,
                                            cudf::table const& values,
                                            std::vector<operation> const& ops,
                                            Options options = Options{});
}  // namespace sort
}  // namespace groupby
}  // namespace cudf

/**
 * @brief Returns the first index of each unique row. Assumes the data is
 * already sorted
 *
 * @param[in]  input_table          The input columns whose rows are sorted.
 * @param[in]  context              The options for controlling treatment of
 * nulls context->flag_null_sort_behavior GDF_NULL_AS_LARGEST = Nulls are
 * treated as largest, GDF_NULL_AS_SMALLEST = Nulls are treated as smallest,
 *
 * @returns A non-nullable column of `GDF_INT32` elements containing the indices of the first occurrences of each unique row.
 */
gdf_column
gdf_unique_indices(cudf::table const& input_table, gdf_context const& context);

/**
 * @brief Sorts a set of columns based on specified "key" columns. Returns a
 * column containing the offset to the start of each set of unique keys.
 *
 * @param[in]  input_table           The input columns whose rows will be
 * grouped.
 * @param[in]  num_key_cols             The number of key columns.
 * @param[in]  key_col_indices          The indices of the of the key columns by
 * which data will be grouped.
 * @param[in]  context                  The context used to control how nulls
 * are treated in group by context->flag_null_sort_behavior GDF_NULL_AS_LARGEST
 * = Nulls are treated as largest, GDF_NULL_AS_SMALLEST = Nulls are treated as
 * smallest, context-> flag_groupby_include_nulls false = Nulls keys are ignored
 * (Pandas style), true = Nulls keys are treated as values. NULL keys will
 * compare as equal NULL == NULL (SQL style)
 *
 * @returns A tuple containing:
 *          - A cudf::table containing a set of columns sorted by the key
 * columns.
 *          - A non-nullable column of `GDF_INT32` elements containing the indices of the first occurrences of each unique row.
 */
std::pair<cudf::table,
          gdf_column>
gdf_group_by_without_aggregations(cudf::table const& input_table,
                                  gdf_size_type num_key_cols,
                                  gdf_index_type const* key_col_indices,
                                  gdf_context* context);

#endif
