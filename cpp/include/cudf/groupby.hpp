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

#include <cudf/types.hpp>

#include <utility>
#include <vector>

namespace cudf {
namespace experimental {
namespace groupby {
/**
 * @brief Options for controlling behavior of the groupby operation.
 */
struct Options {
  Options(bool _ignore_null_keys) : ignore_null_keys{_ignore_null_keys} {}

  Options() = default;

  /**
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
   * @note The behavior for a Pandas groupby operation is
   * `ignore_null_keys == true`.
   * @note The behavior for a SQL groupby operation is
   * `ignore_null_keys == false`.
   *
   */
  bool const ignore_null_keys{true};
};

/**
 * @brief Base class for specifying the desired aggregation in an
 * `aggregation_request`.
 *
 * Other kinds of aggregations may derive from this class to encapsulate
 * additional information needed to compute the aggregation.
 */
struct aggregation {
  /**
   * @brief The aggregation operations that may be performed
   */
  enum Kind { SUM, MIN, MAX, COUNT, MEAN, MEDIAN, QUANTILE };
  Kind kind;  ///< The kind of aggregation to perform
};

/**
 * @brief Encapsulates the request for groupby aggregation operation(s) to be
 * performed on a column.
 */
struct aggregation_request {
  column_view values;
  std::vector<std::unique_ptr<aggregation>> aggregations;
};

/**
 * @brief Groups together equivalent rows in `keys` and performs the requested
 * aggregation(s) on corresponding values.
 *
 * The values to aggregate and the aggregations to perform are specifed in an
 * `aggregation_request`. Each request contains a `column_view` of values to
 * aggregate and a set of `aggregation`s to perform on those elements.
 *
 * For each `aggregation` in a request, `values[i]` will be aggregated with all
 * other `values[j]` where rows `i` and `j` in `keys` are equivalent.
 *
 * The `size()` of the request column must equal `keys.num_rows()`.
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
 * @param keys The table of keys
 * @param requests The set of columns to aggregate and the aggregations to
 * perform
 * @param options Controls behavior of the groupby 
 * @param mr Memory resource used to allocate the returned table and columns
 * @return Pair containing a table of the unique rows from `keys` and a set of
 * `column`s containing the result(s) of the requested aggregations.
 */
std::pair<std::unique_ptr<table>, std::vector<std::unique_ptr<column>>> groupby(
    table_view const& keys,
    std::vector<aggregation_request> const& requests,
    Options options = Options{},
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace groupby
}  // namespace experimental
}  // namespace cudf