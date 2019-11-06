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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <utility>
#include <vector>

namespace cudf {
namespace experimental {

/**
 * @brief Interpolation method to use, when the desired quantile lies between
 * two data points i and j
 *
 * TODO: Move this somewhere else once quantiles are ported to libcudf++
 *
 */
struct interpolation {
  enum type {
    LINEAR = 0,  ///< Linear interpolation between i and j
    LOWER,       ///< Lower data point (i)
    HIGHER,      ///< Higher data point (j)
    MIDPOINT,    ///< (i + j)/2
    NEAREST      ///< i or j, whichever is nearest
  };
};

namespace groupby {
/**
 * @brief Base class for specifying the desired aggregation in an
 * `aggregation_request`.
 *
 * Other kinds of aggregations may derive from this class to encapsulate
 * additional information needed to compute the aggregation.
 */
struct aggregation {
  /**
   * @brief Possible aggregation operations
   */
  enum Kind { SUM, MIN, MAX, COUNT, MEAN, MEDIAN, QUANTILE };

  aggregation(aggregation::Kind a) : kind{a} {}
  Kind kind;  ///< The aggregation to perform
};

/**
 * @brief Derived class for specifying a quantile aggregation
 */
struct quantile_aggregation : aggregation {
  quantile_aggregation(std::vector<double> const& quantiles,
                       interpolation::type interpolation)
      : aggregation{QUANTILE},
        quantiles{quantiles},
        interpolation{interpolation} {}
  std::vector<double> quantiles;      ///< Desired quantile(s)
  interpolation::type interpolation;  ///< Desired interpolation
};

/// Factory to create a SUM aggregation
std::unique_ptr<aggregation> make_sum_aggregation();

/// Factory to create a MIN aggregation
std::unique_ptr<aggregation> make_min_aggregation();

/// Factory to create a MAX aggregation
std::unique_ptr<aggregation> make_max_aggregation();

/// Factory to create a COUNT aggregation
std::unique_ptr<aggregation> make_count_aggregation();

/// Factory to create a MEAN aggregation
std::unique_ptr<aggregation> make_mean_aggregation();

/// Factory to create a MEDIAN aggregation
std::unique_ptr<aggregation> make_median_aggregation();

/**
 * @brief Factory to create a QUANTILE aggregation
 *
 * @param quantiles The desired quantiles
 * @param interpolation The desired interpolation
 */
std::unique_ptr<aggregation> make_quantile_aggregation(
    std::vector<double> const& quantiles, interpolation::type interpolation);

/**
 * @brief Request for groupby aggregation(s) to perform on a column.
 */
struct aggregation_request {
  column_view values;  ///< The elements to aggregate
  std::vector<std::unique_ptr<aggregation>>
      aggregations;  ///< Desired aggregations
};

/**
 * @brief The result(s) of an `aggregation_request`
 */
struct aggregation_result {
  /// Pairs containing columns of aggregation results and their corresponding
  /// aggregation
  std::vector<std::pair<std::unique_ptr<column>, std::unique_ptr<aggregation>>>
      results{};
};

/**
 * @brief Groups values by keys and computes aggregations on those groups.
 */
class groupby {
 public:
  groupby() = delete;
  ~groupby() = default;
  groupby(groupby const&) = delete;
  groupby(groupby&&) = delete;
  groupby& operator=(groupby const&) = delete;
  groupby& operator=(groupby&&) = delete;

  /**
   * @brief Construct a groupby object with the specified `keys`
   *
   * @note This object does *not* maintain the lifetime of `keys`. It is the
   * users responsibility to ensure the `groupby` object does not outlive
   * `keys`.
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
   * For each `aggregation` in a request, `values[i]` will be aggregated with
   * all other `values[j]` where rows `i` and `j` in `keys` are equivalent.
   *
   * The `size()` of the request column must equal `keys.num_rows()`.
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
   * @return Pair containing the table with each groups unique key and
   * a set of aggregation_results for each request in the same order as was
   * specified in `requests`.
   */
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> aggregate(
      std::vector<aggregation_request> const& requests,
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

  /**
   * @brief Dispatches to the appropriate implementation to satisfy the
   * aggregation requests.
   */
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>>
  dispatch_aggregation(std::vector<aggregation_request> const& requests,
                       cudaStream_t stream,
                       rmm::mr::device_memory_resource* mr);
};
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf