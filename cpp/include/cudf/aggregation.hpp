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

#include <functional>
#include <memory>
#include <vector>

/**
 * @file aggregation.hpp
 * @brief Representation for specifying desired aggregations from
 * aggregation-based APIs, e.g., groupby, reductions, rolling, etc.
 *
 * @note Not all aggregation APIs support all aggregation operations. See
 * individual function documentation to see what aggregations are supported.
 */

namespace cudf {
/**
 * @addtogroup aggregation_factories
 * @{
 * @file
 */

/**
 * @brief Base class for specifying the desired aggregation in an
 * `aggregation_request`.
 *
 * Other kinds of aggregations may derive from this class to encapsulate
 * additional information needed to compute the aggregation.
 */
class aggregation {
 public:
  /**
   * @brief Possible aggregation operations
   */
  enum Kind {
    SUM,             ///< sum reduction
    PRODUCT,         ///< product reduction
    MIN,             ///< min reduction
    MAX,             ///< max reduction
    COUNT_VALID,     ///< count number of valid elements
    COUNT_ALL,       ///< count number of elements
    ANY,             ///< any reduction
    ALL,             ///< all reduction
    SUM_OF_SQUARES,  ///< sum of squares reduction
    MEAN,            ///< arithmetic mean reduction
    VARIANCE,        ///< groupwise variance
    STD,             ///< groupwise standard deviation
    MEDIAN,          ///< median reduction
    QUANTILE,        ///< compute specified quantile(s)
    ARGMAX,          ///< Index of max element
    ARGMIN,          ///< Index of min element
    NUNIQUE,         ///< count number of unique elements
    NTH_ELEMENT,     ///< get the nth element
    ROW_NUMBER,      ///< get row-number of element
    COLLECT,         ///< collect values into a list
    LEAD,            ///< window function, accesses row at specified offset following current row
    LAG,             ///< window function, accesses row at specified offset preceding current row
    PTX,             ///< PTX UDF based reduction
    CUDA             ///< CUDA UDf based reduction
  };

  aggregation(aggregation::Kind a) : kind{a} {}
  Kind kind;  ///< The aggregation to perform

  virtual bool is_equal(aggregation const& other) const { return kind == other.kind; }

  virtual size_t do_hash() const { return std::hash<int>{}(kind); }

  virtual std::unique_ptr<aggregation> clone() const
  {
    return std::make_unique<aggregation>(*this);
  }

  virtual ~aggregation() = default;
};

enum class udf_type : bool { CUDA, PTX };

/// Factory to create a SUM aggregation
std::unique_ptr<aggregation> make_sum_aggregation();

/// Factory to create a PRODUCT aggregation
std::unique_ptr<aggregation> make_product_aggregation();

/// Factory to create a MIN aggregation
std::unique_ptr<aggregation> make_min_aggregation();

/// Factory to create a MAX aggregation
std::unique_ptr<aggregation> make_max_aggregation();

/**
 * @brief Factory to create a COUNT aggregation
 *
 * @param null_handling Indicates if null values will be counted.
 */
std::unique_ptr<aggregation> make_count_aggregation(
  null_policy null_handling = null_policy::EXCLUDE);

/// Factory to create a ANY aggregation
std::unique_ptr<aggregation> make_any_aggregation();

/// Factory to create a ALL aggregation
std::unique_ptr<aggregation> make_all_aggregation();

/// Factory to create a SUM_OF_SQUARES aggregation
std::unique_ptr<aggregation> make_sum_of_squares_aggregation();

/// Factory to create a MEAN aggregation
std::unique_ptr<aggregation> make_mean_aggregation();

/**
 * @brief Factory to create a VARIANCE aggregation
 *
 * @param ddof Delta degrees of freedom. The divisor used in calculation of
 *             `variance` is `N - ddof`, where `N` is the population size.
 */
std::unique_ptr<aggregation> make_variance_aggregation(size_type ddof = 1);

/**
 * @brief Factory to create a STD aggregation
 *
 * @param ddof Delta degrees of freedom. The divisor used in calculation of
 *             `std` is `N - ddof`, where `N` is the population size.
 */
std::unique_ptr<aggregation> make_std_aggregation(size_type ddof = 1);

/// Factory to create a MEDIAN aggregation
std::unique_ptr<aggregation> make_median_aggregation();

/**
 * @brief Factory to create a QUANTILE aggregation
 *
 * @param quantiles The desired quantiles
 * @param interpolation The desired interpolation
 */
std::unique_ptr<aggregation> make_quantile_aggregation(std::vector<double> const& q,
                                                       interpolation i = interpolation::LINEAR);

/**
 * @brief Factory to create an `argmax` aggregation
 *
 * `argmax` returns the index of the maximum element.
 */
std::unique_ptr<aggregation> make_argmax_aggregation();

/**
 * @brief Factory to create an `argmin` aggregation
 *
 * `argmin` returns the index of the minimum element.
 */
std::unique_ptr<aggregation> make_argmin_aggregation();

/**
 * @brief Factory to create a `nunique` aggregation
 *
 * `nunique` returns the number of unique elements.
 * @param null_handling Indicates if null values will be counted.
 */
std::unique_ptr<aggregation> make_nunique_aggregation(
  null_policy null_handling = null_policy::EXCLUDE);

/**
 * @brief Factory to create a `nth_element` aggregation
 *
 * `nth_element` returns the n'th element of the group/series.
 *
 * If @p n is not within the range `[-group_size, group_size)`, the result of
 * the respective group will be null. Negative indices `[-group_size, -1]`
 * corresponds to `[0, group_size-1]` indices respectively where `group_size` is
 * the size of each group.
 *
 * @param n index of nth element in each group.
 * @param null_handling Indicates to include/exclude nulls during indexing.
 */
std::unique_ptr<aggregation> make_nth_element_aggregation(
  size_type n, null_policy null_handling = null_policy::INCLUDE);

/// Factory to create a ROW_NUMBER aggregation
std::unique_ptr<aggregation> make_row_number_aggregation();

/// Factory to create a COLLECT_NUMBER aggregation
std::unique_ptr<aggregation> make_collect_aggregation();

/// Factory to create a LAG aggregation
std::unique_ptr<aggregation> make_lag_aggregation(size_type offset);

/// Factory to create a LEAD aggregation
std::unique_ptr<aggregation> make_lead_aggregation(size_type offset);

/**
 * @brief Factory to create an aggregation base on UDF for PTX or CUDA
 *
 * @param[in] type: either udf_type::PTX or udf_type::CUDA
 * @param[in] user_defined_aggregator A string containing the aggregator code
 * @param[in] output_type expected output type
 *
 * @return aggregation unique pointer housing user_defined_aggregator string.
 */
std::unique_ptr<aggregation> make_udf_aggregation(udf_type type,
                                                  std::string const& user_defined_aggregator,
                                                  data_type output_type);

/** @} */  // end of group
}  // namespace cudf
