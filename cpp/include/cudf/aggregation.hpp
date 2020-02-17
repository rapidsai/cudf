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
#include <vector>

/**
 * @file aggregation.hpp
 * @brief Representation for specifying desired aggregations from
 * aggregation-based APIs, e.g., groupby, reductions, rolling, etc.
 *
 * @note Not all aggregation APIs support all aggregation operations. See
 * individual function documentation to see what aggregations are supported.
 *
 */

namespace cudf {
namespace experimental {
/**
 * @brief Base class for abstract representation of an aggregation.
 */
class aggregation;

enum class udf_type : bool {
   CUDA,
   PTX
};

// @brief Enum to describe include nulls or exclude nulls in an aggregation
enum class include_nulls : bool {
   YES, 
   NO
};

/// Factory to create a SUM aggregation
std::unique_ptr<aggregation> make_sum_aggregation();

/// Factory to create a PRODUCT aggregation
std::unique_ptr<aggregation> make_product_aggregation();

/// Factory to create a MIN aggregation
std::unique_ptr<aggregation> make_min_aggregation();

/// Factory to create a MAX aggregation
std::unique_ptr<aggregation> make_max_aggregation();

/// Factory to create a COUNT aggregation
std::unique_ptr<aggregation> make_count_aggregation();

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
std::unique_ptr<aggregation> make_quantile_aggregation(
    std::vector<double> const& q, interpolation i = interpolation::LINEAR);

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
*/
std::unique_ptr<aggregation>
make_nunique_aggregation(include_nulls _include_nulls = include_nulls::NO);
/**
 * @brief Factory to create a aggregation base on UDF for PTX or CUDA
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

}  // namespace experimental
}  // namespace cudf
