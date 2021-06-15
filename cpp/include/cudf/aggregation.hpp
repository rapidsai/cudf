/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

// forward declaration
namespace detail {
class simple_aggregations_collector;
class aggregation_finalizer;
}  // namespace detail
/**
 * @brief Abstract base class for specifying the desired aggregation in an
 * `aggregation_request`.
 *
 * All aggregations must derive from this class to implement the pure virtual
 * functions and potentially encapsulate additional information needed to
 * compute the aggregation.
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
    ROW_NUMBER,      ///< get row-number of current index (relative to rolling window)
    COLLECT_LIST,    ///< collect values into a list
    COLLECT_SET,     ///< collect values into a list without duplicate entries
    MERGE_LISTS,     ///< merge multiple lists values into one list
    MERGE_SETS,      ///< merge multiple lists values into one list then drop duplicate entries
    LEAD,            ///< window function, accesses row at specified offset following current row
    LAG,             ///< window function, accesses row at specified offset preceding current row
    PTX,             ///< PTX  UDF based reduction
    CUDA             ///< CUDA UDF based reduction
  };

  aggregation() = delete;
  aggregation(aggregation::Kind a) : kind{a} {}
  Kind kind;  ///< The aggregation to perform
  virtual ~aggregation() = default;

  virtual bool is_equal(aggregation const& other) const { return kind == other.kind; }
  virtual size_t do_hash() const { return std::hash<int>{}(kind); }
  virtual std::unique_ptr<aggregation> clone() const = 0;

  // override functions for compound aggregations
  virtual std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const = 0;
  virtual void finalize(cudf::detail::aggregation_finalizer& finalizer) const         = 0;
};

/**
 * @brief Derived class intended for enforcing operation-specific restrictions
 * when calling various cudf functions.
 *
 * As an example, rolling_window will only accept rolling_aggregation inputs,
 * and the appropriate derived classes (sum_aggregation, mean_aggregation, etc)
 * derive from this interface to represent these valid options.
 */
class rolling_aggregation : public virtual aggregation {
 public:
  ~rolling_aggregation() = default;

 protected:
  rolling_aggregation() {}
};

enum class udf_type : bool { CUDA, PTX };

/// Factory to create a SUM aggregation
template <typename Base = aggregation>
std::unique_ptr<Base> make_sum_aggregation();

/// Factory to create a PRODUCT aggregation
template <typename Base = aggregation>
std::unique_ptr<Base> make_product_aggregation();

/// Factory to create a MIN aggregation
template <typename Base = aggregation>
std::unique_ptr<Base> make_min_aggregation();

/// Factory to create a MAX aggregation
template <typename Base = aggregation>
std::unique_ptr<Base> make_max_aggregation();

/**
 * @brief Factory to create a COUNT aggregation
 *
 * @param null_handling Indicates if null values will be counted.
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_count_aggregation(null_policy null_handling = null_policy::EXCLUDE);

/// Factory to create an ANY aggregation
template <typename Base = aggregation>
std::unique_ptr<Base> make_any_aggregation();

/// Factory to create a ALL aggregation
template <typename Base = aggregation>
std::unique_ptr<Base> make_all_aggregation();

/// Factory to create a SUM_OF_SQUARES aggregation
template <typename Base = aggregation>
std::unique_ptr<Base> make_sum_of_squares_aggregation();

/// Factory to create a MEAN aggregation
template <typename Base = aggregation>
std::unique_ptr<Base> make_mean_aggregation();

/**
 * @brief Factory to create a VARIANCE aggregation
 *
 * @param ddof Delta degrees of freedom. The divisor used in calculation of
 *             `variance` is `N - ddof`, where `N` is the population size.
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_variance_aggregation(size_type ddof = 1);

/**
 * @brief Factory to create a STD aggregation
 *
 * @param ddof Delta degrees of freedom. The divisor used in calculation of
 *             `std` is `N - ddof`, where `N` is the population size.
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_std_aggregation(size_type ddof = 1);

/// Factory to create a MEDIAN aggregation
template <typename Base = aggregation>
std::unique_ptr<Base> make_median_aggregation();

/**
 * @brief Factory to create a QUANTILE aggregation
 *
 * @param quantiles The desired quantiles
 * @param interpolation The desired interpolation
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_quantile_aggregation(std::vector<double> const& q,
                                                interpolation i = interpolation::LINEAR);

/**
 * @brief Factory to create an `argmax` aggregation
 *
 * `argmax` returns the index of the maximum element.
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_argmax_aggregation();

/**
 * @brief Factory to create an `argmin` aggregation
 *
 * `argmin` returns the index of the minimum element.
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_argmin_aggregation();

/**
 * @brief Factory to create a `nunique` aggregation
 *
 * `nunique` returns the number of unique elements.
 * @param null_handling Indicates if null values will be counted.
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_nunique_aggregation(null_policy null_handling = null_policy::EXCLUDE);

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
template <typename Base = aggregation>
std::unique_ptr<Base> make_nth_element_aggregation(
  size_type n, null_policy null_handling = null_policy::INCLUDE);

/// Factory to create a ROW_NUMBER aggregation
template <typename Base = aggregation>
std::unique_ptr<Base> make_row_number_aggregation();

/**
 * @brief Factory to create a COLLECT_LIST aggregation
 *
 * `COLLECT_LIST` returns a list column of all included elements in the group/series.
 *
 * If `null_handling` is set to `EXCLUDE`, null elements are dropped from each
 * of the list rows.
 *
 * @param null_handling Indicates whether to include/exclude nulls in list elements.
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_collect_list_aggregation(
  null_policy null_handling = null_policy::INCLUDE);

/**
 * @brief Factory to create a COLLECT_SET aggregation.
 *
 * `COLLECT_SET` returns a lists column of all included elements in the group/series. Within each
 * list, the duplicated entries are dropped out such that each entry appears only once.
 *
 * If `null_handling` is set to `EXCLUDE`, null elements are dropped from each
 * of the list rows.
 *
 * @param null_handling Indicates whether to include/exclude nulls during collection
 * @param nulls_equal Flag to specify whether null entries within each list should be considered
 *        equal.
 * @param nans_equal Flag to specify whether NaN values in floating point column should be
 *        considered equal.
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_collect_set_aggregation(null_policy null_handling = null_policy::INCLUDE,
                                                   null_equality nulls_equal = null_equality::EQUAL,
                                                   nan_equality nans_equal = nan_equality::UNEQUAL);

/**
 * @brief Factory to create a MERGE_LISTS aggregation.
 *
 * Given a lists column, this aggregation merges all the lists corresponding to the same key value
 * into one list. It is designed specificly to merge the partial results of multiple (distributed)
 * groupby `COLLECT_LIST` aggregations into a final `COLLECT_LIST` result. As such, it requires the
 * input lists column to be non-nullable (the child column containing list entries is not subjected
 * to this requirement).
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_merge_lists_aggregation();

/**
 * @brief Factory to create a MERGE_SETS aggregation.
 *
 * Given a lists column, this aggregation firstly merges all the lists corresponding to the same key
 * value into one list, then it drops all the duplicate entries in each lists, producing a lists
 * column containing non-repeated entries.
 *
 * This aggregation is designed specificly to merge the partial results of multiple (distributed)
 * groupby `COLLECT_LIST` or `COLLECT_SET` aggregations into a final `COLLECT_SET` result. As such,
 * it requires the input lists column to be non-nullable (the child column containing list entries
 * is not subjected to this requirement).
 *
 * In practice, the input (partial results) to this aggregation should be generated by (distributed)
 * `COLLECT_LIST` aggregations, not `COLLECT_SET`, to avoid unnecessarily removing duplicate entries
 * for the partial results.
 *
 * @param nulls_equal Flag to specify whether nulls within each list should be considered equal
 *        during dropping duplicate list entries.
 * @param nans_equal Flag to specify whether NaN values in floating point column should be
 *        considered equal during dropping duplicate list entries.
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_merge_sets_aggregation(null_equality nulls_equal = null_equality::EQUAL,
                                                  nan_equality nans_equal = nan_equality::UNEQUAL);

/// Factory to create a LAG aggregation
template <typename Base = aggregation>
std::unique_ptr<Base> make_lag_aggregation(size_type offset);

/// Factory to create a LEAD aggregation
template <typename Base = aggregation>
std::unique_ptr<Base> make_lead_aggregation(size_type offset);

/**
 * @brief Factory to create an aggregation base on UDF for PTX or CUDA
 *
 * @param[in] type: either udf_type::PTX or udf_type::CUDA
 * @param[in] user_defined_aggregator A string containing the aggregator code
 * @param[in] output_type expected output type
 *
 * @return aggregation unique pointer housing user_defined_aggregator string.
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_udf_aggregation(udf_type type,
                                           std::string const& user_defined_aggregator,
                                           data_type output_type);

/** @} */  // end of group
}  // namespace cudf
