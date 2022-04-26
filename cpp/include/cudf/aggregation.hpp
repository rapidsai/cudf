/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
 * @brief Tie-breaker method to use for ranking the column.
 *
 * @see cudf::make_rank_aggregation for more details.
 * @ingroup column_sort
 */
enum class rank_method : int32_t {
  FIRST,    ///< stable sort order ranking (no ties)
  AVERAGE,  ///< mean of first in the group
  MIN,      ///< min of first in the group
  MAX,      ///< max of first in the group
  DENSE     ///< rank always increases by 1 between groups
};

/**
 * @brief Whether returned rank should be percentage or not and
 *  mention the type of percentage normalization.
 *
 */
enum class rank_percentage : int32_t {
  NONE,             ///< rank
  ZERO_NORMALIZED,  ///< rank / count
  ONE_NORMALIZED    ///< (rank - 1) / (count - 1)
};

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
    M2,              ///< sum of squares of differences from the mean
    VARIANCE,        ///< variance
    STD,             ///< standard deviation
    MEDIAN,          ///< median reduction
    QUANTILE,        ///< compute specified quantile(s)
    ARGMAX,          ///< Index of max element
    ARGMIN,          ///< Index of min element
    NUNIQUE,         ///< count number of unique elements
    NTH_ELEMENT,     ///< get the nth element
    ROW_NUMBER,      ///< get row-number of current index (relative to rolling window)
    RANK,            ///< get rank of current index
    COLLECT_LIST,    ///< collect values into a list
    COLLECT_SET,     ///< collect values into a list without duplicate entries
    LEAD,            ///< window function, accesses row at specified offset following current row
    LAG,             ///< window function, accesses row at specified offset preceding current row
    PTX,             ///< PTX  UDF based reduction
    CUDA,            ///< CUDA UDF based reduction
    MERGE_LISTS,     ///< merge multiple lists values into one list
    MERGE_SETS,      ///< merge multiple lists values into one list then drop duplicate entries
    MERGE_M2,        ///< merge partial values of M2 aggregation,
    COVARIANCE,      ///< covariance between two sets of elements
    CORRELATION,     ///< correlation between two sets of elements
    TDIGEST,         ///< create a tdigest from a set of input values
    MERGE_TDIGEST    ///< create a tdigest by merging multiple tdigests together
  };

  aggregation() = delete;
  aggregation(aggregation::Kind a) : kind{a} {}
  Kind kind;  ///< The aggregation to perform
  virtual ~aggregation() = default;

  [[nodiscard]] virtual bool is_equal(aggregation const& other) const { return kind == other.kind; }
  [[nodiscard]] virtual size_t do_hash() const { return std::hash<int>{}(kind); }
  [[nodiscard]] virtual std::unique_ptr<aggregation> clone() const = 0;

  // override functions for compound aggregations
  virtual std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const = 0;
  virtual void finalize(cudf::detail::aggregation_finalizer& finalizer) const         = 0;
};

/**
 * @brief Derived class intended for rolling_window specific aggregation usage.
 *
 * As an example, rolling_window will only accept rolling_aggregation inputs,
 * and the appropriate derived classes (sum_aggregation, mean_aggregation, etc)
 * derive from this interface to represent these valid options.
 */
class rolling_aggregation : public virtual aggregation {
 public:
  ~rolling_aggregation() override = default;

 protected:
  rolling_aggregation() {}
  rolling_aggregation(aggregation::Kind a) : aggregation{a} {}
};

/**
 * @brief Derived class intended for groupby specific aggregation usage.
 */
class groupby_aggregation : public virtual aggregation {
 public:
  ~groupby_aggregation() override = default;

 protected:
  groupby_aggregation() {}
};

/**
 * @brief Derived class intended for groupby specific scan usage.
 */
class groupby_scan_aggregation : public virtual aggregation {
 public:
  ~groupby_scan_aggregation() override = default;

 protected:
  groupby_scan_aggregation() {}
};

/**
 * @brief Derived class intended for reduction usage.
 */
class reduce_aggregation : public virtual aggregation {
 public:
  ~reduce_aggregation() override = default;

 protected:
  reduce_aggregation() {}
};

/**
 * @brief Derived class intended for scan usage.
 */
class scan_aggregation : public virtual aggregation {
 public:
  ~scan_aggregation() override = default;

 protected:
  scan_aggregation() {}
};

/**
 * @brief Derived class intended for segmented reduction usage.
 */
class segmented_reduce_aggregation : public virtual aggregation {
 public:
  ~segmented_reduce_aggregation() override = default;

 protected:
  segmented_reduce_aggregation() {}
};

enum class udf_type : bool { CUDA, PTX };
enum class correlation_type : int32_t { PEARSON, KENDALL, SPEARMAN };

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
 * @brief Factory to create a M2 aggregation
 *
 * A M2 aggregation is sum of squares of differences from the mean. That is:
 *  `M2 = SUM((x - MEAN) * (x - MEAN))`.
 *
 * This aggregation produces the intermediate values that are used to compute variance and standard
 * deviation across multiple discrete sets. See
 * `https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm` for more
 * detail.
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_m2_aggregation();

/**
 * @brief Factory to create a VARIANCE aggregation
 *
 * @param ddof Delta degrees of freedom. The divisor used in calculation of
 *             `variance` is `N - ddof`, where `N` is the population size.
 *
 * @throw cudf::logic_error if input type is chrono or compound types.
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_variance_aggregation(size_type ddof = 1);

/**
 * @brief Factory to create a STD aggregation
 *
 * @param ddof Delta degrees of freedom. The divisor used in calculation of
 *             `std` is `N - ddof`, where `N` is the population size.
 *
 * @throw cudf::logic_error if input type is chrono or compound types.
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
 * @param interp The desired interpolation
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_quantile_aggregation(std::vector<double> const& quantiles,
                                                interpolation interp = interpolation::LINEAR);

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
 * @brief Factory to create a RANK aggregation
 *
 * `RANK` returns a column of size_type or double "ranks" (see note 3 below for how the
 * data type is determined) for a given rank method and column order.
 * If nulls are excluded, the rank will be null for those rows, otherwise a non-nullable column is
 * returned. Double precision column is returned only when percentage!=NONE and when rank method is
 * average.
 *
 * This aggregation only works with "scan" algorithms. The input column into the group or
 * ungrouped scan is an orderby column that orders the rows that the aggregate function ranks.
 * If rows are ordered by more than one column, the orderby input column should be a struct
 * column containing the ordering columns.
 *
 * Note:
 *  1. This method could work faster with the rows that are presorted by the group keys and order_by
 *     columns. Though groupby object does not require order_by column to be sorted, groupby rank
 *     scan aggregation does require the order_by column to be sorted if the keys are sorted.
 *  2. `RANK` aggregations are not compatible with exclusive scans.
 *  3. All rank methods except AVERAGE method and percentage!=NONE returns size_type column.
 *     For AVERAGE method and percentage!=NONE, the return type is double column.
 *
 * @code{.pseudo}
 * Example: Consider a motor-racing statistics dataset, containing the following columns:
 *   1. venue:  (STRING) Location of the race event
 *   2. driver: (STRING) Name of the car driver (abbreviated to 3 characters)
 *   3. time:   (INT32)  Time taken to complete the circuit
 *
 * For the following presorted data:
 *
 *  [ //      venue,           driver,           time
 *    {   "silverstone",  "HAM" ("hamilton"),   15823},
 *    {   "silverstone",  "LEC" ("leclerc"),    15827},
 *    {   "silverstone",  "BOT" ("bottas"),     15834},  // <-- Tied for 3rd place.
 *    {   "silverstone",  "NOR" ("norris"),     15834},  // <-- Tied for 3rd place.
 *    {   "silverstone",  "RIC" ("ricciardo"),  15905},
 *    {      "monza",     "RIC" ("ricciardo"),  12154},
 *    {      "monza",     "NOR" ("norris"),     12156},  // <-- Tied for 2nd place.
 *    {      "monza",     "BOT" ("bottas"),     12156},  // <-- Tied for 2nd place.
 *    {      "monza",     "LEC" ("leclerc"),    12201},
 *    {      "monza",     "PER" ("perez"),      12203}
 *  ]
 *
 * A grouped rank aggregation scan with:
 *   groupby column      : venue
 *   input orderby column: time
 * Produces the following rank column for each methods:
 * first:   {   1,     2,     3,     4,     5,      1,     2,     3,     4,     5}
 * average: {   1,     2,   3.5,   3.5,     5,      1,   2.5,   2.5,     4,     5}
 * min:     {   1,     2,     3,     3,     5,      1,     2,     2,     4,     5}
 * max:     {   1,     2,     4,     4,     5,      1,     3,     3,     4,     5}
 * dense:   {   1,     2,     3,     3,     4,      1,     2,     2,     3,     4}
 * This corresponds to the following grouping and `driver` rows:
 *          { "HAM", "LEC", "BOT", "NOR", "RIC",  "RIC", "NOR", "BOT", "LEC", "PER" }
 *            <----------silverstone----------->|<-------------monza-------------->
 *
 * min rank for each percentage types:
 * NONE:             {   1,      2,     3,     3,     5,      1,     2,     2,     4,     5 }
 * ZERO_NORMALIZED : { 0.16,  0.33,  0.50,  0.50,  0.83,   0.16,  0.33,  0.33,  0.66,  0.83 }
 * ONE_NORMALIZED:   { 0.00,  0.25,  0.50,  0.50,  1.00,   0.00,  0.25,  0.25,  0.75,  1.00 }
 * where count corresponds to the number of rows in the group. @see cudf::rank_percentage
 *
 * @endcode
 *
 * @param method The ranking method used for tie breaking (same values).
 * @param column_order The desired sort order for ranking
 * @param null_handling  flag to include nulls during ranking. If nulls are not included,
 * the corresponding rank will be null.
 * @param null_precedence The desired order of null compared to other elements for column
 * @param percentage enum to denote the type of conversion of ranks to percentage in range (0,1]
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_rank_aggregation(rank_method method,
                                            order column_order         = order::ASCENDING,
                                            null_policy null_handling  = null_policy::EXCLUDE,
                                            null_order null_precedence = null_order::AFTER,
                                            rank_percentage percentage = rank_percentage::NONE);

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
 * @brief Factory to create a COLLECT_SET aggregation
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

/**
 * @brief Factory to create a MERGE_LISTS aggregation.
 *
 * Given a lists column, this aggregation merges all the lists corresponding to the same key value
 * into one list. It is designed specifically to merge the partial results of multiple (distributed)
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
 * This aggregation is designed specifically to merge the partial results of multiple (distributed)
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

/**
 * @brief Factory to create a MERGE_M2 aggregation
 *
 * Merges the results of `M2` aggregations on independent sets into a new `M2` value equivalent to
 * if a single `M2` aggregation was done across all of the sets at once. This aggregation is only
 * valid on structs whose members are the result of the `COUNT_VALID`, `MEAN`, and `M2` aggregations
 * on the same sets. The output of this aggregation is a struct containing the merged `COUNT_VALID`,
 * `MEAN`, and `M2` aggregations.
 *
 * The input `M2` aggregation values are expected to be all non-negative numbers, since they
 * were output from `M2` aggregation.
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_merge_m2_aggregation();

/**
 * @brief Factory to create a COVARIANCE aggregation
 *
 * Compute covariance between two columns.
 * The input columns are child columns of a non-nullable struct columns.
 * @param min_periods Minimum number of non-null observations required to produce a result.
 * @param ddof Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N is
 *        the number of non-null observations.
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_covariance_aggregation(size_type min_periods = 1, size_type ddof = 1);

/**
 * @brief Factory to create a CORRELATION aggregation
 *
 * Compute correlation coefficient between two columns.
 * The input columns are child columns of a non-nullable struct columns.
 *
 * @param type correlation_type
 * @param min_periods Minimum number of non-null observations required to produce a result.
 */
template <typename Base = aggregation>
std::unique_ptr<Base> make_correlation_aggregation(correlation_type type,
                                                   size_type min_periods = 1);

/**
 * @brief Factory to create a TDIGEST aggregation
 *
 * Produces a tdigest (https://arxiv.org/pdf/1902.04023.pdf) column from input values.
 * The input aggregation values are expected to be fixed-width numeric types.
 *
 * The tdigest column produced is of the following structure:
 *
 * struct {
 *   // centroids for the digest
 *   list {
 *    struct {
 *      double    // mean
 *      double    // weight
 *    },
 *    ...
 *   }
 *   // these are from the input stream, not the centroids. they are used
 *   // during the percentile_approx computation near the beginning or
 *   // end of the quantiles
 *   double       // min
 *   double       // max
 * }
 *
 * Each output row is a single tdigest.  The length of the row is the "size" of the
 * tdigest, each element of which represents a weighted centroid (mean, weight).
 *
 * @param max_centroids Parameter controlling compression level and accuracy on subsequent
 * queries on the output tdigest data.  `max_centroids` places an upper bound on the size of
 * the computed tdigests: A value of 1000 will result in a tdigest containing no
 * more than 1000 centroids (32 bytes each). Higher result in more accurate tdigest information.
 *
 * @returns A TDIGEST aggregation object.
 */
template <typename Base>
std::unique_ptr<Base> make_tdigest_aggregation(int max_centroids = 1000);

/**
 * @brief Factory to create a MERGE_TDIGEST aggregation
 *
 * Merges the results from a previous aggregation resulting from a `make_tdigest_aggregation`
 * or `make_merge_tdigest_aggregation` to produce a new a tdigest
 * (https://arxiv.org/pdf/1902.04023.pdf) column.
 *
 * The tdigest column produced is of the following structure:
 *
 * struct {
 *   // centroids for the digest
 *   list {
 *    struct {
 *      double    // mean
 *      double    // weight
 *    },
 *    ...
 *   }
 *   // these are from the input stream, not the centroids. they are used
 *   // during the percentile_approx computation near the beginning or
 *   // end of the quantiles
 *   double       // min
 *   double       // max
 * }
 *
 * Each output row is a single tdigest.  The length of the row is the "size" of the
 * tdigest, each element of which represents a weighted centroid (mean, weight).
 *
 * @param max_centroids Parameter controlling compression level and accuracy on subsequent
 * queries on the output tdigest data.  `max_centroids` places an upper bound on the size of
 * the computed tdigests: A value of 1000 will result in a tdigest containing no
 * more than 1000 centroids (32 bytes each). Higher result in more accurate tdigest information.
 *
 * @returns A MERGE_TDIGEST aggregation object.
 */
template <typename Base>
std::unique_ptr<Base> make_merge_tdigest_aggregation(int max_centroids = 1000);

/** @} */  // end of group
}  // namespace cudf
