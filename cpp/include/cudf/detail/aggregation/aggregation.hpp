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

#include <cudf/aggregation.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <functional>
#include <numeric>

namespace cudf {
namespace detail {

// Visitor pattern
class simple_aggregations_collector {  // Declares the interface for the simple aggregations
                                       // collector
 public:
  // Declare overloads for each kind of a agg to dispatch
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class sum_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class product_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class min_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class max_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class count_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class any_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class all_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(
    data_type col_type, class sum_of_squares_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class mean_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class m2_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class var_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class std_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class median_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class quantile_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class argmax_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class argmin_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class nunique_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class nth_element_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class row_number_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(
    data_type col_type, class collect_list_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class collect_set_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class lead_lag_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class udf_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class merge_lists_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class merge_sets_aggregation const& agg);
  virtual std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                          class merge_m2_aggregation const& agg);
};

class aggregation_finalizer {  // Declares the interface for the finalizer
 public:
  // Declare overloads for each kind of a agg to dispatch
  virtual void visit(aggregation const& agg);
  virtual void visit(class sum_aggregation const& agg);
  virtual void visit(class product_aggregation const& agg);
  virtual void visit(class min_aggregation const& agg);
  virtual void visit(class max_aggregation const& agg);
  virtual void visit(class count_aggregation const& agg);
  virtual void visit(class any_aggregation const& agg);
  virtual void visit(class all_aggregation const& agg);
  virtual void visit(class sum_of_squares_aggregation const& agg);
  virtual void visit(class mean_aggregation const& agg);
  virtual void visit(class m2_aggregation const& agg);
  virtual void visit(class var_aggregation const& agg);
  virtual void visit(class std_aggregation const& agg);
  virtual void visit(class median_aggregation const& agg);
  virtual void visit(class quantile_aggregation const& agg);
  virtual void visit(class argmax_aggregation const& agg);
  virtual void visit(class argmin_aggregation const& agg);
  virtual void visit(class nunique_aggregation const& agg);
  virtual void visit(class nth_element_aggregation const& agg);
  virtual void visit(class row_number_aggregation const& agg);
  virtual void visit(class collect_list_aggregation const& agg);
  virtual void visit(class collect_set_aggregation const& agg);
  virtual void visit(class lead_lag_aggregation const& agg);
  virtual void visit(class udf_aggregation const& agg);
  virtual void visit(class merge_lists_aggregation const& agg);
  virtual void visit(class merge_sets_aggregation const& agg);
  virtual void visit(class merge_m2_aggregation const& agg);
};

/**
 * @brief Derived class for specifying a sum aggregation
 */
class sum_aggregation final : public rolling_aggregation {
 public:
  sum_aggregation() : aggregation(SUM) {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<sum_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived class for specifying a product aggregation
 */
class product_aggregation final : public aggregation {
 public:
  product_aggregation() : aggregation(PRODUCT) {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<product_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived class for specifying a min aggregation
 */
class min_aggregation final : public rolling_aggregation {
 public:
  min_aggregation() : aggregation(MIN) {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<min_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived class for specifying a max aggregation
 */
class max_aggregation final : public rolling_aggregation {
 public:
  max_aggregation() : aggregation(MAX) {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<max_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived class for specifying a count aggregation
 */
class count_aggregation final : public rolling_aggregation {
 public:
  count_aggregation(aggregation::Kind kind) : aggregation(kind) {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<count_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived class for specifying an any aggregation
 */
class any_aggregation final : public aggregation {
 public:
  any_aggregation() : aggregation(ANY) {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<any_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived class for specifying an all aggregation
 */
class all_aggregation final : public aggregation {
 public:
  all_aggregation() : aggregation(ALL) {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<all_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived class for specifying a sum_of_squares aggregation
 */
class sum_of_squares_aggregation final : public aggregation {
 public:
  sum_of_squares_aggregation() : aggregation(SUM_OF_SQUARES) {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<sum_of_squares_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived class for specifying a mean aggregation
 */
class mean_aggregation final : public rolling_aggregation {
 public:
  mean_aggregation() : aggregation(MEAN) {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<mean_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived class for specifying a m2 aggregation
 */
class m2_aggregation : public aggregation {
 public:
  m2_aggregation() : aggregation{M2} {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<m2_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived class for specifying a standard deviation/variance aggregation
 */
class std_var_aggregation : public aggregation {
 public:
  size_type _ddof;  ///< Delta degrees of freedom

  bool is_equal(aggregation const& _other) const override
  {
    if (!this->aggregation::is_equal(_other)) { return false; }
    auto const& other = dynamic_cast<std_var_aggregation const&>(_other);
    return _ddof == other._ddof;
  }

  size_t do_hash() const override { return this->aggregation::do_hash() ^ hash_impl(); }

 protected:
  std_var_aggregation(aggregation::Kind k, size_type ddof) : aggregation(k), _ddof{ddof}
  {
    CUDF_EXPECTS(k == aggregation::STD or k == aggregation::VARIANCE,
                 "std_var_aggregation can accept only STD, VARIANCE");
  }

  size_type hash_impl() const { return std::hash<size_type>{}(_ddof); }
};

/**
 * @brief Derived class for specifying a variance aggregation
 */
class var_aggregation final : public std_var_aggregation {
 public:
  var_aggregation(size_type ddof) : std_var_aggregation{aggregation::VARIANCE, ddof} {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<var_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived class for specifying a standard deviation aggregation
 */
class std_aggregation final : public std_var_aggregation {
 public:
  std_aggregation(size_type ddof) : std_var_aggregation{aggregation::STD, ddof} {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<std_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived class for specifying a median aggregation
 */
class median_aggregation final : public aggregation {
 public:
  median_aggregation() : aggregation(MEDIAN) {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<median_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived class for specifying a quantile aggregation
 */
class quantile_aggregation final : public aggregation {
 public:
  quantile_aggregation(std::vector<double> const& q, interpolation i)
    : aggregation{QUANTILE}, _quantiles{q}, _interpolation{i}
  {
  }
  std::vector<double> _quantiles;  ///< Desired quantile(s)
  interpolation _interpolation;    ///< Desired interpolation

  bool is_equal(aggregation const& _other) const override
  {
    if (!this->aggregation::is_equal(_other)) { return false; }

    auto const& other = dynamic_cast<quantile_aggregation const&>(_other);

    return _interpolation == other._interpolation &&
           std::equal(_quantiles.begin(), _quantiles.end(), other._quantiles.begin());
  }

  size_t do_hash() const override { return this->aggregation::do_hash() ^ hash_impl(); }

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<quantile_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }

 private:
  size_t hash_impl() const
  {
    return std::hash<int>{}(static_cast<int>(_interpolation)) ^
           std::accumulate(
             _quantiles.cbegin(), _quantiles.cend(), size_t{0}, [](size_t a, double b) {
               return a ^ std::hash<double>{}(b);
             });
  }
};

/**
 * @brief Derived class for specifying an argmax aggregation
 */
class argmax_aggregation final : public rolling_aggregation {
 public:
  argmax_aggregation() : aggregation(ARGMAX) {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<argmax_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived class for specifying an argmin aggregation
 */
class argmin_aggregation final : public rolling_aggregation {
 public:
  argmin_aggregation() : aggregation(ARGMIN) {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<argmin_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived class for specifying a nunique aggregation
 */
class nunique_aggregation final : public aggregation {
 public:
  nunique_aggregation(null_policy null_handling)
    : aggregation{NUNIQUE}, _null_handling{null_handling}
  {
  }

  null_policy _null_handling;  ///< include or exclude nulls

  bool is_equal(aggregation const& _other) const override
  {
    if (!this->aggregation::is_equal(_other)) { return false; }
    auto const& other = dynamic_cast<nunique_aggregation const&>(_other);
    return _null_handling == other._null_handling;
  }

  size_t do_hash() const override { return this->aggregation::do_hash() ^ hash_impl(); }

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<nunique_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }

 private:
  size_t hash_impl() const { return std::hash<int>{}(static_cast<int>(_null_handling)); }
};

/**
 * @brief Derived class for specifying a nth element aggregation
 */
class nth_element_aggregation final : public aggregation {
 public:
  nth_element_aggregation(size_type n, null_policy null_handling)
    : aggregation{NTH_ELEMENT}, _n{n}, _null_handling{null_handling}
  {
  }

  size_type _n;                ///< nth index to return
  null_policy _null_handling;  ///< include or exclude nulls

  bool is_equal(aggregation const& _other) const override
  {
    if (!this->aggregation::is_equal(_other)) { return false; }
    auto const& other = dynamic_cast<nth_element_aggregation const&>(_other);
    return _n == other._n and _null_handling == other._null_handling;
  }

  size_t do_hash() const override { return this->aggregation::do_hash() ^ hash_impl(); }

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<nth_element_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }

 private:
  size_t hash_impl() const
  {
    return std::hash<size_type>{}(_n) ^ std::hash<int>{}(static_cast<int>(_null_handling));
  }
};

/**
 * @brief Derived class for specifying a row_number aggregation
 */
class row_number_aggregation final : public rolling_aggregation {
 public:
  row_number_aggregation() : aggregation(ROW_NUMBER) {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<row_number_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived aggregation class for specifying COLLECT_LIST aggregation
 */
class collect_list_aggregation final : public rolling_aggregation {
 public:
  explicit collect_list_aggregation(null_policy null_handling = null_policy::INCLUDE)
    : aggregation{COLLECT_LIST}, _null_handling{null_handling}
  {
  }

  null_policy _null_handling;  ///< include or exclude nulls

  bool is_equal(aggregation const& _other) const override
  {
    if (!this->aggregation::is_equal(_other)) { return false; }
    auto const& other = dynamic_cast<collect_list_aggregation const&>(_other);
    return (_null_handling == other._null_handling);
  }

  size_t do_hash() const override { return this->aggregation::do_hash() ^ hash_impl(); }

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<collect_list_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }

 private:
  size_t hash_impl() const { return std::hash<int>{}(static_cast<int>(_null_handling)); }
};

/**
 * @brief Derived aggregation class for specifying COLLECT_SET aggregation
 */
class collect_set_aggregation final : public rolling_aggregation {
 public:
  explicit collect_set_aggregation(null_policy null_handling = null_policy::INCLUDE,
                                   null_equality nulls_equal = null_equality::EQUAL,
                                   nan_equality nans_equal   = nan_equality::UNEQUAL)
    : aggregation{COLLECT_SET},
      _null_handling{null_handling},
      _nulls_equal(nulls_equal),
      _nans_equal(nans_equal)
  {
  }

  null_policy _null_handling;  ///< include or exclude nulls
  null_equality _nulls_equal;  ///< whether to consider nulls as equal values
  nan_equality _nans_equal;    ///< whether to consider NaNs as equal value (applicable only to
                               ///< floating point types)

  bool is_equal(aggregation const& _other) const override
  {
    if (!this->aggregation::is_equal(_other)) { return false; }
    auto const& other = dynamic_cast<collect_set_aggregation const&>(_other);
    return (_null_handling == other._null_handling && _nulls_equal == other._nulls_equal &&
            _nans_equal == other._nans_equal);
  }

  size_t do_hash() const override { return this->aggregation::do_hash() ^ hash_impl(); }

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<collect_set_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }

 protected:
  size_t hash_impl() const
  {
    return std::hash<int>{}(static_cast<int>(_null_handling) ^ static_cast<int>(_nulls_equal) ^
                            static_cast<int>(_nans_equal));
  }
};

/**
 * @brief Derived aggregation class for specifying LEAD/LAG window aggregations
 */
class lead_lag_aggregation final : public rolling_aggregation {
 public:
  lead_lag_aggregation(Kind kind, size_type offset)
    : aggregation{offset < 0 ? (kind == LAG ? LEAD : LAG) : kind}, row_offset{std::abs(offset)}
  {
  }

  bool is_equal(aggregation const& _other) const override
  {
    if (!this->aggregation::is_equal(_other)) { return false; }
    auto const& other = dynamic_cast<lead_lag_aggregation const&>(_other);
    return (row_offset == other.row_offset);
  }

  size_t do_hash() const override { return this->aggregation::do_hash() ^ hash_impl(); }

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<lead_lag_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }

  size_type row_offset;

 private:
  size_t hash_impl() const { return std::hash<size_type>()(row_offset); }
};

/**
 * @brief Derived class for specifying a custom aggregation
 * specified in udf
 */
class udf_aggregation final : public rolling_aggregation {
 public:
  udf_aggregation(aggregation::Kind type,
                  std::string const& user_defined_aggregator,
                  data_type output_type)
    : aggregation{type},
      _source{user_defined_aggregator},
      _operator_name{(type == aggregation::PTX) ? "rolling_udf_ptx" : "rolling_udf_cuda"},
      _function_name{"rolling_udf"},
      _output_type{output_type}
  {
    CUDF_EXPECTS(type == aggregation::PTX or type == aggregation::CUDA,
                 "udf_aggregation can accept only PTX, CUDA");
  }

  bool is_equal(aggregation const& _other) const override
  {
    if (!this->aggregation::is_equal(_other)) { return false; }
    auto const& other = dynamic_cast<udf_aggregation const&>(_other);
    return (_source == other._source and _operator_name == other._operator_name and
            _function_name == other._function_name and _output_type == other._output_type);
  }

  size_t do_hash() const override { return this->aggregation::do_hash() ^ hash_impl(); }

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<udf_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }

  std::string const _source;
  std::string const _operator_name;
  std::string const _function_name;
  data_type _output_type;

 protected:
  size_t hash_impl() const
  {
    return std::hash<std::string>{}(_source) ^ std::hash<std::string>{}(_operator_name) ^
           std::hash<std::string>{}(_function_name) ^
           std::hash<int>{}(static_cast<int32_t>(_output_type.id()));
  }
};

/**
 * @brief Derived aggregation class for specifying MERGE_LISTS aggregation
 */
class merge_lists_aggregation final : public aggregation {
 public:
  explicit merge_lists_aggregation() : aggregation{MERGE_LISTS} {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<merge_lists_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Derived aggregation class for specifying MERGE_SETS aggregation
 */
class merge_sets_aggregation final : public aggregation {
 public:
  explicit merge_sets_aggregation(null_equality nulls_equal, nan_equality nans_equal)
    : aggregation{MERGE_SETS}, _nulls_equal(nulls_equal), _nans_equal(nans_equal)
  {
  }

  null_equality _nulls_equal;  ///< whether to consider nulls as equal value
  nan_equality _nans_equal;    ///< whether to consider NaNs as equal value (applicable only to
                               ///< floating point types)

  bool is_equal(aggregation const& _other) const override
  {
    if (!this->aggregation::is_equal(_other)) { return false; }
    auto const& other = dynamic_cast<merge_sets_aggregation const&>(_other);
    return (_nulls_equal == other._nulls_equal && _nans_equal == other._nans_equal);
  }

  size_t do_hash() const override { return this->aggregation::do_hash() ^ hash_impl(); }

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<merge_sets_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }

 protected:
  size_t hash_impl() const
  {
    return std::hash<int>{}(static_cast<int>(_nulls_equal) ^ static_cast<int>(_nans_equal));
  }
};

/**
 * @brief Derived aggregation class for specifying MERGE_M2 aggregation
 */
class merge_m2_aggregation final : public aggregation {
 public:
  explicit merge_m2_aggregation() : aggregation{MERGE_M2} {}

  std::unique_ptr<aggregation> clone() const override
  {
    return std::make_unique<merge_m2_aggregation>(*this);
  }
  std::vector<std::unique_ptr<aggregation>> get_simple_aggregations(
    data_type col_type, cudf::detail::simple_aggregations_collector& collector) const override
  {
    return collector.visit(col_type, *this);
  }
  void finalize(aggregation_finalizer& finalizer) const override { finalizer.visit(*this); }
};

/**
 * @brief Sentinel value used for `ARGMAX` aggregation.
 *
 * The output column for an `ARGMAX` aggregation is initialized with the
 * sentinel value to indicate an unused element.
 */
constexpr size_type ARGMAX_SENTINEL{-1};

/**
 * @brief Sentinel value used for `ARGMIN` aggregation.
 *
 * The output column for an `ARGMIN` aggregation is initialized with the
 * sentinel value to indicate an unused element.
 */
constexpr size_type ARGMIN_SENTINEL{-1};

/**
 * @brief Determines accumulator type based on input type and aggregation.
 *
 * @tparam Source The type on which the aggregation is computed
 * @tparam k The aggregation performed
 */
template <typename Source, aggregation::Kind k, typename Enable = void>
struct target_type_impl {
  using type = void;
};

// Computing MIN of Source, use Source accumulator
template <typename Source>
struct target_type_impl<Source, aggregation::MIN> {
  using type = Source;
};

// Computing MAX of Source, use Source accumulator
template <typename Source>
struct target_type_impl<Source, aggregation::MAX> {
  using type = Source;
};

// Always use size_type accumulator for COUNT_VALID
template <typename Source>
struct target_type_impl<Source, aggregation::COUNT_VALID> {
  using type = cudf::size_type;
};

// Always use size_type accumulator for COUNT_ALL
template <typename Source>
struct target_type_impl<Source, aggregation::COUNT_ALL> {
  using type = cudf::size_type;
};

// Computing ANY of any type, use bool accumulator
template <typename Source>
struct target_type_impl<Source, aggregation::ANY> {
  using type = bool;
};

// Computing ALL of any type, use bool accumulator
template <typename Source>
struct target_type_impl<Source, aggregation::ALL> {
  using type = bool;
};

// Always use `double` for MEAN
// Except for chrono types where result is chrono. (Use FloorDiv)
// TODO: MEAN should be only be enabled for duration types - not for timestamps
template <typename Source, aggregation::Kind k>
struct target_type_impl<
  Source,
  k,
  std::enable_if_t<is_fixed_width<Source>() && !is_chrono<Source>() && (k == aggregation::MEAN)>> {
  using type = double;
};

template <typename Source, aggregation::Kind k>
struct target_type_impl<Source,
                        k,
                        std::enable_if_t<is_chrono<Source>() && (k == aggregation::MEAN)>> {
  using type = Source;
};

constexpr bool is_sum_product_agg(aggregation::Kind k)
{
  return (k == aggregation::SUM) || (k == aggregation::PRODUCT) ||
         (k == aggregation::SUM_OF_SQUARES);
}

// Summing/Multiplying integers of any type, always use int64_t accumulator
template <typename Source, aggregation::Kind k>
struct target_type_impl<
  Source,
  k,
  std::enable_if_t<std::is_integral<Source>::value && is_sum_product_agg(k)>> {
  using type = int64_t;
};

// Summing fixed_point numbers, always use the decimal64 accumulator
template <typename Source, aggregation::Kind k>
struct target_type_impl<
  Source,
  k,
  std::enable_if_t<cudf::is_fixed_point<Source>() && (k == aggregation::SUM)>> {
  using type = numeric::decimal64;
};

// Summing/Multiplying float/doubles, use same type accumulator
template <typename Source, aggregation::Kind k>
struct target_type_impl<
  Source,
  k,
  std::enable_if_t<std::is_floating_point<Source>::value && is_sum_product_agg(k)>> {
  using type = Source;
};

// Summing/Multiplying chrono types, use same type accumulator
// TODO: Sum/Product should only be enabled for duration types - not for timestamps
template <typename Source, aggregation::Kind k>
struct target_type_impl<Source, k, std::enable_if_t<is_chrono<Source>() && is_sum_product_agg(k)>> {
  using type = Source;
};

// Always use `double` for M2
template <typename SourceType>
struct target_type_impl<SourceType, aggregation::M2> {
  using type = double;
};

// Always use `double` for VARIANCE
template <typename SourceType>
struct target_type_impl<SourceType, aggregation::VARIANCE> {
  using type = double;
};

// Always use `double` for STD
template <typename SourceType>
struct target_type_impl<SourceType, aggregation::STD> {
  using type = double;
};

// Always use `double` for quantile
template <typename Source>
struct target_type_impl<Source, aggregation::QUANTILE> {
  using type = double;
};

// MEDIAN is a special case of a QUANTILE
template <typename Source>
struct target_type_impl<Source, aggregation::MEDIAN> {
  using type = typename target_type_impl<Source, aggregation::QUANTILE>::type;
};

// Always use `size_type` for ARGMAX index
template <typename Source>
struct target_type_impl<Source, aggregation::ARGMAX> {
  using type = size_type;
};

// Always use `size_type` for ARGMIN index
template <typename Source>
struct target_type_impl<Source, aggregation::ARGMIN> {
  using type = size_type;
};

// Always use size_type accumulator for NUNIQUE
template <typename Source>
struct target_type_impl<Source, aggregation::NUNIQUE> {
  using type = cudf::size_type;
};

// Always use Source for NTH_ELEMENT
template <typename Source>
struct target_type_impl<Source, aggregation::NTH_ELEMENT> {
  using type = Source;
};

// Always use size_type accumulator for ROW_NUMBER
template <typename Source>
struct target_type_impl<Source, aggregation::ROW_NUMBER> {
  using type = cudf::size_type;
};

// Always use list for COLLECT_LIST
template <typename Source>
struct target_type_impl<Source, aggregation::COLLECT_LIST> {
  using type = cudf::list_view;
};

// Always use list for COLLECT_SET
template <typename Source>
struct target_type_impl<Source, aggregation::COLLECT_SET> {
  using type = cudf::list_view;
};

// Always use Source for LEAD
template <typename Source>
struct target_type_impl<Source, aggregation::LEAD> {
  using type = Source;
};

// Always use Source for LAG
template <typename Source>
struct target_type_impl<Source, aggregation::LAG> {
  using type = Source;
};

// Always use list for MERGE_LISTS
template <typename Source>
struct target_type_impl<Source, aggregation::MERGE_LISTS> {
  using type = cudf::list_view;
};

// Always use list for MERGE_SETS
template <typename Source>
struct target_type_impl<Source, aggregation::MERGE_SETS> {
  using type = cudf::list_view;
};

// Always use struct for MERGE_M2
template <typename SourceType>
struct target_type_impl<SourceType, aggregation::MERGE_M2> {
  using type = cudf::struct_view;
};

/**
 * @brief Helper alias to get the accumulator type for performing aggregation
 * `k` on elements of type `Source`
 *
 * @tparam Source The type on which the aggregation is computed
 * @tparam k The aggregation performed
 */
template <typename Source, aggregation::Kind k>
using target_type_t = typename target_type_impl<Source, k>::type;

template <aggregation::Kind k>
struct kind_to_type_impl {
  using type = aggregation;
};

template <aggregation::Kind k>
using kind_to_type = typename kind_to_type_impl<k>::type;

#ifndef AGG_KIND_MAPPING
#define AGG_KIND_MAPPING(k, Type) \
  template <>                     \
  struct kind_to_type_impl<k> {   \
    using type = Type;            \
  }
#endif

AGG_KIND_MAPPING(aggregation::QUANTILE, quantile_aggregation);
AGG_KIND_MAPPING(aggregation::STD, std_aggregation);
AGG_KIND_MAPPING(aggregation::VARIANCE, var_aggregation);

/**
 * @brief Dispatches `k` as a non-type template parameter to a callable,  `f`.
 *
 * @tparam F Type of callable
 * @param k The `aggregation::Kind` value to dispatch
 * aram f The callable that accepts an `aggregation::Kind` non-type template
 * argument.
 * @param args Parameter pack forwarded to the `operator()` invocation
 * @return Forwards the return value of the callable.
 */
#pragma nv_exec_check_disable
template <typename F, typename... Ts>
CUDA_HOST_DEVICE_CALLABLE decltype(auto) aggregation_dispatcher(aggregation::Kind k,
                                                                F&& f,
                                                                Ts&&... args)
{
  switch (k) {
    case aggregation::SUM:
      return f.template operator()<aggregation::SUM>(std::forward<Ts>(args)...);
    case aggregation::PRODUCT:
      return f.template operator()<aggregation::PRODUCT>(std::forward<Ts>(args)...);
    case aggregation::MIN:
      return f.template operator()<aggregation::MIN>(std::forward<Ts>(args)...);
    case aggregation::MAX:
      return f.template operator()<aggregation::MAX>(std::forward<Ts>(args)...);
    case aggregation::COUNT_VALID:
      return f.template operator()<aggregation::COUNT_VALID>(std::forward<Ts>(args)...);
    case aggregation::COUNT_ALL:
      return f.template operator()<aggregation::COUNT_ALL>(std::forward<Ts>(args)...);
    case aggregation::ANY:
      return f.template operator()<aggregation::ANY>(std::forward<Ts>(args)...);
    case aggregation::ALL:
      return f.template operator()<aggregation::ALL>(std::forward<Ts>(args)...);
    case aggregation::SUM_OF_SQUARES:
      return f.template operator()<aggregation::SUM_OF_SQUARES>(std::forward<Ts>(args)...);
    case aggregation::MEAN:
      return f.template operator()<aggregation::MEAN>(std::forward<Ts>(args)...);
    case aggregation::M2: return f.template operator()<aggregation::M2>(std::forward<Ts>(args)...);
    case aggregation::VARIANCE:
      return f.template operator()<aggregation::VARIANCE>(std::forward<Ts>(args)...);
    case aggregation::STD:
      return f.template operator()<aggregation::STD>(std::forward<Ts>(args)...);
    case aggregation::MEDIAN:
      return f.template operator()<aggregation::MEDIAN>(std::forward<Ts>(args)...);
    case aggregation::QUANTILE:
      return f.template operator()<aggregation::QUANTILE>(std::forward<Ts>(args)...);
    case aggregation::ARGMAX:
      return f.template operator()<aggregation::ARGMAX>(std::forward<Ts>(args)...);
    case aggregation::ARGMIN:
      return f.template operator()<aggregation::ARGMIN>(std::forward<Ts>(args)...);
    case aggregation::NUNIQUE:
      return f.template operator()<aggregation::NUNIQUE>(std::forward<Ts>(args)...);
    case aggregation::NTH_ELEMENT:
      return f.template operator()<aggregation::NTH_ELEMENT>(std::forward<Ts>(args)...);
    case aggregation::ROW_NUMBER:
      return f.template operator()<aggregation::ROW_NUMBER>(std::forward<Ts>(args)...);
    case aggregation::COLLECT_LIST:
      return f.template operator()<aggregation::COLLECT_LIST>(std::forward<Ts>(args)...);
    case aggregation::COLLECT_SET:
      return f.template operator()<aggregation::COLLECT_SET>(std::forward<Ts>(args)...);
    case aggregation::LEAD:
      return f.template operator()<aggregation::LEAD>(std::forward<Ts>(args)...);
    case aggregation::LAG:
      return f.template operator()<aggregation::LAG>(std::forward<Ts>(args)...);
    case aggregation::MERGE_LISTS:
      return f.template operator()<aggregation::MERGE_LISTS>(std::forward<Ts>(args)...);
    case aggregation::MERGE_SETS:
      return f.template operator()<aggregation::MERGE_SETS>(std::forward<Ts>(args)...);
    case aggregation::MERGE_M2:
      return f.template operator()<aggregation::MERGE_M2>(std::forward<Ts>(args)...);
    default: {
#ifndef __CUDA_ARCH__
      CUDF_FAIL("Unsupported aggregation.");
#else
      cudf_assert(false && "Unsupported aggregation.");

      // The following code will never be reached, but the compiler generates a
      // warning if there isn't a return value.

      // Need to find out what the return type is in order to have a default
      // return value and solve the compiler warning for lack of a default
      // return
      using return_type =
        decltype(f.template operator()<aggregation::SUM>(std::forward<Ts>(args)...));
      return return_type();
#endif
    }
  }
}

template <typename Element>
struct dispatch_aggregation {
#pragma nv_exec_check_disable
  template <aggregation::Kind k, typename F, typename... Ts>
  CUDA_HOST_DEVICE_CALLABLE decltype(auto) operator()(F&& f, Ts&&... args) const
  {
    return f.template operator()<Element, k>(std::forward<Ts>(args)...);
  }
};

struct dispatch_source {
#pragma nv_exec_check_disable
  template <typename Element, typename F, typename... Ts>
  CUDA_HOST_DEVICE_CALLABLE decltype(auto) operator()(aggregation::Kind k,
                                                      F&& f,
                                                      Ts&&... args) const
  {
    return aggregation_dispatcher(
      k, dispatch_aggregation<Element>{}, std::forward<F>(f), std::forward<Ts>(args)...);
  }
};

/**
 * @brief Dispatches both a type and `aggregation::Kind` template parameters to
 * a callable.
 *
 * This function expects a callable `f` with an `operator()` template accepting
 * two template parameters. The first is a type dispatched from `type`. The
 * second is an `aggregation::Kind` dispatched from `k`.
 *
 * @param type The `data_type` used to dispatch a type for the first template
 * parameter of the callable `F`
 * @param k The `aggregation::Kind` used to dispatch an `aggregation::Kind`
 * non-type template parameter for the second template parameter of the callable
 * @param args Parameter pack forwarded to the `operator()` invocation
 * `F`.
 */
#pragma nv_exec_check_disable
template <typename F, typename... Ts>
CUDA_HOST_DEVICE_CALLABLE constexpr decltype(auto) dispatch_type_and_aggregation(
  data_type type, aggregation::Kind k, F&& f, Ts&&... args)
{
  return type_dispatcher(type, dispatch_source{}, k, std::forward<F>(f), std::forward<Ts>(args)...);
}
/**
 * @brief Returns the target `data_type` for the specified aggregation  k
 * performed on elements of type  source_type.
 *
 * aram source_type The element type to be aggregated
 * aram k The aggregation
 * @return data_type The target_type of  k performed on  source_type
 * elements
 */
data_type target_type(data_type source_type, aggregation::Kind k);

/**
 * @brief Indicates whether the specified aggregation `k` is valid to perform on
 * the type `Source`.
 *
 * @tparam Source Type on which the aggregation is performed
 * @tparam k The aggregation to perform
 */
template <typename Source, aggregation::Kind k>
constexpr inline bool is_valid_aggregation()
{
  return (not std::is_void<target_type_t<Source, k>>::value);
}

/**
 * @brief Indicates whether the specified aggregation `k` is valid to perform on
 * the `data_type` `source`.
 *
 * @param source Source `data_type` on which the aggregation is performed
 * @param k The aggregation to perform
 */
bool is_valid_aggregation(data_type source, aggregation::Kind k);

}  // namespace detail
}  // namespace cudf
