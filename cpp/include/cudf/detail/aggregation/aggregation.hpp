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

#include <cudf/aggregation.hpp>
#include <cudf/detail/utilities/release_assert.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <functional>
#include <numeric>

namespace cudf {
namespace detail {
/**
 * @brief A wrapper to simplify inheritance of virtual methods from aggregation
 *
 * Derived aggregations are required to implement operator==() and hash_impl().
 *
 * https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
 */
template <class Derived>
class derived_aggregation : public aggregation {
 public:
  derived_aggregation(aggregation::Kind a) : aggregation(a) {}

  bool is_equal(aggregation const& other) const override
  {
    if (this->aggregation::is_equal(other)) {
      // Dispatch to operator== using static polymorphism
      return static_cast<Derived const&>(*this) == static_cast<Derived const&>(other);
    } else {
      return false;
    }
  }

  size_t do_hash() const override
  {
    // Dispatch to hash_impl() using static polymorphism
    return this->aggregation::do_hash() ^ static_cast<Derived const&>(*this).hash_impl();
  }

  std::unique_ptr<aggregation> clone() const override
  {
    // Dispatch to copy constructor using static polymorphism
    return std::make_unique<Derived>(static_cast<Derived const&>(*this));
  }
};

/**
 * @brief Derived class for specifying a quantile aggregation
 */
struct quantile_aggregation final : derived_aggregation<quantile_aggregation> {
  quantile_aggregation(std::vector<double> const& q, interpolation i)
    : derived_aggregation{QUANTILE}, _quantiles{q}, _interpolation{i}
  {
  }
  std::vector<double> _quantiles;  ///< Desired quantile(s)
  interpolation _interpolation;    ///< Desired interpolation

 protected:
  friend class derived_aggregation<quantile_aggregation>;

  bool operator==(quantile_aggregation const& other) const
  {
    return _interpolation == other._interpolation and
           std::equal(_quantiles.begin(), _quantiles.end(), other._quantiles.begin());
  }

  size_t hash_impl() const
  {
    return std::hash<int>{}(static_cast<int>(_interpolation)) ^
           std::accumulate(
             _quantiles.cbegin(), _quantiles.cend(), size_t{0}, [](size_t a, double b) {
               return a ^ std::hash<double>{}(b);
             });
  }
};

struct lead_lag_aggregation final : derived_aggregation<lead_lag_aggregation> {
  lead_lag_aggregation(Kind kind, size_type offset)
    : derived_aggregation{offset < 0 ? (kind == LAG ? LEAD : LAG) : kind},
      row_offset{std::abs(offset)}
  {
  }

  size_type row_offset;

 protected:
  friend class derived_aggregation<lead_lag_aggregation>;

  bool operator==(lead_lag_aggregation const& rhs) const { return row_offset == rhs.row_offset; }

  size_t hash_impl() const { return std::hash<size_type>()(row_offset); }
};

/**
 * @brief Derived class for specifying a standard deviation/variance aggregation
 */
struct std_var_aggregation final : derived_aggregation<std_var_aggregation> {
  std_var_aggregation(aggregation::Kind k, size_type ddof) : derived_aggregation{k}, _ddof{ddof} {}
  size_type _ddof;  ///< Delta degrees of freedom

 protected:
  friend class derived_aggregation<std_var_aggregation>;

  bool operator==(std_var_aggregation const& other) const { return _ddof == other._ddof; }

  size_t hash_impl() const { return std::hash<size_type>{}(_ddof); }
};

/**
 * @brief Derived class for specifying a nunique aggregation
 */
struct nunique_aggregation final : derived_aggregation<nunique_aggregation> {
  nunique_aggregation(aggregation::Kind k, null_policy null_handling)
    : derived_aggregation{k}, _null_handling{null_handling}
  {
  }
  null_policy _null_handling;  ///< include or exclude nulls

 protected:
  friend class derived_aggregation<nunique_aggregation>;

  bool operator==(nunique_aggregation const& other) const
  {
    return _null_handling == other._null_handling;
  }

  size_t hash_impl() const { return std::hash<int>{}(static_cast<int>(_null_handling)); }
};

/**
 * @brief Derived class for specifying a nth element aggregation
 */
struct nth_element_aggregation final : derived_aggregation<nth_element_aggregation> {
  nth_element_aggregation(aggregation::Kind k, size_type n, null_policy null_handling)
    : derived_aggregation{k}, _n{n}, _null_handling{null_handling}
  {
  }
  size_type _n;                ///< nth index to return
  null_policy _null_handling;  ///< include or exclude nulls

 protected:
  friend class derived_aggregation<nth_element_aggregation>;

  bool operator==(nth_element_aggregation const& other) const
  {
    return _n == other._n and _null_handling == other._null_handling;
  }

  size_t hash_impl() const
  {
    return std::hash<size_type>{}(_n) ^ std::hash<int>{}(static_cast<int>(_null_handling));
  }
};

/**
 * @brief Derived class for specifying a custom aggregation
 * specified in udf
 */
struct udf_aggregation final : derived_aggregation<udf_aggregation> {
  udf_aggregation(aggregation::Kind type,
                  std::string const& user_defined_aggregator,
                  data_type output_type)
    : derived_aggregation{type},
      _source{user_defined_aggregator},
      _operator_name{(type == aggregation::PTX) ? "rolling_udf_ptx" : "rolling_udf_cuda"},
      _function_name{"rolling_udf"},
      _output_type{output_type}
  {
  }
  std::string const _source;
  std::string const _operator_name;
  std::string const _function_name;
  data_type _output_type;

 protected:
  friend class derived_aggregation<udf_aggregation>;

  bool operator==(udf_aggregation const& other) const
  {
    return _source == other._source and _operator_name == other._operator_name and
           _function_name == other._function_name and _output_type == other._output_type;
  }

  size_t hash_impl() const
  {
    return std::hash<std::string>{}(_source) ^ std::hash<std::string>{}(_operator_name) ^
           std::hash<std::string>{}(_function_name) ^
           std::hash<int>{}(static_cast<int32_t>(_output_type.id()));
  }
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
 **/
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
struct target_type_impl<Source,
                        k,
                        std::enable_if_t<!is_chrono<Source>() && (k == aggregation::MEAN)>> {
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

// Always use list for COLLECT
template <typename Source>
struct target_type_impl<Source, aggregation::COLLECT> {
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
AGG_KIND_MAPPING(aggregation::STD, std_var_aggregation);
AGG_KIND_MAPPING(aggregation::VARIANCE, std_var_aggregation);

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
    case aggregation::COLLECT:
      return f.template operator()<aggregation::COLLECT>(std::forward<Ts>(args)...);
    case aggregation::LEAD:
      return f.template operator()<aggregation::LEAD>(std::forward<Ts>(args)...);
    case aggregation::LAG:
      return f.template operator()<aggregation::LAG>(std::forward<Ts>(args)...);
    default: {
#ifndef __CUDA_ARCH__
      CUDF_FAIL("Unsupported aggregation.");
#else
      release_assert(false && "Unsupported aggregation.");

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
  CUDA_HOST_DEVICE_CALLABLE decltype(auto) operator()(F&& f, Ts&&... args) const noexcept
  {
    return f.template operator()<Element, k>(std::forward<Ts>(args)...);
  }
};

struct dispatch_source {
#pragma nv_exec_check_disable
  template <typename Element, typename F, typename... Ts>
  CUDA_HOST_DEVICE_CALLABLE decltype(auto) operator()(aggregation::Kind k,
                                                      F&& f,
                                                      Ts&&... args) const noexcept
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
