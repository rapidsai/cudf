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

#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/types.hpp>

namespace cudf {
namespace experimental {
/**
 * @brief Base class for specifying the desired aggregation in an
 * `aggregation_request`.
 *
 * This type is meant to be opaque in the public interface.
 *
 * Other kinds of aggregations may derive from this class to encapsulate
 * additional information needed to compute the aggregation.
 */
class aggregation {
 public:
  /**
   * @brief Possible aggregation operations
   */
  enum Kind { SUM, MIN, MAX, COUNT, MEAN, MEDIAN, QUANTILE };

  aggregation(aggregation::Kind a) : kind{a} {}
  Kind kind;  ///< The aggregation to perform
};
namespace detail {
/**
 * @brief Derived class for specifying a quantile aggregation
 */
struct quantile_aggregation : aggregation {
  quantile_aggregation(std::vector<double> const& q,
                       experimental::interpolation i)
      : aggregation{QUANTILE}, _quantiles{q}, _interpolation{i} {}
  std::vector<double> _quantiles;              ///< Desired quantile(s)
  experimental::interpolation _interpolation;  ///< Desired interpolation
};

/**
 * @brief Maps an `aggregation::Kind` value to it's corresponding binary
 * operator.
 *
 * @note Not all values of `aggregation::Kind` have a valid corresponding binary
 * operator. For these values `E`,
 * `std::is_same_v<corresponding_operator<E>::type, void>`.
 *
 * @tparam k The `aggregation::Kind` value to map to its corresponding operator
 */
template <aggregation::Kind k> struct corresponding_operator { using type = void; };
template <> struct corresponding_operator<aggregation::MIN> { using type = DeviceMin; };
template <> struct corresponding_operator<aggregation::MAX> { using type = DeviceMax; };
template <> struct corresponding_operator<aggregation::SUM> { using type = DeviceSum; };
template <> struct corresponding_operator<aggregation::COUNT> { using type = DeviceSum; };
template <aggregation::Kind k>
using corresponding_operator_t = typename corresponding_operator<k>::type;

/**---------------------------------------------------------------------------*
 * @brief Determines accumulator type based on input type and aggregation.
 *
 * @tparam SourceType The type on which the aggregation is computed
 * @tparam k The aggregation performed
 *---------------------------------------------------------------------------**/
template <typename SourceType, aggregation::Kind k, typename Enable = void>
struct target_type_impl { using type = void; };

// Computing MIN of SourceType, use SourceType accumulator
template <typename SourceType>
struct target_type_impl<SourceType, aggregation::MIN> { using type = SourceType; };

// Computing MAX of SourceType, use SourceType accumulator
template <typename SourceType>
struct target_type_impl<SourceType, aggregation::MAX> { using type = SourceType; };

// Always use size_type accumulator for COUNT
template <typename SourceType>
struct target_type_impl<SourceType, aggregation::COUNT> { using type = cudf::size_type; };

// Always use `double` for MEAN
template <typename SourceType>
struct target_type_impl<SourceType, aggregation::MEAN> { using type = double; };

// Summing integers of any type, always use int64_t accumulator
template <typename SourceType>
struct target_type_impl<SourceType, aggregation::SUM,
                   std::enable_if_t<std::is_integral<SourceType>::value>> {
  using type = int64_t;
};

// Summing float/doubles, use same type accumulator
template <typename SourceType>
struct target_type_impl<
    SourceType, aggregation::SUM,
    std::enable_if_t<std::is_floating_point<SourceType>::value>> {
  using type = SourceType;
};

// Always use `double` for quantile 
template <typename SourceType>
struct target_type_impl<SourceType, aggregation::QUANTILE> { using type = double; };

// MEDIAN is a special case of a QUANTILE  
template <typename SourceType>
struct target_type_impl<SourceType, aggregation::MEDIAN> {
   using type = target_type_impl<SourceType, aggregation::QUANTILE>; 
};

/**
 * @brief Helper alias to get the accumulator type for performing aggregation 
 * `k` on elements of type `SourceType`
 *
 * @tparam SourceType The type on which the aggregation is computed
 * @tparam k The aggregation performed
 */
template <typename SourceType, aggregation::Kind k>
using target_type_t = typename target_type_impl<SourceType, k>::type;

/**
 * @brief Dispatches  k as a non-type template parameter to a callable,  f.
 *
 * @tparam F Type of callable
 * @param k The `aggregation::Kind` value to dispatch
 * aram f The callable that accepts an `aggregation::Kind` non-type template
 * argument.
 * @return Forwards the return value of the callable.
 */
template <typename F>
decltype(auto) aggregation_dispatcher(aggregation::Kind k, F f){
    switch(k){
        case aggregation::SUM:      return f.template operator()<aggregation::SUM>();
        case aggregation::MIN:      return f.template operator()<aggregation::MIN>();
        case aggregation::MAX:      return f.template operator()<aggregation::MAX>();
        case aggregation::COUNT:    return f.template operator()<aggregation::COUNT>();
        case aggregation::MEAN:     return f.template operator()<aggregation::MEAN>();
        case aggregation::MEDIAN:   return f.template operator()<aggregation::MEDIAN>();
        case aggregation::QUANTILE: return f.template operator()<aggregation::QUANTILE>();
        default:                    CUDF_FAIL("Unsupported aggregation kind");
    }
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

}  // namespace detail
}  // namespace experimental
}  // namespace cudf
