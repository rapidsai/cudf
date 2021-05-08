/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/fixed_point/fixed_point.hpp>

#include <cudf/wrappers/timestamps.hpp>

#include <cudf/strings/string_view.cuh>

#include <cudf/wrappers/durations.hpp>

#include <cudf/utilities/traits.hpp>

#include "conversion_type_select.cuh"

#include <tuple>

namespace cudf {
namespace io {
namespace detail {

enum class io_type {
  ORC,
  PARQUET
};

template <io_type IO>
struct conversion_map;

template <>
struct conversion_map<io_type::ORC> {
using types = std::tuple<
std::pair<cudf::timestamp_s, cudf::timestamp_ms>,
std::pair<cudf::timestamp_us, cudf::timestamp_ms>,
std::pair<cudf::timestamp_ns, cudf::timestamp_ms>,
std::pair<cudf::duration_s, cudf::duration_ms>,
std::pair<cudf::duration_us, cudf::duration_ms>,
std::pair<cudf::duration_ns, cudf::duration_ms>>;
};

template <>
struct conversion_map<io_type::PARQUET> {
using types = std::tuple<
std::pair<cudf::timestamp_s, cudf::timestamp_ms>,
std::pair<cudf::timestamp_ns, cudf::timestamp_us>,
std::pair<cudf::duration_s, cudf::duration_ms>,
std::pair<cudf::duration_ns, cudf::duration_us>>;
};

template <typename conversion>
class type_conversion {
  using type_selector = ConversionTypeSelect<typename conversion::types>;

  public:
  template <typename T>
  using type = typename type_selector::template type<T>;

  template <typename T>
  static constexpr __device__
  typename type_selector::template type<T> convert(const T& elem) {
    using Type = typename type_selector::template type<T>;
    if constexpr (cudf::is_duration<T>()) {
      return cuda::std::chrono::duration_cast<Type>(elem);
    } else if constexpr (cudf::is_timestamp<T>()) {
      using Duration = typename Type::duration;
      return cuda::std::chrono::time_point_cast<Duration>(elem);
    } else {
      return elem;
    }
    return Type{};
  }
};

template<class T> struct dependent_false : std::false_type {};

template <typename T>
class extrema_type {
  private:
  using integral_extrema_type =
      typename std::conditional_t<
        std::is_signed_v<T>,
        int64_t,
        uint64_t>;

  using arithmetic_extrema_type =
      typename std::conditional_t<
        std::is_integral_v<T>,
        integral_extrema_type,
        double>;

  using non_arithmetic_extrema_type =
    typename std::conditional_t<
      cudf::is_fixed_point<T>()
      or cudf::is_duration<T>()
      or cudf::is_timestamp<T>(),
      int64_t,
      typename std::conditional_t<
        std::is_same_v<T,string_view>,
        string_view,
        void
      >
   >;

  //unsigned int/bool -> uint64_t
  //signed int        -> int64_t
  //float/double      -> double
  //decimal32/64      -> int64_t
  //duration_[T]      -> int64_t
  //string_view       -> string_view
  //timestamp_[T]     -> int64_t

  public:
  //Does type T have an extrema?
  static constexpr bool is_supported = 
    std::is_arithmetic_v<T>
    or std::is_same_v<T,string_view>
    or cudf::is_duration<T>()
    or cudf::is_timestamp<T>()
    or cudf::is_fixed_point<T>();

  using type =
    typename std::conditional_t<
      std::is_arithmetic_v<T>,
      arithmetic_extrema_type,
      non_arithmetic_extrema_type>;

  __device__ static
  type convert(const T& val) {
    if constexpr (std::is_arithmetic_v<T> or std::is_same_v<T,string_view>) {
      return val;
    } else if constexpr (cudf::is_fixed_point<T>()) {
      return val.value();
    } else if constexpr (cudf::is_duration<T>()) {
      return val.count();
    } else if constexpr (cudf::is_timestamp<T>()) {
      return val.time_since_epoch().count();
    } else {
      static_assert(dependent_false<T>::value, "aggregation_type does not exist");
    }
    return type{};
  }
};

template <typename T>
class aggregation_type {
  private:
  using integral_aggregation_type =
      typename std::conditional_t<
        std::is_signed_v<T>,
        int64_t,
        uint64_t>;

  using arithmetic_aggregation_type =
      typename std::conditional_t<
        std::is_integral_v<T>,
        integral_aggregation_type,
        double>;

  using non_arithmetic_aggregation_type =
    typename std::conditional_t<
      cudf::is_fixed_point<T>()
      or cudf::is_duration<T>()
      or cudf::is_timestamp<T>()//To be disabled with static_assert
      or std::is_same_v<T,string_view>,
      int64_t,
      void
   >;

  //unsigned int/bool -> uint64_t
  //signed int        -> int64_t
  //float/double      -> double
  //decimal32/64      -> int64_t
  //duration_[T]      -> int64_t
  //string_view       -> int64_t
  //NOTE : timestamps do not have an aggregation type

  public:
  //Does type T aggregate?
  static constexpr bool is_supported =
    std::is_arithmetic_v<T>
    or std::is_same_v<T,string_view>
    or cudf::is_duration<T>()
    or cudf::is_fixed_point<T>();

  using type =
    typename std::conditional_t<
      std::is_arithmetic_v<T>,
      arithmetic_aggregation_type,
      non_arithmetic_aggregation_type>;

  __device__ static
  type convert(const T& val) {
    if constexpr(std::is_same_v<T,string_view>) {
      return val.size_bytes();
    } else if constexpr (std::is_integral_v<T>) {
      return val;
    } else if constexpr (std::is_floating_point_v<T>) {
      return isnan(val)? 0 : val;
    } else if constexpr (cudf::is_fixed_point<T>()) {
      return val.value();
    } else if constexpr (cudf::is_duration<T>()) {
      return val.count();
    } else {
      static_assert(dependent_false<T>::value, "aggregation_type for timestamps do not exist");
    }
    return type{};
  }
};

template <typename T>
__inline__ __device__ constexpr
std::enable_if_t<std::is_same_v<T,string_view>, T>
minimum_identity(void) {
  return string_view::max();
}

template <typename T>
__inline__ __device__ constexpr
std::enable_if_t<not std::is_same_v<T,string_view>, T>
minimum_identity(void) {
  return std::numeric_limits<T>::max();
}

template <typename T>
__inline__ __device__ constexpr
std::enable_if_t<std::is_same_v<T,string_view>, T>
maximum_identity(void) {
  return string_view::min();
}

template <typename T>
__inline__ __device__ constexpr
std::enable_if_t<not std::is_same_v<T,string_view>, T>
maximum_identity(void) {
  return std::numeric_limits<T>::lowest();
}

template <typename T, io_type IO>
class statistics_type_category {
  //ORC does not calculate the statistics of unsigned integers except bools
  //and durations
  static constexpr bool ignore_aggregated_statistics = 
    (IO == io_type::ORC) and
    (((not std::is_same_v<T,bool>) and
    (std::is_integral_v<T> and std::is_unsigned_v<T>)) or
    cudf::is_duration<T>());

  public:
  //Types that calculate the sum of elements encountered
  static constexpr bool aggregated_statistics =
    (not ignore_aggregated_statistics) and
    aggregation_type<T>::is_supported;

  //Types for which sum does not make sense
  static constexpr bool non_aggregated_statistics = cudf::is_timestamp<T>();

  //Do not calculate statistics for any other type
  static constexpr bool ignored_statistics = not (aggregated_statistics or non_aggregated_statistics);
};

}  // namespace detail
}  // namespace io
}  // namespace cudf
