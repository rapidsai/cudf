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

#include "statistics.cuh"

#include "statistics_type_identification.cuh"

#include <cudf/fixed_point/fixed_point.hpp>

#include <cudf/wrappers/timestamps.hpp>

#include <math_constants.h>

namespace cudf {
namespace io {

class union_member {
  template <typename U, typename V>
  using reference_type = std::conditional_t<std::is_const_v<U>, const V&, V&>;

  public:
  template <typename T, typename U>
  using type = std::conditional_t<std::is_same_v<std::remove_cv_t<T>,string_view>,
    reference_type<U, string_stats>,
    reference_type<U, T>>;

  template <typename T, typename U>
  __device__ static
  std::enable_if_t<std::is_integral_v<T> and std::is_unsigned_v<T>, type<T,U>>
  get(U& val) {
    return val.u_val;
  }

  template <typename T, typename U>
  __device__ static
  std::enable_if_t<std::is_integral_v<T> and std::is_signed_v<T>, type<T,U>>
  get(U& val) {
    return val.i_val;
  }

  template <typename T, typename U>
  __device__ static
  std::enable_if_t<std::is_floating_point_v<T>, type<T,U>>
  get(U& val) {
    return val.fp_val;
  }

  template <typename T, typename U>
  __device__ static
  std::enable_if_t<std::is_same_v<T,string_view>, type<T,U>>
  get(U& val) {
    return val.str_val;
  }

};

template <typename T, typename Enable = void>
struct typed_statistics_chunk{};

template <typename T>
struct typed_statistics_chunk<T,
  typename std::enable_if_t<detail::aggregation_type<T>::is_supported>>{

  using E = typename detail::extrema_type<T>::type;
  using A = typename detail::aggregation_type<T>::type;

  uint32_t num_rows;        //!< number of non-null values in chunk
  uint32_t non_nulls;        //!< number of non-null values in chunk
  uint32_t null_count;       //!< number of null values in chunk

  E minimum_value;
  E maximum_value;
  A aggregate;

  uint8_t has_minmax;  //!< Nonzero if min_value and max_values are valid
  uint8_t has_sum;     //!< Nonzero if sum is valid

  __device__
  typed_statistics_chunk(const uint32_t _num_rows = 0) :
    num_rows(_num_rows),
    non_nulls(0),
    null_count(0),
    minimum_value(detail::minimum_identity<E>()),
    maximum_value(detail::maximum_identity<E>()),
    aggregate(0),
    has_minmax(false),
    has_sum(false)//Set to true when storing
    {}

  __device__
  void reduce(const T& elem) {
    non_nulls++;
    minimum_value = thrust::min<E>(minimum_value, detail::extrema_type<T>::convert(elem));
    maximum_value = thrust::max<E>(maximum_value, detail::extrema_type<T>::convert(elem));
    aggregate += detail::aggregation_type<T>::convert(elem);
    has_minmax = true;
  }

  __device__
  void reduce(const statistics_chunk& chunk) {
    if (chunk.has_minmax) {
      minimum_value = thrust::min<E>(minimum_value, union_member::get<E>(chunk.min_value));
      maximum_value = thrust::max<E>(maximum_value, union_member::get<E>(chunk.max_value));
    }
    if (chunk.has_sum) {
      aggregate += detail::aggregation_type<A>::convert(union_member::get<A>(chunk.sum));
    }
    non_nulls += chunk.non_nulls;
    null_count += chunk.null_count;
    num_rows += (chunk.non_nulls + chunk.null_count);
  }
};

template <typename T>
struct typed_statistics_chunk<T,
  typename std::enable_if<cudf::is_timestamp<T>()>::type>{
  using E = typename detail::extrema_type<T>::type;

  uint32_t num_rows;        //!< number of non-null values in chunk
  uint32_t non_nulls;        //!< number of non-null values in chunk
  uint32_t null_count;       //!< number of null values in chunk

  E minimum_value;
  E maximum_value;

  uint8_t has_minmax;  //!< Nonzero if min_value and max_values are valid
  uint8_t has_sum;     //!< Nonzero if sum is valid

  __device__
  typed_statistics_chunk(const uint32_t _num_rows = 0) :
    num_rows(_num_rows),
    non_nulls(0),
    null_count(0),
    minimum_value(detail::minimum_identity<E>()),
    maximum_value(detail::maximum_identity<E>()),
    has_minmax(false),
    has_sum(false)//Set to true when storing
    {}

  __device__
  void reduce(const T& elem) {
    non_nulls++;
    minimum_value = thrust::min<E>(minimum_value, detail::extrema_type<T>::convert(elem));
    maximum_value = thrust::max<E>(maximum_value, detail::extrema_type<T>::convert(elem));
    has_minmax = true;
  }

  __device__
  void reduce(const statistics_chunk& chunk) {
    if (chunk.has_minmax) {
      minimum_value = thrust::min<E>(minimum_value, union_member::get<E>(chunk.min_value));
      maximum_value = thrust::max<E>(maximum_value, union_member::get<E>(chunk.max_value));
    }
    non_nulls += chunk.non_nulls;
    null_count += chunk.null_count;
    num_rows += (chunk.non_nulls + chunk.null_count);
  }
};

template <typename T>
__inline__ __device__
statistics_chunk
get_untyped_chunk(const typed_statistics_chunk<T> &chunk) {
  statistics_chunk stat;
  stat.non_nulls = chunk.non_nulls;
  stat.null_count = chunk.num_rows - chunk.non_nulls;
  stat.has_minmax = chunk.has_minmax;
  stat.has_sum = chunk.has_minmax; // If a valid input was encountered we assume that the sum is valid
  if (chunk.has_minmax) {
    using E = typename detail::extrema_type<T>::type;
    if constexpr (std::is_floating_point_v<E>) {
      union_member::get<E>(stat.min_value) = (chunk.minimum_value != 0.0 )? chunk.minimum_value : CUDART_NEG_ZERO;
      union_member::get<E>(stat.max_value) = (chunk.maximum_value != 0.0 )? chunk.maximum_value : CUDART_ZERO;
    } else {
      union_member::get<E>(stat.min_value) = chunk.minimum_value;
      union_member::get<E>(stat.max_value) = chunk.maximum_value;
    }
    if constexpr (detail::aggregation_type<T>::is_supported) {
      using A = typename detail::aggregation_type<T>::type;
      union_member::get<A>(stat.sum) = chunk.aggregate;
    }
  }
  return stat;
}

}  // namespace io
}  // namespace cudf
