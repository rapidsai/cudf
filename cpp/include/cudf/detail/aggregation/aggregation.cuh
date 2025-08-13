/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <type_traits>

namespace cudf {
namespace detail {
template <typename T>
CUDF_HOST_DEVICE constexpr bool is_product_supported()
{
  return is_numeric<T>();
}

/**
 * @brief Maps an `aggregation::Kind` value to its corresponding binary operator
 *
 * @note Not all values of `aggregation::Kind` have a valid corresponding binary
 * operator. For these values `E`,
 * `std::is_same_v<corresponding_operator<E>::type, void>`.
 *
 * @tparam k The `aggregation::Kind` value to map to its corresponding operator
 */
template <aggregation::Kind k>
struct corresponding_operator {
  using type = void;
};

template <>
struct corresponding_operator<aggregation::MIN> {
  using type = DeviceMin;
};
template <>
struct corresponding_operator<aggregation::MAX> {
  using type = DeviceMax;
};
template <>
struct corresponding_operator<aggregation::ARGMIN> {
  using type = DeviceMin;
};
template <>
struct corresponding_operator<aggregation::ARGMAX> {
  using type = DeviceMax;
};
template <>
struct corresponding_operator<aggregation::ANY> {
  using type = DeviceMax;
};
template <>
struct corresponding_operator<aggregation::ALL> {
  using type = DeviceMin;
};
template <>
struct corresponding_operator<aggregation::SUM> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::SUM_WITH_OVERFLOW> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::PRODUCT> {
  using type = DeviceProduct;
};
template <>
struct corresponding_operator<aggregation::SUM_OF_SQUARES> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::STD> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::VARIANCE> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::MEAN> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::COUNT_VALID> {
  using type = DeviceCount;
};
template <>
struct corresponding_operator<aggregation::COUNT_ALL> {
  using type = DeviceCount;
};

template <aggregation::Kind k>
using corresponding_operator_t = typename corresponding_operator<k>::type;

template <aggregation::Kind k>
constexpr bool has_corresponding_operator()
{
  return !std::is_same_v<typename corresponding_operator<k>::type, void>;
}

}  // namespace detail
}  // namespace cudf
