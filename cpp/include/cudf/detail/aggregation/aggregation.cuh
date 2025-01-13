/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

#include <type_traits>

namespace cudf {
namespace detail {
template <typename T>
CUDF_HOST_DEVICE constexpr bool is_product_supported()
{
  return is_numeric<T>();
}

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

/**
 * @brief Dispatched functor to initialize a column with the identity of an
 * aggregation operation.
 *
 * Given a type `T` and `aggregation kind k`, determines and sets the value of
 * each element of the passed column to the appropriate initial value for the
 * aggregation.
 *
 * The initial values set as per aggregation are:
 * SUM: 0
 * COUNT_VALID: 0 and VALID
 * COUNT_ALL:   0 and VALID
 * MIN: Max value of type `T`
 * MAX: Min value of type `T`
 * ARGMAX: `ARGMAX_SENTINEL`
 * ARGMIN: `ARGMIN_SENTINEL`
 *
 * Only works on columns of fixed-width types.
 */
struct identity_initializer {
 private:
  template <typename T, aggregation::Kind k>
  static constexpr bool is_supported()
  {
    return cudf::is_fixed_width<T>() and
           (k == aggregation::SUM or k == aggregation::MIN or k == aggregation::MAX or
            k == aggregation::COUNT_VALID or k == aggregation::COUNT_ALL or
            k == aggregation::ARGMAX or k == aggregation::ARGMIN or
            k == aggregation::SUM_OF_SQUARES or k == aggregation::STD or
            k == aggregation::VARIANCE or
            (k == aggregation::PRODUCT and is_product_supported<T>()));
  }

  template <typename T, aggregation::Kind k>
  std::enable_if_t<not std::is_same_v<corresponding_operator_t<k>, void>, T>
  identity_from_operator()
  {
    using DeviceType = device_storage_type_t<T>;
    return corresponding_operator_t<k>::template identity<DeviceType>();
  }

  template <typename T, aggregation::Kind k>
  std::enable_if_t<std::is_same_v<corresponding_operator_t<k>, void>, T> identity_from_operator()
  {
    CUDF_FAIL("Unable to get identity/sentinel from device operator");
  }

  template <typename T, aggregation::Kind k>
  T get_identity()
  {
    if (k == aggregation::ARGMAX || k == aggregation::ARGMIN) {
      if constexpr (cudf::is_timestamp<T>())
        return k == aggregation::ARGMAX ? T{typename T::duration(ARGMAX_SENTINEL)}
                                        : T{typename T::duration(ARGMIN_SENTINEL)};
      else {
        using DeviceType = device_storage_type_t<T>;
        return k == aggregation::ARGMAX ? static_cast<DeviceType>(ARGMAX_SENTINEL)
                                        : static_cast<DeviceType>(ARGMIN_SENTINEL);
      }
    }
    return identity_from_operator<T, k>();
  }

 public:
  template <typename T, aggregation::Kind k>
  std::enable_if_t<is_supported<T, k>(), void> operator()(mutable_column_view const& col,
                                                          rmm::cuda_stream_view stream)
  {
    using DeviceType = device_storage_type_t<T>;
    thrust::fill(rmm::exec_policy(stream),
                 col.begin<DeviceType>(),
                 col.end<DeviceType>(),
                 get_identity<DeviceType, k>());
  }

  template <typename T, aggregation::Kind k>
  std::enable_if_t<not is_supported<T, k>(), void> operator()(mutable_column_view const& col,
                                                              rmm::cuda_stream_view stream)
  {
    CUDF_FAIL("Unsupported aggregation for initializing values");
  }
};

/**
 * @brief Initializes each column in a table with a corresponding identity value
 * of an aggregation operation.
 *
 * The `i`th column will be initialized with the identity value of the `i`th
 * aggregation operation in `aggs`.
 *
 * @throw cudf::logic_error if column type and corresponding agg are incompatible
 * @throw cudf::logic_error if column type is not fixed-width
 *
 * @param table The table of columns to initialize.
 * @param aggs A span of aggregation operations corresponding to the table
 * columns. The aggregations determine the identity value for each column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
void initialize_with_identity(mutable_table_view& table,
                              host_span<cudf::aggregation::Kind const> aggs,
                              rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace cudf
