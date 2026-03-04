/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>

namespace cudf {
namespace io {

/**
 * @brief Lookup table to compute power of ten
 */
static const __device__ __constant__ int32_t powers_of_ten[10] = {
  1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};

struct get_period {
  template <typename T>
  int32_t operator()()
  {
    if constexpr (is_chrono<T>()) { return T::period::den; }
    CUDF_FAIL("Invalid, non chrono type");
  }
};

/**
 * @brief Function that translates cuDF time unit to clock frequency
 */
inline int32_t to_clockrate(type_id timestamp_type_id)
{
  return timestamp_type_id == type_id::EMPTY
           ? 0
           : type_dispatcher(data_type{timestamp_type_id}, get_period{});
}

}  // namespace io
}  // namespace cudf
