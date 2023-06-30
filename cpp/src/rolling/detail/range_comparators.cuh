/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <thrust/functional.h>

#include <cmath>
#include <limits>

namespace cudf::detail {

/**
 * @brief Comparator for numeric order-by columns, handling floating point NaN values.
 *
 * This is required for binary search through sorted vectors that contain NaN values.
 * With ascending sort, NaN values are stored at the end of the sequence, even
 * greater than infinity.
 * But thrust::less would have trouble locating it because:
 * 1. thrust::less(NaN, 10) returns false
 * 2. thrust::less(10, NaN) also returns false
 *
 * This comparator honours the position of NaN values vis-à-vis non-NaN values.
 *
 */
struct nan_aware_less {
  template <typename T, CUDF_ENABLE_IF(not cudf::is_floating_point<T>())>
  __host__ __device__ bool operator()(T const& lhs, T const& rhs) const
  {
    return thrust::less<T>{}(lhs, rhs);
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_floating_point<T>())>
  __host__ __device__ bool operator()(T const& lhs, T const& rhs) const
  {
    if (std::isnan(lhs)) { return false; }
    return std::isnan(rhs) ? true : thrust::less<T>{}(lhs, rhs);
  }
};

/**
 * @brief Comparator for numeric order-by columns, handling floating point NaN values. *
 *
 * This is required for binary search through sorted vectors that contain NaN values.
 * With descending sort, NaN values are stored at the beginning of the sequence, even
 * greater than infinity.
 * But thrust::greater would have trouble locating it because:
 * 1. thrust::greater(NaN, 10) returns false
 * 2. thrust::greater(10, NaN) also returns false
 *
 * This comparator honours the position of NaN values vis-à-vis non-NaN values.
 *
 */
struct nan_aware_greater {
  template <typename T>
  __host__ __device__ bool operator()(T const& lhs, T const& rhs) const
  {
    return nan_aware_less{}(rhs, lhs);
  }
};
}  // namespace cudf::detail
