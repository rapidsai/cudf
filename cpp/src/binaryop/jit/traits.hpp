/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

// Include Jitify's cstddef header first
#include <cstddef>

#include <cuda/std/climits>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

namespace cudf {
namespace binops {
namespace jit {

// -------------------------------------------------------------------------
// type_traits cannot tell the difference between float and double
template <typename Type>
constexpr bool isFloat = false;

template <typename T>
constexpr bool is_timestamp_v =
  cuda::std::is_same_v<cudf::timestamp_D, T> || cuda::std::is_same_v<cudf::timestamp_s, T> ||
  cuda::std::is_same_v<cudf::timestamp_ms, T> || cuda::std::is_same_v<cudf::timestamp_us, T> ||
  cuda::std::is_same_v<cudf::timestamp_ns, T>;

template <typename T>
constexpr bool is_duration_v =
  cuda::std::is_same_v<cudf::duration_D, T> || cuda::std::is_same_v<cudf::duration_s, T> ||
  cuda::std::is_same_v<cudf::duration_ms, T> || cuda::std::is_same_v<cudf::duration_us, T> ||
  cuda::std::is_same_v<cudf::duration_ns, T>;

template <typename T>
constexpr bool is_chrono_v = is_timestamp_v<T> || is_duration_v<T>;

template <>
constexpr bool isFloat<float> = true;

template <typename Type>
constexpr bool isDouble = false;

template <>
constexpr bool isDouble<double> = true;

}  // namespace jit
}  // namespace binops
}  // namespace cudf
