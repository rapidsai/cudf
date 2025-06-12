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

#include <cudf/utilities/export.hpp>
#include <cudf/wrappers/durations.hpp>

/**
 * @file timestamps.hpp
 * @brief Concrete type definitions for int32_t and int64_t timestamps in
 * varying resolutions as durations since the UNIX epoch.
 */
namespace CUDF_EXPORT cudf {
namespace detail {
// TODO: Use chrono::utc_clock when available in libcu++?
template <class Duration>
using time_point = cuda::std::chrono::sys_time<Duration>;  ///< Time point type

/**
 * @brief A wrapper around a column of time_point in varying resolutions
 *
 * @tparam Duration The underlying duration type
 */
template <class Duration>
using timestamp = time_point<Duration>;
}  // namespace detail

/**
 * @addtogroup timestamp_classes
 * @{
 * @file
 */

/**
 * @brief Type alias representing a cudf::duration_D (int32_t) since the unix epoch.
 */
using timestamp_D = detail::timestamp<cudf::duration_D>;
/**
 * @brief Type alias representing a cudf::duration_h (int32_t) since the unix epoch.
 */
using timestamp_h = detail::timestamp<cudf::duration_h>;
/**
 * @brief Type alias representing a cudf::duration_m (int32_t) since the unix epoch.
 */
using timestamp_m = detail::timestamp<cudf::duration_m>;
/**
 * @brief Type alias representing a cudf::duration_s (int64_t) since the unix epoch.
 */
using timestamp_s = detail::timestamp<cudf::duration_s>;
/**
 * @brief Type alias representing a cudf::duration_ms (int64_t) since the unix epoch.
 */
using timestamp_ms = detail::timestamp<cudf::duration_ms>;
/**
 * @brief Type alias representing a cudf::duration_us (int64_t) since the unix epoch.
 */
using timestamp_us = detail::timestamp<cudf::duration_us>;
/**
 * @brief Type alias representing a cudf::duration_ns (int64_t) since the unix epoch.
 */
using timestamp_ns = detail::timestamp<cudf::duration_ns>;

static_assert(sizeof(timestamp_D) == sizeof(typename timestamp_D::rep));
static_assert(sizeof(timestamp_h) == sizeof(typename timestamp_h::rep));
static_assert(sizeof(timestamp_m) == sizeof(typename timestamp_m::rep));
static_assert(sizeof(timestamp_s) == sizeof(typename timestamp_s::rep));
static_assert(sizeof(timestamp_ms) == sizeof(typename timestamp_ms::rep));
static_assert(sizeof(timestamp_us) == sizeof(typename timestamp_us::rep));
static_assert(sizeof(timestamp_ns) == sizeof(typename timestamp_ns::rep));

/**
 * @brief Maps the timestamp type to its rep type
 *
 * Use this "type function" with the `using` type alias:
 * @code
 * using Type = timestamp_rep_type_t<ElementType>;
 * @endcode
 *
 * @tparam T The literal timestamp type
 * @tparam N The default type to return if T is not a timestamp type
 */
// clang-format off
template <typename T, typename N=T>
using timestamp_rep_type_t =
  std::conditional_t<std::is_same_v<timestamp_D, T>,  timestamp_D::rep,
  std::conditional_t<std::is_same_v<timestamp_h, T>,  timestamp_h::rep,
  std::conditional_t<std::is_same_v<timestamp_m, T>,  timestamp_m::rep,
  std::conditional_t<std::is_same_v<timestamp_s, T>,  timestamp_s::rep,
  std::conditional_t<std::is_same_v<timestamp_ms, T>, timestamp_ms::rep,
  std::conditional_t<std::is_same_v<timestamp_us, T>, timestamp_us::rep,
  std::conditional_t<std::is_same_v<timestamp_ns, T>, timestamp_ns::rep, N>>>>>>>;
// clang-format on

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
