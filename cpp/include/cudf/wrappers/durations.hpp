/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cuda/std/chrono>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup timestamp_classes Timestamp
 * @{
 * @file durations.hpp
 * @brief Concrete type definitions for int32_t and int64_t durations in varying resolutions.
 */

/**
 * @brief Type alias representing an int32_t duration of days.
 */
using duration_D = cuda::std::chrono::duration<int32_t, cuda::std::chrono::days::period>;
/**
 * @brief Type alias representing an int32_t duration of hours.
 */
using duration_h = cuda::std::chrono::duration<int32_t, cuda::std::chrono::hours::period>;
/**
 * @brief Type alias representing an int32_t duration of minutes.
 */
using duration_m = cuda::std::chrono::duration<int32_t, cuda::std::chrono::minutes::period>;
/**
 * @brief Type alias representing an int64_t duration of seconds.
 */
using duration_s = cuda::std::chrono::duration<int64_t, cuda::std::chrono::seconds::period>;
/**
 * @brief Type alias representing an int64_t duration of milliseconds.
 */
using duration_ms = cuda::std::chrono::duration<int64_t, cuda::std::chrono::milliseconds::period>;
/**
 * @brief Type alias representing an int64_t duration of microseconds.
 */
using duration_us = cuda::std::chrono::duration<int64_t, cuda::std::chrono::microseconds::period>;
/**
 * @brief Type alias representing an int64_t duration of nanoseconds.
 */
using duration_ns = cuda::std::chrono::duration<int64_t, cuda::std::chrono::nanoseconds::period>;

static_assert(sizeof(duration_D) == sizeof(typename duration_D::rep));
static_assert(sizeof(duration_h) == sizeof(typename duration_h::rep));
static_assert(sizeof(duration_m) == sizeof(typename duration_m::rep));
static_assert(sizeof(duration_s) == sizeof(typename duration_s::rep));
static_assert(sizeof(duration_ms) == sizeof(typename duration_ms::rep));
static_assert(sizeof(duration_us) == sizeof(typename duration_us::rep));
static_assert(sizeof(duration_ns) == sizeof(typename duration_ns::rep));

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
