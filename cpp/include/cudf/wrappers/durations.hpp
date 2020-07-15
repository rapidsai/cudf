/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <limits>

#define _LIBCUDACXX_USE_CXX20_CHRONO
#define _LIBCUDACXX_USE_CXX17_TYPE_TRAITS

#include <simt/chrono>

/**
 * @file durations.hpp
 * @brief Concrete type definitions for int32_t and int64_t durations in varying resolutions.
 **/
namespace cudf {

/**
 * @addtogroup timestamp_classes Timestamp
 * @{
 */
/**
 * @brief Type alias representing an int32_t duration of days.
 **/
using duration_D = simt::std::chrono::duration<int32_t, simt::std::chrono::days::period>;
/**
 * @brief Type alias representing an int64_t duration of seconds.
 **/
using duration_s = simt::std::chrono::duration<int64_t, simt::std::chrono::seconds::period>;
/**
 * @brief Type alias representing an int64_t duration of milliseconds.
 **/
using duration_ms = simt::std::chrono::duration<int64_t, simt::std::chrono::milliseconds::period>;
/**
 * @brief Type alias representing an int64_t duration of microseconds.
 **/
using duration_us = simt::std::chrono::duration<int64_t, simt::std::chrono::microseconds::period>;
/**
 * @brief Type alias representing an int64_t duration of nanoseconds.
 **/
using duration_ns = simt::std::chrono::duration<int64_t, simt::std::chrono::nanoseconds::period>;

static_assert(sizeof(duration_D) == sizeof(typename duration_D::rep), "");
static_assert(sizeof(duration_s) == sizeof(typename duration_s::rep), "");
static_assert(sizeof(duration_ms) == sizeof(typename duration_ms::rep), "");
static_assert(sizeof(duration_us) == sizeof(typename duration_us::rep), "");
static_assert(sizeof(duration_ns) == sizeof(typename duration_ns::rep), "");

/** @} */  // end of group
}  // namespace cudf

namespace std {
/**
 * @brief Specialization of std::numeric_limits for cudf::detail::duration
 *
 * Pass through to return the limits of the underlying numeric representation.
 **/
#define DURATION_LIMITS(TypeName)                                             \
  template <>                                                                 \
  struct numeric_limits<TypeName> {                                           \
    static constexpr TypeName max() noexcept { return TypeName::max(); }      \
    static constexpr TypeName lowest() noexcept                               \
    {                                                                         \
      return TypeName(std::numeric_limits<typename TypeName::rep>::lowest()); \
    }                                                                         \
    static constexpr TypeName min() noexcept { return TypeName::min(); }      \
  }

DURATION_LIMITS(cudf::duration_D);
DURATION_LIMITS(cudf::duration_s);
DURATION_LIMITS(cudf::duration_ms);
DURATION_LIMITS(cudf::duration_us);
DURATION_LIMITS(cudf::duration_ns);

#undef DURATION_LIMITS

}  // namespace std
