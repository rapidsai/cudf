/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

/**---------------------------------------------------------------------------*
 * @file timestamps.hpp
 * @brief Concrete type definitions for int32_t and int64_t timestamps in
 * varying resolutions as durations since the UNIX epoch.
 *---------------------------------------------------------------------------**/
namespace cudf {

namespace detail {

// TODO: Use chrono::utc_clock when available in libcu++?
template <class Duration>
using time_point =
    simt::std::chrono::time_point<simt::std::chrono::system_clock, Duration>;

template <class Duration>
struct timestamp : time_point<Duration> {
  constexpr timestamp() : time_point<Duration>(Duration()){};
  constexpr timestamp(Duration d) : time_point<Duration>(d){};
  constexpr timestamp(typename Duration::rep r)
      : time_point<Duration>(Duration(r)){};
  /**
   * @brief Constructs a new timestamp by copying the contents of another
   * `time_point` and converting its duration value if necessary.
   *
   * @param other The `timestamp` to copy
   */
  template <class FromDuration>
  inline constexpr explicit timestamp(time_point<FromDuration> const& other)
      : time_point<Duration>(simt::std::chrono::duration_cast<Duration>(
            other.time_since_epoch())){};
};
}  // namespace detail

/**---------------------------------------------------------------------------*
 * @brief Type alias representing an int32_t duration of days since the unix
 * epoch.
 *---------------------------------------------------------------------------**/
using timestamp_D = detail::timestamp<
    simt::std::chrono::duration<int32_t, simt::std::ratio<86400>>>;
/**---------------------------------------------------------------------------*
 * @brief Type alias representing an int64_t duration of seconds since the
 * unix epoch.
 *---------------------------------------------------------------------------**/
using timestamp_s = detail::timestamp<
    simt::std::chrono::duration<int64_t, simt::std::ratio<1>>>;
/**---------------------------------------------------------------------------*
 * @brief Type alias representing an int64_t duration of milliseconds since
 * the unix epoch.
 *---------------------------------------------------------------------------**/
using timestamp_ms =
    detail::timestamp<simt::std::chrono::duration<int64_t, simt::std::milli>>;
/**---------------------------------------------------------------------------*
 * @brief Type alias representing an int64_t duration of microseconds since
 * the unix epoch.
 *---------------------------------------------------------------------------**/
using timestamp_us =
    detail::timestamp<simt::std::chrono::duration<int64_t, simt::std::micro>>;
/**---------------------------------------------------------------------------*
 * @brief Type alias representing an int64_t duration of nanoseconds since
 * the unix epoch.
 *---------------------------------------------------------------------------**/
using timestamp_ns =
    detail::timestamp<simt::std::chrono::duration<int64_t, simt::std::nano>>;

static_assert(sizeof(timestamp_D) == sizeof(typename timestamp_D::rep), "");
static_assert(sizeof(timestamp_s) == sizeof(typename timestamp_s::rep), "");
static_assert(sizeof(timestamp_ms) == sizeof(typename timestamp_ms::rep), "");
static_assert(sizeof(timestamp_us) == sizeof(typename timestamp_us::rep), "");
static_assert(sizeof(timestamp_ns) == sizeof(typename timestamp_ns::rep), "");

}  // namespace cudf

namespace std {
/**---------------------------------------------------------------------------*
 * @brief Specialization of std::numeric_limits for cudf::detail::timestamp
 *
 * Pass through to return the limits of the underlying numeric representation.
 *--------------------------------------------------------------------------**/
#define TIMESTAMP_LIMITS(TypeName)                                  \
  template <>                                                       \
  struct numeric_limits<TypeName> {                                 \
    static constexpr TypeName max() noexcept {                      \
      return std::numeric_limits<typename TypeName::rep>::max();    \
    }                                                               \
    static constexpr TypeName lowest() noexcept {                   \
      return std::numeric_limits<typename TypeName::rep>::lowest(); \
    }                                                               \
    static constexpr TypeName min() noexcept {                      \
      return std::numeric_limits<typename TypeName::rep>::min();    \
    }                                                               \
  }

TIMESTAMP_LIMITS(cudf::timestamp_D);
TIMESTAMP_LIMITS(cudf::timestamp_s);
TIMESTAMP_LIMITS(cudf::timestamp_ms);
TIMESTAMP_LIMITS(cudf::timestamp_us);
TIMESTAMP_LIMITS(cudf::timestamp_ns);

#undef TIMESTAMP_LIMITS

}  // namespace std
