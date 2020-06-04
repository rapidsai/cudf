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
namespace detail {

template <typename Rep, typename Period>
using chrono_duration = simt::std::chrono::duration<Rep, Period>;

template <typename Rep, typename Period>
struct cudf_duration : chrono_duration<Rep, Period> {
  using ChronoDurationT = chrono_duration<Rep, Period>;

  constexpr cudf_duration() = default;

  constexpr cudf_duration(ChronoDurationT const& d) : ChronoDurationT(d) {}

  // Implicitly convert a tick count to a cudf_duration
  constexpr cudf_duration(typename cudf_duration::rep r) : ChronoDurationT(r){};

  /**
   * @brief Constructs a new duration by copying the contents of another `duration` and converting
   * its duration value if necessary.
   *
   * This is required as a higher resolution duration period cannot be assigned to
   * a lower resolution duration period naturally. Such truncations may result in loss
   * of time precision.
   *
   * @param other The `duration` to copy
   */
  template <typename Rep2, typename Period2>
  inline constexpr explicit cudf_duration(chrono_duration<Rep2, Period2> const& other)
    : ChronoDurationT(simt::std::chrono::duration_cast<ChronoDurationT>(other)){};
};
}  // namespace detail

/**
 * @addtogroup duration_classes Duration
 * @{
 */

/**
 * @brief Type alias representing an int32_t duration of days.
 **/
using duration_D = detail::cudf_duration<int32_t, simt::std::ratio<86400>>;
/**
 * @brief Type alias representing an int64_t duration of seconds.
 **/
using duration_s = detail::cudf_duration<int64_t, simt::std::ratio<1>>;
/**
 * @brief Type alias representing an int64_t duration of milliseconds.
 **/
using duration_ms = detail::cudf_duration<int64_t, simt::std::milli>;
/**
 * @brief Type alias representing an int64_t duration of microseconds.
 **/
using duration_us = detail::cudf_duration<int64_t, simt::std::micro>;
/**
 * @brief Type alias representing an int64_t duration of nanoseconds.
 **/
using duration_ns = detail::cudf_duration<int64_t, simt::std::nano>;

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
#define DURATION_LIMITS(TypeName)                                   \
  template <>                                                       \
  struct numeric_limits<TypeName> {                                 \
    static constexpr TypeName max() noexcept                        \
    {                                                               \
      return std::numeric_limits<typename TypeName::rep>::max();    \
    }                                                               \
    static constexpr TypeName lowest() noexcept                     \
    {                                                               \
      return std::numeric_limits<typename TypeName::rep>::lowest(); \
    }                                                               \
    static constexpr TypeName min() noexcept                        \
    {                                                               \
      return std::numeric_limits<typename TypeName::rep>::min();    \
    }                                                               \
  }

DURATION_LIMITS(cudf::duration_D);
DURATION_LIMITS(cudf::duration_s);
DURATION_LIMITS(cudf::duration_ms);
DURATION_LIMITS(cudf::duration_us);
DURATION_LIMITS(cudf::duration_ns);

#undef DURATION_LIMITS

}  // namespace std
