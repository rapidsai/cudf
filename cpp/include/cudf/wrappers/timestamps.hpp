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

// Figure out a way to avoid the macro redefinitions in <simt/../details/__config>
#ifndef __CUDACC__
#  undef __host__
#  undef __device__
#endif

#include <simt/chrono>

/**---------------------------------------------------------------------------*
 * @file timestamps.hpp
 * @brief Concrete type definitions for int32_t and int64_t timestamps in
 * varying resolutions as durations since the UNIX epoch.
 *---------------------------------------------------------------------------**/
namespace cudf {
namespace exp {

namespace detail {

    template <class Rep, intmax_t Num, intmax_t Denom = 1>
    using duration_st =
        simt::std::chrono::duration<Rep, simt::std::ratio<Num, Denom>>;

    template <class Rep, intmax_t Num, intmax_t Denom = 1>
    struct timestamp : duration_st<Rep, Num, Denom> {
        /// Initialize timestamp to 1/1/1970:00:00:00
        constexpr timestamp() : duration_st<Rep, Num, Denom>(0) {};
        constexpr timestamp(Rep t) : duration_st<Rep, Num, Denom>(t) {};
    };
} // detail

using timestamp_D = detail::timestamp<int32_t, 86400>;
using timestamp_s = detail::timestamp<int64_t, 1, 1>;
using timestamp_ms = detail::timestamp<int64_t, 1, 1000>;
using timestamp_us = detail::timestamp<int64_t, 1, 1000000>;
using timestamp_ns = detail::timestamp<int64_t, 1, 1000000000>;

}  // exp
}  // cudf
