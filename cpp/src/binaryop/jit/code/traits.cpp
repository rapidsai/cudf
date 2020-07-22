/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

namespace cudf {
namespace binops {
namespace jit {
namespace code {
const char* traits =
  R"***(
#pragma once
    #include <cstdint>
    #include <simt/type_traits>

    // -------------------------------------------------------------------------
    // Simplifying std::is_integral
    template <typename T>
    constexpr bool is_integral_v = simt::std::is_integral<T>::value;

    // Simplifying std::is_floating_point
    template <typename T>
    constexpr bool is_floating_point_v = simt::std::is_floating_point<T>::value;

    // -------------------------------------------------------------------------
    // type_traits cannot tell the difference between float and double
    template <typename Type>
    constexpr bool isFloat = false;

    template <typename T>
    constexpr bool is_timestamp_v =
        simt::std::is_same<cudf::timestamp_D, T>::value ||
        simt::std::is_same<cudf::timestamp_s, T>::value ||
        simt::std::is_same<cudf::timestamp_ms, T>::value ||
        simt::std::is_same<cudf::timestamp_us, T>::value ||
        simt::std::is_same<cudf::timestamp_ns, T>::value;

    template <typename T>
    constexpr bool is_duration_v =
        simt::std::is_same<cudf::duration_D, T>::value ||
        simt::std::is_same<cudf::duration_s, T>::value ||
        simt::std::is_same<cudf::duration_ms, T>::value ||
        simt::std::is_same<cudf::duration_us, T>::value ||
        simt::std::is_same<cudf::duration_ns, T>::value;

    template <typename T>
    constexpr bool is_chrono_v = is_timestamp_v<T> || is_duration_v<T>;

    template <>
    constexpr bool isFloat<float> = true;

    template <typename Type>
    constexpr bool isDouble = false;

    template <>
    constexpr bool isDouble<double> = true;
)***";

}  // namespace code
}  // namespace jit
}  // namespace binops
}  // namespace cudf
