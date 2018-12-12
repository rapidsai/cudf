/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Eyal Rozenberg <eyalroz@blazingdb.com>
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

#ifndef UTIL_MISCELLANY_HPP_
#define UTIL_MISCELLANY_HPP_

#include <cstdlib> // for std::div
#include <type_traits> // for std::enable_if

extern "C" {
#include <cudf/types.h>
}

namespace gdf {

namespace util {

/**
* Divides the left-hand-side by the right-hand-side, rounding up
* to an integral multiple of the right-hand-side, e.g. (9,5) -> 2 , (10,5) -> 2, (11,5) -> 3.
*
* @param dividend the number to divide
* @param divisor the number of by which to divide
* @return The least integer multiple of {@link divisor} which is greater-or-equal to
* the non-integral division dividend/divisor.
*
* @note sensitive to overflow, i.e. if dividend > std::numeric_limits<S>::max() - divisor,
* the result will be incorrect
*/
template <typename S, typename T>
constexpr inline S div_rounding_up_unsafe(const S& dividend, const T& divisor) {
    return (dividend + divisor - 1) / divisor;
}


/**
* Divides the left-hand-side by the right-hand-side, rounding up
* to an integral multiple of the right-hand-side, e.g. (9,5) -> 2 , (10,5) -> 2, (11,5) -> 3.
*
* @param dividend the number to divide
* @param divisor the number of by which to divide
* @return The least integer multiple of {@link divisor} which is greater-or-equal to
* the non-integral division dividend/divisor.
*
* @note will not overflow, and may _or may not_ be slower than the intuitive
* approach of using (dividend + divisor - 1) / divisor
*/
template <typename I>
constexpr inline I div_rounding_up_safe(I dividend, I divisor);

template <typename I>
constexpr inline typename std::enable_if<std::is_signed<I>::value, I>::type
div_rounding_up_safe(I dividend, I divisor)
{
#if cplusplus >= 201402L
    auto div_result = std::div(dividend, divisor);
    return div_result.quot + !(!div_result.rem);
#else
    // Hopefully the compiler will optimize the two calls away.
    return std::div(dividend, divisor).quot + !(!std::div(dividend, divisor).rem);
#endif
}

// This variant will be used for unsigned types
template <typename I>
constexpr inline I div_rounding_up_safe(I dividend, I divisor)
{
    // TODO: This could probably be implemented faster
    return (dividend > divisor) ?
        1 + div_rounding_up_unsafe(dividend - divisor, divisor) :
        (dividend > 0);
}

} // namespace util
} // namespace gdf


#endif // UTIL_MISCELLANY_HPP_
