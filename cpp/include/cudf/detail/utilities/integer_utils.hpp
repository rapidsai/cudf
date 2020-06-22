/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Eyal Rozenberg <eyalroz@blazingdb.com>
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

/**
 * @file Utility code involving integer arithmetic
 *
 */

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace cudf {
//! Utility functions
namespace util {
/**
 * Finds the smallest integer not less than `number_to_round` and modulo `S` is
 * zero. This function assumes that `number_to_round` is non-negative and
 * `modulus` is positive.
 */
template <typename S>
inline S round_up_safe(S number_to_round, S modulus)
{
  auto remainder = number_to_round % modulus;
  if (remainder == 0) { return number_to_round; }
  auto rounded_up = number_to_round - remainder + modulus;
  if (rounded_up < number_to_round) {
    throw std::invalid_argument("Attempt to round up beyond the type's maximum value");
  }
  return rounded_up;
}

/**
 * Finds the largest integer not greater than `number_to_round` and modulo `S` is
 * zero. This function assumes that `number_to_round` is non-negative and
 * `modulus` is positive.
 */
template <typename S>
inline S round_down_safe(S number_to_round, S modulus)
{
  auto remainder    = number_to_round % modulus;
  auto rounded_down = number_to_round - remainder;
  return rounded_down;
}

/**
 * Divides the left-hand-side by the right-hand-side, rounding up
 * to an integral multiple of the right-hand-side, e.g. (9,5) -> 2 , (10,5) -> 2, (11,5) -> 3.
 *
 * @param dividend the number to divide
 * @param divisor the number by which to divide
 * @return The least integer multiple of {@link divisor} which is greater than or equal to
 * the non-integral division dividend/divisor.
 *
 * @note sensitive to overflow, i.e. if dividend > std::numeric_limits<S>::max() - divisor,
 * the result will be incorrect
 */
template <typename S, typename T>
constexpr inline S div_rounding_up_unsafe(const S& dividend, const T& divisor) noexcept
{
  return (dividend + divisor - 1) / divisor;
}

namespace detail {
template <typename I>
constexpr inline I div_rounding_up_safe(std::integral_constant<bool, false>,
                                        I dividend,
                                        I divisor) noexcept
{
  // TODO: This could probably be implemented faster
  return (dividend > divisor) ? 1 + div_rounding_up_unsafe(dividend - divisor, divisor)
                              : (dividend > 0);
}

template <typename I>
constexpr inline I div_rounding_up_safe(std::integral_constant<bool, true>,
                                        I dividend,
                                        I divisor) noexcept
{
  auto quotient  = dividend / divisor;
  auto remainder = dividend % divisor;
  return quotient + (remainder != 0);
}

}  // namespace detail

/**
 * Divides the left-hand-side by the right-hand-side, rounding up
 * to an integral multiple of the right-hand-side, e.g. (9,5) -> 2 , (10,5) -> 2, (11,5) -> 3.
 *
 * @param dividend the number to divide
 * @param divisor the number of by which to divide
 * @return The least integer multiple of {@link divisor} which is greater than or equal to
 * the non-integral division dividend/divisor.
 *
 * @note will not overflow, and may _or may not_ be slower than the intuitive
 * approach of using (dividend + divisor - 1) / divisor
 */
template <typename I>
constexpr inline I div_rounding_up_safe(I dividend, I divisor) noexcept
{
  using i_is_a_signed_type = std::integral_constant<bool, std::is_signed<I>::value>;
  return detail::div_rounding_up_safe(i_is_a_signed_type{}, dividend, divisor);
}

template <typename I>
constexpr inline bool is_a_power_of_two(I val) noexcept
{
  static_assert(std::is_integral<I>::value, "This function only applies to integral types");
  return ((val - 1) & val) == 0;
}

/**
 * @brief Return the absolute value of a number.
 *
 * This calls `std::abs()` which performs equivalent: `(value < 0) ? -value : value`.
 *
 * This was created to prevent compile errors calling `std::abs()` with unsigned integers.
 * An example compile error appears as follows:
 * @code{.pseudo}
 * error: more than one instance of overloaded function "std::abs" matches the argument list:
 *          function "abs(int)"
 *          function "std::abs(long)"
 *          function "std::abs(long long)"
 *          function "std::abs(double)"
 *          function "std::abs(float)"
 *          function "std::abs(long double)"
 *          argument types are: (uint64_t)
 * @endcode
 *
 * Not all cases could be if-ed out using `std::is_signed<T>::value` and satisfy the compiler.
 *
 * @param value Numeric value can be either integer or float type.
 * @return Absolute value if value type is signed.
 */
template <typename T>
std::enable_if_t<std::is_signed<T>::value, T> constexpr inline absolute_value(T value)
{
  return std::abs(value);
}
// Unsigned type just returns itself.
template <typename T>
std::enable_if_t<!std::is_signed<T>::value, T> constexpr inline absolute_value(T value)
{
  return value;
}

}  // namespace util

}  // namespace cudf
