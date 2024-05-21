/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuda/std/type_traits>

namespace numeric {

/**
 * @addtogroup floating_conversion
 * @{
 * @file
 * @brief fixed_point <--> floating-point conversion functions.
 */

namespace detail {

/**
 * @brief Recursively calculate a signed large power of 10 (>= 10^19) that can only be stored in an
 * 128bit integer
 *
 * @note Intended to be run at compile time.
 *
 * @tparam Exp10 The power of 10 to calculate
 * @return Returns 10^Exp10
 */
template <int Exp10>
constexpr __uint128_t large_power_of_10()
{
  // Stop at 10^19 to speed up compilation; literals can be used for smaller powers of 10.
  static_assert(Exp10 >= 19);
  if constexpr (Exp10 == 19)
    return __uint128_t(10000000000000000000ULL);
  else
    return large_power_of_10<Exp10 - 1>() * __uint128_t(10);
}

/**
 * @brief Divide by a power of 10 that fits within a 32bit integer.
 *
 * @tparam T Type of value to be divided-from.
 * @param value The number to be divided-from.
 * @param exp10 The power-of-10 of the denominator, from 0 to 9 inclusive.
 * @return Returns value / 10^exp10
 */
template <typename T, typename cuda::std::enable_if_t<cuda::std::is_unsigned_v<T>>* = nullptr>
CUDF_HOST_DEVICE inline T divide_power10_32bit(T value, int exp10)
{
  // Computing division this way is much faster than the alternatives.
  // Division is not implemented in GPU hardware, and the compiler will often implement it as a
  // multiplication of the reciprocal of the denominator, requiring a conversion to floating point.
  // Ths is especially slow for larger divides that have to use the FP64 pipeline, where threads
  // bottleneck.

  // Instead, if the compiler can see exactly what number it is dividing by, it can
  // produce much more optimal assembly, doing bit shifting, multiplies by a constant, etc.
  // For the compiler to see the value though, array lookup (with exp10 as the index)
  // is not sufficient: We have to use a switch statement. Although this introduces a branch,
  // it is still much faster than doing the divide any other way.
  // Perhaps an array can be used in C++23 with the assume attribute?

  // Since we're optimizing division this way, we have to do this for multiplication as well.
  // That's because doing them in different ways (switch, array, runtime-computation, etc.)
  // increases the register pressure on all kernels that use fixed_point types, specifically slowing
  // down some of the PYMOD and join benchmarks.

  // This is split up into separate functions for 32-, 64-, and 128-bit denominators.
  // That way we limit the templated, inlined code generation to the exponents that are
  // capable of being represented. Combining them together into a single function again
  // introduces too much pressure on the kernels that use this code, slowing down their benchmarks.
  // It also dramatically slows down the compile time.

  switch (exp10) {
    case 0: return value;
    case 1: return value / 10U;
    case 2: return value / 100U;
    case 3: return value / 1000U;
    case 4: return value / 10000U;
    case 5: return value / 100000U;
    case 6: return value / 1000000U;
    case 7: return value / 10000000U;
    case 8: return value / 100000000U;
    case 9: return value / 1000000000U;
    default: return 0;
  }
}

/**
 * @brief Divide by a power of 10 that fits within a 64bit integer.
 *
 * @tparam T Type of value to be divided-from.
 * @param value The number to be divided-from.
 * @param exp10 The power-of-10 of the denominator, from 0 to 19 inclusive.
 * @return Returns value / 10^exp10
 */
template <typename T, typename cuda::std::enable_if_t<cuda::std::is_unsigned_v<T>>* = nullptr>
CUDF_HOST_DEVICE inline T divide_power10_64bit(T value, int exp10)
{
  // See comments in divide_power10_32bit() for discussion.
  switch (exp10) {
    case 0: return value;
    case 1: return value / 10U;
    case 2: return value / 100U;
    case 3: return value / 1000U;
    case 4: return value / 10000U;
    case 5: return value / 100000U;
    case 6: return value / 1000000U;
    case 7: return value / 10000000U;
    case 8: return value / 100000000U;
    case 9: return value / 1000000000U;
    case 10: return value / 10000000000ULL;
    case 11: return value / 100000000000ULL;
    case 12: return value / 1000000000000ULL;
    case 13: return value / 10000000000000ULL;
    case 14: return value / 100000000000000ULL;
    case 15: return value / 1000000000000000ULL;
    case 16: return value / 10000000000000000ULL;
    case 17: return value / 100000000000000000ULL;
    case 18: return value / 1000000000000000000ULL;
    case 19: return value / 10000000000000000000ULL;
    default: return 0;
  }
}

/**
 * @brief Divide by a power of 10 that fits within a 128bit integer.
 *
 * @tparam T Type of value to be divided-from.
 * @param value The number to be divided-from.
 * @param exp10 The power-of-10 of the denominator, from 0 to 38 inclusive.
 * @return Returns value / 10^exp10.
 */
template <typename T, typename cuda::std::enable_if_t<cuda::std::is_unsigned_v<T>>* = nullptr>
CUDF_HOST_DEVICE inline constexpr T divide_power10_128bit(T value, int exp10)
{
  // See comments in divide_power10_32bit() for an introduction.
  switch (exp10) {
    case 0: return value;
    case 1: return value / 10U;
    case 2: return value / 100U;
    case 3: return value / 1000U;
    case 4: return value / 10000U;
    case 5: return value / 100000U;
    case 6: return value / 1000000U;
    case 7: return value / 10000000U;
    case 8: return value / 100000000U;
    case 9: return value / 1000000000U;
    case 10: return value / 10000000000ULL;
    case 11: return value / 100000000000ULL;
    case 12: return value / 1000000000000ULL;
    case 13: return value / 10000000000000ULL;
    case 14: return value / 100000000000000ULL;
    case 15: return value / 1000000000000000ULL;
    case 16: return value / 10000000000000000ULL;
    case 17: return value / 100000000000000000ULL;
    case 18: return value / 1000000000000000000ULL;
    case 19: return value / 10000000000000000000ULL;
    case 20: return value / large_power_of_10<20>();
    case 21: return value / large_power_of_10<21>();
    case 22: return value / large_power_of_10<22>();
    case 23: return value / large_power_of_10<23>();
    case 24: return value / large_power_of_10<24>();
    case 25: return value / large_power_of_10<25>();
    case 26: return value / large_power_of_10<26>();
    case 27: return value / large_power_of_10<27>();
    case 28: return value / large_power_of_10<28>();
    case 29: return value / large_power_of_10<29>();
    case 30: return value / large_power_of_10<30>();
    case 31: return value / large_power_of_10<31>();
    case 32: return value / large_power_of_10<32>();
    case 33: return value / large_power_of_10<33>();
    case 34: return value / large_power_of_10<34>();
    case 35: return value / large_power_of_10<35>();
    case 36: return value / large_power_of_10<36>();
    case 37: return value / large_power_of_10<37>();
    case 38: return value / large_power_of_10<38>();
    default: return 0;
  }
}

/**
 * @brief Multiply by a power of 10 that fits within a 32bit integer.
 *
 * @tparam T Type of value to be multiplied.
 * @param value The number to be multiplied.
 * @param exp10 The power-of-10 of the multiplier, from 0 to 9 inclusive.
 * @return Returns value * 10^exp10
 */
template <typename T, typename cuda::std::enable_if_t<cuda::std::is_unsigned_v<T>>* = nullptr>
CUDF_HOST_DEVICE inline constexpr T multiply_power10_32bit(T value, int exp10)
{
  // See comments in divide_power10_32bit() for discussion.
  switch (exp10) {
    case 0: return value;
    case 1: return value * 10U;
    case 2: return value * 100U;
    case 3: return value * 1000U;
    case 4: return value * 10000U;
    case 5: return value * 100000U;
    case 6: return value * 1000000U;
    case 7: return value * 10000000U;
    case 8: return value * 100000000U;
    case 9: return value * 1000000000U;
    default: return 0;
  }
}

/**
 * @brief Multiply by a power of 10 that fits within a 64bit integer.
 *
 * @tparam T Type of value to be multiplied.
 * @param value The number to be multiplied.
 * @param exp10 The power-of-10 of the multiplier, from 0 to 19 inclusive.
 * @return Returns value * 10^exp10
 */
template <typename T, typename cuda::std::enable_if_t<cuda::std::is_unsigned_v<T>>* = nullptr>
CUDF_HOST_DEVICE inline constexpr T multiply_power10_64bit(T value, int exp10)
{
  // See comments in divide_power10_32bit() for discussion.
  switch (exp10) {
    case 0: return value;
    case 1: return value * 10U;
    case 2: return value * 100U;
    case 3: return value * 1000U;
    case 4: return value * 10000U;
    case 5: return value * 100000U;
    case 6: return value * 1000000U;
    case 7: return value * 10000000U;
    case 8: return value * 100000000U;
    case 9: return value * 1000000000U;
    case 10: return value * 10000000000ULL;
    case 11: return value * 100000000000ULL;
    case 12: return value * 1000000000000ULL;
    case 13: return value * 10000000000000ULL;
    case 14: return value * 100000000000000ULL;
    case 15: return value * 1000000000000000ULL;
    case 16: return value * 10000000000000000ULL;
    case 17: return value * 100000000000000000ULL;
    case 18: return value * 1000000000000000000ULL;
    case 19: return value * 10000000000000000000ULL;
    default: return 0;
  }
}

/**
 * @brief Multiply by a power of 10 that fits within a 128bit integer.
 *
 * @tparam T Type of value to be multiplied.
 * @param value The number to be multiplied.
 * @param exp10 The power-of-10 of the multiplier, from 0 to 38 inclusive.
 * @return Returns value * 10^exp10.
 */
template <typename T, typename cuda::std::enable_if_t<cuda::std::is_unsigned_v<T>>* = nullptr>
CUDF_HOST_DEVICE inline constexpr T multiply_power10_128bit(T value, int exp10)
{
  // See comments in divide_power10_128bit() for discussion.
  switch (exp10) {
    case 0: return value;
    case 1: return value * 10U;
    case 2: return value * 100U;
    case 3: return value * 1000U;
    case 4: return value * 10000U;
    case 5: return value * 100000U;
    case 6: return value * 1000000U;
    case 7: return value * 10000000U;
    case 8: return value * 100000000U;
    case 9: return value * 1000000000U;
    case 10: return value * 10000000000ULL;
    case 11: return value * 100000000000ULL;
    case 12: return value * 1000000000000ULL;
    case 13: return value * 10000000000000ULL;
    case 14: return value * 100000000000000ULL;
    case 15: return value * 1000000000000000ULL;
    case 16: return value * 10000000000000000ULL;
    case 17: return value * 100000000000000000ULL;
    case 18: return value * 1000000000000000000ULL;
    case 19: return value * 10000000000000000000ULL;
    case 20: return value * large_power_of_10<20>();
    case 21: return value * large_power_of_10<21>();
    case 22: return value * large_power_of_10<22>();
    case 23: return value * large_power_of_10<23>();
    case 24: return value * large_power_of_10<24>();
    case 25: return value * large_power_of_10<25>();
    case 26: return value * large_power_of_10<26>();
    case 27: return value * large_power_of_10<27>();
    case 28: return value * large_power_of_10<28>();
    case 29: return value * large_power_of_10<29>();
    case 30: return value * large_power_of_10<30>();
    case 31: return value * large_power_of_10<31>();
    case 32: return value * large_power_of_10<32>();
    case 33: return value * large_power_of_10<33>();
    case 34: return value * large_power_of_10<34>();
    case 35: return value * large_power_of_10<35>();
    case 36: return value * large_power_of_10<36>();
    case 37: return value * large_power_of_10<37>();
    case 38: return value * large_power_of_10<38>();
    default: return 0;
  }
}

/**
 * @brief Multiply an integer by a power of 10.
 *
 * @note Use this function if you have no a-priori knowledge of what exp10 might be.
 * If you do, prefer calling the bit-size-specific versions
 *
 * @tparam Rep Representation type needed for integer exponentiation
 * @tparam T Integral type of value to be multiplied.
 * @param value The number to be multiplied.
 * @param exp10 The power-of-10 of the multiplier.
 * @return Returns value * 10^exp10
 */
template <typename Rep,
          typename T,
          typename cuda::std::enable_if_t<(cuda::std::is_unsigned_v<T>)>* = nullptr>
CUDF_HOST_DEVICE inline constexpr T multiply_power10(T value, int exp10)
{
  // Use this function if you have no knowledge of what exp10 might be
  // If you do, prefer calling the bit-size-specific versions
  if constexpr (sizeof(Rep) <= 4) {
    return multiply_power10_32bit(value, exp10);
  } else if constexpr (sizeof(Rep) <= 8) {
    return multiply_power10_64bit(value, exp10);
  } else {
    return multiply_power10_128bit(value, exp10);
  }
}

/**
 * @brief Divide an integer by a power of 10.
 *
 * @note Use this function if you have no a-priori knowledge of what exp10 might be.
 * If you do, prefer calling the bit-size-specific versions
 *
 * @tparam Rep Representation type needed for integer exponentiation
 * @tparam T Integral type of value to be divided-from.
 * @param value The number to be divided-from.
 * @param exp10 The power-of-10 of the denominator.
 * @return Returns value / 10^exp10
 */
template <typename Rep,
          typename T,
          typename cuda::std::enable_if_t<(cuda::std::is_unsigned_v<T>)>* = nullptr>
CUDF_HOST_DEVICE inline constexpr T divide_power10(T value, int exp10)
{
  // Use this function if you have no knowledge of what exp10 might be
  // If you do, prefer calling the bit-size-specific versions
  if constexpr (sizeof(Rep) <= 4) {
    return divide_power10_32bit(value, exp10);
  } else if constexpr (sizeof(Rep) <= 8) {
    return divide_power10_64bit(value, exp10);
  } else {
    return divide_power10_128bit(value, exp10);
  }
}

}  // namespace detail

/** @} */  // end of group
}  // namespace numeric
