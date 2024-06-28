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

#include <cudf/utilities/traits.hpp>

#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cstring>

namespace numeric {

/**
 * @addtogroup floating_conversion
 * @{
 * @file
 * @brief fixed_point <--> floating-point conversion functions.
 */

namespace detail {

/**
 * @brief Helper struct for getting and setting the components of a floating-point value
 *
 * @tparam FloatingType Type of floating-point value
 */
template <typename FloatingType, CUDF_ENABLE_IF(cuda::std::is_floating_point_v<FloatingType>)>
struct floating_converter {
  // This struct assumes we're working with IEEE 754 floating-point values.
  // Details on the IEEE-754 floating-point format:
  // Format: https://learn.microsoft.com/en-us/cpp/build/ieee-floating-point-representation
  // Float Visualizer: https://www.h-schmidt.net/FloatConverter/IEEE754.html
  static_assert(cuda::std::numeric_limits<FloatingType>::is_iec559, "Assumes IEEE 754");

  /// Unsigned int type with same size as floating type
  using IntegralType =
    cuda::std::conditional_t<cuda::std::is_same_v<FloatingType, float>, uint32_t, uint64_t>;

  // The high bit is the sign bit (0 for positive, 1 for negative).
  /// How many bits in the floating type
  static constexpr int num_floating_bits = sizeof(FloatingType) * CHAR_BIT;
  /// The index of the sign bit
  static constexpr int sign_bit_index = num_floating_bits - 1;
  /// The mask to select the sign bit
  static constexpr IntegralType sign_mask = (IntegralType(1) << sign_bit_index);

  // The low 23 / 52 bits (for float / double) are the mantissa.
  // The mantissa is normalized. There is an understood 1 bit to the left of the binary point.
  // The value of the mantissa is in the range [1, 2).
  /// # mantissa bits (-1 for understood bit)
  static constexpr int num_mantissa_bits = cuda::std::numeric_limits<FloatingType>::digits - 1;
  /// The mask for the understood bit
  static constexpr IntegralType understood_bit_mask = (IntegralType(1) << num_mantissa_bits);
  /// The mask to select the mantissa
  static constexpr IntegralType mantissa_mask = understood_bit_mask - 1;

  // And in between are the bits used to store the biased power-of-2 exponent.
  /// # exponents bits (-1 for sign bit)
  static constexpr int num_exponent_bits = num_floating_bits - num_mantissa_bits - 1;
  /// The mask for the exponents, unshifted
  static constexpr IntegralType unshifted_exponent_mask =
    (IntegralType(1) << num_exponent_bits) - 1;
  /// The mask to select the exponents
  static constexpr IntegralType exponent_mask = unshifted_exponent_mask << num_mantissa_bits;

  // To store positive and negative exponents as unsigned values, the stored value for
  // the power-of-2 is exponent + bias. The bias is 127 for floats and 1023 for doubles.
  /// 127 / 1023 for float / double
  static constexpr IntegralType exponent_bias =
    cuda::std::numeric_limits<FloatingType>::max_exponent - 1;

  /**
   * @brief Reinterpret the bits of a floating-point value as an integer
   *
   * @param floating The floating-point value to cast
   * @return An integer with bits identical to the input
   */
  CUDF_HOST_DEVICE inline static IntegralType bit_cast_to_integer(FloatingType floating)
  {
    // Convert floating to integer
    IntegralType integer_rep;
    memcpy(&integer_rep, &floating, sizeof(floating));
    return integer_rep;
  }

  /**
   * @brief Reinterpret the bits of an integer as floating-point value
   *
   * @param integer The integer to cast
   * @return A floating-point value with bits identical to the input
   */
  CUDF_HOST_DEVICE inline static FloatingType bit_cast_to_floating(IntegralType integer)
  {
    // Convert back to float
    FloatingType floating;
    memcpy(&floating, &integer, sizeof(floating));
    return floating;
  }

  /**
   * @brief Extracts the integral significand of a bit-casted floating-point number
   *
   * @param integer_rep The bit-casted floating value to extract the exponent from
   * @return The integral significand, bit-shifted to a (large) whole number
   */
  CUDF_HOST_DEVICE inline static IntegralType get_base2_value(IntegralType integer_rep)
  {
    // Extract the significand, setting the high bit for the understood 1/2
    return (integer_rep & mantissa_mask) | understood_bit_mask;
  }

  /**
   * @brief Extracts the sign bit of a bit-casted floating-point number
   *
   * @param integer_rep The bit-casted floating value to extract the exponent from
   * @return The sign bit
   */
  CUDF_HOST_DEVICE inline static bool get_is_negative(IntegralType integer_rep)
  {
    // Extract the sign bit:
    return static_cast<bool>(sign_mask & integer_rep);
  }

  /**
   * @brief Extracts the exponent of a bit-casted floating-point number
   *
   * @note This returns INT_MIN for +/-0, +/-inf, NaN's, and denormals
   * For all of these cases, the decimal fixed_point number should be set to zero
   *
   * @param integer_rep The bit-casted floating value to extract the exponent from
   * @return The stored base-2 exponent, or INT_MIN for special values
   */
  CUDF_HOST_DEVICE inline static int get_exp2(IntegralType integer_rep)
  {
    // First extract the exponent bits and handle its special values.
    // To minimize branching, all of these special cases will return INT_MIN.
    // For all of these cases, the decimal fixed_point number should be set to zero.
    auto const exponent_bits = integer_rep & exponent_mask;
    if (exponent_bits == 0) {
      // Because of the understood set-bit not stored in the mantissa, it is not possible
      // to store the value zero directly. Instead both +/-0 and denormals are represented with
      // the exponent bits set to zero.
      // Thus it's fastest to just floor (generally unwanted) denormals to zero.
      return INT_MIN;
    } else if (exponent_bits == exponent_mask) {
      //+/-inf and NaN values are stored with all of the exponent bits set.
      // As none of these are representable by integers, we'll return the same value for all cases.
      return INT_MIN;
    }

    // Extract the exponent value: shift the bits down and subtract the bias.
    using SignedIntegralType                       = cuda::std::make_signed_t<IntegralType>;
    SignedIntegralType const shifted_exponent_bits = exponent_bits >> num_mantissa_bits;
    return shifted_exponent_bits - static_cast<SignedIntegralType>(exponent_bias);
  }

  /**
   * @brief Sets the sign bit of a positive floating-point number
   *
   * @param floating The floating-point value to set the sign of. Must be positive.
   * @param is_negative The sign bit to set for the floating-point number
   * @return The input floating-point value with the chosen sign
   */
  CUDF_HOST_DEVICE inline static FloatingType set_is_negative(FloatingType floating,
                                                              bool is_negative)
  {
    // Convert floating to integer
    IntegralType integer_rep = bit_cast_to_integer(floating);

    // Set the sign bit. Note that the input floating-point number must be positive (bit = 0).
    integer_rep |= (IntegralType(is_negative) << sign_bit_index);

    // Convert back to float
    return bit_cast_to_floating(integer_rep);
  }

  /**
   * @brief Adds to the base-2 exponent of a floating-point number
   *
   * @param floating The floating value to add to the exponent of. Must be positive.
   * @param exp2 The power-of-2 to add to the floating-point number
   * @return The input floating-point value * 2^exp2
   */
  CUDF_HOST_DEVICE inline static FloatingType add_exp2(FloatingType floating, int exp2)
  {
    // Convert floating to integer
    auto integer_rep = bit_cast_to_integer(floating);

    // Extract the currently stored (biased) exponent
    auto exponent_bits = integer_rep & exponent_mask;
    auto stored_exp2   = exponent_bits >> num_mantissa_bits;

    // Add the additional power-of-2
    stored_exp2 += exp2;

    // Check for exponent over/under-flow.
    // Note that the input floating-point number is always positive, so we don't have to
    // worry about the sign here; the sign will be set later in set_is_negative()
    if (stored_exp2 <= 0) {
      return 0.0;
    } else if (stored_exp2 >= unshifted_exponent_mask) {
      return cuda::std::numeric_limits<FloatingType>::infinity();
    } else {
      // Clear existing exponent bits and set new ones
      exponent_bits = stored_exp2 << num_mantissa_bits;
      integer_rep &= (~exponent_mask);
      integer_rep |= exponent_bits;

      // Convert back to float
      return bit_cast_to_floating(integer_rep);
    }
  }
};

/**
 * @brief Determine the number of significant bits in an integer
 *
 * @tparam T Type of input integer value. Must be either uint32_t, uint64_t, or __uint128_t
 * @param value The integer whose bits are being counted
 * @return The number of significant bits: the # of bits - # of leading zeroes
 */
template <typename T,
          CUDF_ENABLE_IF(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
                         std::is_same_v<T, __uint128_t>)>
CUDF_HOST_DEVICE inline int count_significant_bits(T value)
{
#ifdef __CUDA_ARCH__
  if constexpr (std::is_same_v<T, uint64_t>) {
    return 64 - __clzll(static_cast<int64_t>(value));
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return 32 - __clz(static_cast<int32_t>(value));
  } else if constexpr (std::is_same_v<T, __uint128_t>) {
    // 128 bit type, must break up into high and low components
    auto const high_bits = static_cast<int64_t>(value >> 64);
    auto const low_bits  = static_cast<int64_t>(value);
    return 128 - (__clzll(high_bits) + static_cast<int>(high_bits == 0) * __clzll(low_bits));
  }
#else
  // Undefined behavior to call __builtin_clzll() with zero in gcc and clang
  if (value == 0) { return 0; }

  if constexpr (std::is_same_v<T, uint64_t>) {
    return 64 - __builtin_clzll(value);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return 32 - __builtin_clz(value);
  } else if constexpr (std::is_same_v<T, __uint128_t>) {
    // 128 bit type, must break up into high and low components
    auto const high_bits = static_cast<uint64_t>(value >> 64);
    if (high_bits == 0) {
      return 64 - __builtin_clzll(static_cast<uint64_t>(value));
    } else {
      return 128 - __builtin_clzll(high_bits);
    }
  }
#endif
}

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
