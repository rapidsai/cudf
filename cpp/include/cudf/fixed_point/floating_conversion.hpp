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
  // the power-of-2 is exponent + bias. The bias is 126 for floats and 1022 for doubles.
  /// 126 / 1022 for float / double
  static constexpr IntegralType exponent_bias =
    cuda::std::numeric_limits<FloatingType>::max_exponent - 2;

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
   * @param floating The floating value to add to the exponent of
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
    exponent_bits = stored_exp2 << num_mantissa_bits;

    // Clear existing exponent bits and set new ones
    integer_rep &= (~exponent_mask);
    integer_rep |= exponent_bits;

    // Convert back to float
    return bit_cast_to_floating(integer_rep);
  }
};

/**
 * @brief Determine the number of significant bits in an integer
 *
 * @tparam T Type of input integer value. Must be either uint32_t, uint64_t, or __uint128_t
 * @param value The integer whose bits are being counted
 * @return The number of significant bits: the # of bits - # of leading zeroes
 */
template <typename T, CUDF_ENABLE_IF(cuda::std::is_unsigned_v<T>)>
CUDF_HOST_DEVICE inline int count_significant_bits(T value)
{
  static_assert(
    std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> || std::is_same_v<T, __uint128_t>,
    "Unimplemented type");

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

}  // namespace detail

/** @} */  // end of group
}  // namespace numeric
