/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/assert.cuh>
#include <cudf/fixed_point/temporary.hpp>
#include <cudf/types.hpp>

#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <string>

/// `fixed_point` and supporting types
namespace numeric {

/**
 * @addtogroup fixed_point_classes
 * @{
 * @file
 * @brief Class definition for fixed point data type
 */

/// The scale type for fixed_point
enum scale_type : int32_t {};

/**
 * @brief Scoped enumerator to use when constructing `fixed_point`
 *
 * Examples:
 * ```cpp
 * using decimal32 = fixed_point<int32_t, Radix::BASE_10>;
 * using binary64  = fixed_point<int64_t, Radix::BASE_2>;
 * ```
 */
enum class Radix : int32_t { BASE_2 = 2, BASE_10 = 10 };

/**
 * @brief Returns `true` if the representation type is supported by `fixed_point`
 *
 * @tparam T The representation type
 * @return `true` if the type is supported by `fixed_point` implementation
 */
template <typename T>
constexpr inline auto is_supported_representation_type()
{
  return cuda::std::is_same_v<T, int32_t> ||  //
         cuda::std::is_same_v<T, int64_t> ||  //
         cuda::std::is_same_v<T, __int128_t>;
}

/**
 * @brief Returns `true` if the value type is supported for constructing a `fixed_point`
 *
 * @tparam T The construction value type
 * @return `true` if the value type is supported to construct a `fixed_point` type
 */
template <typename T>
constexpr inline auto is_supported_construction_value_type()
{
  return cuda::std::is_integral<T>() || cuda::std::is_floating_point_v<T>;
}

/** @} */  // end of group

// Helper functions for `fixed_point` type
namespace detail {
/**
 * @brief A function for integer exponentiation by squaring
 *
 * https://simple.wikipedia.org/wiki/Exponentiation_by_squaring <br>
 * Note: this is the iterative equivalent of the recursive definition (faster) <br>
 * Quick-bench: http://quick-bench.com/Wg7o7HYQC9FW5M0CO0wQAjSwP_Y
 *
 * @tparam Rep Representation type for return type
 * @tparam Base The base to be exponentiated
 * @param exponent The exponent to be used for exponentiation
 * @return Result of `Base` to the power of `exponent` of type `Rep`
 */
template <typename Rep,
          Radix Base,
          typename T,
          typename cuda::std::enable_if_t<(cuda::std::is_same_v<int32_t, T> &&
                                           cuda::std::is_integral<Rep>()
                                           )>* = nullptr>
CUDF_HOST_DEVICE inline Rep ipow(T exponent)
{
  cudf_assert(exponent >= 0 && "integer exponentiation with negative exponent is not possible.");
  if (exponent == 0) { return static_cast<Rep>(1); }

  auto extra  = static_cast<Rep>(1);
  auto square = static_cast<Rep>(Base);
  while (exponent > 1) {
    if (exponent & 1 /* odd */) {
      extra *= square;
      exponent -= 1;
    }
    exponent /= 2;
    square *= square;
  }
  return square * extra;
}

/** @brief Function that performs a `right shift` scale "times" on the `val`
 *
 * Note: perform this operation when constructing with positive scale
 *
 * @tparam Rep Representation type needed for integer exponentiation
 * @tparam Rad The radix which will act as the base in the exponentiation
 * @tparam T Type for value `val` being shifted and the return type
 * @param val The value being shifted
 * @param scale The amount to shift the value by
 * @return Shifted value of type T
 */
template <typename Rep, Radix Rad, typename T>
CUDF_HOST_DEVICE inline constexpr T right_shift(T const& val, scale_type const& scale)
{
  return val / ipow<Rep, Rad>(static_cast<int32_t>(scale));
}

/** @brief Function that performs a `left shift` scale "times" on the `val`
 *
 * Note: perform this operation when constructing with negative scale
 *
 * @tparam Rep Representation type needed for integer exponentiation
 * @tparam Rad The radix which will act as the base in the exponentiation
 * @tparam T Type for value `val` being shifted and the return type
 * @param val The value being shifted
 * @param scale The amount to shift the value by
 * @return Shifted value of type T
 */
template <typename Rep, Radix Rad, typename T>
CUDF_HOST_DEVICE inline constexpr T left_shift(T const& val, scale_type const& scale)
{
  return val * ipow<Rep, Rad>(static_cast<int32_t>(-scale));
}

/** @brief Function that performs a `right` or `left shift`
 * scale "times" on the `val`
 *
 * Note: Function will call the correct right or left shift based
 * on the sign of `val`
 *
 * @tparam Rep Representation type needed for integer exponentiation
 * @tparam Rad The radix which will act as the base in the exponentiation
 * @tparam T Type for value `val` being shifted and the return type
 * @param val The value being shifted
 * @param scale The amount to shift the value by
 * @return Shifted value of type T
 */
template <typename Rep, Radix Rad, typename T>
CUDF_HOST_DEVICE inline constexpr T shift(T const& val, scale_type const& scale)
{
  if (scale == 0) { return val; }
  if (scale > 0) { return right_shift<Rep, Rad>(val, scale); }
  return left_shift<Rep, Rad>(val, scale);
}

/** @brief Performs (nearly) lossless bit/base-10-digit shifting for floating-point -> integral conversion
 *
 * Note: Intended to only be called by convert_floating_to_integral_base10()
 * Note: Losses will occur if chosen scale factor truncates values
 *
 * @tparam FloatingType The type of floating point object we are converting from
 * @param integer_rep The initial (unsigned) integer representation of the floating point value
 * @param exponent2 The number of powers of 2 to apply to convert from floating point
 * @param exponent10 The number of powers of 10 to apply to reach the desired scale factor
 * @return Shifted integral representation of the floating point value
 */
template <typename FloatingType, typename cuda::std::enable_if_t<
  cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE __uint128_t floating_to_integral_shifting(__uint128_t integer_rep, int exponent2, int exponent10)
{
  // Adapted and modified from: https://github.com/apache/arrow/blob/main/cpp/src/arrow/util/decimal.cc#L90
  // exponent2:  the scale of the float.   Floating value = input value    *  2^exponent2
  // exponent10: the scale of the decimal. Decimal  value = returned value * 10^exponent10

  // To finish converting the input "float" to a base-10 integer, we need to multiply it by 2^exponent2
  // And to represent our number with the desired scale factor, we need to divide it by 10^exponent10

  // However, if we apply these powers all at once, we may under/overflow, or otherwise lose precision
  // Instead apply the powers iteratively: we have up to 53 (double) bits of precision on our input, and
  // 128 bits of space (in __uint128_t) to play with

  // This code is branchy, but all data in a given data column frequently tends to have similar scales:
  // Much of the time all of the threads will take the same branches

  //Test for shortcuts (exponent10 == 0) checked prior to input
  if (exponent2 == 0) {
    //Apply the scale and we're done
    return shift<__uint128_t, Radix::BASE_10>(integer_rep, scale_type(exponent10));
  }

  if ((exponent2 > 0) == (exponent10 < 0)) {
    //These shifts are in the same direction: order doesn't matter, no need to iterate
    //Bit shift first
    (exponent2 >= 0) ? (integer_rep <<= exponent2) : (integer_rep >>= -exponent2);
    
    //Apply the scale and we're done
    return shift<__uint128_t, Radix::BASE_10>(integer_rep, scale_type(exponent10));
  }

  // To uniquely represent double/float need 17 / 9 digits 
  // To represent 10^17 / 10^9, you need 57 / 30 bits
  // +1 decimal digit (4 bits) buffer: cushion for shifting (see below): 61 / 34 bits to represent #
  static constexpr auto num_representation_bits = std::is_same_v<FloatingType, double> ? 61 : 34;

  // Calculate max amount of decimal places we can safely shift at once
  // double/float: 128 bits of room - 61 / 34 for rep = 67 / 94 bits remaining
  // 2^67 / 2^94 is approximately 1.4*10^20 / 2*10^28
  // Thus we can shift up to 20 / 28 decimal places at a time (for double/float) without losing precision

  // It's important to keep these 2^N and 10^M shifts about as equal in magnitude as possible, 
  // or else we can lose precision by overflow or truncation of our bits outside of the 128 bit range
  // 2^10 = ~10^3, so we want to shift by 10 bits for every 3 decimal places we shift
  
  // To simplify the math, round the double/float max-decimal-place shift of 20 / 28 down to 18 / 27
  // 10^18 = ~2^60, and 10^27 = ~2^90, so our max bit shifts 60 / 90 bits for double/float
  static constexpr int max_digits_shift = std::is_same_v<FloatingType, double> ? 18 : 27;
  static constexpr int max_bits_shift = max_digits_shift * 10 / 3; //60, 90

  // Shift input bits so our data is lined up to the top of the lower num_representation_bits bit range
  // Starting with the data at a predefined position simplifies the logic later
  // Why line-up to the top of the range?: Truncate minimal info on intermediary right-shifts
  static constexpr int init_shift = num_representation_bits - std::numeric_limits<FloatingType>::digits; //8, 11
  integer_rep <<= init_shift;
  exponent2 -= init_shift; // We need to apply these powers of 2 later to get our base-10 value

  if (exponent2 > 0) { //exponent10 > 0
    //Alternately multiply by 2's and divide by 10's until both all factors of 2 and 10 are applied
    while (exponent2 > max_bits_shift) {
      //We need to shift more bits than we have room: iterate

      //First shift the max number of bits left (our data is in the low bits): Multiply by 2s
      integer_rep <<= max_bits_shift;
      exponent2 -= max_bits_shift;

      //Then tens-shift our data right: Divide by 10s
      if(exponent10 <= max_digits_shift) {
        //Last 10s-shift: Shift all remaining decimal places, shift all remaining bits, then bail
        integer_rep = right_shift<__uint128_t, Radix::BASE_10>(integer_rep, scale_type(exponent10));
        return integer_rep << exponent2;
      }

      //More decimal places to shift than we have room: Shift the max number of 10s
      integer_rep = right_shift<__uint128_t, Radix::BASE_10>(integer_rep, scale_type(max_digits_shift));
      exponent10 -= max_digits_shift;
    }

    //Last bit-shift: Shift all remaining bits, apply the remaining scale, then bail
    integer_rep <<= exponent2;
    //NOTE: If you don't want to truncate information, round here (user-supplied scale may be too high)
    return right_shift<__uint128_t, Radix::BASE_10>(integer_rep, scale_type(exponent10));

  } else { //exponent2 < 0 & exponent10 < 0
    //Alternately multiply by 10s and divide by 2s until both all factors of 2 and 10 are applied
    int exponent2_magnitude = -exponent2;
    while (-exponent10 > max_digits_shift) {
      //We need to shift more tens than we have room: iterate

      //First shift the max number of tens left (our data is in the low bits): Multiply by 10s
      integer_rep = left_shift<__uint128_t, Radix::BASE_10>(integer_rep, scale_type(-max_digits_shift));
      exponent10 += max_digits_shift;

      //Then bit shift our data right: Divide by 2s
      if(exponent2_magnitude <= max_bits_shift) {
        //Last bit-shift: Shift all remaining bits, apply the remaining scale, then bail
        integer_rep >>= exponent2_magnitude;
        return left_shift<__uint128_t, Radix::BASE_10>(integer_rep, scale_type(exponent10));
      }

      //More bits to shift than we have room: Shift the max number of 2s
      integer_rep >>= max_bits_shift;
      exponent2_magnitude -= max_bits_shift;
    }

    //Last 10s-shift: Shift all remaining decimal places, shift all remaining bits, then bail
    integer_rep = left_shift<__uint128_t, Radix::BASE_10>(integer_rep, scale_type(exponent10));
    //NOTE: If you don't want to truncate information, round here (user-supplied scale may be too high)
    return (integer_rep >> exponent2_magnitude);
  }
}

/** @brief Performs (nearly) lossless bit/base-10-digit shifting for floating-point -> integral conversion
 *
 * Note: Intended to only be called by convert_floating_to_integral()
 * Note: Losses will occur if chosen scale factor truncates values
 *
 * @tparam Rep The type of integer we are converting to to store the value
 * @tparam FloatingType The type of floating point object we are converting from
 * @param input The floating point value to convert
 * @param scale The desired base-10 scale factor for the resulting integer: value = returned-int * 10^scale
 * @return Integral representation of the floating point value, given the desired scale. 
 */
template <typename Rep, typename FloatingType, typename cuda::std::enable_if_t<
  cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE Rep convert_floating_to_integral_base10(FloatingType const& input, scale_type const& scale)
{
  //Shortcut: If no scale to apply, then cast and bail
  auto const casted_input = static_cast<Rep>(input);
  if(scale == 0) {
    return casted_input;
  }

  //Shortcut: If input is whole number representable by an int (magnitude not too large): scale and bail
  if(static_cast<FloatingType>(casted_input) == input) {
    return shift<Rep, Radix::BASE_10>(casted_input, scale);
  }

  //Extract mantissa and exponent from floating point value
  int exponent2 = 0;
  FloatingType significand; //magnitude btw 0.5 & 1
  if constexpr (std::is_same_v<FloatingType, float>) {
    significand = frexpf(input, &exponent2);
  } else {
    significand = frexp(input, &exponent2);
  }
  
  //If scale is such that we lose precision anyway, no need for exact result: fall back on old algorithm
  //However, old algorithm only works if 10^scale is representable in Rep
  auto const approximate_input_scale = exponent2 * 3/10; //2^10 = ~10^3
  static constexpr auto shortcut_precision_limit = std::numeric_limits<FloatingType>::digits10 - 1; //5, 14
  auto const int_scale = static_cast<int32_t>(scale);
  bool const is_conversion_safe = (numeric::detail::abs(int_scale) < std::numeric_limits<Rep>::digits10);
  if(is_conversion_safe && (approximate_input_scale - int_scale <= shortcut_precision_limit)) {
    return shift<Rep, Radix::BASE_10>(input, scale);
  }

  // Multiply significand by 2^M (M = #mantissa_bits) to get an exact (unsigned) integer.
  // Ignoring sign for now due to following bit shifts
  static constexpr auto num_mantissa_bits = std::numeric_limits<FloatingType>::digits; //significand bits + 1
  FloatingType shifted_mantissa;
  if constexpr (std::is_same_v<FloatingType, float>) {
    shifted_mantissa = ldexpf(significand, num_mantissa_bits);
  } else {
    shifted_mantissa = ldexp(significand, num_mantissa_bits);
  }
  auto const sign_term = (input < 0) ? -1 : 1;
  auto const integer_rep = static_cast<__uint128_t>(sign_term * shifted_mantissa); 
  
  // Figure out how many remaining powers of 2 need to be applied
  int const remaining_powers2 = exponent2 - num_mantissa_bits;

  // Shift data by powers of 2 and 10 as needed
  auto const magnitude = floating_to_integral_shifting<FloatingType>(integer_rep, remaining_powers2, int_scale);

  // Reapply the sign and return
  return sign_term * static_cast<Rep>(magnitude);
}

/** @brief Converts floating point -> integral, given the desired scale factor
 *
 * Note: For base-10, uses the new (nearly) lossless conversion routine, else falls back on older routine
 *
 * @tparam Rep The type of integer we are converting to to store the value
 * @tparam Rad The base of the scale factor
 * @tparam FloatingType The type of floating point object we are converting from
 * @param value The floating point value to convert
 * @param scale The desired scale factor for the resulting integer: value = returned-int * Rad^scale
 * @return Integral representation of the floating point value, given the desired scale. 
 */
template <typename Rep, Radix Rad, typename FloatingType, typename cuda::std::enable_if_t<
  cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE inline Rep convert_floating_to_integral(FloatingType const& value, scale_type const& scale)
{
  if constexpr (Rad == Radix::BASE_10) {
    return static_cast<Rep>(convert_floating_to_integral_base10<Rep, FloatingType>(value, scale));
  } else { //Fall back on old algorithm
    return static_cast<Rep>(shift<Rep, Rad>(value, scale));
  }
}

/** @brief Determine the number of leading zeroes for 128bit unsigned integers
 *
 * @param value The (unsigned) integer whose bits are being counted
 * @return The number of leading zeroes in the integer, with range from 0 -> 128 inclusive
 */
CUDF_HOST_DEVICE inline int count_leading_zeroes(__uint128_t value)
{
#ifdef __CUDA_ARCH__
  auto const high_bits = static_cast<int64_t>(value >> 64);
  return __clzll(high_bits) + (high_bits == 0) * __clzll(static_cast<int64_t>(value));
#else
  //Undefined behavior to call __builtin_clzll() with zero in gcc and clang
  if(value == 0)
    return 128;

  auto const high_bits = static_cast<uint64_t>(value >> 64);
  if (high_bits == 0)
    return 64 + __builtin_clzll(static_cast<uint64_t>(value)); //low bits
  else
    return __builtin_clzll(high_bits);
#endif
}

/** @brief Performs lossless bit/base-10-digit shifting for integral -> floating-point conversion
 *
 * Note: Intended to only be called by convert_integral_to_floating()
 *
 * @tparam FloatingType The type of floating point object we are converting to
 * @param integer_rep The integer to convert
 * @param exponent10 The number of powers of 10 to apply to undo the scale factor
 * @return Floating-point representation of the scaled integral value
 */
template <typename FloatingType, typename cuda::std::enable_if_t<
  cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE FloatingType integral_to_floating_shifting(__uint128_t integer_rep, int exponent10)
{
  //This is the reverse algorithm of floating_to_integral_shifting(), see discussion there for more details

  //Shortcut: If can losslessly apply 10s, do that, cast to float, and bail
  if(numeric::detail::abs(exponent10) <= std::numeric_limits<__uint128_t>::digits10) { //guard against overflow
    if(exponent10 > 0) {
      //Lossless if (integer_rep * n) / n == integer_rep
      auto const n = ipow<__uint128_t, Radix::BASE_10>(exponent10);
      auto const shifted = integer_rep * n;
      if(integer_rep == (shifted / n))
        return static_cast<FloatingType>(shifted);
    } else {
      //Lossless if (integer_rep / n) * n == integer_rep
      auto const n = ipow<__uint128_t, Radix::BASE_10>(-exponent10);
      auto const shifted = integer_rep / n;
      if(integer_rep == (shifted * n))
        return static_cast<FloatingType>(shifted);
    }
  }

  //Control variables (see discussion in floating_to_integral_shifting())
  static constexpr auto num_representation_bits = std::is_same_v<FloatingType, double> ? 61 : 34;
  static constexpr int max_digits_shift = std::is_same_v<FloatingType, double> ? 18 : 27;
  static constexpr int max_bits_shift = max_digits_shift * 10 / 3; //60, 90

  //Shift input so that our data is lined up to the top of the lower num_representation_bits bit range
  //Starting with the data at a predefined position simplifies the logic later
  //If this truncates bits, it was more than we could store in our floating point value anyway
  //Why line-up to the top of the range: don't truncate info on right-shifts
  int const num_leading_zeros = count_leading_zeroes(integer_rep);
  int exponent2 = 128 - num_representation_bits - num_leading_zeros;
  (exponent2 >= 0) ? (integer_rep >>= exponent2) : (integer_rep <<= -exponent2);

  if (exponent10 > 0) {
    //Alternately multiply by 10s and divide by 2s until all factors of 10 are applied
    while(exponent10 > max_digits_shift) {
      //We need to shift more tens than we have room: iterate

      //First shift the max number of tens left (our data is in the low bits): Multiply by 10s
      integer_rep = left_shift<__uint128_t, Radix::BASE_10>(integer_rep, scale_type(-max_digits_shift));
      exponent10 -= max_digits_shift;

      //Then bit shift our data right: Divide by max # of 2s
      integer_rep >>= max_bits_shift;
      exponent2 += max_bits_shift;
    }

    //Last 10s-shift: Shift all remaining decimal places
    integer_rep = left_shift<__uint128_t, Radix::BASE_10>(integer_rep, scale_type(-exponent10));

  } else { //exponent10 < 0
    //Alternately divide by 10s and multiply by 2s until all factors of 10 are applied
    int exponent10_magnitude = -exponent10;
    do {
      //Max bit shift left to give us the most room for shifting 10s: Multiply by 2s
      integer_rep <<= max_bits_shift;
      exponent2 -= max_bits_shift;

      //Now tens-shift our data right: Divide by 10s
      if(exponent10_magnitude <= max_digits_shift) {
        //Last 10s-shift: Shift all remaining decimal places
        integer_rep = right_shift<__uint128_t, Radix::BASE_10>(integer_rep, scale_type(exponent10_magnitude));
        break;
      }

      //More decimal places to shift than we have room: Shift the max number of 10s
      integer_rep = right_shift<__uint128_t, Radix::BASE_10>(integer_rep, scale_type(max_digits_shift));
      exponent10_magnitude -= max_digits_shift;
    } while (true);
  }

  //Done with applying scale factor: cast to floating point
  auto const floating = static_cast<FloatingType>(integer_rep); //IEEE-754 rounds to even

  //Apply the remaining powers of 2 and return
  if constexpr (std::is_same_v<FloatingType, float>) {
    return ldexpf(floating, exponent2);
  } else {
    return ldexp(floating, exponent2);
  }
}

/** @brief Performs lossless bit/base-10-digit shifting for integral -> floating-point conversion
 *
 * @tparam FloatingType The type of floating point object we are converting to
 * @tparam Rep The type of integer we are converting from
 * @param value The integer to convert
 * @param scale The base-10 scale factor for the input integer
 * @return Floating-point representation of the scaled integral value
 */
template <typename FloatingType, typename Rep, typename cuda::std::enable_if_t<
  cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE inline FloatingType convert_integral_to_floating(Rep const& value, scale_type const& scale)
{
  //Shortcut: If no scale to apply, cast and we're done
  if (scale == 0) {
    return static_cast<FloatingType>(value);
  }

  //Convert to unsigned for bit shifting
  auto const sign_term = (value >= 0) ? 1 : -1;
  auto const unsigned_value = static_cast<__uint128_t>(sign_term*value);

  //Shift by powers of 2 and 10, cast to float, reapply sign
  return sign_term * integral_to_floating_shifting<FloatingType>(unsigned_value, static_cast<int32_t>(scale));
}

}  // namespace detail

/**
 * @addtogroup fixed_point_classes
 * @{
 * @file
 * @brief Class definition for fixed point data type
 */

/**
 * @brief Helper struct for constructing `fixed_point` when value is already shifted
 *
 * Example:
 * ```cpp
 * using decimal32 = fixed_point<int32_t, Radix::BASE_10>;
 * auto n = decimal32{scaled_integer{1001, 3}}; // n = 1.001
 * ```
 *
 * @tparam Rep The representation type (either `int32_t` or `int64_t`)
 */
template <typename Rep,
          typename cuda::std::enable_if_t<is_supported_representation_type<Rep>()>* = nullptr>
struct scaled_integer {
  Rep value;         ///< The value of the fixed point number
  scale_type scale;  ///< The scale of the value
  /**
   * @brief Constructor for `scaled_integer`
   *
   * @param v The value of the fixed point number
   * @param s The scale of the value
   */
  CUDF_HOST_DEVICE inline explicit scaled_integer(Rep v, scale_type s) : value{v}, scale{s} {}
};

/**
 * @brief A type for representing a number with a fixed amount of precision
 *
 * Currently, only binary and decimal `fixed_point` numbers are supported.
 * Binary operations can only be performed with other `fixed_point` numbers
 *
 * @tparam Rep The representation type (either `int32_t` or `int64_t`)
 * @tparam Rad The radix/base (either `Radix::BASE_2` or `Radix::BASE_10`)
 */
template <typename Rep, Radix Rad>
class fixed_point {
  Rep _value{};
  scale_type _scale;

 public:
  using rep = Rep;  ///< The representation type

  /**
   * @brief Constructor that will perform shifting to store value appropriately (from floating point
   * types)
   *
   * @tparam T The floating point type that you are constructing from
   * @param value The value that will be constructed from
   * @param scale The exponent that is applied to Rad to perform shifting
   */
  template <typename T,
            typename cuda::std::enable_if_t<cuda::std::is_floating_point<T>() &&
                                            is_supported_representation_type<Rep>()>* = nullptr>
  CUDF_HOST_DEVICE inline explicit fixed_point(T const& value, scale_type const& scale)
    : _value{detail::convert_floating_to_integral<Rep, Rad>(value, scale)}, _scale{scale}
  {
  }

  /**
   * @brief Constructor that will perform shifting to store value appropriately (from integral
   * types)
   *
   * @tparam T The integral type that you are constructing from
   * @param value The value that will be constructed from
   * @param scale The exponent that is applied to Rad to perform shifting
   */
  template <typename T,
            typename cuda::std::enable_if_t<cuda::std::is_integral<T>() &&
                                            is_supported_representation_type<Rep>()>* = nullptr>
  CUDF_HOST_DEVICE inline explicit fixed_point(T const& value, scale_type const& scale)
    // `value` is cast to `Rep` to avoid overflow in cases where
    // constructing to `Rep` that is wider than `T`
    : _value{detail::shift<Rep, Rad>(static_cast<Rep>(value), scale)}, _scale{scale}
  {
  }

  /**
   * @brief Constructor that will not perform shifting (assumes value already shifted)
   *
   * @param s scaled_integer that contains scale and already shifted value
   */
  CUDF_HOST_DEVICE inline explicit fixed_point(scaled_integer<Rep> s)
    : _value{s.value}, _scale{s.scale}
  {
  }

  /**
   * @brief "Scale-less" constructor that constructs `fixed_point` number with a specified
   * value and scale of zero
   *
   * @tparam T The value type being constructing from
   * @param value The value that will be constructed from
   */
  template <typename T,
            typename cuda::std::enable_if_t<is_supported_construction_value_type<T>()>* = nullptr>
  CUDF_HOST_DEVICE inline fixed_point(T const& value)
    : _value{static_cast<Rep>(value)}, _scale{scale_type{0}}
  {
  }

  /**
   * @brief Default constructor that constructs `fixed_point` number with a
   * value and scale of zero
   */
  CUDF_HOST_DEVICE inline fixed_point() : _scale{scale_type{0}} {}

  /**
   * @brief Explicit conversion operator for casting to floating point types
   *
   * @tparam U The floating point type that is being explicitly converted to
   * @return The `fixed_point` number in base 10 (aka human readable format)
   */
  template <typename U,
            typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<U>>* = nullptr>
  explicit constexpr operator U() const
  {
    if constexpr (Rad == Radix::BASE_10)
      return detail::convert_integral_to_floating<U>(_value, _scale);
    else
      return detail::shift<Rep, Rad>(static_cast<U>(_value), scale_type{-_scale});
  }

  /**
   * @brief Explicit conversion operator for casting to integral types
   *
   * @tparam U The integral type that is being explicitly converted to
   * @return The `fixed_point` number in base 10 (aka human readable format)
   */
  template <typename U, typename cuda::std::enable_if_t<cuda::std::is_integral_v<U>>* = nullptr>
  explicit constexpr operator U() const
  {
    // Cast to the larger of the two types (of U and Rep) before converting to Rep because in
    // certain cases casting to U before shifting will result in integer overflow (i.e. if U =
    // int32_t, Rep = int64_t and _value > 2 billion)
    auto const value = std::common_type_t<U, Rep>(_value);
    return static_cast<U>(detail::shift<Rep, Rad>(value, scale_type{-_scale}));
  }

  /**
   * @brief Converts the `fixed_point` number to a `scaled_integer`
   *
   * @return The `scaled_integer` representation of the `fixed_point` number
   */
  CUDF_HOST_DEVICE inline operator scaled_integer<Rep>() const
  {
    return scaled_integer<Rep>{_value, _scale};
  }

  /**
   * @brief Method that returns the underlying value of the `fixed_point` number
   *
   * @return The underlying value of the `fixed_point` number
   */
  CUDF_HOST_DEVICE inline rep value() const { return _value; }

  /**
   * @brief Method that returns the scale of the `fixed_point` number
   *
   * @return The scale of the `fixed_point` number
   */
  CUDF_HOST_DEVICE inline scale_type scale() const { return _scale; }

  /**
   * @brief Explicit conversion operator to `bool`
   *
   * @return The `fixed_point` value as a boolean (zero is `false`, nonzero is `true`)
   */
  CUDF_HOST_DEVICE inline explicit constexpr operator bool() const
  {
    return static_cast<bool>(_value);
  }

  /**
   * @brief operator +=
   *
   * @tparam Rep1 Representation type of the operand `rhs`
   * @tparam Rad1 Radix (base) type of the operand `rhs`
   * @param rhs The number being added to `this`
   * @return The sum
   */
  template <typename Rep1, Radix Rad1>
  CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1>& operator+=(fixed_point<Rep1, Rad1> const& rhs)
  {
    *this = *this + rhs;
    return *this;
  }

  /**
   * @brief operator *=
   *
   * @tparam Rep1 Representation type of the operand `rhs`
   * @tparam Rad1 Radix (base) type of the operand `rhs`
   * @param rhs The number being multiplied to `this`
   * @return The product
   */
  template <typename Rep1, Radix Rad1>
  CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1>& operator*=(fixed_point<Rep1, Rad1> const& rhs)
  {
    *this = *this * rhs;
    return *this;
  }

  /**
   * @brief operator -=
   *
   * @tparam Rep1 Representation type of the operand `rhs`
   * @tparam Rad1 Radix (base) type of the operand `rhs`
   * @param rhs The number being subtracted from `this`
   * @return The difference
   */
  template <typename Rep1, Radix Rad1>
  CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1>& operator-=(fixed_point<Rep1, Rad1> const& rhs)
  {
    *this = *this - rhs;
    return *this;
  }

  /**
   * @brief operator /=
   *
   * @tparam Rep1 Representation type of the operand `rhs`
   * @tparam Rad1 Radix (base) type of the operand `rhs`
   * @param rhs The number being divided from `this`
   * @return The quotient
   */
  template <typename Rep1, Radix Rad1>
  CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1>& operator/=(fixed_point<Rep1, Rad1> const& rhs)
  {
    *this = *this / rhs;
    return *this;
  }

  /**
   * @brief operator ++ (post-increment)
   *
   * @return The incremented result
   */
  CUDF_HOST_DEVICE inline fixed_point<Rep, Rad>& operator++()
  {
    *this = *this + fixed_point<Rep, Rad>{1, scale_type{_scale}};
    return *this;
  }

  /**
   * @brief operator + (for adding two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are added.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are added.
   *
   * @tparam Rep1 Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1 Radix (base) type of the operand `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return The resulting `fixed_point` sum
   */
  template <typename Rep1, Radix Rad1>
  CUDF_HOST_DEVICE inline friend fixed_point<Rep1, Rad1> operator+(
    fixed_point<Rep1, Rad1> const& lhs, fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator - (for subtracting two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are subtracted.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are subtracted.
   *
   * @tparam Rep1 Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1 Radix (base) type of the operand `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return The resulting `fixed_point` difference
   */
  template <typename Rep1, Radix Rad1>
  CUDF_HOST_DEVICE inline friend fixed_point<Rep1, Rad1> operator-(
    fixed_point<Rep1, Rad1> const& lhs, fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator * (for multiplying two `fixed_point` numbers)
   *
   * `_scale`s are added and `_value`s are multiplied.
   *
   * @tparam Rep1 Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1 Radix (base) type of the operand `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return The resulting `fixed_point` product
   */
  template <typename Rep1, Radix Rad1>
  CUDF_HOST_DEVICE inline friend fixed_point<Rep1, Rad1> operator*(
    fixed_point<Rep1, Rad1> const& lhs, fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator / (for dividing two `fixed_point` numbers)
   *
   * `_scale`s are subtracted and `_value`s are divided.
   *
   * @tparam Rep1 Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1 Radix (base) type of the operand `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return The resulting `fixed_point` quotient
   */
  template <typename Rep1, Radix Rad1>
  CUDF_HOST_DEVICE inline friend fixed_point<Rep1, Rad1> operator/(
    fixed_point<Rep1, Rad1> const& lhs, fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator % (for computing the modulo operation of two `fixed_point` numbers)
   *
   * If `_scale`s are equal, the modulus is computed directly.
   * If `_scale`s are not equal, the number with larger `_scale` is shifted to the
   * smaller `_scale`, and then the modulus is computed.
   *
   * @tparam Rep1 Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1 Radix (base) type of the operand `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return The resulting `fixed_point` number
   */
  template <typename Rep1, Radix Rad1>
  CUDF_HOST_DEVICE inline friend fixed_point<Rep1, Rad1> operator%(
    fixed_point<Rep1, Rad1> const& lhs, fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator == (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are compared.
   *
   * @tparam Rep1 Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1 Radix (base) type of the operand `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return true if `lhs` and `rhs` are equal, false if not
   */
  template <typename Rep1, Radix Rad1>
  CUDF_HOST_DEVICE inline friend bool operator==(fixed_point<Rep1, Rad1> const& lhs,
                                                 fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator != (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are compared.
   *
   * @tparam Rep1 Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1 Radix (base) type of the operand `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return true if `lhs` and `rhs` are not equal, false if not
   */
  template <typename Rep1, Radix Rad1>
  CUDF_HOST_DEVICE inline friend bool operator!=(fixed_point<Rep1, Rad1> const& lhs,
                                                 fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator <= (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are compared.
   *
   * @tparam Rep1 Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1 Radix (base) type of the operand `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return true if `lhs` less than or equal to `rhs`, false if not
   */
  template <typename Rep1, Radix Rad1>
  CUDF_HOST_DEVICE inline friend bool operator<=(fixed_point<Rep1, Rad1> const& lhs,
                                                 fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator >= (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are compared.
   *
   * @tparam Rep1 Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1 Radix (base) type of the operand `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return true if `lhs` greater than or equal to `rhs`, false if not
   */
  template <typename Rep1, Radix Rad1>
  CUDF_HOST_DEVICE inline friend bool operator>=(fixed_point<Rep1, Rad1> const& lhs,
                                                 fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator < (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are compared.
   *
   * @tparam Rep1 Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1 Radix (base) type of the operand `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return true if `lhs` less than `rhs`, false if not
   */
  template <typename Rep1, Radix Rad1>
  CUDF_HOST_DEVICE inline friend bool operator<(fixed_point<Rep1, Rad1> const& lhs,
                                                fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator > (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are compared.
   *
   * @tparam Rep1 Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1 Radix (base) type of the operand `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return true if `lhs` greater than `rhs`, false if not
   */
  template <typename Rep1, Radix Rad1>
  CUDF_HOST_DEVICE inline friend bool operator>(fixed_point<Rep1, Rad1> const& lhs,
                                                fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief Method for creating a `fixed_point` number with a new `scale`
   *
   * The `fixed_point` number returned will have the same value, underlying representation and
   * radix as `this`, the only thing changed is the scale.
   *
   * @param scale The `scale` of the returned `fixed_point` number
   * @return `fixed_point` number with a new `scale`
   */
  CUDF_HOST_DEVICE inline fixed_point<Rep, Rad> rescaled(scale_type scale) const
  {
    if (scale == _scale) { return *this; }
    Rep const value = detail::shift<Rep, Rad>(_value, scale_type{scale - _scale});
    return fixed_point<Rep, Rad>{scaled_integer<Rep>{value, scale}};
  }

  /**
   * @brief Returns a string representation of the fixed_point value.
   */
  explicit operator std::string() const
  {
    if (_scale < 0) {
      auto const av = detail::abs(_value);
      Rep const n   = detail::exp10<Rep>(-_scale);
      Rep const f   = av % n;
      auto const num_zeros =
        std::max(0, (-_scale - static_cast<int32_t>(detail::to_string(f).size())));
      auto const zeros = std::string(num_zeros, '0');
      auto const sign  = _value < 0 ? std::string("-") : std::string();
      return sign + detail::to_string(av / n) + std::string(".") + zeros +
             detail::to_string(av % n);
    }
    auto const zeros = std::string(_scale, '0');
    return detail::to_string(_value) + zeros;
  }
};

/**
 *  @brief Function for identifying integer overflow when adding
 *
 * @tparam Rep Type of integer to check for overflow on
 * @tparam T Types of lhs and rhs (ensures they are the same type)
 * @param lhs Left hand side of addition
 * @param rhs Right hand side of addition
 * @return true if addition causes overflow, false otherwise
 */
template <typename Rep, typename T>
CUDF_HOST_DEVICE inline auto addition_overflow(T lhs, T rhs)
{
  return rhs > 0 ? lhs > cuda::std::numeric_limits<Rep>::max() - rhs
                 : lhs < cuda::std::numeric_limits<Rep>::min() - rhs;
}

/** @brief Function for identifying integer overflow when subtracting
 *
 * @tparam Rep Type of integer to check for overflow on
 * @tparam T Types of lhs and rhs (ensures they are the same type)
 * @param lhs Left hand side of subtraction
 * @param rhs Right hand side of subtraction
 * @return true if subtraction causes overflow, false otherwise
 */
template <typename Rep, typename T>
CUDF_HOST_DEVICE inline auto subtraction_overflow(T lhs, T rhs)
{
  return rhs > 0 ? lhs < cuda::std::numeric_limits<Rep>::min() + rhs
                 : lhs > cuda::std::numeric_limits<Rep>::max() + rhs;
}

/** @brief Function for identifying integer overflow when dividing
 *
 * @tparam Rep Type of integer to check for overflow on
 * @tparam T Types of lhs and rhs (ensures they are the same type)
 * @param lhs Left hand side of division
 * @param rhs Right hand side of division
 * @return true if division causes overflow, false otherwise
 */
template <typename Rep, typename T>
CUDF_HOST_DEVICE inline auto division_overflow(T lhs, T rhs)
{
  return lhs == cuda::std::numeric_limits<Rep>::min() && rhs == -1;
}

/** @brief Function for identifying integer overflow when multiplying
 *
 * @tparam Rep Type of integer to check for overflow on
 * @tparam T Types of lhs and rhs (ensures they are the same type)
 * @param lhs Left hand side of multiplication
 * @param rhs Right hand side of multiplication
 * @return true if multiplication causes overflow, false otherwise
 */
template <typename Rep, typename T>
CUDF_HOST_DEVICE inline auto multiplication_overflow(T lhs, T rhs)
{
  auto const min = cuda::std::numeric_limits<Rep>::min();
  auto const max = cuda::std::numeric_limits<Rep>::max();
  if (rhs > 0) { return lhs > max / rhs || lhs < min / rhs; }
  if (rhs < -1) { return lhs > min / rhs || lhs < max / rhs; }
  return rhs == -1 && lhs == min;
}

// PLUS Operation
template <typename Rep1, Radix Rad1>
CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1> operator+(fixed_point<Rep1, Rad1> const& lhs,
                                                          fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  auto const sum   = lhs.rescaled(scale)._value + rhs.rescaled(scale)._value;

#if defined(__CUDACC_DEBUG__)

  assert(!addition_overflow<Rep1>(lhs.rescaled(scale)._value, rhs.rescaled(scale)._value) &&
         "fixed_point overflow");

#endif

  return fixed_point<Rep1, Rad1>{scaled_integer<Rep1>{sum, scale}};
}

// MINUS Operation
template <typename Rep1, Radix Rad1>
CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1> operator-(fixed_point<Rep1, Rad1> const& lhs,
                                                          fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  auto const diff  = lhs.rescaled(scale)._value - rhs.rescaled(scale)._value;

#if defined(__CUDACC_DEBUG__)

  assert(!subtraction_overflow<Rep1>(lhs.rescaled(scale)._value, rhs.rescaled(scale)._value) &&
         "fixed_point overflow");

#endif

  return fixed_point<Rep1, Rad1>{scaled_integer<Rep1>{diff, scale}};
}

// MULTIPLIES Operation
template <typename Rep1, Radix Rad1>
CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1> operator*(fixed_point<Rep1, Rad1> const& lhs,
                                                          fixed_point<Rep1, Rad1> const& rhs)
{
#if defined(__CUDACC_DEBUG__)

  assert(!multiplication_overflow<Rep1>(lhs._value, rhs._value) && "fixed_point overflow");

#endif

  return fixed_point<Rep1, Rad1>{
    scaled_integer<Rep1>(lhs._value * rhs._value, scale_type{lhs._scale + rhs._scale})};
}

// DIVISION Operation
template <typename Rep1, Radix Rad1>
CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1> operator/(fixed_point<Rep1, Rad1> const& lhs,
                                                          fixed_point<Rep1, Rad1> const& rhs)
{
#if defined(__CUDACC_DEBUG__)

  assert(!division_overflow<Rep1>(lhs._value, rhs._value) && "fixed_point overflow");

#endif

  return fixed_point<Rep1, Rad1>{
    scaled_integer<Rep1>(lhs._value / rhs._value, scale_type{lhs._scale - rhs._scale})};
}

// EQUALITY COMPARISON Operation
template <typename Rep1, Radix Rad1>
CUDF_HOST_DEVICE inline bool operator==(fixed_point<Rep1, Rad1> const& lhs,
                                        fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value == rhs.rescaled(scale)._value;
}

// EQUALITY NOT COMPARISON Operation
template <typename Rep1, Radix Rad1>
CUDF_HOST_DEVICE inline bool operator!=(fixed_point<Rep1, Rad1> const& lhs,
                                        fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value != rhs.rescaled(scale)._value;
}

// LESS THAN OR EQUAL TO Operation
template <typename Rep1, Radix Rad1>
CUDF_HOST_DEVICE inline bool operator<=(fixed_point<Rep1, Rad1> const& lhs,
                                        fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value <= rhs.rescaled(scale)._value;
}

// GREATER THAN OR EQUAL TO Operation
template <typename Rep1, Radix Rad1>
CUDF_HOST_DEVICE inline bool operator>=(fixed_point<Rep1, Rad1> const& lhs,
                                        fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value >= rhs.rescaled(scale)._value;
}

// LESS THAN Operation
template <typename Rep1, Radix Rad1>
CUDF_HOST_DEVICE inline bool operator<(fixed_point<Rep1, Rad1> const& lhs,
                                       fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value < rhs.rescaled(scale)._value;
}

// GREATER THAN Operation
template <typename Rep1, Radix Rad1>
CUDF_HOST_DEVICE inline bool operator>(fixed_point<Rep1, Rad1> const& lhs,
                                       fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value > rhs.rescaled(scale)._value;
}

// MODULO OPERATION
template <typename Rep1, Radix Rad1>
CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1> operator%(fixed_point<Rep1, Rad1> const& lhs,
                                                          fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale     = std::min(lhs._scale, rhs._scale);
  auto const remainder = lhs.rescaled(scale)._value % rhs.rescaled(scale)._value;
  return fixed_point<Rep1, Rad1>{scaled_integer<Rep1>{remainder, scale}};
}

using decimal32  = fixed_point<int32_t, Radix::BASE_10>;     ///<  32-bit decimal fixed point
using decimal64  = fixed_point<int64_t, Radix::BASE_10>;     ///<  64-bit decimal fixed point
using decimal128 = fixed_point<__int128_t, Radix::BASE_10>;  ///< 128-bit decimal fixed point

/** @} */  // end of group
}  // namespace numeric
