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
#include <cuda/std/utility>

#include <string.h>

#include <algorithm>
#include <cassert>
#include <cmath>
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
 * @brief Helper struct containing common constants for setting and extracting
 * the components of a floating point value.
 */
template <typename FloatingType,
          typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
struct FloatingComponentConstants {
  // This struct assumes we're working with IEEE 754 floating point values.
  // Details on the IEEE-754 floating point format:
  // Format: https://learn.microsoft.com/en-us/cpp/build/ieee-floating-point-representation
  // Float Visualizer: https://www.h-schmidt.net/FloatConverter/IEEE754.html
  static_assert(cuda::std::numeric_limits<FloatingType>::is_iec559, "Assumes IEEE 754");

  static constexpr bool is_float = cuda::std::is_same_v<FloatingType, float>;  ///< Is-float or not
  using IntegralType =
    cuda::std::conditional_t<is_float, uint32_t, uint64_t>;  ///< Unsigned int type with same size
                                                             ///< as floating type

  // The high bit is the sign bit (0 for positive #"s, 1 for negative #'s).
  static constexpr int num_floating_bits =
    sizeof(FloatingType) * CHAR_BIT;  ///< How many bits in the floating type
  static constexpr int sign_bit_index = num_floating_bits - 1;  ///< The index of the sign bit
  static constexpr IntegralType sign_mask =
    (IntegralType(1) << sign_bit_index);  ///< The mask to select the sign bit

  // The low 23 / 52 bits (for float / double) are the mantissa.
  // There is an additional bit of value 1 that isn't stored in the mantissa.
  // Instead it is "understood" that the value of the mantissa is between 0.5 and 1.
  static constexpr int num_mantissa_bits =
    cuda::std::numeric_limits<FloatingType>::digits - 1;  ///< -1 for understood bit
  static constexpr IntegralType understood_bit_mask =
    (IntegralType(1) << num_mantissa_bits);  ///< The mask for the understood bit
  static constexpr IntegralType mantissa_mask =
    understood_bit_mask - 1;  ///< The mask to select the mantissa

  // And in between are the bits used to store the biased power-of-2 exponent.
  static constexpr int num_exponent_bits =
    num_floating_bits - num_mantissa_bits - 1;  ///< -1: sign bit
  static constexpr IntegralType unshifted_exponent_mask =
    (IntegralType(1) << num_exponent_bits) - 1;  ///< The mask for the exponents, unshifted
  static constexpr IntegralType exponent_mask =
    unshifted_exponent_mask << num_mantissa_bits;  ///< The mask to select the exponents

  // To store positive and negative exponents as unsigned values, the stored value for
  // the power-of-2 is exponent + bias. The bias is 126 for floats and 1022 for doubles.
  static constexpr IntegralType exponent_bias =
    cuda::std::numeric_limits<FloatingType>::max_exponent - 2;  ///< 126 / 1022 for float / double
};

/**
 * @brief A function for integer exponentiation by squaring.
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
                                           is_supported_representation_type<Rep>())>* = nullptr>
CUDF_HOST_DEVICE inline Rep ipow(T exponent)
{
  cudf_assert(exponent >= 0 && "integer exponentiation with negative exponent is not possible.");

  if constexpr (Base == numeric::Radix::BASE_2) { return static_cast<Rep>(1) << exponent; }

  // Note: Including an array here introduces too much register pressure
  // https://simple.wikipedia.org/wiki/Exponentiation_by_squaring
  // This is the iterative equivalent of the recursive definition (faster)
  // Quick-bench for squaring: http://quick-bench.com/Wg7o7HYQC9FW5M0CO0wQAjSwP_Y
  if (exponent == 0) { return static_cast<Rep>(1); }
  auto extra  = static_cast<Rep>(1);
  auto square = static_cast<Rep>(Base);
  while (exponent > 1) {
    if (exponent & 1) { extra *= square; }
    exponent >>= 1;
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

/** @brief Extracts the sign, exponent, and integral significand of a floating point number.
 *
 * @note This returns all zero's for +/-0 and denormals, and zero's with an exponent of
 * INT_MIN for the unrepresentable values +/-inf and NaN's.
 *
 * @tparam FloatingType Type of input floating point value
 * @param floating The floating value to extract the components from
 * @return A tuple containing the sign (bool), exponent (power-of-2), and integral significand.
 */
template <typename FloatingType,
          typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE inline auto extract_components(FloatingType floating)
{
  // This function is implemented using integer and bit operations, which is much
  // faster than using the FPU functions for each component individually. This is especially
  // true for doubles, as bottlenecking on the FP64 GPU pipeline can cut performance in half.

  // See comments in FloatingComponentConstants about the floating point format.
  using Constants    = FloatingComponentConstants<FloatingType>;
  using IntegralType = typename Constants::IntegralType;

  // Convert floating to integer
  auto const integer_rep = [&]() {
    IntegralType integer;
    memcpy(&integer, &floating, sizeof(floating));
    return integer;
  }();

  // First extract the exponent bits and handle its special values
  auto const exponent_bits = integer_rep & Constants::exponent_mask;
  if (exponent_bits == 0) {
    // Because of the understood set-bit not stored in the mantissa, it is not possible
    // to store the value zero directly. Instead both +/-0 and denormals are represented with
    // the exponent bits set to zero.
    // Thus it's fastest to just floor (generally unwanted) denormals to zero.
    return cuda::std::tuple{false, 0, IntegralType(0)};
  } else if (exponent_bits == Constants::exponent_mask) {
    //+/-inf and NaN values are stored with all of the exponent bits set.
    // As none of these are representable by integers, we'll return the same value for all cases.
    return cuda::std::tuple{false, INT_MIN, IntegralType(0)};
  }

  // Extract the exponent value: shift the bits down and subtract the bias.
  using SignedIntegralType                       = cuda::std::make_signed_t<IntegralType>;
  SignedIntegralType const shifted_exponent_bits = exponent_bits >> Constants::num_mantissa_bits;
  int const exp2 =
    shifted_exponent_bits - static_cast<SignedIntegralType>(Constants::exponent_bias);

  // Extract the sign bit:
  bool const is_positive = ((Constants::sign_mask & integer_rep) == 0);

  // Extract the significand, setting the high bit for the understood 1/2
  IntegralType const significand =
    (integer_rep & Constants::mantissa_mask) | Constants::understood_bit_mask;

  // Return the tuple of values
  return cuda::std::tuple{is_positive, exp2, significand};
}

/** @brief Sets the sign and adds to the exponent of a floating point number.
 *
 * @tparam FloatingType Type of input floating point value.
 * @param floating The floating value to set the components of.
 * @param is_negative The sign bit to set for the floating point number.
 * @param exp2 The power-of-2 to add to the floating point number.
 * @return A floating point value with the mantissa of the original, but the added sign and
 * exponent.
 */
template <typename FloatingType,
          typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<FloatingType>>* = nullptr>
CUDF_HOST_DEVICE inline FloatingType add_sign_and_exp2(FloatingType floating,
                                                       bool is_negative,
                                                       int exp2)
{
  // Because IEEE-754 mandates rounding when casting an integer to floating point,
  // the fastest way to set the mantissa is to cast, even though it uses the FPU.
  // However, the fastest way to set the sign and exponent is still to do so manually,
  // especially for doubles as bottlenecking on the FP64 pipeline is very painful.

  // See comments in FloatingComponentConstants about the floating point format.
  using Constants    = FloatingComponentConstants<FloatingType>;
  using IntegralType = typename Constants::IntegralType;

  // Convert floating to integer
  IntegralType integer_rep;
  memcpy(&integer_rep, &floating, sizeof(floating));

  // Set the sign bit
  integer_rep |= (IntegralType(is_negative) << Constants::sign_bit_index);

  // Extract the currently stored (biased) exponent
  auto exponent_bits = integer_rep & Constants::exponent_mask;
  auto stored_exp2   = exponent_bits >> Constants::num_mantissa_bits;

  // Add the additional power-of-2
  stored_exp2 += exp2;
  exponent_bits = stored_exp2 << Constants::num_mantissa_bits;

  // Clear existing exponent bits and set new ones
  integer_rep &= (~Constants::exponent_mask);
  integer_rep |= exponent_bits;

  // Convert back to float
  memcpy(&floating, &integer_rep, sizeof(floating));
  return floating;
}

/** @brief Determine the number of significant bits in an integer
 *
 * @tparam T Type of input integer value.
 * @param value The integer whose bits are being counted
 * @return The number of significant bits: the # of bits - # of leading zeroes
 */
template <typename T, typename cuda::std::enable_if_t<(cuda::std::is_unsigned_v<T>)>* = nullptr>
CUDF_HOST_DEVICE inline int count_significant_bits(T value)
{
#ifdef __CUDA_ARCH__
  if constexpr (std::is_same_v<T, uint64_t>) {
    return 64 - __clzll(static_cast<int64_t>(value));
  } else if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    return 32 - __clz(static_cast<int32_t>(value));
  } else {
    // 128 bit type, must break u[ into high and low components
    auto const high_bits = static_cast<int64_t>(value >> 64);
    auto const low_bits  = static_cast<int64_t>(value);
    return 128 - (__clzll(high_bits) + int(high_bits == 0) * __clzll(low_bits));
  }
#else
  // Undefined behavior to call __builtin_clzll() with zero in gcc and clang
  if (value == 0) { return 0; }

  if constexpr (std::is_same_v<T, uint64_t>) {
    return 64 - __builtin_clzll(value);
  } else if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    return 32 - __builtin_clz(value);
  } else {
    // 128 bit type, must break u[ into high and low components
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
    : _value{static_cast<Rep>(detail::shift<Rep, Rad>(value, scale))}, _scale{scale}
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
