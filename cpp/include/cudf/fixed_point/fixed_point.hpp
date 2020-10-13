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

#define _LIBCUDACXX_USE_CXX17_TYPE_TRAITS

// Note: The <simt/*> versions are used in order for Jitify to work with our fixed_point type.
//       Jitify is needed for several algorithms (binaryop, rolling, etc)
#include <simt/limits>
#include <simt/type_traits>  // add simt namespace

#include <algorithm>
#include <cassert>
#include <cmath>

#include <string>

//! `fixed_point` and supporting types
namespace numeric {
/** \cond HIDDEN_SYMBOLS */
// This is a wrapper struct that enforces "strong typing"
// at the construction site of the type. No implicit
// conversions will be allowed and you will need to use the
// name of the type alias (i.e. scale_type{0})
template <typename T>
struct strong_typedef {
  T _t;
  CUDA_HOST_DEVICE_CALLABLE explicit constexpr strong_typedef(T t) : _t(t) {}
  CUDA_HOST_DEVICE_CALLABLE operator T() const { return _t; }
};
/** \endcond */

using scale_type = strong_typedef<int32_t>;

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

template <typename T>
constexpr inline auto is_supported_representation_type()
{
  return simt::std::is_same<T, int32_t>::value || simt::std::is_same<T, int64_t>::value;
}

template <typename T>
constexpr inline auto is_supported_construction_value_type()
{
  return simt::std::is_integral<T>::value || simt::std::is_floating_point<T>::value;
}

// Helper functions for `fixed_point` type
namespace detail {
/**
 * @brief A function for integer exponentiation by squaring
 *
 * https://simple.wikipedia.org/wiki/Exponentiation_by_squaring <br>
 * Note: this is the iterative equivalent of the recursive definition (faster) <br>
 * Quick-bench: http://quick-bench.com/Wg7o7HYQC9FW5M0CO0wQAjSwP_Y <br>
 * `exponent` comes from `using scale_type = strong_typedef<int32_t>` <br>
 *
 * @tparam Rep Representation type for return type
 * @tparam Base The base to be exponentiated
 * @param exponent The exponent to be used for exponentiation
 * @return Result of `Base` to the power of `exponent` of type `Rep`
 */
template <typename Rep,
          Radix Base,
          typename T,
          typename simt::std::enable_if_t<(simt::std::is_same<int32_t, T>::value &&
                                           is_supported_representation_type<Rep>())>* = nullptr>
CUDA_HOST_DEVICE_CALLABLE Rep ipow(T exponent)
{
  if (exponent == 0) return static_cast<Rep>(1);
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

/** @brief Helper function to negate strongly typed scale_type
 *
 * @param scale The scale to be negated
 * @return The negated scale
 */
CUDA_HOST_DEVICE_CALLABLE
auto negate(scale_type const& scale) { return scale_type{-scale}; }

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
CUDA_HOST_DEVICE_CALLABLE constexpr T right_shift(T const& val, scale_type const& scale)
{
  return val / ipow<Rep, Rad>(scale._t);
}

/** @brief Function that performs a rounding `right shift` scale "times" on the `val`
 *
 * The scaled integer equivalent of 0.5 is added to the value before truncating such that
 * any remaining fractional part will be rounded away from zero.
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
CUDA_HOST_DEVICE_CALLABLE constexpr T right_shift_rounded(T const& val, scale_type const& scale)
{
  Rep const factor = ipow<Rep, Rad>(scale._t);
  Rep const half   = factor / 2;
  return (val >= 0 ? val + half : val - half) / factor;
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
CUDA_HOST_DEVICE_CALLABLE constexpr T left_shift(T const& val, scale_type const& scale)
{
  return val * ipow<Rep, Rad>(-scale._t);
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
CUDA_HOST_DEVICE_CALLABLE constexpr T shift(T const& val, scale_type const& scale)
{
  if (scale == 0)
    return val;
  else if (scale > 0)
    return right_shift<Rep, Rad>(val, scale);
  else
    return left_shift<Rep, Rad>(val, scale);
}

/** @brief Function that performs precise shift to avoid "lossiness"
 * inherent in floating point values
 *
 * Example: `auto n = fixed_point<int32_t, Radix::BASE_10>{1.001, scale_type{-3}}`
 * will construct n to have a value of 1 without the precise shift
 *
 * @tparam Rep Representation type needed for integer exponentiation
 * @tparam Rad The radix which will act as the base in the exponentiation
 * @tparam T Type for value `val` being shifted and the return type
 * @param value The value being shifted
 * @param scale The amount to shift the value by
 * @return Shifted value of type T
 */
template <typename Rep,
          Radix Rad,
          typename T,
          typename simt::std::enable_if_t<simt::std::is_integral<T>::value>* = nullptr>
CUDA_HOST_DEVICE_CALLABLE auto shift_with_precise_round(T const& value, scale_type const& scale)
  -> Rep
{
  if (scale == 0)
    return value;
  else if (scale > 0)
    return right_shift_rounded<Rep, Rad>(value, scale);
  else
    return left_shift<Rep, Rad>(value, scale);
}

/** @brief Function that performs precise shift to avoid "lossiness"
 * inherent in floating point values
 *
 * Example: `auto n = fixed_point<int32_t, Radix::BASE_10>{1.001, scale_type{-3}}`
 * will construct n to have a value of 1 without the precise shift
 *
 * @tparam Rep Representation type needed for integer exponentiation
 * @tparam Rad The radix which will act as the base in the exponentiation
 * @tparam T Type for value `val` being shifted and the return type
 * @param value The value being shifted
 * @param scale The amount to shift the value by
 * @return Shifted value of type T
 */
template <typename Rep,
          Radix Rad,
          typename T,
          typename simt::std::enable_if_t<simt::std::is_floating_point<T>::value>* = nullptr>
CUDA_HOST_DEVICE_CALLABLE auto shift_with_precise_round(T const& value, scale_type const& scale)
  -> Rep
{
  if (scale == 0) return value;
  T const factor = ipow<int64_t, Rad>(std::abs(scale));
  return std::roundf(scale <= 0 ? value * factor : value / factor);
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
          typename simt::std::enable_if_t<is_supported_representation_type<Rep>()>* = nullptr>
struct scaled_integer {
  Rep value;
  scale_type scale;
  CUDA_HOST_DEVICE_CALLABLE explicit scaled_integer(Rep v, scale_type s) : value(v), scale(s) {}
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
  Rep _value;
  scale_type _scale;

 public:
  using rep = Rep;

  /**
   * @brief Constructor that will perform shifting to store value appropriately
   *
   * @tparam T The type that you are constructing from (integral or floating)
   * @param value The value that will be constructed from
   * @param scale The exponent that is applied to Rad to perform shifting
   */
  template <typename T,
            typename simt::std::enable_if_t<is_supported_construction_value_type<T>() &&
                                            is_supported_representation_type<Rep>()>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE explicit fixed_point(T const& value, scale_type const& scale)
    : _value{detail::shift_with_precise_round<Rep, Rad>(value, scale)}, _scale{scale}
  {
  }

  /**
   * @brief Constructor that will not perform shifting (assumes value already
   * shifted)
   *
   * @param s scaled_integer that contains scale and already shifted value
   */
  CUDA_HOST_DEVICE_CALLABLE
  explicit fixed_point(scaled_integer<Rep> s) : _value{s.value}, _scale{s.scale} {}

  /**
   * @brief "Scale-less" constructor that constructs `fixed_point` number with a specified
   * value and scale of zero
   */
  template <typename T,
            typename simt::std::enable_if_t<is_supported_construction_value_type<T>()>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE fixed_point(T const& value)
    : _value{static_cast<Rep>(value)}, _scale{scale_type{0}}
  {
  }

  /**
   * @brief Default constructor that constructs `fixed_point` number with a
   * value and scale of zero
   */
  CUDA_HOST_DEVICE_CALLABLE
  fixed_point() : _value{0}, _scale{scale_type{0}} {}

  /**
   * @brief Explicit conversion operator
   *
   * @tparam U The type that is being explicitly converted to (integral or floating)
   * @return The `fixed_point` number in base 10 (aka human readable format)
   */
  template <typename U,
            typename simt::std::enable_if_t<is_supported_construction_value_type<U>()>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE explicit constexpr operator U() const
  {
    return detail::shift<Rep, Rad>(static_cast<U>(_value), detail::negate(_scale));
  }

  CUDA_HOST_DEVICE_CALLABLE operator scaled_integer<Rep>() const
  {
    return scaled_integer<Rep>{_value, _scale};
  }

  /**
   * @brief Explicit conversion operator to `bool`
   *
   * @return The `fixed_point` value as a boolean (zero is `false`, nonzero is `true`)
   */
  CUDA_HOST_DEVICE_CALLABLE explicit constexpr operator bool() const
  {
    return static_cast<bool>(_value);
  }

  /**
   * @brief operator +=
   *
   * @tparam Rep1 Representation type of number being added to `this`
   * @tparam Rad1 Radix (base) type of number being added to `this`
   * @return The sum
   */
  template <typename Rep1, Radix Rad1>
  CUDA_HOST_DEVICE_CALLABLE fixed_point<Rep1, Rad1>& operator+=(fixed_point<Rep1, Rad1> const& rhs)
  {
    *this = *this + rhs;
    return *this;
  }

  /**
   * @brief operator *=
   *
   * @tparam Rep1 Representation type of number being added to `this`
   * @tparam Rad1 Radix (base) type of number being added to `this`
   * @return The product
   */
  template <typename Rep1, Radix Rad1>
  CUDA_HOST_DEVICE_CALLABLE fixed_point<Rep1, Rad1>& operator*=(fixed_point<Rep1, Rad1> const& rhs)
  {
    *this = *this * rhs;
    return *this;
  }

  /**
   * @brief operator -=
   *
   * @tparam Rep1 Representation type of number being added to `this`
   * @tparam Rad1 Radix (base) type of number being added to `this`
   * @return The difference
   */
  template <typename Rep1, Radix Rad1>
  CUDA_HOST_DEVICE_CALLABLE fixed_point<Rep1, Rad1>& operator-=(fixed_point<Rep1, Rad1> const& rhs)
  {
    *this = *this - rhs;
    return *this;
  }

  /**
   * @brief operator /=
   *
   * @tparam Rep1 Representation type of number being added to `this`
   * @tparam Rad1 Radix (base) type of number being added to `this`
   * @return The quotient
   */
  template <typename Rep1, Radix Rad1>
  CUDA_HOST_DEVICE_CALLABLE fixed_point<Rep1, Rad1>& operator/=(fixed_point<Rep1, Rad1> const& rhs)
  {
    *this = *this / rhs;
    return *this;
  }

  /**
   * @brief operator ++ (post-increment)
   *
   * @return The incremented result
   */
  CUDA_HOST_DEVICE_CALLABLE
  fixed_point<Rep, Rad>& operator++()
  {
    *this = *this + fixed_point<Rep, Rad>{1, scale_type{_scale}};
    return *this;
  }

  /**
   * @brief operator + (for adding two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are added <br>
   * If `_scale`s are not equal, number with smaller `_scale` is shifted to the
   * greater `_scale`, and then `_value`s are added
   *
   * @tparam Rep1 Representation type of number being added to `this`
   * @tparam Rad1 Radix (base) type of number being added to `this`
   * @return The resulting `fixed_point` sum
   */
  template <typename Rep1, Radix Rad1>
  CUDA_HOST_DEVICE_CALLABLE friend fixed_point<Rep1, Rad1> operator+(
    fixed_point<Rep1, Rad1> const& lhs, fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator - (for subtracting two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are subtracted <br>
   * If `_scale`s are not equal, number with smaller `_scale` is shifted to the
   * greater `_scale`, and then `_value`s are subtracted
   *
   * @tparam Rep1 Representation type of number being added to `this`
   * @tparam Rad1 Radix (base) type of number being added to `this`
   * @return The resulting `fixed_point` difference
   */
  template <typename Rep1, Radix Rad1>
  CUDA_HOST_DEVICE_CALLABLE friend fixed_point<Rep1, Rad1> operator-(
    fixed_point<Rep1, Rad1> const& lhs, fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator * (for multiplying two `fixed_point` numbers)
   *
   * `_scale`s are added and `_value`s are multiplied
   *
   * @tparam Rep1 Representation type of number being added to `this`
   * @tparam Rad1 Radix (base) type of number being added to `this`
   * @return The resulting `fixed_point` product
   */
  template <typename Rep1, Radix Rad1>
  CUDA_HOST_DEVICE_CALLABLE friend fixed_point<Rep1, Rad1> operator*(
    fixed_point<Rep1, Rad1> const& lhs, fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator / (for dividing two `fixed_point` numbers)
   *
   * `_scale`s are subtracted and `_value`s are divided
   *
   * @tparam Rep1 Representation type of number being added to `this`
   * @tparam Rad1 Radix (base) type of number being added to `this`
   * @return The resulting `fixed_point` quotient
   */
  template <typename Rep1, Radix Rad1>
  CUDA_HOST_DEVICE_CALLABLE friend fixed_point<Rep1, Rad1> operator/(
    fixed_point<Rep1, Rad1> const& lhs, fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator == (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared <br>
   * If `_scale`s are not equal, number with smaller `_scale` is shifted to the
   * greater `_scale`, and then `_value`s are compared
   *
   * @tparam Rep1 Representation type of number being added to `this`
   * @tparam Rad1 Radix (base) type of number being added to `this`
   * @return true if `lhs` and `rhs` are equal, false if not
   */
  template <typename Rep1, Radix Rad1>
  CUDA_HOST_DEVICE_CALLABLE friend bool operator==(fixed_point<Rep1, Rad1> const& lhs,
                                                   fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator != (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared <br>
   * If `_scale`s are not equal, number with smaller `_scale` is shifted to the
   * greater `_scale`, and then `_value`s are compared
   *
   * @tparam Rep1 Representation type of number being added to `this`
   * @tparam Rad1 Radix (base) type of number being added to `this`
   * @return true if `lhs` and `rhs` are not equal, false if not
   */
  template <typename Rep1, Radix Rad1>
  CUDA_HOST_DEVICE_CALLABLE friend bool operator!=(fixed_point<Rep1, Rad1> const& lhs,
                                                   fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator <= (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared <br>
   * If `_scale`s are not equal, number with smaller `_scale` is shifted to the
   * greater `_scale`, and then `_value`s are compared
   *
   * @tparam Rep1 Representation type of number being added to `this`
   * @tparam Rad1 Radix (base) type of number being added to `this`
   * @return true if `lhs` less than or equal to `rhs`, false if not
   */
  template <typename Rep1, Radix Rad1>
  CUDA_HOST_DEVICE_CALLABLE friend bool operator<=(fixed_point<Rep1, Rad1> const& lhs,
                                                   fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator >= (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared <br>
   * If `_scale`s are not equal, number with smaller `_scale` is shifted to the
   * greater `_scale`, and then `_value`s are compared
   *
   * @tparam Rep1 Representation type of number being added to `this`
   * @tparam Rad1 Radix (base) type of number being added to `this`
   * @return true if `lhs` greater than or equal to `rhs`, false if not
   */
  template <typename Rep1, Radix Rad1>
  CUDA_HOST_DEVICE_CALLABLE friend bool operator>=(fixed_point<Rep1, Rad1> const& lhs,
                                                   fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator < (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared <br>
   * If `_scale`s are not equal, number with smaller `_scale` is shifted to the
   * greater `_scale`, and then `_value`s are compared
   *
   * @tparam Rep1 Representation type of number being added to `this`
   * @tparam Rad1 Radix (base) type of number being added to `this`
   * @return true if `lhs` less than `rhs`, false if not
   */
  template <typename Rep1, Radix Rad1>
  CUDA_HOST_DEVICE_CALLABLE friend bool operator<(fixed_point<Rep1, Rad1> const& lhs,
                                                  fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief operator > (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared <br>
   * If `_scale`s are not equal, number with smaller `_scale` is shifted to the
   * greater `_scale`, and then `_value`s are compared
   *
   * @tparam Rep1 Representation type of number being added to `this`
   * @tparam Rad1 Radix (base) type of number being added to `this`
   * @return true if `lhs` greater than `rhs`, false if not
   */
  template <typename Rep1, Radix Rad1>
  CUDA_HOST_DEVICE_CALLABLE friend bool operator>(fixed_point<Rep1, Rad1> const& lhs,
                                                  fixed_point<Rep1, Rad1> const& rhs);

  /**
   * @brief Method for creating a `fixed_point` number with a new `scale`
   *
   * The `fixed_point` number returned will have the same value, underlying representation and
   * radix as `this`, the only thing changed is the scale
   *
   * @param scale The `scale` of the returned `fixed_point` number
   * @return `fixed_point` number with a new `scale`
   */
  CUDA_HOST_DEVICE_CALLABLE fixed_point<Rep, Rad> rescaled(scale_type scale) const
  {
    if (scale == _scale) return *this;
    Rep const value =
      detail::shift_with_precise_round<Rep, Rad>(_value, scale_type{scale - _scale});
    return fixed_point<Rep, Rad>{scaled_integer<Rep>{value, scale}};
  }
};  // namespace numeric

/** @brief Function that converts Rep to `std::string`
 *
 * @tparam Rep Representation type
 * @return String-ified Rep
 */
template <typename Rep>
std::string print_rep()
{
  if (simt::std::is_same<Rep, int32_t>::value)
    return "int32_t";
  else if (simt::std::is_same<Rep, int64_t>::value)
    return "int64_t";
  else
    return "unknown type";
}

/** @brief Function for identifying integer overflow when adding
 *
 * @tparam Rep Type of integer to check for overflow on
 * @tparam T Types of lhs and rhs (ensures they are the same type)
 * @param lhs Left hand side of addition
 * @param rhs Right hand side of addition
 * @return true if addition causes overflow, false otherwise
 */
template <typename Rep, typename T>
CUDA_HOST_DEVICE_CALLABLE auto addition_overflow(T lhs, T rhs)
{
  return rhs > 0 ? lhs > simt::std::numeric_limits<Rep>::max() - rhs
                 : lhs < simt::std::numeric_limits<Rep>::min() - rhs;
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
CUDA_HOST_DEVICE_CALLABLE auto subtraction_overflow(T lhs, T rhs)
{
  return rhs > 0 ? lhs < simt::std::numeric_limits<Rep>::min() + rhs
                 : lhs > simt::std::numeric_limits<Rep>::max() + rhs;
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
CUDA_HOST_DEVICE_CALLABLE auto division_overflow(T lhs, T rhs)
{
  return lhs == simt::std::numeric_limits<Rep>::min() && rhs == -1;
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
CUDA_HOST_DEVICE_CALLABLE auto multiplication_overflow(T lhs, T rhs)
{
  auto const min = simt::std::numeric_limits<Rep>::min();
  auto const max = simt::std::numeric_limits<Rep>::max();
  if (rhs > 0)
    return lhs > max / rhs || lhs < min / rhs;
  else if (rhs < -1)
    return lhs > min / rhs || lhs < max / rhs;
  else
    return rhs == -1 && lhs == min;
}

// PLUS Operation
template <typename Rep1, Radix Rad1>
CUDA_HOST_DEVICE_CALLABLE fixed_point<Rep1, Rad1> operator+(fixed_point<Rep1, Rad1> const& lhs,
                                                            fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  auto const sum   = lhs.rescaled(scale)._value + rhs.rescaled(scale)._value;

#if defined(__CUDACC_DEBUG__)

  assert(("fixed_point overflow of underlying representation type " + print_rep<Rep1>(),
          !addition_overflow<Rep1>(lhs.rescaled(scale)._value, rhs.rescaled(scale)._value)));

#endif

  return fixed_point<Rep1, Rad1>{scaled_integer<Rep1>{sum, scale}};
}

// MINUS Operation
template <typename Rep1, Radix Rad1>
CUDA_HOST_DEVICE_CALLABLE fixed_point<Rep1, Rad1> operator-(fixed_point<Rep1, Rad1> const& lhs,
                                                            fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  auto const diff  = lhs.rescaled(scale)._value - rhs.rescaled(scale)._value;

#if defined(__CUDACC_DEBUG__)

  assert(("fixed_point overflow of underlying representation type " + print_rep<Rep1>(),
          !subtraction_overflow<Rep1>(lhs.rescaled(scale)._value, rhs.rescaled(scale)._value)));

#endif

  return fixed_point<Rep1, Rad1>{scaled_integer<Rep1>{diff, scale}};
}

// MULTIPLIES Operation
template <typename Rep1, Radix Rad1>
CUDA_HOST_DEVICE_CALLABLE fixed_point<Rep1, Rad1> operator*(fixed_point<Rep1, Rad1> const& lhs,
                                                            fixed_point<Rep1, Rad1> const& rhs)
{
#if defined(__CUDACC_DEBUG__)

  assert(("fixed_point overflow of underlying representation type " + print_rep<Rep1>(),
          !multiplication_overflow<Rep1>(lhs._value, rhs._value)));

#endif

  return fixed_point<Rep1, Rad1>{
    scaled_integer<Rep1>(lhs._value * rhs._value, scale_type{lhs._scale + rhs._scale})};
}

// DIVISION Operation
template <typename Rep1, Radix Rad1>
CUDA_HOST_DEVICE_CALLABLE fixed_point<Rep1, Rad1> operator/(fixed_point<Rep1, Rad1> const& lhs,
                                                            fixed_point<Rep1, Rad1> const& rhs)
{
#if defined(__CUDACC_DEBUG__)

  assert(("fixed_point overflow of underlying representation type " + print_rep<Rep1>(),
          !division_overflow<Rep1>(lhs._value, rhs._value)));

#endif

  return fixed_point<Rep1, Rad1>{scaled_integer<Rep1>(std::roundf(lhs._value * 1.0 / rhs._value),
                                                      scale_type{lhs._scale - rhs._scale})};
}

// EQUALITY COMPARISON Operation
template <typename Rep1, Radix Rad1>
CUDA_HOST_DEVICE_CALLABLE bool operator==(fixed_point<Rep1, Rad1> const& lhs,
                                          fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value == rhs.rescaled(scale)._value;
}

// EQUALITY NOT COMPARISON Operation
template <typename Rep1, Radix Rad1>
CUDA_HOST_DEVICE_CALLABLE bool operator!=(fixed_point<Rep1, Rad1> const& lhs,
                                          fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value != rhs.rescaled(scale)._value;
}

// LESS THAN OR EQUAL TO Operation
template <typename Rep1, Radix Rad1>
CUDA_HOST_DEVICE_CALLABLE bool operator<=(fixed_point<Rep1, Rad1> const& lhs,
                                          fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value <= rhs.rescaled(scale)._value;
}

// GREATER THAN OR EQUAL TO Operation
template <typename Rep1, Radix Rad1>
CUDA_HOST_DEVICE_CALLABLE bool operator>=(fixed_point<Rep1, Rad1> const& lhs,
                                          fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value >= rhs.rescaled(scale)._value;
}

// LESS THAN Operation
template <typename Rep1, Radix Rad1>
CUDA_HOST_DEVICE_CALLABLE bool operator<(fixed_point<Rep1, Rad1> const& lhs,
                                         fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value < rhs.rescaled(scale)._value;
}

// GREATER THAN Operation
template <typename Rep1, Radix Rad1>
CUDA_HOST_DEVICE_CALLABLE bool operator>(fixed_point<Rep1, Rad1> const& lhs,
                                         fixed_point<Rep1, Rad1> const& rhs)
{
  auto const scale = std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value > rhs.rescaled(scale)._value;
}

using decimal32 = fixed_point<int32_t, Radix::BASE_10>;
using decimal64 = fixed_point<int64_t, Radix::BASE_10>;

/** @} */  // end of group
}  // namespace numeric
