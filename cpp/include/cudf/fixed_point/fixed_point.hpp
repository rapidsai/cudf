/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/assert.cuh>
#include <cudf/fixed_point/temporary.hpp>
#include <cudf/types.hpp>

#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>

/// `fixed_point` and supporting types
namespace CUDF_EXPORT numeric {

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
 * @brief Compile-time switch that enables sticky overflow tracking on a `fixed_point`
 *
 * When `Track == overflow_tracking::on`, the `fixed_point` value carries an extra `bool`
 * that is set whenever an arithmetic operation (or scale-change) on the value would
 * overflow the underlying integer representation. The flag is sticky: it propagates
 * through `+`, `-`, `*`, `/`, `%` and `rescaled()` so a downstream consumer can ask
 * whether any overflow has occurred along the entire chain of operations that
 * produced the value.
 *
 * The default, `overflow_tracking::off`, leaves `fixed_point` byte-for-byte identical
 * to the historical layout — there is zero runtime or storage overhead. The
 * `decimal32_safe` / `decimal64_safe` / `decimal128_safe` aliases instantiate the
 * `on` variant for callers (e.g. velox-cudf) that need overflow detection without
 * requiring a separate libcudf build.
 */
enum class overflow_tracking : bool { off = false, on = true };

/**
 * @brief Returns `true` if the representation type is supported by `fixed_point`
 *
 * @tparam T The representation type
 * @return `true` if the type is supported by `fixed_point` implementation
 */
template <typename T>
CUDF_HOST_DEVICE constexpr inline auto is_supported_representation_type()
{
  return cuda::std::is_same_v<T, int32_t> ||  //
         cuda::std::is_same_v<T, int64_t> ||  //
         cuda::std::is_same_v<T, __int128_t>;
}

/** @} */  // end of group

// Helper functions for `fixed_point` type
namespace detail {

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
                                           cuda::std::is_integral_v<Rep>)>* = nullptr>
CUDF_HOST_DEVICE inline constexpr Rep ipow(T exponent)
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

// Used by `fixed_point` overflow tracking; defined after `multiplication_overflow` /
// `division_overflow` in this header.
template <typename Rep, Radix Rad, typename T>
CUDF_HOST_DEVICE constexpr bool shift_overflows(T const& val, scale_type const& scale);

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
 * @tparam Rep   The representation type (either `int32_t` or `int64_t`)
 * @tparam Rad   The radix/base (either `Radix::BASE_2` or `Radix::BASE_10`)
 * @tparam Track Whether to carry a sticky overflow flag through arithmetic and
 *               scale-change operations. Defaults to `overflow_tracking::off`,
 *               which keeps the layout and runtime behavior identical to a
 *               non-tracking `fixed_point`.
 *
 * @note Sticky overflow tracking lives at the **value-type** level. The flag
 *       propagates automatically through every operator that takes a
 *       `fixed_point` value (`+`, `-`, `*`, `/`, `%`, comparisons, `rescaled()`),
 *       which means binaryops, transforms, scans and any reduction expressed
 *       on top of the value-level operators carry it for free. Aggregations
 *       that bypass the value layer and atomically update the **raw integer
 *       storage** (e.g. `cudf::detail::atomic_add(&target.element<DeviceTarget>(...), ...)`
 *       in `cudf/detail/aggregation/device_aggregators.cuh`) do **not**
 *       propagate the per-element bool. For groupby/reduce overflow detection,
 *       use the existing `aggregation::SUM_WITH_OVERFLOW` pattern, which
 *       maintains a sidecar overflow column rather than relying on the
 *       per-element flag.
 */
template <typename Rep, Radix Rad, overflow_tracking Track = overflow_tracking::off>
class fixed_point {
  Rep _value{};
  scale_type _scale;

  // Storage helpers used to keep `sizeof(fixed_point<..., off>)` identical to the
  // original non-tracking layout. When `Track == on`, `_overflow` carries a bool;
  // otherwise it is an empty type and `[[no_unique_address]]` collapses it to zero
  // bytes (no ABI change for `decimal32`/`decimal64`/`decimal128`).
  struct _no_overflow_flag {};
  struct _overflow_flag_storage {
    bool value{false};
  };

  static constexpr bool _tracks_overflow = (Track == overflow_tracking::on);
  using _overflow_storage_t =
    cuda::std::conditional_t<_tracks_overflow, _overflow_flag_storage, _no_overflow_flag>;
  [[no_unique_address]] _overflow_storage_t _overflow{};

  // Grant matching same-Rep/Rad/Track instantiations access to `_overflow` so the
  // free-function operators below can read and update the sticky flag.
  template <typename, Radix, overflow_tracking>
  friend class fixed_point;

 public:
  using rep                       = Rep;     ///< The representation type
  static constexpr auto rad       = Rad;     ///< The base
  static constexpr auto track     = Track;   ///< The overflow-tracking mode

  /**
   * @brief Constructor that will perform shifting to store value appropriately (from integral
   * types)
   *
   * @tparam T The integral type that you are constructing from
   * @param value The value that will be constructed from
   * @param scale The exponent that is applied to Rad to perform shifting
   */
  template <typename T,
            typename cuda::std::enable_if_t<cuda::std::is_integral_v<T> &&
                                            is_supported_representation_type<Rep>()>* = nullptr>
  CUDF_HOST_DEVICE inline explicit fixed_point(T const& value, scale_type const& scale)
    // `value` is cast to `Rep` to avoid overflow in cases where
    // constructing to `Rep` that is wider than `T`
    : _value{detail::shift<Rep, Rad>(static_cast<Rep>(value), scale)}, _scale{scale}
  {
    if constexpr (_tracks_overflow) {
      _overflow.value = detail::shift_overflows<Rep, Rad>(static_cast<Rep>(value), scale);
    }
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
   * @brief Constructor from a pre-scaled integer plus an overflow flag
   *
   * This is intended for conversions that already computed the scaled integer
   * representation and independently detected overflow (e.g. float <-> decimal
   * conversion helpers).
   */
  CUDF_HOST_DEVICE inline explicit fixed_point(scaled_integer<Rep> s, bool overflow)
    requires(Track == overflow_tracking::on)
    : _value{s.value}, _scale{s.scale}
  {
    _overflow.value = overflow;
  }

  /**
   * @brief "Scale-less" constructor that constructs `fixed_point` number with a specified
   * value and scale of zero
   *
   * @tparam T The value type being constructing from
   * @param value The value that will be constructed from
   */
  template <typename T, typename cuda::std::enable_if_t<cuda::std::is_integral_v<T>>* = nullptr>
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
   * @brief Explicit conversion operator for casting to integral types
   *
   * @tparam U The integral type that is being explicitly converted to
   * @return The `fixed_point` number in base 10 (aka human readable format)
   */
  template <typename U, typename cuda::std::enable_if_t<cuda::std::is_integral_v<U>>* = nullptr>
  CUDF_HOST_DEVICE explicit constexpr operator U() const
  {
    // Cast to the larger of the two types (of U and Rep) before converting to Rep because in
    // certain cases casting to U before shifting will result in integer overflow (i.e. if U =
    // int32_t, Rep = int64_t and _value > 2 billion)
    auto const value = cuda::std::common_type_t<U, Rep>(_value);
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
  CUDF_HOST_DEVICE [[nodiscard]] inline rep value() const { return _value; }

  /**
   * @brief Method that returns the scale of the `fixed_point` number
   *
   * @return The scale of the `fixed_point` number
   */
  CUDF_HOST_DEVICE [[nodiscard]] inline scale_type scale() const { return _scale; }

  /**
   * @brief Whether fixed-point overflow was detected while producing this value
   *
   * Only callable when `Track == overflow_tracking::on` (e.g. `decimal*_safe` aliases).
   * Sticky: once set, propagates through operations that combine this value with others.
   *
   * Note: the per-element flag is propagated by the value-level `+`, `-`, `*`, `/`,
   * `%` and `rescaled()` operations on `fixed_point`. Aggregations that bypass the
   * value-level operators (e.g. atomic adds on the raw integer storage in
   * `cudf::detail::atomic_add(&target.element<DeviceTarget>(...), ...)`) do not
   * carry the flag through; for groupby/reduce overflow detection see the
   * `aggregation::SUM_WITH_OVERFLOW` pattern in `device_aggregators.cuh`.
   */
  [[nodiscard]] CUDF_HOST_DEVICE inline bool overflow_occurred() const noexcept
    requires(Track == overflow_tracking::on)
  {
    return _overflow.value;
  }

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
   * @tparam Rep1   Representation type of the operand `rhs`
   * @tparam Rad1   Radix (base) type of the operand `rhs`
   * @tparam Track1 Overflow-tracking mode of the operand `rhs` (must match this)
   * @param rhs The number being added to `this`
   * @return The sum
   */
  template <typename Rep1, Radix Rad1, overflow_tracking Track1>
  CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1, Track1>& operator+=(
    fixed_point<Rep1, Rad1, Track1> const& rhs)
  {
    *this = *this + rhs;
    return *this;
  }

  /**
   * @brief operator *=
   *
   * @tparam Rep1   Representation type of the operand `rhs`
   * @tparam Rad1   Radix (base) type of the operand `rhs`
   * @tparam Track1 Overflow-tracking mode of the operand `rhs` (must match this)
   * @param rhs The number being multiplied to `this`
   * @return The product
   */
  template <typename Rep1, Radix Rad1, overflow_tracking Track1>
  CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1, Track1>& operator*=(
    fixed_point<Rep1, Rad1, Track1> const& rhs)
  {
    *this = *this * rhs;
    return *this;
  }

  /**
   * @brief operator -=
   *
   * @tparam Rep1   Representation type of the operand `rhs`
   * @tparam Rad1   Radix (base) type of the operand `rhs`
   * @tparam Track1 Overflow-tracking mode of the operand `rhs` (must match this)
   * @param rhs The number being subtracted from `this`
   * @return The difference
   */
  template <typename Rep1, Radix Rad1, overflow_tracking Track1>
  CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1, Track1>& operator-=(
    fixed_point<Rep1, Rad1, Track1> const& rhs)
  {
    *this = *this - rhs;
    return *this;
  }

  /**
   * @brief operator /=
   *
   * @tparam Rep1   Representation type of the operand `rhs`
   * @tparam Rad1   Radix (base) type of the operand `rhs`
   * @tparam Track1 Overflow-tracking mode of the operand `rhs` (must match this)
   * @param rhs The number being divided from `this`
   * @return The quotient
   */
  template <typename Rep1, Radix Rad1, overflow_tracking Track1>
  CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1, Track1>& operator/=(
    fixed_point<Rep1, Rad1, Track1> const& rhs)
  {
    *this = *this / rhs;
    return *this;
  }

  /**
   * @brief operator ++ (post-increment)
   *
   * @return The incremented result
   */
  CUDF_HOST_DEVICE inline fixed_point<Rep, Rad, Track>& operator++()
  {
    *this = *this + fixed_point<Rep, Rad, Track>{1, scale_type{_scale}};
    return *this;
  }

  /**
   * @brief operator + (for adding two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are added.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are added.
   *
   * @tparam Rep1   Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1   Radix (base) type of the operand `lhs` and `rhs`
   * @tparam Track1 Overflow-tracking mode of `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return The resulting `fixed_point` sum
   */
  template <typename Rep1, Radix Rad1, overflow_tracking Track1>
  CUDF_HOST_DEVICE inline friend fixed_point<Rep1, Rad1, Track1> operator+(
    fixed_point<Rep1, Rad1, Track1> const& lhs, fixed_point<Rep1, Rad1, Track1> const& rhs);

  /**
   * @brief operator - (for subtracting two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are subtracted.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are subtracted.
   *
   * @tparam Rep1   Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1   Radix (base) type of the operand `lhs` and `rhs`
   * @tparam Track1 Overflow-tracking mode of `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return The resulting `fixed_point` difference
   */
  template <typename Rep1, Radix Rad1, overflow_tracking Track1>
  CUDF_HOST_DEVICE inline friend fixed_point<Rep1, Rad1, Track1> operator-(
    fixed_point<Rep1, Rad1, Track1> const& lhs, fixed_point<Rep1, Rad1, Track1> const& rhs);

  /**
   * @brief operator * (for multiplying two `fixed_point` numbers)
   *
   * `_scale`s are added and `_value`s are multiplied.
   *
   * @tparam Rep1   Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1   Radix (base) type of the operand `lhs` and `rhs`
   * @tparam Track1 Overflow-tracking mode of `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return The resulting `fixed_point` product
   */
  template <typename Rep1, Radix Rad1, overflow_tracking Track1>
  CUDF_HOST_DEVICE inline friend fixed_point<Rep1, Rad1, Track1> operator*(
    fixed_point<Rep1, Rad1, Track1> const& lhs, fixed_point<Rep1, Rad1, Track1> const& rhs);

  /**
   * @brief operator / (for dividing two `fixed_point` numbers)
   *
   * `_scale`s are subtracted and `_value`s are divided.
   *
   * @tparam Rep1   Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1   Radix (base) type of the operand `lhs` and `rhs`
   * @tparam Track1 Overflow-tracking mode of `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return The resulting `fixed_point` quotient
   */
  template <typename Rep1, Radix Rad1, overflow_tracking Track1>
  CUDF_HOST_DEVICE inline friend fixed_point<Rep1, Rad1, Track1> operator/(
    fixed_point<Rep1, Rad1, Track1> const& lhs, fixed_point<Rep1, Rad1, Track1> const& rhs);

  /**
   * @brief operator % (for computing the modulo operation of two `fixed_point` numbers)
   *
   * If `_scale`s are equal, the modulus is computed directly.
   * If `_scale`s are not equal, the number with larger `_scale` is shifted to the
   * smaller `_scale`, and then the modulus is computed.
   *
   * @tparam Rep1   Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1   Radix (base) type of the operand `lhs` and `rhs`
   * @tparam Track1 Overflow-tracking mode of `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return The resulting `fixed_point` number
   */
  template <typename Rep1, Radix Rad1, overflow_tracking Track1>
  CUDF_HOST_DEVICE inline friend fixed_point<Rep1, Rad1, Track1> operator%(
    fixed_point<Rep1, Rad1, Track1> const& lhs, fixed_point<Rep1, Rad1, Track1> const& rhs);

  /**
   * @brief operator == (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are compared.
   *
   * @tparam Rep1   Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1   Radix (base) type of the operand `lhs` and `rhs`
   * @tparam Track1 Overflow-tracking mode of `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return true if `lhs` and `rhs` are equal, false if not
   */
  template <typename Rep1, Radix Rad1, overflow_tracking Track1>
  CUDF_HOST_DEVICE inline friend bool operator==(fixed_point<Rep1, Rad1, Track1> const& lhs,
                                                 fixed_point<Rep1, Rad1, Track1> const& rhs);

  /**
   * @brief operator != (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are compared.
   *
   * @tparam Rep1   Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1   Radix (base) type of the operand `lhs` and `rhs`
   * @tparam Track1 Overflow-tracking mode of `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return true if `lhs` and `rhs` are not equal, false if not
   */
  template <typename Rep1, Radix Rad1, overflow_tracking Track1>
  CUDF_HOST_DEVICE inline friend bool operator!=(fixed_point<Rep1, Rad1, Track1> const& lhs,
                                                 fixed_point<Rep1, Rad1, Track1> const& rhs);

  /**
   * @brief operator <= (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are compared.
   *
   * @tparam Rep1   Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1   Radix (base) type of the operand `lhs` and `rhs`
   * @tparam Track1 Overflow-tracking mode of `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return true if `lhs` less than or equal to `rhs`, false if not
   */
  template <typename Rep1, Radix Rad1, overflow_tracking Track1>
  CUDF_HOST_DEVICE inline friend bool operator<=(fixed_point<Rep1, Rad1, Track1> const& lhs,
                                                 fixed_point<Rep1, Rad1, Track1> const& rhs);

  /**
   * @brief operator >= (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are compared.
   *
   * @tparam Rep1   Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1   Radix (base) type of the operand `lhs` and `rhs`
   * @tparam Track1 Overflow-tracking mode of `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return true if `lhs` greater than or equal to `rhs`, false if not
   */
  template <typename Rep1, Radix Rad1, overflow_tracking Track1>
  CUDF_HOST_DEVICE inline friend bool operator>=(fixed_point<Rep1, Rad1, Track1> const& lhs,
                                                 fixed_point<Rep1, Rad1, Track1> const& rhs);

  /**
   * @brief operator < (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are compared.
   *
   * @tparam Rep1   Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1   Radix (base) type of the operand `lhs` and `rhs`
   * @tparam Track1 Overflow-tracking mode of `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return true if `lhs` less than `rhs`, false if not
   */
  template <typename Rep1, Radix Rad1, overflow_tracking Track1>
  CUDF_HOST_DEVICE inline friend bool operator<(fixed_point<Rep1, Rad1, Track1> const& lhs,
                                                fixed_point<Rep1, Rad1, Track1> const& rhs);

  /**
   * @brief operator > (for comparing two `fixed_point` numbers)
   *
   * If `_scale`s are equal, `_value`s are compared.
   * If `_scale`s are not equal, the number with the larger `_scale` is shifted to the
   * smaller `_scale`, and then the `_value`s are compared.
   *
   * @tparam Rep1   Representation type of the operand `lhs` and `rhs`
   * @tparam Rad1   Radix (base) type of the operand `lhs` and `rhs`
   * @tparam Track1 Overflow-tracking mode of `lhs` and `rhs`
   * @param lhs The left hand side operand
   * @param rhs The right hand side operand
   * @return true if `lhs` greater than `rhs`, false if not
   */
  template <typename Rep1, Radix Rad1, overflow_tracking Track1>
  CUDF_HOST_DEVICE inline friend bool operator>(fixed_point<Rep1, Rad1, Track1> const& lhs,
                                                fixed_point<Rep1, Rad1, Track1> const& rhs);

  /**
   * @brief Method for creating a `fixed_point` number with a new `scale`
   *
   * The `fixed_point` number returned will have the same value, underlying representation and
   * radix as `this`, the only thing changed is the scale.
   *
   * @param scale The `scale` of the returned `fixed_point` number
   * @return `fixed_point` number with a new `scale`
   */
  CUDF_HOST_DEVICE [[nodiscard]] inline fixed_point<Rep, Rad, Track> rescaled(
    scale_type scale) const
  {
    if (scale == _scale) { return *this; }
    auto const scale_delta = scale_type{scale - _scale};
    Rep const value        = detail::shift<Rep, Rad>(_value, scale_delta);
    fixed_point<Rep, Rad, Track> result{scaled_integer<Rep>{value, scale}};
    if constexpr (_tracks_overflow) {
      result._overflow.value =
        _overflow.value || detail::shift_overflows<Rep, Rad>(_value, scale_delta);
    }
    return result;
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

namespace detail {

/**
 * @brief Whether `shift<Rep, Rad>(val, scale)` incurs signed-integer overflow in the mul/div
 * steps (same conditions as `multiplication_overflow` / `division_overflow` on intermediates).
 */
template <typename Rep, Radix Rad, typename T>
CUDF_HOST_DEVICE inline constexpr bool shift_overflows(T const& val, scale_type const& scale)
{
  auto const v = static_cast<Rep>(val);
  if (scale == 0) { return false; }
  if (scale > 0) {
    Rep const divisor = ipow<Rep, Rad>(static_cast<int32_t>(scale));
    return division_overflow<Rep>(v, divisor);
  }
  Rep const multiplier = ipow<Rep, Rad>(static_cast<int32_t>(-scale));
  return multiplication_overflow<Rep>(v, multiplier);
}

/**
 * @brief Run binary integer-overflow predicate once; assert under `__CUDACC_DEBUG__`.
 *
 * Unconditionally defined: only the call sites in `fixed_point` operator overloads
 * decide (via `if constexpr (Track == overflow_tracking::on)` / under
 * `__CUDACC_DEBUG__`) whether to instantiate it. Unused instantiations are free.
 *
 * @tparam Rep1 Representation type
 * @tparam F Function type `bool (Rep1, Rep1)` (e.g. `&addition_overflow<Rep1, Rep1>`)
 * @param overflow_fn Predicate on the operation's integer operands
 * @param lhs_value Left-hand integer operand at common scale (or lhs._value for `*`/`/`)
 * @param rhs_value Right-hand integer operand
 * @return Predicate result for sticky `fixed_point` overflow tracking
 */
template <typename Rep1, typename F>
CUDF_HOST_DEVICE inline bool fixed_point_op_overflow_check(F overflow_fn,
                                                           Rep1 lhs_value,
                                                           Rep1 rhs_value)
{
  bool const op_overflow = static_cast<bool>(overflow_fn(lhs_value, rhs_value));
#if defined(__CUDACC_DEBUG__)
  assert(!op_overflow && "fixed_point overflow");
#endif
  return op_overflow;
}

}  // namespace detail

// PLUS Operation
template <typename Rep1, Radix Rad1, overflow_tracking Track1>
CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1, Track1> operator+(
  fixed_point<Rep1, Rad1, Track1> const& lhs, fixed_point<Rep1, Rad1, Track1> const& rhs)
{
  auto const scale = cuda::std::min(lhs._scale, rhs._scale);
  auto const lhs_r = lhs.rescaled(scale);
  auto const rhs_r = rhs.rescaled(scale);
  auto const sum   = lhs_r._value + rhs_r._value;
  auto result      = fixed_point<Rep1, Rad1, Track1>{scaled_integer<Rep1>{sum, scale}};

  if constexpr (Track1 == overflow_tracking::on) {
    bool const op_overflow = detail::fixed_point_op_overflow_check<Rep1>(
      &addition_overflow<Rep1, Rep1>, lhs_r._value, rhs_r._value);
    result._overflow.value = op_overflow || lhs_r._overflow.value || rhs_r._overflow.value;
  } else {
#if defined(__CUDACC_DEBUG__)
    static_cast<void>(detail::fixed_point_op_overflow_check<Rep1>(
      &addition_overflow<Rep1, Rep1>, lhs_r._value, rhs_r._value));
#endif
  }
  return result;
}

// MINUS Operation
template <typename Rep1, Radix Rad1, overflow_tracking Track1>
CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1, Track1> operator-(
  fixed_point<Rep1, Rad1, Track1> const& lhs, fixed_point<Rep1, Rad1, Track1> const& rhs)
{
  auto const scale = cuda::std::min(lhs._scale, rhs._scale);
  auto const lhs_r = lhs.rescaled(scale);
  auto const rhs_r = rhs.rescaled(scale);
  auto const diff  = lhs_r._value - rhs_r._value;
  auto result      = fixed_point<Rep1, Rad1, Track1>{scaled_integer<Rep1>{diff, scale}};

  if constexpr (Track1 == overflow_tracking::on) {
    bool const op_overflow = detail::fixed_point_op_overflow_check<Rep1>(
      &subtraction_overflow<Rep1, Rep1>, lhs_r._value, rhs_r._value);
    result._overflow.value = op_overflow || lhs_r._overflow.value || rhs_r._overflow.value;
  } else {
#if defined(__CUDACC_DEBUG__)
    static_cast<void>(detail::fixed_point_op_overflow_check<Rep1>(
      &subtraction_overflow<Rep1, Rep1>, lhs_r._value, rhs_r._value));
#endif
  }
  return result;
}

// MULTIPLIES Operation
template <typename Rep1, Radix Rad1, overflow_tracking Track1>
CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1, Track1> operator*(
  fixed_point<Rep1, Rad1, Track1> const& lhs, fixed_point<Rep1, Rad1, Track1> const& rhs)
{
  auto result = fixed_point<Rep1, Rad1, Track1>{
    scaled_integer<Rep1>(lhs._value * rhs._value, scale_type{lhs._scale + rhs._scale})};

  if constexpr (Track1 == overflow_tracking::on) {
    bool const op_overflow = detail::fixed_point_op_overflow_check<Rep1>(
      &multiplication_overflow<Rep1, Rep1>, lhs._value, rhs._value);
    result._overflow.value = op_overflow || lhs._overflow.value || rhs._overflow.value;
  } else {
#if defined(__CUDACC_DEBUG__)
    static_cast<void>(detail::fixed_point_op_overflow_check<Rep1>(
      &multiplication_overflow<Rep1, Rep1>, lhs._value, rhs._value));
#endif
  }
  return result;
}

// DIVISION Operation
template <typename Rep1, Radix Rad1, overflow_tracking Track1>
CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1, Track1> operator/(
  fixed_point<Rep1, Rad1, Track1> const& lhs, fixed_point<Rep1, Rad1, Track1> const& rhs)
{
  auto result = fixed_point<Rep1, Rad1, Track1>{
    scaled_integer<Rep1>(lhs._value / rhs._value, scale_type{lhs._scale - rhs._scale})};

  if constexpr (Track1 == overflow_tracking::on) {
    bool const op_overflow = detail::fixed_point_op_overflow_check<Rep1>(
      &division_overflow<Rep1, Rep1>, lhs._value, rhs._value);
    result._overflow.value = op_overflow || lhs._overflow.value || rhs._overflow.value;
  } else {
#if defined(__CUDACC_DEBUG__)
    static_cast<void>(detail::fixed_point_op_overflow_check<Rep1>(
      &division_overflow<Rep1, Rep1>, lhs._value, rhs._value));
#endif
  }
  return result;
}

// EQUALITY COMPARISON Operation
template <typename Rep1, Radix Rad1, overflow_tracking Track1>
CUDF_HOST_DEVICE inline bool operator==(fixed_point<Rep1, Rad1, Track1> const& lhs,
                                        fixed_point<Rep1, Rad1, Track1> const& rhs)
{
  auto const scale = cuda::std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value == rhs.rescaled(scale)._value;
}

// EQUALITY NOT COMPARISON Operation
template <typename Rep1, Radix Rad1, overflow_tracking Track1>
CUDF_HOST_DEVICE inline bool operator!=(fixed_point<Rep1, Rad1, Track1> const& lhs,
                                        fixed_point<Rep1, Rad1, Track1> const& rhs)
{
  auto const scale = cuda::std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value != rhs.rescaled(scale)._value;
}

// LESS THAN OR EQUAL TO Operation
template <typename Rep1, Radix Rad1, overflow_tracking Track1>
CUDF_HOST_DEVICE inline bool operator<=(fixed_point<Rep1, Rad1, Track1> const& lhs,
                                        fixed_point<Rep1, Rad1, Track1> const& rhs)
{
  auto const scale = cuda::std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value <= rhs.rescaled(scale)._value;
}

// GREATER THAN OR EQUAL TO Operation
template <typename Rep1, Radix Rad1, overflow_tracking Track1>
CUDF_HOST_DEVICE inline bool operator>=(fixed_point<Rep1, Rad1, Track1> const& lhs,
                                        fixed_point<Rep1, Rad1, Track1> const& rhs)
{
  auto const scale = cuda::std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value >= rhs.rescaled(scale)._value;
}

// LESS THAN Operation
template <typename Rep1, Radix Rad1, overflow_tracking Track1>
CUDF_HOST_DEVICE inline bool operator<(fixed_point<Rep1, Rad1, Track1> const& lhs,
                                       fixed_point<Rep1, Rad1, Track1> const& rhs)
{
  auto const scale = cuda::std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value < rhs.rescaled(scale)._value;
}

// GREATER THAN Operation
template <typename Rep1, Radix Rad1, overflow_tracking Track1>
CUDF_HOST_DEVICE inline bool operator>(fixed_point<Rep1, Rad1, Track1> const& lhs,
                                       fixed_point<Rep1, Rad1, Track1> const& rhs)
{
  auto const scale = cuda::std::min(lhs._scale, rhs._scale);
  return lhs.rescaled(scale)._value > rhs.rescaled(scale)._value;
}

// MODULO OPERATION
template <typename Rep1, Radix Rad1, overflow_tracking Track1>
CUDF_HOST_DEVICE inline fixed_point<Rep1, Rad1, Track1> operator%(
  fixed_point<Rep1, Rad1, Track1> const& lhs, fixed_point<Rep1, Rad1, Track1> const& rhs)
{
  auto const scale     = cuda::std::min(lhs._scale, rhs._scale);
  auto const lhs_r     = lhs.rescaled(scale);
  auto const rhs_r     = rhs.rescaled(scale);
  auto const remainder = lhs_r._value % rhs_r._value;
  auto result          = fixed_point<Rep1, Rad1, Track1>{scaled_integer<Rep1>{remainder, scale}};
  if constexpr (Track1 == overflow_tracking::on) {
    result._overflow.value = lhs_r._overflow.value || rhs_r._overflow.value;
  }
  return result;
}

using decimal32  = fixed_point<int32_t, Radix::BASE_10>;     ///<  32-bit decimal fixed point
using decimal64  = fixed_point<int64_t, Radix::BASE_10>;     ///<  64-bit decimal fixed point
using decimal128 = fixed_point<__int128_t, Radix::BASE_10>;  ///< 128-bit decimal fixed point

// -----------------------------------------------------------------------------
// Overflow-tracking aliases
// -----------------------------------------------------------------------------
// These instantiate the same `fixed_point` class template with `Track == on`
// and so participate in every operator overload above. They are wired into the
// runtime type system as `type_id::DECIMAL{32,64,128}_SAFE`, which means they
// flow through `binary_operation`, `transform`, scans and any code path that
// dispatches via `cudf::type_dispatcher`. The on-device storage of a
// `column<decimal*_safe>` is still the raw signed integer (see
// `cudf::device_storage_type_t<>`); the sticky bit is purely a value-type
// concept used inside element-wise kernels.
//
// Aggregations whose update step bypasses the value-level operators (e.g.
// atomic adds on the raw integer storage in
// `cudf::detail::atomic_add(&target.element<DeviceTarget>(...), ...)`) will
// **not** carry the sticky bit through. Use the existing
// `aggregation::SUM_WITH_OVERFLOW` pattern in
// `cpp/include/cudf/detail/aggregation/device_aggregators.cuh` for groupby and
// reduce overflow detection.

/// 32-bit decimal fixed point with sticky overflow tracking
using decimal32_safe = fixed_point<int32_t, Radix::BASE_10, overflow_tracking::on>;
/// 64-bit decimal fixed point with sticky overflow tracking
using decimal64_safe = fixed_point<int64_t, Radix::BASE_10, overflow_tracking::on>;
/// 128-bit decimal fixed point with sticky overflow tracking
using decimal128_safe = fixed_point<__int128_t, Radix::BASE_10, overflow_tracking::on>;

/** @} */  // end of group
}  // namespace CUDF_EXPORT numeric
