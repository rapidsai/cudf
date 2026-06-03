/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/fixed_point/detail/floating_conversion.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/type_traits>
#include <cuda/std/utility>

/**
 * @file safe_arithmetic.hpp
 * @brief Overflow-aware free functions for `numeric::fixed_point`.
 *
 * These free functions intentionally live outside of `fixed_point` itself
 * (mirroring how `floating_conversion.hpp` lives outside `fixed_point` for the
 * float<->decimal conversion path). They take regular, non-tracking
 * `fixed_point` operands and return a `safe_result` that pairs the computed
 * value with an `overflow` flag describing whether the operation -- including
 * any internal rescale -- would have wrapped the underlying integer storage.
 *
 * When overflow is detected the actual arithmetic is skipped: computing the
 * wrapped result would be invalid (callers are not expected to consume a value
 * whose `overflow` flag is set) and, for signed integer types, performing it
 * would invoke undefined behavior. In that case `value` holds a defined
 * placeholder (zero) and `overflow` is `true`.
 *
 * The intent is that overflow-aware code paths (the `binary_operation_safe`
 * kernel today; specialized reduce/groupby aggregations in the future) compose
 * these primitives explicitly, instead of relying on a sticky bit baked into
 * the value type.
 */

namespace CUDF_EXPORT numeric {
namespace detail {

/**
 * @brief Result of an overflow-checked `fixed_point` operation.
 *
 * @tparam Rep Storage type of the wrapped `fixed_point` value
 * @tparam Rad Radix of the wrapped `fixed_point` value
 */
template <typename Rep, Radix Rad>
struct safe_result {
  fixed_point<Rep, Rad> value;  ///< Computed `fixed_point` value
  bool overflow;                ///< Whether the producing operation overflowed
};

/**
 * @brief Whether `shift<Rep, Rad>(val, scale)` would incur signed-integer overflow
 *
 * Mirrors the overflow conditions of `multiplication_overflow` /
 * `division_overflow` on the intermediate scale factor.
 *
 * @tparam Rep Representation type
 * @tparam Rad Radix
 * @tparam T   Type of the value being shifted (typically `Rep`)
 * @param val The value being shifted
 * @param scale The amount to shift the value by
 * @return true if the shift would overflow `Rep`, false otherwise
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
 * @brief Rescale a `fixed_point` value, reporting whether the underlying shift overflows
 *
 * Equivalent to `x.rescaled(new_scale)` but surfaces the shift overflow flag
 * instead of dropping it.
 *
 * @tparam Rep Storage type
 * @tparam Rad Radix
 * @param x The value to rescale
 * @param new_scale The target scale
 * @return `{rescaled_value, overflow}`; `rescaled_value` is zero on overflow
 */
template <typename Rep, Radix Rad>
CUDF_HOST_DEVICE inline safe_result<Rep, Rad> safe_rescaled(fixed_point<Rep, Rad> x,
                                                            scale_type new_scale)
{
  if (new_scale == x.scale()) { return safe_result<Rep, Rad>{x, false}; }
  auto const scale_delta = scale_type{new_scale - x.scale()};
  // Skip the shift when it would overflow: the rescaled value is meaningless
  // and performing the multiply would be signed-integer-overflow UB.
  if (shift_overflows<Rep, Rad>(x.value(), scale_delta)) {
    return safe_result<Rep, Rad>{fixed_point<Rep, Rad>{scaled_integer<Rep>{Rep{0}, new_scale}},
                                 true};
  }
  Rep const value = shift<Rep, Rad>(x.value(), scale_delta);
  return safe_result<Rep, Rad>{fixed_point<Rep, Rad>{scaled_integer<Rep>{value, new_scale}}, false};
}

/**
 * @brief Overflow-checked addition of two `fixed_point` values
 *
 * Rescales both operands to the smaller of their two scales (matching
 * `operator+`), then performs the add. The returned `overflow` flag is the
 * disjunction of any rescale overflow and the integer add overflow. When
 * overflow is detected the add is skipped and `value` is zero.
 */
template <typename Rep, Radix Rad>
CUDF_HOST_DEVICE inline safe_result<Rep, Rad> safe_add(fixed_point<Rep, Rad> lhs,
                                                       fixed_point<Rep, Rad> rhs)
{
  auto const common_scale = cuda::std::min(lhs.scale(), rhs.scale());
  auto const lhs_r        = safe_rescaled(lhs, common_scale);
  auto const rhs_r        = safe_rescaled(rhs, common_scale);
  Rep const lv            = lhs_r.value.value();
  Rep const rv            = rhs_r.value.value();
  bool const overflow     = lhs_r.overflow || rhs_r.overflow || addition_overflow<Rep>(lv, rv);
  Rep const sum           = overflow ? Rep{0} : lv + rv;
  return safe_result<Rep, Rad>{fixed_point<Rep, Rad>{scaled_integer<Rep>{sum, common_scale}},
                               overflow};
}

/**
 * @brief Overflow-checked subtraction of two `fixed_point` values
 *
 * When overflow is detected the subtract is skipped and `value` is zero.
 */
template <typename Rep, Radix Rad>
CUDF_HOST_DEVICE inline safe_result<Rep, Rad> safe_sub(fixed_point<Rep, Rad> lhs,
                                                       fixed_point<Rep, Rad> rhs)
{
  auto const common_scale = cuda::std::min(lhs.scale(), rhs.scale());
  auto const lhs_r        = safe_rescaled(lhs, common_scale);
  auto const rhs_r        = safe_rescaled(rhs, common_scale);
  Rep const lv            = lhs_r.value.value();
  Rep const rv            = rhs_r.value.value();
  bool const overflow     = lhs_r.overflow || rhs_r.overflow || subtraction_overflow<Rep>(lv, rv);
  Rep const diff          = overflow ? Rep{0} : lv - rv;
  return safe_result<Rep, Rad>{fixed_point<Rep, Rad>{scaled_integer<Rep>{diff, common_scale}},
                               overflow};
}

/**
 * @brief Overflow-checked multiplication of two `fixed_point` values
 *
 * No rescale is needed -- the result scale is `lhs.scale() + rhs.scale()`.
 * When overflow is detected the multiply is skipped and `value` is zero.
 */
template <typename Rep, Radix Rad>
CUDF_HOST_DEVICE inline safe_result<Rep, Rad> safe_mul(fixed_point<Rep, Rad> lhs,
                                                       fixed_point<Rep, Rad> rhs)
{
  Rep const lv        = lhs.value();
  Rep const rv        = rhs.value();
  bool const overflow = multiplication_overflow<Rep>(lv, rv);
  Rep const prod      = overflow ? Rep{0} : lv * rv;
  scale_type const out_scale{lhs.scale() + rhs.scale()};
  return safe_result<Rep, Rad>{fixed_point<Rep, Rad>{scaled_integer<Rep>{prod, out_scale}},
                               overflow};
}

/**
 * @brief Overflow-checked division of two `fixed_point` values
 *
 * Two failure modes are reported as overflow: `INT_MIN / -1` (caught by
 * `division_overflow`) and division by zero. In both cases the divide is
 * skipped -- so we never invoke signed-integer divide overflow or
 * divide-by-zero UB -- and a zero-valued result is returned.
 */
template <typename Rep, Radix Rad>
CUDF_HOST_DEVICE inline safe_result<Rep, Rad> safe_div(fixed_point<Rep, Rad> lhs,
                                                       fixed_point<Rep, Rad> rhs)
{
  Rep const lv = lhs.value();
  Rep const rv = rhs.value();
  scale_type const out_scale{lhs.scale() - rhs.scale()};
  // Short-circuit on a zero divisor before touching `division_overflow` (which
  // would itself divide) or the divide below.
  bool const overflow = (rv == Rep{0}) || division_overflow<Rep>(lv, rv);
  Rep const quot      = overflow ? Rep{0} : lv / rv;
  return safe_result<Rep, Rad>{fixed_point<Rep, Rad>{scaled_integer<Rep>{quot, out_scale}},
                               overflow};
}

/**
 * @brief Overflow-checked modulo of two `fixed_point` values
 *
 * The op-level failure modes are divide-by-zero and the `INT_MIN % -1`
 * signed-overflow boundary; the rescale to the common scale can also overflow.
 * All are OR'd into the returned flag, and whenever any of them is set the `%`
 * is skipped and a zero-valued result is returned (so we never invoke
 * `%`-by-zero or signed-overflow UB).
 */
template <typename Rep, Radix Rad>
CUDF_HOST_DEVICE inline safe_result<Rep, Rad> safe_mod(fixed_point<Rep, Rad> lhs,
                                                       fixed_point<Rep, Rad> rhs)
{
  auto const common_scale = cuda::std::min(lhs.scale(), rhs.scale());
  auto const lhs_r        = safe_rescaled(lhs, common_scale);
  auto const rhs_r        = safe_rescaled(rhs, common_scale);
  Rep const lv            = lhs_r.value.value();
  Rep const rv            = rhs_r.value.value();
  bool const overflow =
    lhs_r.overflow || rhs_r.overflow || (rv == Rep{0}) || division_overflow<Rep>(lv, rv);
  Rep const remainder = overflow ? Rep{0} : lv % rv;
  return safe_result<Rep, Rad>{fixed_point<Rep, Rad>{scaled_integer<Rep>{remainder, common_scale}},
                               overflow};
}

/**
 * @brief Overflow-checked positive modulo, matching `ops::PMod` semantics for decimals.
 *
 * Implements `rem = x % y; (rem < 0) ? (rem + y) % y : rem`. The intermediate
 * `rem + y` can overflow even though `%` itself cannot. If the base modulo
 * already overflowed (incl. divide-by-zero), or the correcting add overflows,
 * the correction is skipped and a zero-valued result is returned.
 */
template <typename Rep, Radix Rad>
CUDF_HOST_DEVICE inline safe_result<Rep, Rad> safe_pmod(fixed_point<Rep, Rad> lhs,
                                                        fixed_point<Rep, Rad> rhs)
{
  auto const m = safe_mod(lhs, rhs);
  if (m.overflow || !(m.value.value() < Rep{0})) { return m; }

  // `m.overflow` is false here, so `safe_mod` saw a non-zero divisor: `rv != 0`.
  auto const rhs_r    = safe_rescaled(rhs, m.value.scale());
  Rep const mv        = m.value.value();
  Rep const rv        = rhs_r.value.value();
  bool const overflow = rhs_r.overflow || addition_overflow<Rep>(mv, rv);
  Rep const corrected = overflow ? Rep{0} : (mv + rv) % rv;
  return safe_result<Rep, Rad>{
    fixed_point<Rep, Rad>{scaled_integer<Rep>{corrected, m.value.scale()}}, overflow};
}

/**
 * @brief Overflow-checked Python-style modulo: `((x % y) + y) % y`.
 *
 * The intermediate add can overflow; the final `%` cannot. If the base modulo
 * already overflowed (incl. divide-by-zero), or the correcting add overflows,
 * the correction is skipped and a zero-valued result is returned, so we never
 * invoke `%`-by-zero or signed-overflow UB.
 */
template <typename Rep, Radix Rad>
CUDF_HOST_DEVICE inline safe_result<Rep, Rad> safe_pymod(fixed_point<Rep, Rad> lhs,
                                                         fixed_point<Rep, Rad> rhs)
{
  auto const m = safe_mod(lhs, rhs);
  if (m.overflow) { return m; }

  // `m.overflow` is false here, so `safe_mod` saw a non-zero divisor: `rv != 0`.
  auto const rhs_r    = safe_rescaled(rhs, m.value.scale());
  Rep const mv        = m.value.value();
  Rep const rv        = rhs_r.value.value();
  bool const overflow = rhs_r.overflow || addition_overflow<Rep>(mv, rv);
  Rep const corrected = overflow ? Rep{0} : (mv + rv) % rv;
  return safe_result<Rep, Rad>{
    fixed_point<Rep, Rad>{scaled_integer<Rep>{corrected, m.value.scale()}}, overflow};
}

/**
 * @brief Overflow-checked floating-point -> `fixed_point` conversion
 *
 * Uses `convert_floating_to_integral<Rep, true>` (in
 * `floating_conversion.hpp`) for base-10 decimals, which saturates and reports
 * overflow. For base-2 radixes there is no checked path today, so `overflow`
 * is always `false`.
 *
 * @tparam Fixed    Target `fixed_point` instantiation
 * @tparam Floating Source floating-point type
 * @param floating  The floating-point value to convert
 * @param scale     The desired scale of the result
 * @return `{fixed_point_value, overflow}`
 */
template <typename Fixed,
          typename Floating,
          CUDF_ENABLE_IF(cuda::std::is_floating_point_v<Floating>&& cudf::is_fixed_point<Fixed>())>
CUDF_HOST_DEVICE inline safe_result<typename Fixed::rep, Fixed::rad> safe_convert_floating_to_fixed(
  Floating floating, scale_type scale)
{
  using Rep = typename Fixed::rep;
  if constexpr (Fixed::rad == Radix::BASE_10) {
    auto const [value, overflow] = convert_floating_to_integral<Rep, true>(floating, scale);
    return safe_result<Rep, Fixed::rad>{Fixed{scaled_integer<Rep>{value, scale}}, overflow};
  } else {
    Rep const value = static_cast<Rep>(shift<Rep, Fixed::rad>(floating, scale));
    return safe_result<Rep, Fixed::rad>{Fixed{scaled_integer<Rep>{value, scale}}, false};
  }
}

}  // namespace detail
}  // namespace CUDF_EXPORT numeric
