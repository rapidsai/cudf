/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/fixed_point/detail/floating_conversion.hpp>
#include <cudf/fixed_point/detail/safe_arithmetic.hpp>
#include <cudf/fixed_point/fixed_point.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

using namespace numeric;

struct SafeArithmeticTest : public cudf::test::BaseFixture {};
struct FloatingConversionOverflowTest : public cudf::test::BaseFixture {};

// ---------------------------------------------------------------------------
// safe_add
// ---------------------------------------------------------------------------

TEST_F(SafeArithmeticTest, AddNoOverflow)
{
  decimal64 const a{scaled_integer<int64_t>{1, scale_type{0}}};
  decimal64 const b{scaled_integer<int64_t>{2, scale_type{0}}};
  auto const r = detail::safe_add(a, b);
  EXPECT_FALSE(r.overflow);
  EXPECT_EQ(r.value.value(), int64_t{3});
}

TEST_F(SafeArithmeticTest, AddOverflowSetsFlag)
{
  auto constexpr near_max = std::numeric_limits<int64_t>::max() - 100;
  decimal64 const a{scaled_integer<int64_t>{near_max, scale_type{0}}};
  decimal64 const b{scaled_integer<int64_t>{200, scale_type{0}}};
  auto const r = detail::safe_add(a, b);
  EXPECT_TRUE(r.overflow);
}

TEST_F(SafeArithmeticTest, AddRescaleOverflowSetsFlag)
{
  // Rescaling lhs from scale 0 down to scale -3 multiplies by 10^3; near-max value overflows.
  decimal64 const a{
    scaled_integer<int64_t>{std::numeric_limits<int64_t>::max() / 2, scale_type{0}}};
  decimal64 const b{scaled_integer<int64_t>{0, scale_type{-3}}};
  auto const r = detail::safe_add(a, b);
  EXPECT_TRUE(r.overflow);
}

// ---------------------------------------------------------------------------
// safe_sub
// ---------------------------------------------------------------------------

TEST_F(SafeArithmeticTest, SubNoOverflow)
{
  decimal64 const a{scaled_integer<int64_t>{10, scale_type{0}}};
  decimal64 const b{scaled_integer<int64_t>{3, scale_type{0}}};
  auto const r = detail::safe_sub(a, b);
  EXPECT_FALSE(r.overflow);
  EXPECT_EQ(r.value.value(), int64_t{7});
}

TEST_F(SafeArithmeticTest, SubOverflowSetsFlag)
{
  auto constexpr near_min = std::numeric_limits<int64_t>::min() + 100;
  decimal64 const a{scaled_integer<int64_t>{near_min, scale_type{0}}};
  decimal64 const b{scaled_integer<int64_t>{200, scale_type{0}}};
  auto const r = detail::safe_sub(a, b);
  EXPECT_TRUE(r.overflow);
}

// ---------------------------------------------------------------------------
// safe_mul
// ---------------------------------------------------------------------------

TEST_F(SafeArithmeticTest, MulNoOverflow)
{
  decimal64 const a{scaled_integer<int64_t>{2, scale_type{0}}};
  decimal64 const b{scaled_integer<int64_t>{3, scale_type{0}}};
  auto const r = detail::safe_mul(a, b);
  EXPECT_FALSE(r.overflow);
  EXPECT_EQ(r.value.value(), int64_t{6});
}

TEST_F(SafeArithmeticTest, MulOverflowSetsFlag)
{
  decimal64 const a{scaled_integer<int64_t>{1'000'000'000'000LL, scale_type{0}}};
  decimal64 const b{scaled_integer<int64_t>{1'000'000'000'000LL, scale_type{0}}};
  auto const r = detail::safe_mul(a, b);
  EXPECT_TRUE(r.overflow);
}

TEST_F(SafeArithmeticTest, Mul32OverflowSetsFlag)
{
  decimal32 const a{scaled_integer<int32_t>{1'000'000, scale_type{0}}};
  decimal32 const b{scaled_integer<int32_t>{1'000'000, scale_type{0}}};
  auto const r = detail::safe_mul(a, b);
  EXPECT_TRUE(r.overflow);
}

// ---------------------------------------------------------------------------
// safe_div
// ---------------------------------------------------------------------------

TEST_F(SafeArithmeticTest, DivNoOverflow)
{
  decimal64 const a{scaled_integer<int64_t>{10, scale_type{0}}};
  decimal64 const b{scaled_integer<int64_t>{2, scale_type{0}}};
  auto const r = detail::safe_div(a, b);
  EXPECT_FALSE(r.overflow);
  EXPECT_EQ(r.value.value(), int64_t{5});
}

TEST_F(SafeArithmeticTest, DivIntMinByNegativeOneOverflows)
{
  // INT64_MIN / -1 is the canonical signed-integer division overflow.
  decimal64 const a{
    scaled_integer<int64_t>{std::numeric_limits<int64_t>::min(), scale_type{0}}};
  decimal64 const b{scaled_integer<int64_t>{-1, scale_type{0}}};
  auto const r = detail::safe_div(a, b);
  EXPECT_TRUE(r.overflow);
}

// ---------------------------------------------------------------------------
// safe_mod / safe_pmod / safe_pymod
// ---------------------------------------------------------------------------

TEST_F(SafeArithmeticTest, ModNoOverflow)
{
  decimal64 const a{scaled_integer<int64_t>{7, scale_type{0}}};
  decimal64 const b{scaled_integer<int64_t>{3, scale_type{0}}};
  auto const r = detail::safe_mod(a, b);
  EXPECT_FALSE(r.overflow);
  EXPECT_EQ(r.value.value(), int64_t{1});
}

TEST_F(SafeArithmeticTest, PModMatchesOpsPModSemantics)
{
  // -7 % 3 should be -1 from %, then pmod produces 2.
  decimal64 const a{scaled_integer<int64_t>{-7, scale_type{0}}};
  decimal64 const b{scaled_integer<int64_t>{3, scale_type{0}}};
  auto const r = detail::safe_pmod(a, b);
  EXPECT_FALSE(r.overflow);
  EXPECT_EQ(r.value.value(), int64_t{2});
}

TEST_F(SafeArithmeticTest, PyModMatchesOpsPyModSemantics)
{
  // ((x % y) + y) % y == 2 for x = -7, y = 3
  decimal64 const a{scaled_integer<int64_t>{-7, scale_type{0}}};
  decimal64 const b{scaled_integer<int64_t>{3, scale_type{0}}};
  auto const r = detail::safe_pymod(a, b);
  EXPECT_FALSE(r.overflow);
  EXPECT_EQ(r.value.value(), int64_t{2});
}

// ---------------------------------------------------------------------------
// safe_rescaled
// ---------------------------------------------------------------------------

TEST_F(SafeArithmeticTest, RescaledNoOpDoesNotSetFlag)
{
  decimal64 const a{
    scaled_integer<int64_t>{std::numeric_limits<int64_t>::max() / 2, scale_type{0}}};
  auto const r = detail::safe_rescaled(a, scale_type{0});
  EXPECT_FALSE(r.overflow);
  EXPECT_EQ(r.value.value(), std::numeric_limits<int64_t>::max() / 2);
}

TEST_F(SafeArithmeticTest, RescaledShiftOverflowSetsFlag)
{
  decimal64 const a{
    scaled_integer<int64_t>{std::numeric_limits<int64_t>::max() / 2, scale_type{0}}};
  // Rescaling to a more negative scale multiplies by a power of 10 and overflows.
  auto const r = detail::safe_rescaled(a, scale_type{-3});
  EXPECT_TRUE(r.overflow);
}

// ---------------------------------------------------------------------------
// safe_convert_floating_to_fixed
// ---------------------------------------------------------------------------

TEST_F(SafeArithmeticTest, ConvertFloatingToDecimal32DetectsPositiveOverflow)
{
  auto const r = detail::safe_convert_floating_to_fixed<decimal32>(1e20, scale_type{0});
  EXPECT_TRUE(r.overflow);
  EXPECT_EQ(r.value.value(), std::numeric_limits<int32_t>::max());
}

TEST_F(SafeArithmeticTest, ConvertFloatingToDecimal32DetectsNegativeOverflow)
{
  auto const r = detail::safe_convert_floating_to_fixed<decimal32>(-1e20, scale_type{0});
  EXPECT_TRUE(r.overflow);
  EXPECT_EQ(r.value.value(), std::numeric_limits<int32_t>::min());
}

TEST_F(SafeArithmeticTest, ConvertFloatingToDecimal64DetectsPositiveOverflowViaScale)
{
  // Overflow via scale factor multiplication even for a "moderate" input.
  // scale -19 implies multiplying by 10^19 in the decimal rep.
  auto const r = detail::safe_convert_floating_to_fixed<decimal64>(1.0, scale_type{-19});
  EXPECT_TRUE(r.overflow);
  EXPECT_EQ(r.value.value(), std::numeric_limits<int64_t>::max());
}

TEST_F(SafeArithmeticTest, ConvertFloatingToDecimal64NoOverflow)
{
  auto const r = detail::safe_convert_floating_to_fixed<decimal64>(123.456, scale_type{-3});
  EXPECT_FALSE(r.overflow);
  EXPECT_EQ(r.value.value(), int64_t{123456});
}

TEST_F(SafeArithmeticTest, ConvertFloatingToDecimal64DetectsNegativeOverflowViaScale)
{
  // Mirrors the positive-overflow-via-scale test for the negative branch:
  // -1.0 at scale -19 implies multiplying by 10^19, which exceeds INT64 range.
  auto const r = detail::safe_convert_floating_to_fixed<decimal64>(-1.0, scale_type{-19});
  EXPECT_TRUE(r.overflow);
  EXPECT_EQ(r.value.value(), std::numeric_limits<int64_t>::min());
}

TEST_F(SafeArithmeticTest, ConvertFloatToDecimal32NoOverflow)
{
  // 1.5f is exactly representable in float, and 1.5 at scale -1 is the integer 15.
  auto const r = detail::safe_convert_floating_to_fixed<decimal32>(1.5f, scale_type{-1});
  EXPECT_FALSE(r.overflow);
  EXPECT_EQ(r.value.value(), int32_t{15});
}

TEST_F(SafeArithmeticTest, ConvertFloatToDecimal32DetectsOverflow)
{
  // Float overflow path exercises the 32-bit ShiftingRep branch.
  auto const r = detail::safe_convert_floating_to_fixed<decimal32>(1e20f, scale_type{0});
  EXPECT_TRUE(r.overflow);
  EXPECT_EQ(r.value.value(), std::numeric_limits<int32_t>::max());
}

TEST_F(SafeArithmeticTest, ConvertFloatingToDecimal128DetectsOverflow)
{
  // INT128_MAX is roughly 1.7e38; 1.0 at scale -39 implies 10^39 and overflows.
  auto const r = detail::safe_convert_floating_to_fixed<decimal128>(1.0, scale_type{-39});
  EXPECT_TRUE(r.overflow);
  EXPECT_EQ(r.value.value(), std::numeric_limits<__int128_t>::max());
}

TEST_F(SafeArithmeticTest, ConvertFloatingToDecimal128NoOverflow)
{
  // 1.0 at scale -30 is 10^30, which fits in INT128.
  auto const r = detail::safe_convert_floating_to_fixed<decimal128>(1.0, scale_type{-30});
  EXPECT_FALSE(r.overflow);
}

TEST_F(SafeArithmeticTest, ConvertFloatingToDecimal64PositiveScaleNoOverflow)
{
  // pow10 > 0 path: 12300.0 at scale +2 is 12300 / 10^2 = 123.
  auto const r = detail::safe_convert_floating_to_fixed<decimal64>(12300.0, scale_type{2});
  EXPECT_FALSE(r.overflow);
  EXPECT_EQ(r.value.value(), int64_t{123});
}

TEST_F(SafeArithmeticTest, ConvertFloatingToDecimal64PositiveScaleDetectsOverflow)
{
  // pow10 > 0 path with input far above INT64*10^pow10:
  // 1e25 / 10^5 = 1e20 > INT64_MAX (~9.2e18).
  auto const r = detail::safe_convert_floating_to_fixed<decimal64>(1e25, scale_type{5});
  EXPECT_TRUE(r.overflow);
  EXPECT_EQ(r.value.value(), std::numeric_limits<int64_t>::max());
}

TEST_F(SafeArithmeticTest, ConvertFloatingZeroIsExact)
{
  auto const r = detail::safe_convert_floating_to_fixed<decimal64>(0.0, scale_type{-3});
  EXPECT_FALSE(r.overflow);
  EXPECT_EQ(r.value.value(), int64_t{0});
}

TEST_F(SafeArithmeticTest, ConvertFloatingNegativeZeroIsExact)
{
  auto const r = detail::safe_convert_floating_to_fixed<decimal64>(-0.0, scale_type{-3});
  EXPECT_FALSE(r.overflow);
  EXPECT_EQ(r.value.value(), int64_t{0});
}

// ---------------------------------------------------------------------------
// Composability: callers OR the overflow flags from a chain explicitly.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-width overflow coverage: decimal32 / decimal64 / decimal128
//
// `safe_*` operations preserve the input `Rep`; they do not upcast the
// representation type. These compile-time assertions verify that contract,
// and the runtime tests exercise overflow detection on each width.
// ---------------------------------------------------------------------------

namespace {

// `safe_result` returned by each operation must keep the operand `Rep`/`Rad`.
// Use `declval` to keep the check in an unevaluated context (the `fixed_point`
// default constructor is not `constexpr`).
template <typename Fixed>
constexpr bool safe_ops_preserve_rep()
{
  using Rep          = typename Fixed::rep;
  constexpr auto Rad = Fixed::rad;
  using Expected     = numeric::detail::safe_result<Rep, Rad>;
  return std::is_same_v<decltype(detail::safe_add(std::declval<Fixed>(), std::declval<Fixed>())),
                        Expected> &&
         std::is_same_v<decltype(detail::safe_sub(std::declval<Fixed>(), std::declval<Fixed>())),
                        Expected> &&
         std::is_same_v<decltype(detail::safe_mul(std::declval<Fixed>(), std::declval<Fixed>())),
                        Expected> &&
         std::is_same_v<decltype(detail::safe_div(std::declval<Fixed>(), std::declval<Fixed>())),
                        Expected>;
}

static_assert(safe_ops_preserve_rep<decimal32>(),
              "safe_* over decimal32 must preserve int32_t representation");
static_assert(safe_ops_preserve_rep<decimal64>(),
              "safe_* over decimal64 must preserve int64_t representation");
static_assert(safe_ops_preserve_rep<decimal128>(),
              "safe_* over decimal128 must preserve __int128_t representation");

}  // namespace

// --- decimal32 -------------------------------------------------------------

TEST_F(SafeArithmeticTest, Add32OverflowSetsFlag)
{
  auto constexpr near_max = std::numeric_limits<int32_t>::max() - 100;
  decimal32 const a{scaled_integer<int32_t>{near_max, scale_type{0}}};
  decimal32 const b{scaled_integer<int32_t>{200, scale_type{0}}};
  auto const r = detail::safe_add(a, b);
  EXPECT_TRUE(r.overflow);
}

TEST_F(SafeArithmeticTest, Sub32OverflowSetsFlag)
{
  auto constexpr near_min = std::numeric_limits<int32_t>::min() + 100;
  decimal32 const a{scaled_integer<int32_t>{near_min, scale_type{0}}};
  decimal32 const b{scaled_integer<int32_t>{200, scale_type{0}}};
  auto const r = detail::safe_sub(a, b);
  EXPECT_TRUE(r.overflow);
}

TEST_F(SafeArithmeticTest, Div32IntMinByNegativeOneOverflows)
{
  decimal32 const a{
    scaled_integer<int32_t>{std::numeric_limits<int32_t>::min(), scale_type{0}}};
  decimal32 const b{scaled_integer<int32_t>{-1, scale_type{0}}};
  auto const r = detail::safe_div(a, b);
  EXPECT_TRUE(r.overflow);
}

// --- decimal128 ------------------------------------------------------------

TEST_F(SafeArithmeticTest, Add128OverflowSetsFlag)
{
  auto const near_max = std::numeric_limits<__int128_t>::max() - __int128_t{100};
  decimal128 const a{scaled_integer<__int128_t>{near_max, scale_type{0}}};
  decimal128 const b{scaled_integer<__int128_t>{200, scale_type{0}}};
  auto const r = detail::safe_add(a, b);
  EXPECT_TRUE(r.overflow);
}

TEST_F(SafeArithmeticTest, Sub128OverflowSetsFlag)
{
  auto const near_min = std::numeric_limits<__int128_t>::min() + __int128_t{100};
  decimal128 const a{scaled_integer<__int128_t>{near_min, scale_type{0}}};
  decimal128 const b{scaled_integer<__int128_t>{200, scale_type{0}}};
  auto const r = detail::safe_sub(a, b);
  EXPECT_TRUE(r.overflow);
}

TEST_F(SafeArithmeticTest, Mul128OverflowSetsFlag)
{
  // 10^20 * 10^20 = 10^40, which exceeds INT128_MAX (~1.7e38).
  auto const ten_pow_20 = static_cast<__int128_t>(1'000'000'000'000'000'000LL) * __int128_t{100};
  decimal128 const a{scaled_integer<__int128_t>{ten_pow_20, scale_type{0}}};
  decimal128 const b{scaled_integer<__int128_t>{ten_pow_20, scale_type{0}}};
  auto const r = detail::safe_mul(a, b);
  EXPECT_TRUE(r.overflow);
}

TEST_F(SafeArithmeticTest, Mul128NoOverflow)
{
  decimal128 const a{scaled_integer<__int128_t>{__int128_t{12345}, scale_type{0}}};
  decimal128 const b{scaled_integer<__int128_t>{__int128_t{6789}, scale_type{0}}};
  auto const r = detail::safe_mul(a, b);
  EXPECT_FALSE(r.overflow);
  EXPECT_EQ(r.value.value(), __int128_t{12345} * __int128_t{6789});
}

TEST_F(SafeArithmeticTest, Div128IntMinByNegativeOneOverflows)
{
  decimal128 const a{
    scaled_integer<__int128_t>{std::numeric_limits<__int128_t>::min(), scale_type{0}}};
  decimal128 const b{scaled_integer<__int128_t>{-1, scale_type{0}}};
  auto const r = detail::safe_div(a, b);
  EXPECT_TRUE(r.overflow);
}

TEST_F(SafeArithmeticTest, CallerComposesOverflowAcrossChain)
{
  // (a * b) overflows; the caller is responsible for OR'ing flags as the
  // chain progresses.
  decimal64 const a{scaled_integer<int64_t>{1'000'000'000'000LL, scale_type{0}}};
  decimal64 const b{scaled_integer<int64_t>{1'000'000'000'000LL, scale_type{0}}};
  decimal64 const c{scaled_integer<int64_t>{0, scale_type{0}}};

  auto const m1 = detail::safe_mul(a, b);  // overflow here
  auto const m2 = detail::safe_add(m1.value, c);  // add itself doesn't overflow
  bool const composed_overflow = m1.overflow || m2.overflow;
  EXPECT_TRUE(composed_overflow);
}

// ---------------------------------------------------------------------------
// Floating <-> decimal conversion overflow primitives
//
// `safe_convert_floating_to_fixed` (above) exercises the conversion path
// end-to-end, but the lower-level `CheckOverflow=true` primitives in
// `floating_conversion.hpp` (`checked_left_shift`, `multiply_power10_saturating`,
// `checked_narrow_cast`) and the top-level `convert_floating_to_integral`
// gain little direct test coverage from that. The tests below pin down each
// primitive's contract: on overflow, return the saturated max with the flag
// set; otherwise return the exact result with the flag clear.
// ---------------------------------------------------------------------------

// --- checked_left_shift<true>(value, bit_shift) ----------------------------

TEST_F(FloatingConversionOverflowTest, CheckedLeftShiftZeroIsIdentity)
{
  auto const [v, ovf] = numeric::detail::checked_left_shift<true>(uint64_t{42}, 0);
  EXPECT_EQ(v, uint64_t{42});
  EXPECT_FALSE(ovf);
}

TEST_F(FloatingConversionOverflowTest, CheckedLeftShiftWithinRange)
{
  auto const [v, ovf] = numeric::detail::checked_left_shift<true>(uint64_t{0x1234}, 4);
  EXPECT_EQ(v, uint64_t{0x12340});
  EXPECT_FALSE(ovf);
}

TEST_F(FloatingConversionOverflowTest, CheckedLeftShiftAtBoundaryDoesNotOverflow)
{
  // value == max >> bit_shift is the largest input that still fits after the shift.
  constexpr int bit_shift = 8;
  auto const at_boundary  = std::numeric_limits<uint32_t>::max() >> bit_shift;
  auto const [v, ovf]     = numeric::detail::checked_left_shift<true>(at_boundary, bit_shift);
  EXPECT_EQ(v, static_cast<uint32_t>(at_boundary << bit_shift));
  EXPECT_FALSE(ovf);
}

TEST_F(FloatingConversionOverflowTest, CheckedLeftShiftJustOverBoundarySaturates)
{
  constexpr int bit_shift = 8;
  auto const just_over    = (std::numeric_limits<uint32_t>::max() >> bit_shift) + 1u;
  auto const [v, ovf]     = numeric::detail::checked_left_shift<true>(just_over, bit_shift);
  EXPECT_EQ(v, std::numeric_limits<uint32_t>::max());
  EXPECT_TRUE(ovf);
}

TEST_F(FloatingConversionOverflowTest, CheckedLeftShiftTooLargeBitShiftSaturates)
{
  // Shifting by >= digits is UB on the underlying operator and saturates here.
  auto const [v, ovf] =
    numeric::detail::checked_left_shift<true>(uint64_t{1}, std::numeric_limits<uint64_t>::digits);
  EXPECT_EQ(v, std::numeric_limits<uint64_t>::max());
  EXPECT_TRUE(ovf);
}

TEST_F(FloatingConversionOverflowTest, CheckedLeftShiftNegativeBitShiftSaturates)
{
  // Negative shifts aren't expected from callers; the checked path treats them
  // as overflow rather than falling into the underlying UB path.
  auto const [v, ovf] = numeric::detail::checked_left_shift<true>(uint64_t{1}, -1);
  EXPECT_EQ(v, std::numeric_limits<uint64_t>::max());
  EXPECT_TRUE(ovf);
}

TEST_F(FloatingConversionOverflowTest, CheckedLeftShiftUncheckedNeverOverflows)
{
  // CheckOverflow=false (the default) must always report overflow=false.
  auto const [v, ovf] = numeric::detail::checked_left_shift(uint64_t{0xFFFF}, 60);
  EXPECT_FALSE(ovf);
  // value is whatever guarded_left_shift returns; only the flag is tested here.
  (void)v;
}

// --- multiply_power10_saturating<T, true>(value, pow10) --------------------

TEST_F(FloatingConversionOverflowTest, MultiplyPower10ZeroPow10IsIdentity)
{
  auto const [v, ovf] = numeric::detail::multiply_power10_saturating<uint64_t, true>(123, 0);
  EXPECT_EQ(v, uint64_t{123});
  EXPECT_FALSE(ovf);
}

TEST_F(FloatingConversionOverflowTest, MultiplyPower10NegativePow10IsIdentity)
{
  auto const [v, ovf] = numeric::detail::multiply_power10_saturating<uint64_t, true>(123, -5);
  EXPECT_EQ(v, uint64_t{123});
  EXPECT_FALSE(ovf);
}

TEST_F(FloatingConversionOverflowTest, MultiplyPower10WithinRange)
{
  auto const [v, ovf] = numeric::detail::multiply_power10_saturating<uint64_t, true>(123, 6);
  EXPECT_EQ(v, uint64_t{123'000'000});
  EXPECT_FALSE(ovf);
}

TEST_F(FloatingConversionOverflowTest, MultiplyPower10AtBoundaryDoesNotOverflow)
{
  // value == max / 10^pow10 is the largest input that fits after the multiply.
  // For uint32_t and pow10=3: 4294967295 / 1000 == 4294967.
  uint32_t const at_boundary = std::numeric_limits<uint32_t>::max() / 1000u;
  auto const [v, ovf] =
    numeric::detail::multiply_power10_saturating<uint32_t, true>(at_boundary, 3);
  EXPECT_EQ(v, at_boundary * 1000u);
  EXPECT_FALSE(ovf);
}

TEST_F(FloatingConversionOverflowTest, MultiplyPower10JustOverBoundarySaturates)
{
  uint32_t const just_over = (std::numeric_limits<uint32_t>::max() / 1000u) + 1u;
  auto const [v, ovf] =
    numeric::detail::multiply_power10_saturating<uint32_t, true>(just_over, 3);
  EXPECT_EQ(v, std::numeric_limits<uint32_t>::max());
  EXPECT_TRUE(ovf);
}

TEST_F(FloatingConversionOverflowTest, MultiplyPower10HugePow10SaturatesNonzero)
{
  // 10^pow10 itself doesn't fit in uint32_t (10^10 > 2^32); divide_power10 returns 0,
  // so the threshold is 0 and any nonzero value saturates.
  auto const [v, ovf] = numeric::detail::multiply_power10_saturating<uint32_t, true>(1, 12);
  EXPECT_EQ(v, std::numeric_limits<uint32_t>::max());
  EXPECT_TRUE(ovf);
}

TEST_F(FloatingConversionOverflowTest, MultiplyPower10ZeroValueNeverOverflows)
{
  auto const [v, ovf] = numeric::detail::multiply_power10_saturating<uint32_t, true>(0, 12);
  EXPECT_EQ(v, uint32_t{0});
  EXPECT_FALSE(ovf);
}

// --- checked_narrow_cast<To, true>(value) ----------------------------------

TEST_F(FloatingConversionOverflowTest, CheckedNarrowCastFits)
{
  auto const [v, ovf] = numeric::detail::checked_narrow_cast<uint32_t, true>(uint64_t{12345});
  EXPECT_EQ(v, uint32_t{12345});
  EXPECT_FALSE(ovf);
}

TEST_F(FloatingConversionOverflowTest, CheckedNarrowCastAtBoundaryDoesNotOverflow)
{
  auto const at_max =
    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
  auto const [v, ovf] = numeric::detail::checked_narrow_cast<uint32_t, true>(at_max);
  EXPECT_EQ(v, std::numeric_limits<uint32_t>::max());
  EXPECT_FALSE(ovf);
}

TEST_F(FloatingConversionOverflowTest, CheckedNarrowCastOverflowSaturates)
{
  auto const just_over =
    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1u;
  auto const [v, ovf] = numeric::detail::checked_narrow_cast<uint32_t, true>(just_over);
  EXPECT_EQ(v, std::numeric_limits<uint32_t>::max());
  EXPECT_TRUE(ovf);
}

TEST_F(FloatingConversionOverflowTest, CheckedNarrowCastUncheckedNeverOverflows)
{
  // CheckOverflow=false truncates without ever setting the flag.
  auto const big      = uint64_t{0x1'0000'0000ULL};
  auto const [v, ovf] = numeric::detail::checked_narrow_cast<uint32_t>(big);
  EXPECT_EQ(v, uint32_t{0});  // low 32 bits
  EXPECT_FALSE(ovf);
}

// --- convert_floating_to_integral<Rep, true>(floating, scale) --------------
//
// The end-to-end conversion entry point. These tests target paths that the
// `safe_convert_floating_to_fixed` wrapper tests above don't exercise directly,
// in particular the sign-reapply branches that distinguish INT_MIN-exact from
// representational overflow.

TEST_F(FloatingConversionOverflowTest, ConvertIntegralZero)
{
  auto const [v, ovf] =
    numeric::detail::convert_floating_to_integral<int64_t, true>(0.0, scale_type{-3});
  EXPECT_EQ(v, int64_t{0});
  EXPECT_FALSE(ovf);
}

TEST_F(FloatingConversionOverflowTest, ConvertIntegralPositiveFits)
{
  auto const [v, ovf] =
    numeric::detail::convert_floating_to_integral<int64_t, true>(123.456, scale_type{-3});
  EXPECT_EQ(v, int64_t{123456});
  EXPECT_FALSE(ovf);
}

TEST_F(FloatingConversionOverflowTest, ConvertIntegralPositiveSaturatesToMax)
{
  // |1e20| >> INT32_MAX (~2.15e9) so the sign-reapply path saturates to max with overflow=true.
  auto const [v, ovf] =
    numeric::detail::convert_floating_to_integral<int32_t, true>(1e20, scale_type{0});
  EXPECT_EQ(v, std::numeric_limits<int32_t>::max());
  EXPECT_TRUE(ovf);
}

TEST_F(FloatingConversionOverflowTest, ConvertIntegralNegativeIntMinExactDoesNotOverflow)
{
  // INT32_MIN == -2^31 = -2147483648 is exactly representable in IEEE 754 double.
  // magnitude_u == umin_mag, which the checked path maps to {INT32_MIN, overflow=shifting_ovf}.
  // No shifting overflow occurs for this input, so the flag should remain false.
  auto const int32_min_as_double = static_cast<double>(std::numeric_limits<int32_t>::min());
  auto const [v, ovf]            = numeric::detail::convert_floating_to_integral<int32_t, true>(
    int32_min_as_double, scale_type{0});
  EXPECT_EQ(v, std::numeric_limits<int32_t>::min());
  EXPECT_FALSE(ovf);
}

TEST_F(FloatingConversionOverflowTest, ConvertIntegralNegativeJustBelowIntMinSaturates)
{
  // One step past INT64_MIN: magnitude_u > umin_mag -> {INT64_MIN, true}.
  auto const [v, ovf] =
    numeric::detail::convert_floating_to_integral<int64_t, true>(-1e20, scale_type{0});
  EXPECT_EQ(v, std::numeric_limits<int64_t>::min());
  EXPECT_TRUE(ovf);
}

TEST_F(FloatingConversionOverflowTest, ConvertIntegralOverflowViaNegativeScale)
{
  // Scale -19 multiplies by 10^19 in the decimal rep, which exceeds INT64 range.
  auto const [v, ovf] =
    numeric::detail::convert_floating_to_integral<int64_t, true>(1.0, scale_type{-19});
  EXPECT_EQ(v, std::numeric_limits<int64_t>::max());
  EXPECT_TRUE(ovf);
}

TEST_F(FloatingConversionOverflowTest, ConvertIntegralOverflowViaPositiveScale)
{
  // pow10 > 0 path: 1e25 / 10^5 == 1e20 > INT64_MAX (~9.2e18).
  auto const [v, ovf] =
    numeric::detail::convert_floating_to_integral<int64_t, true>(1e25, scale_type{5});
  EXPECT_EQ(v, std::numeric_limits<int64_t>::max());
  EXPECT_TRUE(ovf);
}

TEST_F(FloatingConversionOverflowTest, ConvertIntegralUncheckedSameValueAsChecked)
{
  // The unchecked path must produce the same Rep value as the checked path's
  // first element for inputs that don't overflow (no regression in the
  // unchecked code path after the refactor).
  double const x        = 123.456;
  auto const checked    = numeric::detail::convert_floating_to_integral<int64_t, true>(
    x, scale_type{-3});
  auto const unchecked  = numeric::detail::convert_floating_to_integral<int64_t>(
    x, scale_type{-3});
  EXPECT_EQ(checked.first, unchecked);
}

CUDF_TEST_PROGRAM_MAIN()
