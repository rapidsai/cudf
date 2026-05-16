/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/fixed_point/detail/safe_arithmetic.hpp>
#include <cudf/fixed_point/fixed_point.hpp>

#include <limits>

using namespace numeric;

struct SafeArithmeticTest : public cudf::test::BaseFixture {};

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

// ---------------------------------------------------------------------------
// Composability: callers OR the overflow flags from a chain explicitly.
// ---------------------------------------------------------------------------

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

CUDF_TEST_PROGRAM_MAIN()
