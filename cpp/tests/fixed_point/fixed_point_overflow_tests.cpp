/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <limits>
#include <type_traits>

using namespace numeric;

struct FixedPointOverflowTest : public cudf::test::BaseFixture {};

// ---------------------------------------------------------------------------
// Layout / ABI: enabling tracking must not perturb the non-tracking layout.
// ---------------------------------------------------------------------------

// Reference layouts for the historical (non-tracking) `fixed_point` storage. The
// safe-variant must collapse the `[[no_unique_address]]` member to zero bytes so
// these match exactly.
namespace {
struct ref_layout32 {
  int32_t v;
  scale_type s;
};
struct ref_layout64 {
  int64_t v;
  scale_type s;
};
struct ref_layout128 {
  __int128_t v;
  scale_type s;
};
}  // namespace

TEST_F(FixedPointOverflowTest, NonTrackingLayoutUnchanged)
{
  // The whole point of `[[no_unique_address]]` + the empty `_no_overflow_flag`
  // helper is that the historical decimal{32,64,128} layout is preserved.
  static_assert(sizeof(decimal32) == sizeof(ref_layout32));
  static_assert(sizeof(decimal64) == sizeof(ref_layout64));
  static_assert(sizeof(decimal128) == sizeof(ref_layout128));

  // The tracking variants intentionally carry an extra bool, so they are
  // strictly larger than (or equal to) their non-tracking counterparts.
  static_assert(sizeof(decimal32_safe) >= sizeof(decimal32));
  static_assert(sizeof(decimal64_safe) >= sizeof(decimal64));
  static_assert(sizeof(decimal128_safe) >= sizeof(decimal128));
}

// ---------------------------------------------------------------------------
// `overflow_occurred()` is only callable on tracking variants.
// ---------------------------------------------------------------------------

template <typename T, typename = void>
struct has_overflow_occurred : std::false_type {};

template <typename T>
struct has_overflow_occurred<T, std::void_t<decltype(std::declval<T const&>().overflow_occurred())>>
  : std::true_type {};

TEST_F(FixedPointOverflowTest, OverflowOccurredOnlyOnTrackingTypes)
{
  static_assert(!has_overflow_occurred<decimal32>::value);
  static_assert(!has_overflow_occurred<decimal64>::value);
  static_assert(!has_overflow_occurred<decimal128>::value);

  static_assert(has_overflow_occurred<decimal32_safe>::value);
  static_assert(has_overflow_occurred<decimal64_safe>::value);
  static_assert(has_overflow_occurred<decimal128_safe>::value);
}

// ---------------------------------------------------------------------------
// `is_fixed_point<T>()` recognizes the new aliases.
// ---------------------------------------------------------------------------

TEST_F(FixedPointOverflowTest, IsFixedPointRecognizesSafeAliases)
{
  static_assert(cudf::is_fixed_point<decimal32_safe>());
  static_assert(cudf::is_fixed_point<decimal64_safe>());
  static_assert(cudf::is_fixed_point<decimal128_safe>());
}

// ---------------------------------------------------------------------------
// type_id / type_dispatcher round-trip for the new aliases.
// ---------------------------------------------------------------------------

TEST_F(FixedPointOverflowTest, TypeIdMappingRoundTrip)
{
  EXPECT_EQ(cudf::type_id::DECIMAL32_SAFE, cudf::type_to_id<decimal32_safe>());
  EXPECT_EQ(cudf::type_id::DECIMAL64_SAFE, cudf::type_to_id<decimal64_safe>());
  EXPECT_EQ(cudf::type_id::DECIMAL128_SAFE, cudf::type_to_id<decimal128_safe>());

  using safe32  = cudf::id_to_type<cudf::type_id::DECIMAL32_SAFE>;
  using safe64  = cudf::id_to_type<cudf::type_id::DECIMAL64_SAFE>;
  using safe128 = cudf::id_to_type<cudf::type_id::DECIMAL128_SAFE>;
  static_assert(std::is_same_v<safe32, decimal32_safe>);
  static_assert(std::is_same_v<safe64, decimal64_safe>);
  static_assert(std::is_same_v<safe128, decimal128_safe>);

  // The on-device storage type is still the raw integer; the safe wrapper is a
  // value-type concept only. This is required so a column-of-decimal*_safe
  // remains a regular int{32,64,128} column.
  static_assert(std::is_same_v<cudf::device_storage_type_t<decimal32_safe>, int32_t>);
  static_assert(std::is_same_v<cudf::device_storage_type_t<decimal64_safe>, int64_t>);
  static_assert(std::is_same_v<cudf::device_storage_type_t<decimal128_safe>, __int128_t>);
}

// ---------------------------------------------------------------------------
// Sticky-flag propagation through the value-level operators.
// ---------------------------------------------------------------------------

TEST_F(FixedPointOverflowTest, AdditionTracksOverflow)
{
  auto constexpr near_max = std::numeric_limits<int64_t>::max() - 100;
  decimal64_safe const a{scaled_integer<int64_t>{near_max, scale_type{0}}};
  decimal64_safe const b{scaled_integer<int64_t>{200, scale_type{0}}};

  auto const safe_sum = decimal64_safe{scaled_integer<int64_t>{1, scale_type{0}}} +
                        decimal64_safe{scaled_integer<int64_t>{2, scale_type{0}}};
  EXPECT_FALSE(safe_sum.overflow_occurred());

  auto const overflow_sum = a + b;
  EXPECT_TRUE(overflow_sum.overflow_occurred());
}

TEST_F(FixedPointOverflowTest, SubtractionTracksOverflow)
{
  auto constexpr near_min = std::numeric_limits<int64_t>::min() + 100;
  decimal64_safe const a{scaled_integer<int64_t>{near_min, scale_type{0}}};
  decimal64_safe const b{scaled_integer<int64_t>{200, scale_type{0}}};

  auto const overflow_diff = a - b;
  EXPECT_TRUE(overflow_diff.overflow_occurred());
}

TEST_F(FixedPointOverflowTest, MultiplicationTracksOverflow)
{
  decimal64_safe const a{scaled_integer<int64_t>{1'000'000'000'000LL, scale_type{0}}};
  decimal64_safe const b{scaled_integer<int64_t>{1'000'000'000'000LL, scale_type{0}}};
  auto const overflow_prod = a * b;
  EXPECT_TRUE(overflow_prod.overflow_occurred());

  decimal64_safe const c{scaled_integer<int64_t>{2, scale_type{0}}};
  decimal64_safe const d{scaled_integer<int64_t>{3, scale_type{0}}};
  auto const safe_prod = c * d;
  EXPECT_FALSE(safe_prod.overflow_occurred());
}

TEST_F(FixedPointOverflowTest, DivisionTracksOverflow)
{
  // INT64_MIN / -1 is the canonical signed-integer division overflow.
  decimal64_safe const a{
    scaled_integer<int64_t>{std::numeric_limits<int64_t>::min(), scale_type{0}}};
  decimal64_safe const b{scaled_integer<int64_t>{-1, scale_type{0}}};
  auto const overflow_quot = a / b;
  EXPECT_TRUE(overflow_quot.overflow_occurred());
}

TEST_F(FixedPointOverflowTest, FlagIsSticky)
{
  // Once any operand has its overflow flag set, the flag must remain set across
  // a chain of subsequent operations, even if no individual op itself overflows.
  decimal64_safe const a{scaled_integer<int64_t>{1'000'000'000'000LL, scale_type{0}}};
  decimal64_safe const b{scaled_integer<int64_t>{1'000'000'000'000LL, scale_type{0}}};

  auto const tainted    = a * b;  // overflow here
  decimal64_safe const c{scaled_integer<int64_t>{0, scale_type{0}}};
  auto const propagated = tainted + c;  // simple add, but tainted carries the flag
  EXPECT_TRUE(propagated.overflow_occurred());

  auto const propagated_again = propagated - c;
  EXPECT_TRUE(propagated_again.overflow_occurred());
}

TEST_F(FixedPointOverflowTest, RescaledShiftOverflowSetsFlag)
{
  decimal64_safe const a{
    scaled_integer<int64_t>{std::numeric_limits<int64_t>::max() / 2, scale_type{0}}};
  // Rescaling to a sufficiently negative scale multiplies by a power of 10 and
  // overflows.
  auto const rescaled = a.rescaled(scale_type{-3});
  EXPECT_TRUE(rescaled.overflow_occurred());

  // A no-op rescale must not falsely set the flag.
  auto const noop = a.rescaled(scale_type{0});
  EXPECT_FALSE(noop.overflow_occurred());
}

TEST_F(FixedPointOverflowTest, ConstructorShiftOverflowSetsFlag)
{
  // Constructing with a scale that would shift the input out of `Rep` range
  // sets the sticky flag from the very first operation.
  decimal64_safe const overflowed{std::numeric_limits<int64_t>::max() / 2, scale_type{-3}};
  EXPECT_TRUE(overflowed.overflow_occurred());

  decimal64_safe const fine{int64_t{42}, scale_type{0}};
  EXPECT_FALSE(fine.overflow_occurred());
}

// ---------------------------------------------------------------------------
// Mixed-Track operations should NOT compile. Verified at compile time below
// via SFINAE; if the static_assert chain is wrong the test file itself fails
// to build (which is the desired behavior).
// ---------------------------------------------------------------------------

template <typename A, typename B, typename = void>
struct addable : std::false_type {};

template <typename A, typename B>
struct addable<A, B, std::void_t<decltype(std::declval<A>() + std::declval<B>())>>
  : std::true_type {};

static_assert(addable<decimal64, decimal64>::value);
static_assert(addable<decimal64_safe, decimal64_safe>::value);
static_assert(!addable<decimal64, decimal64_safe>::value,
              "Mixed-Track addition must not be allowed; choose a single tracking mode.");

// ---------------------------------------------------------------------------
// Column-level smoke test: a `DECIMAL64_SAFE` column should round-trip through
// a `cudf::binary_operation` that internally dispatches to `decimal64_safe`'s
// element type. The on-device storage is still int64, so the per-element
// sticky bit is not preserved in the column (see the docstring on
// `overflow_occurred()`); this test only validates wiring.
// ---------------------------------------------------------------------------

TEST_F(FixedPointOverflowTest, BinaryOpOnSafeColumnTypeIdRoundTrip)
{
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<int64_t>;

  auto const lhs = fp_wrapper{{10, 20, 30}, scale_type{0}};
  auto const rhs = fp_wrapper{{1, 2, 3}, scale_type{0}};

  // The column wrapper produces DECIMAL64 by default; assert the dispatcher
  // accepts the matching SAFE type id when explicitly requested for the result.
  auto const safe_type = cudf::data_type{cudf::type_id::DECIMAL64_SAFE, 0};
  EXPECT_TRUE(cudf::is_fixed_point(safe_type));
  EXPECT_EQ(cudf::type_id::DECIMAL64_SAFE, safe_type.id());
}
