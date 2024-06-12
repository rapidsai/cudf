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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

using namespace numeric;

struct FixedPointTest : public cudf::test::BaseFixture {};

template <typename T>
struct FixedPointTestAllReps : public cudf::test::BaseFixture {};

using RepresentationTypes = ::testing::Types<int32_t, int64_t>;

TYPED_TEST_SUITE(FixedPointTestAllReps, RepresentationTypes);

TYPED_TEST(FixedPointTestAllReps, SimpleDecimalXXConstruction)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  auto num0 = cudf::convert_floating_to_fixed<decimalXX>(1.234567, scale_type(0));
  auto num1 = cudf::convert_floating_to_fixed<decimalXX>(1.234567, scale_type(-1));
  auto num2 = cudf::convert_floating_to_fixed<decimalXX>(1.234567, scale_type(-2));
  auto num3 = cudf::convert_floating_to_fixed<decimalXX>(1.234567, scale_type(-3));
  auto num4 = cudf::convert_floating_to_fixed<decimalXX>(1.234567, scale_type(-4));
  auto num5 = cudf::convert_floating_to_fixed<decimalXX>(1.234567, scale_type(-5));
  auto num6 = cudf::convert_floating_to_fixed<decimalXX>(1.234567, scale_type(-6));

  EXPECT_EQ(1, cudf::convert_fixed_to_floating<double>(num0));
  EXPECT_EQ(1.2, cudf::convert_fixed_to_floating<double>(num1));
  EXPECT_EQ(1.23, cudf::convert_fixed_to_floating<double>(num2));
  EXPECT_EQ(1.234, cudf::convert_fixed_to_floating<double>(num3));
  EXPECT_EQ(1.2345, cudf::convert_fixed_to_floating<double>(num4));
  EXPECT_EQ(1.23456, cudf::convert_fixed_to_floating<double>(num5));
  EXPECT_EQ(1.234567, cudf::convert_fixed_to_floating<double>(num6));
}

TYPED_TEST(FixedPointTestAllReps, SimpleNegativeDecimalXXConstruction)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  auto num0 = cudf::convert_floating_to_fixed<decimalXX>(-1.234567, scale_type(0));
  auto num1 = cudf::convert_floating_to_fixed<decimalXX>(-1.234567, scale_type(-1));
  auto num2 = cudf::convert_floating_to_fixed<decimalXX>(-1.234567, scale_type(-2));
  auto num3 = cudf::convert_floating_to_fixed<decimalXX>(-1.234567, scale_type(-3));
  auto num4 = cudf::convert_floating_to_fixed<decimalXX>(-1.234567, scale_type(-4));
  auto num5 = cudf::convert_floating_to_fixed<decimalXX>(-1.234567, scale_type(-5));
  auto num6 = cudf::convert_floating_to_fixed<decimalXX>(-1.234567, scale_type(-6));

  EXPECT_EQ(-1, cudf::convert_fixed_to_floating<double>(num0));
  EXPECT_EQ(-1.2, cudf::convert_fixed_to_floating<double>(num1));
  EXPECT_EQ(-1.23, cudf::convert_fixed_to_floating<double>(num2));
  EXPECT_EQ(-1.234, cudf::convert_fixed_to_floating<double>(num3));
  EXPECT_EQ(-1.2345, cudf::convert_fixed_to_floating<double>(num4));
  EXPECT_EQ(-1.23456, cudf::convert_fixed_to_floating<double>(num5));
  EXPECT_EQ(-1.234567, cudf::convert_fixed_to_floating<double>(num6));
}

TYPED_TEST(FixedPointTestAllReps, PaddedDecimalXXConstruction)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  auto a = cudf::convert_floating_to_fixed<decimalXX>(1.1, scale_type(-1));
  auto b = cudf::convert_floating_to_fixed<decimalXX>(1.01, scale_type(-2));
  auto c = cudf::convert_floating_to_fixed<decimalXX>(1.001, scale_type(-3));
  auto d = cudf::convert_floating_to_fixed<decimalXX>(1.0001, scale_type(-4));
  auto e = cudf::convert_floating_to_fixed<decimalXX>(1.00001, scale_type(-5));
  auto f = cudf::convert_floating_to_fixed<decimalXX>(1.000001, scale_type(-6));
  auto x = cudf::convert_floating_to_fixed<decimalXX>(1.000123, scale_type(-8));
  auto y = cudf::convert_floating_to_fixed<decimalXX>(0.000123, scale_type(-8));

  EXPECT_EQ(1.1, cudf::convert_fixed_to_floating<double>(a));
  EXPECT_EQ(1.01, cudf::convert_fixed_to_floating<double>(b));
  EXPECT_EQ(1,
            cudf::convert_fixed_to_floating<double>(
              c));  // intentional (inherited problem from floating point)
  EXPECT_EQ(1.0001, cudf::convert_fixed_to_floating<double>(d));
  EXPECT_EQ(1.00001, cudf::convert_fixed_to_floating<double>(e));
  EXPECT_EQ(1,
            cudf::convert_fixed_to_floating<double>(
              f));  // intentional (inherited problem from floating point)

  EXPECT_TRUE(1.000123 - cudf::convert_fixed_to_floating<double>(x) <
              std::numeric_limits<double>::epsilon());
  EXPECT_EQ(0.000123, cudf::convert_fixed_to_floating<double>(y));
}

TYPED_TEST(FixedPointTestAllReps, SimpleBinaryFPConstruction)
{
  using binary_fp = fixed_point<TypeParam, Radix::BASE_2>;

  binary_fp num0{10, scale_type{0}};
  binary_fp num1{10, scale_type{1}};
  binary_fp num2{10, scale_type{2}};
  binary_fp num3{10, scale_type{3}};
  binary_fp num4{10, scale_type{4}};

  auto num5 = cudf::convert_floating_to_fixed<binary_fp>(1.24, scale_type(0));
  auto num6 = cudf::convert_floating_to_fixed<binary_fp>(1.24, scale_type(-1));
  auto num7 = cudf::convert_floating_to_fixed<binary_fp>(1.32, scale_type(-2));
  auto num8 = cudf::convert_floating_to_fixed<binary_fp>(1.41, scale_type(-3));
  auto num9 = cudf::convert_floating_to_fixed<binary_fp>(1.45, scale_type(-4));

  EXPECT_EQ(10, cudf::convert_fixed_to_floating<double>(num0));
  EXPECT_EQ(10, cudf::convert_fixed_to_floating<double>(num1));
  EXPECT_EQ(8, cudf::convert_fixed_to_floating<double>(num2));
  EXPECT_EQ(8, cudf::convert_fixed_to_floating<double>(num3));
  EXPECT_EQ(0, cudf::convert_fixed_to_floating<double>(num4));

  EXPECT_EQ(1, cudf::convert_fixed_to_floating<double>(num5));
  EXPECT_EQ(1, cudf::convert_fixed_to_floating<double>(num6));
  EXPECT_EQ(1.25, cudf::convert_fixed_to_floating<double>(num7));
  EXPECT_EQ(1.375, cudf::convert_fixed_to_floating<double>(num8));
  EXPECT_EQ(1.4375, cudf::convert_fixed_to_floating<double>(num9));
}

TYPED_TEST(FixedPointTestAllReps, MoreSimpleBinaryFPConstruction)
{
  using binary_fp = fixed_point<TypeParam, Radix::BASE_2>;

  auto num0 = cudf::convert_floating_to_fixed<binary_fp>(1.25, scale_type(-2));
  auto num1 = cudf::convert_floating_to_fixed<binary_fp>(2.1, scale_type(-4));

  EXPECT_EQ(1.25, cudf::convert_fixed_to_floating<double>(num0));
  EXPECT_EQ(2.0625, cudf::convert_fixed_to_floating<double>(num1));
}

TYPED_TEST(FixedPointTestAllReps, SimpleDecimalXXMath)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX ONE{1, scale_type{-2}};
  decimalXX TWO{2, scale_type{-2}};
  decimalXX THREE{3, scale_type{-2}};
  decimalXX SIX{6, scale_type{-2}};

  EXPECT_TRUE(ONE + ONE == TWO);

  EXPECT_EQ(ONE + ONE, TWO);
  EXPECT_EQ(ONE * TWO, TWO);
  EXPECT_EQ(THREE * TWO, SIX);
  EXPECT_EQ(THREE - TWO, ONE);
  EXPECT_EQ(TWO / ONE, TWO);
  EXPECT_EQ(SIX / TWO, THREE);

  auto a = cudf::convert_floating_to_fixed<decimalXX>(1.23, scale_type(-2));
  decimalXX b{0, scale_type{0}};

  EXPECT_EQ(a + b, a);
  EXPECT_EQ(a - b, a);
}

TYPED_TEST(FixedPointTestAllReps, ComparisonOperators)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX ONE{1, scale_type{-1}};
  decimalXX TWO{2, scale_type{-2}};
  decimalXX THREE{3, scale_type{-3}};
  decimalXX SIX{6, scale_type{-4}};

  EXPECT_TRUE(ONE + ONE >= TWO);

  EXPECT_TRUE(ONE + ONE <= TWO);
  EXPECT_TRUE(ONE * TWO < THREE);
  EXPECT_TRUE(THREE * TWO > THREE);
  EXPECT_TRUE(THREE - TWO >= ONE);
  EXPECT_TRUE(TWO / ONE < THREE);
  EXPECT_TRUE(SIX / TWO >= ONE);
}

TYPED_TEST(FixedPointTestAllReps, DecimalXXTrickyDivision)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX ONE_1{1, scale_type{1}};
  decimalXX SIX_0{6, scale_type{0}};
  decimalXX SIX_1{6, scale_type{1}};
  decimalXX TEN_0{10, scale_type{0}};
  decimalXX TEN_1{10, scale_type{1}};
  decimalXX SIXTY_1{60, scale_type{1}};

  EXPECT_EQ(static_cast<int32_t>(ONE_1), 0);
  EXPECT_EQ(static_cast<int32_t>(SIX_1), 0);
  EXPECT_EQ(static_cast<int32_t>(TEN_0), 10);
  EXPECT_EQ(static_cast<int32_t>(SIXTY_1), 60);

  EXPECT_EQ(SIXTY_1 / TEN_0, ONE_1);
  EXPECT_EQ(SIXTY_1 / TEN_1, SIX_0);

  auto A = cudf::convert_floating_to_fixed<decimalXX>(34.56, scale_type(-2));
  auto B = cudf::convert_floating_to_fixed<decimalXX>(1.234, scale_type(-3));
  decimalXX C{1, scale_type{-2}};

  EXPECT_EQ(static_cast<int32_t>(A / B), 20);
  EXPECT_EQ(static_cast<int32_t>((A * C) / B), 28);

  decimalXX n{28, scale_type{1}};
  EXPECT_EQ(static_cast<int32_t>(n), 20);
}

TYPED_TEST(FixedPointTestAllReps, DecimalXXRounding)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX ZERO_0{0, scale_type{0}};
  decimalXX ZERO_1{4, scale_type{1}};
  decimalXX THREE_0{3, scale_type{0}};
  decimalXX FOUR_0{4, scale_type{0}};
  decimalXX FIVE_0{5, scale_type{0}};
  decimalXX TEN_0{10, scale_type{0}};
  decimalXX TEN_1{10, scale_type{1}};

  decimalXX FOURTEEN_0{14, scale_type{0}};
  decimalXX FIFTEEN_0{15, scale_type{0}};

  EXPECT_EQ(ZERO_0, ZERO_1);
  EXPECT_EQ(TEN_0, TEN_1);

  EXPECT_EQ(ZERO_1 + TEN_1, TEN_1);
  EXPECT_EQ(FOUR_0 + TEN_1, FOURTEEN_0);
  EXPECT_TRUE(ZERO_0 == ZERO_1);
  EXPECT_TRUE(FIVE_0 != TEN_1);
  EXPECT_TRUE(FIVE_0 + FIVE_0 + FIVE_0 == FIFTEEN_0);
  EXPECT_TRUE(FIVE_0 + FIVE_0 + FIVE_0 != TEN_1);
  EXPECT_TRUE(FIVE_0 * THREE_0 == FIFTEEN_0);
  EXPECT_TRUE(FIVE_0 * THREE_0 != TEN_1);
}

TYPED_TEST(FixedPointTestAllReps, ArithmeticWithDifferentScales)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX a{1, scale_type{0}};
  auto b = cudf::convert_floating_to_fixed<decimalXX>(1.2, scale_type(-1));
  auto c = cudf::convert_floating_to_fixed<decimalXX>(1.23, scale_type(-2));
  auto d = cudf::convert_floating_to_fixed<decimalXX>(1.111, scale_type(-3));

  auto x = cudf::convert_floating_to_fixed<decimalXX>(2.2, scale_type(-1));
  auto y = cudf::convert_floating_to_fixed<decimalXX>(3.43, scale_type(-2));
  auto z = cudf::convert_floating_to_fixed<decimalXX>(4.541, scale_type(-3));

  auto xx = cudf::convert_floating_to_fixed<decimalXX>(0.2, scale_type(-1));
  auto yy = cudf::convert_floating_to_fixed<decimalXX>(0.03, scale_type(-2));
  auto zz = cudf::convert_floating_to_fixed<decimalXX>(0.119, scale_type(-3));

  EXPECT_EQ(a + b, x);
  EXPECT_EQ(a + b + c, y);
  EXPECT_EQ(a + b + c + d, z);
  EXPECT_EQ(b - a, xx);
  EXPECT_EQ(c - b, yy);
  EXPECT_EQ(c - d, zz);
}

TYPED_TEST(FixedPointTestAllReps, RescaledTest)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX num0{1, scale_type{0}};
  auto num1 = cudf::convert_floating_to_fixed<decimalXX>(1.2, scale_type(-1));
  auto num2 = cudf::convert_floating_to_fixed<decimalXX>(1.23, scale_type(-2));
  auto num3 = cudf::convert_floating_to_fixed<decimalXX>(1.234, scale_type(-3));
  auto num4 = cudf::convert_floating_to_fixed<decimalXX>(1.2345, scale_type(-4));
  auto num5 = cudf::convert_floating_to_fixed<decimalXX>(1.23456, scale_type(-5));
  auto num6 = cudf::convert_floating_to_fixed<decimalXX>(1.234567, scale_type(-6));

  EXPECT_EQ(num0, num6.rescaled(scale_type{0}));
  EXPECT_EQ(num1, num6.rescaled(scale_type{-1}));
  EXPECT_EQ(num2, num6.rescaled(scale_type{-2}));
  EXPECT_EQ(num3, num6.rescaled(scale_type{-3}));
  EXPECT_EQ(num4, num6.rescaled(scale_type{-4}));
  EXPECT_EQ(num5, num6.rescaled(scale_type{-5}));
}

TYPED_TEST(FixedPointTestAllReps, RescaledRounding)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX num0{1500, scale_type{0}};
  decimalXX num1{1499, scale_type{0}};
  decimalXX num2{-1499, scale_type{0}};
  decimalXX num3{-1500, scale_type{0}};

  EXPECT_EQ(1000, static_cast<TypeParam>(num0.rescaled(scale_type{3})));
  EXPECT_EQ(1000, static_cast<TypeParam>(num1.rescaled(scale_type{3})));
  EXPECT_EQ(-1000, static_cast<TypeParam>(num2.rescaled(scale_type{3})));
  EXPECT_EQ(-1000, static_cast<TypeParam>(num3.rescaled(scale_type{3})));
}

TYPED_TEST(FixedPointTestAllReps, BoolConversion)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  auto truthy_value = cudf::convert_floating_to_fixed<decimalXX>(1.234567, scale_type(0));
  decimalXX falsy_value{0, scale_type{0}};

  // Test explicit conversions
  EXPECT_EQ(static_cast<bool>(truthy_value), true);
  EXPECT_EQ(static_cast<bool>(falsy_value), false);

  // These operators also *explicitly* convert to bool
  EXPECT_EQ(truthy_value && true, true);
  EXPECT_EQ(true && truthy_value, true);
  EXPECT_EQ(falsy_value || false, false);
  EXPECT_EQ(false || falsy_value, false);
  EXPECT_EQ(!truthy_value, false);
  EXPECT_EQ(!falsy_value, true);
}

// These two overflow tests only work in a Debug build.
// Unfortunately, in a full debug build, the test will each take about
// an hour to run.
// Therefore they are disabled here and can be enabled in an appropriate
// debug build when required.
TEST_F(FixedPointTest, DISABLED_OverflowDecimal32)
{
  // This flag is needed to avoid warnings with ASSERT_DEATH
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  using decimal32 = fixed_point<int32_t, Radix::BASE_10>;

  decimal32 num0{2, scale_type{-9}};
  decimal32 num1{-2, scale_type{-9}};

  ASSERT_DEATH(num0 + num0, ".*");
  ASSERT_DEATH(num1 - num0, ".*");

  decimal32 min{std::numeric_limits<int32_t>::min(), scale_type{0}};
  decimal32 max{std::numeric_limits<int32_t>::max(), scale_type{0}};
  decimal32 NEG_ONE{-1, scale_type{0}};
  decimal32 ONE{1, scale_type{0}};
  decimal32 TWO{2, scale_type{0}};

  ASSERT_DEATH(min / NEG_ONE, ".*");
  ASSERT_DEATH(max * TWO, ".*");
  ASSERT_DEATH(min * TWO, ".*");
  ASSERT_DEATH(max + ONE, ".*");
  ASSERT_DEATH(max - NEG_ONE, ".*");
  ASSERT_DEATH(min - ONE, ".*");
  ASSERT_DEATH(max - NEG_ONE, ".*");
}

// See comment above for OverflowDecimal32 test.
TEST_F(FixedPointTest, DISABLED_OverflowDecimal64)
{
  // This flag is needed to avoid warnings with ASSERT_DEATH
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  using decimal64 = fixed_point<int64_t, Radix::BASE_10>;

  decimal64 num0{5, scale_type{-18}};
  decimal64 num1{-5, scale_type{-18}};

  ASSERT_DEATH(num0 + num0, ".*");
  ASSERT_DEATH(num1 - num0, ".*");

  decimal64 min{std::numeric_limits<int64_t>::min(), scale_type{0}};
  decimal64 max{std::numeric_limits<int64_t>::max(), scale_type{0}};
  decimal64 NEG_ONE{-1, scale_type{0}};
  decimal64 ONE{1, scale_type{0}};
  decimal64 TWO{2, scale_type{0}};

  ASSERT_DEATH(min / NEG_ONE, ".*");
  ASSERT_DEATH(max * TWO, ".*");
  ASSERT_DEATH(min * TWO, ".*");
  ASSERT_DEATH(max + ONE, ".*");
  ASSERT_DEATH(max - NEG_ONE, ".*");
  ASSERT_DEATH(min - ONE, ".*");
  ASSERT_DEATH(max - NEG_ONE, ".*");
}

template <typename ValueType, typename Binop>
void integer_vector_test(ValueType const initial_value,
                         int32_t const size,
                         int32_t const scale,
                         Binop binop)
{
  using decimal32 = fixed_point<int32_t, Radix::BASE_10>;

  std::vector<decimal32> vec1(size);
  std::vector<ValueType> vec2(size);

  std::iota(std::begin(vec1), std::end(vec1), decimal32{initial_value, scale_type{scale}});
  std::iota(std::begin(vec2), std::end(vec2), initial_value);

  auto const res1 =
    std::accumulate(std::cbegin(vec1), std::cend(vec1), decimal32{0, scale_type{scale}});

  auto const res2 = std::accumulate(std::cbegin(vec2), std::cend(vec2), static_cast<ValueType>(0));

  EXPECT_EQ(static_cast<int32_t>(res1), res2);

  std::vector<ValueType> vec3(vec1.size());

  std::transform(std::cbegin(vec1), std::cend(vec1), std::begin(vec3), [](auto const& e) {
    return static_cast<int32_t>(e);
  });

  EXPECT_EQ(vec2, vec3);
}

TEST_F(FixedPointTest, Decimal32IntVector)
{
  integer_vector_test(0, 10, -2, std::plus<>());
  integer_vector_test(0, 1000, -2, std::plus<>());

  integer_vector_test(1, 10, 0, std::multiplies<>());
  integer_vector_test(2, 20, 0, std::multiplies<>());
}

template <typename ValueType, typename Binop>
void float_vector_test(ValueType const initial_value,
                       int32_t const size,
                       int32_t const scale,
                       Binop binop)
{
  using decimal32 = fixed_point<int32_t, Radix::BASE_10>;

  std::vector<decimal32> vec1(size);
  std::vector<ValueType> vec2(size);

  auto decimal_input = cudf::convert_floating_to_fixed<decimal32>(initial_value, scale_type{scale});
  std::iota(std::begin(vec1), std::end(vec1), decimal_input);
  std::iota(std::begin(vec2), std::end(vec2), initial_value);

  auto equal = std::equal(
    std::cbegin(vec1), std::cend(vec1), std::cbegin(vec2), [](auto const& a, auto const& b) {
      return cudf::convert_fixed_to_floating<double>(a) - b <=
             std::numeric_limits<ValueType>::epsilon();
    });

  EXPECT_TRUE(equal);
}

TEST_F(FixedPointTest, Decimal32FloatVector)
{
  float_vector_test(0.1, 1000, -2, std::plus<>());
  float_vector_test(0.15, 1000, -2, std::plus<>());

  float_vector_test(0.1, 10, -2, std::multiplies<>());
  float_vector_test(0.15, 20, -2, std::multiplies<>());
}

struct cast_to_int32_fn {
  using decimal32 = fixed_point<int32_t, Radix::BASE_10>;
  int32_t __host__ __device__ operator()(decimal32 fp) { return static_cast<int32_t>(fp); }
};

TYPED_TEST(FixedPointTestAllReps, FixedPointColumnWrapper)
{
  using namespace numeric;
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;
  using RepType   = TypeParam;

  // fixed_point_column_wrapper
  auto const w = cudf::test::fixed_point_column_wrapper<RepType>{{1, 2, 3, 4}, scale_type{0}};

  // fixed_width_column_wrapper
  auto const ONE   = decimalXX{1, scale_type{0}};
  auto const TWO   = decimalXX{2, scale_type{0}};
  auto const THREE = decimalXX{3, scale_type{0}};
  auto const FOUR  = decimalXX{4, scale_type{0}};

  auto const vec = std::vector<decimalXX>{ONE, TWO, THREE, FOUR};
  auto const col = cudf::test::fixed_width_column_wrapper<decimalXX>(vec.begin(), vec.end());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col, w);
}

TYPED_TEST(FixedPointTestAllReps, NoScaleOrWrongTypeID)
{
  EXPECT_THROW(cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::INT32}, 0),
               cudf::data_type_error);
}

TYPED_TEST(FixedPointTestAllReps, SimpleFixedPointColumnWrapper)
{
  using RepType = cudf::device_storage_type_t<TypeParam>;

  auto const a = cudf::test::fixed_point_column_wrapper<RepType>{{11, 22, 33}, scale_type{-1}};
  auto const b = cudf::test::fixed_point_column_wrapper<RepType>{{110, 220, 330}, scale_type{-2}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(a, b);
}

TEST_F(FixedPointTest, PositiveScaleWithValuesOutsideUnderlyingType32)
{
  // This is testing fixed_point values outside the range of its underlying type.
  // For example, 100,000,000 with scale of 6 is 100,000,000,000,000 (100 trillion) and this is
  // outside the range of a int32_t

  using fp_wrapper = cudf::test::fixed_point_column_wrapper<int32_t>;

  auto const a = fp_wrapper{{100000000}, scale_type{6}};
  auto const b = fp_wrapper{{5000000}, scale_type{7}};
  auto const c = fp_wrapper{{2}, scale_type{0}};

  auto const expected1 = fp_wrapper{{150000000}, scale_type{6}};
  auto const expected2 = fp_wrapper{{50000000}, scale_type{6}};

  auto const type    = cudf::data_type{cudf::type_id::DECIMAL32, 6};
  auto const result1 = cudf::binary_operation(a, b, cudf::binary_operator::ADD, type);
  auto const result2 = cudf::binary_operation(a, c, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, result1->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, result2->view());
}

TEST_F(FixedPointTest, PositiveScaleWithValuesOutsideUnderlyingType64)
{
  // This is testing fixed_point values outside the range of its underlying type.
  // For example, 100,000,000 with scale of 100 is 10 ^ 108 and this is far outside the
  // range of a int64_t

  using fp_wrapper = cudf::test::fixed_point_column_wrapper<int64_t>;

  auto const a = fp_wrapper{{100000000}, scale_type{100}};
  auto const b = fp_wrapper{{5000000}, scale_type{101}};
  auto const c = fp_wrapper{{2}, scale_type{0}};

  auto const expected1 = fp_wrapper{{150000000}, scale_type{100}};
  auto const expected2 = fp_wrapper{{50000000}, scale_type{100}};

  auto const type    = cudf::data_type{cudf::type_id::DECIMAL64, 100};
  auto const result1 = cudf::binary_operation(a, b, cudf::binary_operator::ADD, type);
  auto const result2 = cudf::binary_operation(a, c, cudf::binary_operator::DIV, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, result1->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, result2->view());
}

TYPED_TEST(FixedPointTestAllReps, ExtremelyLargeNegativeScale)
{
  // This is testing fixed_point values with an extremely large negative scale. The fixed_point
  // implementation should be able to handle any scale representable by an int32_t

  using decimalXX  = fixed_point<TypeParam, Radix::BASE_10>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<TypeParam>;

  auto const a = fp_wrapper{{10}, scale_type{-201}};
  auto const b = fp_wrapper{{50}, scale_type{-202}};
  auto const c = fp_wrapper{{2}, scale_type{0}};

  auto const expected1 = fp_wrapper{{150}, scale_type{-202}};
  auto const expected2 = fp_wrapper{{5}, scale_type{-201}};

  auto const type1   = cudf::data_type{cudf::type_to_id<decimalXX>(), -202};
  auto const result1 = cudf::binary_operation(a, b, cudf::binary_operator::ADD, type1);

  auto const type2   = cudf::data_type{cudf::type_to_id<decimalXX>(), -201};
  auto const result2 = cudf::binary_operation(a, c, cudf::binary_operator::DIV, type2);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, result1->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, result2->view());
}

CUDF_TEST_PROGRAM_MAIN()
