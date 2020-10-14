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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

using namespace numeric;

struct FixedPointTest : public cudf::test::BaseFixture {
};

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {
};

using RepresentationTypes = ::testing::Types<int32_t, int64_t>;

TYPED_TEST_CASE(FixedPointTestBothReps, RepresentationTypes);

TYPED_TEST(FixedPointTestBothReps, SimpleDecimalXXConstruction)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX num0{1.234567, scale_type{0}};
  decimalXX num1{1.234567, scale_type{-1}};
  decimalXX num2{1.234567, scale_type{-2}};
  decimalXX num3{1.234567, scale_type{-3}};
  decimalXX num4{1.234567, scale_type{-4}};
  decimalXX num5{1.234567, scale_type{-5}};
  decimalXX num6{1.234567, scale_type{-6}};

  EXPECT_EQ(1, static_cast<double>(num0));
  EXPECT_EQ(1.2, static_cast<double>(num1));
  EXPECT_EQ(1.23, static_cast<double>(num2));
  EXPECT_EQ(1.235, static_cast<double>(num3));    // rounds up
  EXPECT_EQ(1.2346, static_cast<double>(num4));   // rounds up
  EXPECT_EQ(1.23457, static_cast<double>(num5));  // rounds up
  EXPECT_EQ(1.234567, static_cast<double>(num6));
}

TYPED_TEST(FixedPointTestBothReps, SimpleNegativeDecimalXXConstruction)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX num0{-1.234567, scale_type{0}};
  decimalXX num1{-1.234567, scale_type{-1}};
  decimalXX num2{-1.234567, scale_type{-2}};
  decimalXX num3{-1.234567, scale_type{-3}};
  decimalXX num4{-1.234567, scale_type{-4}};
  decimalXX num5{-1.234567, scale_type{-5}};
  decimalXX num6{-1.234567, scale_type{-6}};

  EXPECT_EQ(-1, static_cast<double>(num0));
  EXPECT_EQ(-1.2, static_cast<double>(num1));
  EXPECT_EQ(-1.23, static_cast<double>(num2));
  EXPECT_EQ(-1.235, static_cast<double>(num3));    // rounds up
  EXPECT_EQ(-1.2346, static_cast<double>(num4));   // rounds up
  EXPECT_EQ(-1.23457, static_cast<double>(num5));  // rounds up
  EXPECT_EQ(-1.234567, static_cast<double>(num6));
}

TYPED_TEST(FixedPointTestBothReps, PaddedDecimalXXConstruction)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX a{1.1, scale_type{-1}};
  decimalXX b{1.01, scale_type{-2}};
  decimalXX c{1.001, scale_type{-3}};
  decimalXX d{1.0001, scale_type{-4}};
  decimalXX e{1.00001, scale_type{-5}};
  decimalXX f{1.000001, scale_type{-6}};

  decimalXX x{1.000123, scale_type{-8}};
  decimalXX y{0.000123, scale_type{-8}};

  EXPECT_EQ(1.1, static_cast<double>(a));
  EXPECT_EQ(1.01, static_cast<double>(b));
  EXPECT_EQ(1.001, static_cast<double>(c));
  EXPECT_EQ(1.0001, static_cast<double>(d));
  EXPECT_EQ(1.00001, static_cast<double>(e));
  EXPECT_EQ(1.000001, static_cast<double>(f));

  EXPECT_TRUE(1.000123 - static_cast<double>(x) < std::numeric_limits<double>::epsilon());
  EXPECT_EQ(0.000123, static_cast<double>(y));
}

TYPED_TEST(FixedPointTestBothReps, SimpleBinaryFPConstruction)
{
  using binary_fp = fixed_point<TypeParam, Radix::BASE_2>;

  binary_fp num0{10, scale_type{0}};
  binary_fp num1{10, scale_type{1}};
  binary_fp num2{10, scale_type{2}};
  binary_fp num3{10, scale_type{3}};
  binary_fp num4{10, scale_type{4}};

  binary_fp num5{1.24, scale_type{0}};
  binary_fp num6{1.24, scale_type{-1}};
  binary_fp num7{1.32, scale_type{-2}};
  binary_fp num8{1.41, scale_type{-3}};
  binary_fp num9{1.45, scale_type{-4}};

  EXPECT_EQ(10, static_cast<double>(num0));
  EXPECT_EQ(10, static_cast<double>(num1));
  EXPECT_EQ(12, static_cast<double>(num2));
  EXPECT_EQ(8, static_cast<double>(num3));
  EXPECT_EQ(16, static_cast<double>(num4));

  EXPECT_EQ(1, static_cast<double>(num5));
  EXPECT_EQ(1, static_cast<double>(num6));
  EXPECT_EQ(1.25, static_cast<double>(num7));
  EXPECT_EQ(1.375, static_cast<double>(num8));
  EXPECT_EQ(1.4375, static_cast<double>(num9));
}

TYPED_TEST(FixedPointTestBothReps, MoreSimpleBinaryFPConstruction)
{
  using binary_fp = fixed_point<TypeParam, Radix::BASE_2>;

  binary_fp num0{1.25, scale_type{-2}};
  binary_fp num1{2.1, scale_type{-4}};

  EXPECT_EQ(1.25, static_cast<double>(num0));
  EXPECT_EQ(2.125, static_cast<double>(num1));
}

TYPED_TEST(FixedPointTestBothReps, SimpleDecimalXXMath)
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

  decimalXX a{1.23, scale_type{-2}};
  decimalXX b{0, scale_type{0}};

  EXPECT_EQ(a + b, a);
  EXPECT_EQ(a - b, a);
}

TYPED_TEST(FixedPointTestBothReps, ComparisonOperators)
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

TYPED_TEST(FixedPointTestBothReps, DecimalXXTrickyDivision)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX ONE_1{1, scale_type{1}};
  decimalXX SIX_0{6, scale_type{0}};
  decimalXX SIX_1{6, scale_type{1}};
  decimalXX TEN_0{10, scale_type{0}};
  decimalXX TEN_1{10, scale_type{1}};
  decimalXX SIXTY_1{60, scale_type{1}};

  EXPECT_EQ(static_cast<int32_t>(ONE_1), 0);   // round(1 / 10) = 0
  EXPECT_EQ(static_cast<int32_t>(SIX_1), 10);  // round(6 / 10) = 10
  EXPECT_EQ(static_cast<int32_t>(TEN_0), 10);
  EXPECT_EQ(static_cast<int32_t>(SIXTY_1), 60);

  EXPECT_EQ(SIXTY_1 / TEN_0, TEN_1);
  EXPECT_EQ(SIXTY_1 / TEN_1, SIX_0);

  decimalXX A{34.56, scale_type{-2}};
  decimalXX B{1.234, scale_type{-3}};
  decimalXX C{1, scale_type{-2}};

  EXPECT_EQ(static_cast<int32_t>(A / B), 30);
  EXPECT_EQ(static_cast<int32_t>((A * C) / B), 28);

  decimalXX n{28, scale_type{1}};
  EXPECT_EQ(static_cast<int32_t>(n), 30);
}

TYPED_TEST(FixedPointTestBothReps, DecimalXXRounding)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX ZERO_0{0, scale_type{0}};
  decimalXX ZERO_1{4, scale_type{1}};
  decimalXX THREE_0{3, scale_type{0}};
  decimalXX FOUR_0{4, scale_type{0}};
  decimalXX FIVE_0{5, scale_type{0}};
  decimalXX TEN_0{10, scale_type{0}};
  decimalXX TEN_1{5, scale_type{1}};

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

TYPED_TEST(FixedPointTestBothReps, ArithmeticWithDifferentScales)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX a{1, scale_type{0}};
  decimalXX b{1.2, scale_type{-1}};
  decimalXX c{1.23, scale_type{-2}};
  decimalXX d{1.111, scale_type{-3}};

  decimalXX x{2.2, scale_type{-1}};
  decimalXX y{3.43, scale_type{-2}};
  decimalXX z{4.541, scale_type{-3}};

  decimalXX xx{0.2, scale_type{-1}};
  decimalXX yy{0.03, scale_type{-2}};
  decimalXX zz{0.119, scale_type{-3}};

  EXPECT_EQ(a + b, x);
  EXPECT_EQ(a + b + c, y);
  EXPECT_EQ(a + b + c + d, z);
  EXPECT_EQ(b - a, xx);
  EXPECT_EQ(c - b, yy);
  EXPECT_EQ(c - d, zz);
}

TYPED_TEST(FixedPointTestBothReps, RescaledTest)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX num0{1, scale_type{0}};
  decimalXX num1{1.2, scale_type{-1}};
  decimalXX num2{1.23, scale_type{-2}};
  decimalXX num3{1.235, scale_type{-3}};
  decimalXX num4{1.2346, scale_type{-4}};
  decimalXX num5{1.23457, scale_type{-5}};
  decimalXX num6{1.234567, scale_type{-6}};

  EXPECT_EQ(num0, num6.rescaled(scale_type{0}));
  EXPECT_EQ(num1, num6.rescaled(scale_type{-1}));
  EXPECT_EQ(num2, num6.rescaled(scale_type{-2}));
  EXPECT_EQ(num3, num6.rescaled(scale_type{-3}));
  EXPECT_EQ(num4, num6.rescaled(scale_type{-4}));
  EXPECT_EQ(num5, num6.rescaled(scale_type{-5}));
}

TYPED_TEST(FixedPointTestBothReps, RescaledRounding)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX num0{1500, scale_type{0}};
  decimalXX num1{1499, scale_type{0}};
  decimalXX num2{-1499, scale_type{0}};
  decimalXX num3{-1500, scale_type{0}};

  EXPECT_EQ(2000, static_cast<TypeParam>(num0.rescaled(scale_type{3})));
  EXPECT_EQ(1000, static_cast<TypeParam>(num1.rescaled(scale_type{3})));
  EXPECT_EQ(-1000, static_cast<TypeParam>(num2.rescaled(scale_type{3})));
  EXPECT_EQ(-2000, static_cast<TypeParam>(num3.rescaled(scale_type{3})));
}

TYPED_TEST(FixedPointTestBothReps, DecimalXXThrust)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  std::vector<decimalXX> vec1(1000);
  std::vector<int32_t> vec2(1000);

  std::iota(std::begin(vec1), std::end(vec1), decimalXX{0, scale_type{-2}});
  std::iota(std::begin(vec2), std::end(vec2), 0);

  auto const res1 =
    thrust::reduce(std::cbegin(vec1), std::cend(vec1), decimalXX{0, scale_type{-2}});

  auto const res2 = std::accumulate(std::cbegin(vec2), std::cend(vec2), 0);

  EXPECT_EQ(static_cast<int32_t>(res1), res2);

  std::vector<int32_t> vec3(vec1.size());

  thrust::transform(std::cbegin(vec1), std::cend(vec1), std::begin(vec3), [](auto const& e) {
    return static_cast<int32_t>(e);
  });

  EXPECT_EQ(vec2, vec3);
}

TYPED_TEST(FixedPointTestBothReps, BoolConversion)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  decimalXX truthy_value{1.234567, scale_type{0}};
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

TEST_F(FixedPointTest, OverflowDecimal32)
{
  // This flag is needed to avoid warnings with ASSERT_DEATH
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  using decimal32 = fixed_point<int32_t, Radix::BASE_10>;

#if defined(__CUDACC_DEBUG__)

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

#endif
}

TEST_F(FixedPointTest, OverflowDecimal64)
{
  // This flag is needed to avoid warnings with ASSERT_DEATH
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  using decimal64 = fixed_point<int64_t, Radix::BASE_10>;

#if defined(__CUDACC_DEBUG__)

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

#endif
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

  std::iota(std::begin(vec1), std::end(vec1), decimal32{initial_value, scale_type{scale}});
  std::iota(std::begin(vec2), std::end(vec2), initial_value);

  auto equal = std::equal(
    std::cbegin(vec1), std::cend(vec1), std::cbegin(vec2), [](auto const& a, auto const& b) {
      return static_cast<double>(a) - b <= std::numeric_limits<ValueType>::epsilon();
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

TEST_F(FixedPointTest, DecimalXXThrustOnDevice)
{
  using decimal32 = fixed_point<int32_t, Radix::BASE_10>;

  thrust::device_vector<decimal32> vec1(1000, decimal32{1, scale_type{-2}});

  auto const sum = thrust::reduce(
    rmm::exec_policy(0)->on(0), std::cbegin(vec1), std::cend(vec1), decimal32{0, scale_type{-2}});

  EXPECT_EQ(static_cast<int32_t>(sum), 1000);

  // TODO: Once nvbugs/1990211 is fixed (ExclusiveSum initial_value = 0 bug)
  //       change inclusive scan to run on device (avoid copying to host)
  thrust::host_vector<decimal32> vec1_host = vec1;

  thrust::inclusive_scan(std::cbegin(vec1_host), std::cend(vec1_host), std::begin(vec1_host));

  vec1 = vec1_host;

  std::vector<int32_t> vec2(1000);
  std::iota(std::begin(vec2), std::end(vec2), 1);

  auto const res1 = thrust::reduce(
    rmm::exec_policy(0)->on(0), std::cbegin(vec1), std::cend(vec1), decimal32{0, scale_type{-2}});

  auto const res2 = std::accumulate(std::cbegin(vec2), std::cend(vec2), 0);

  EXPECT_EQ(static_cast<int32_t>(res1), res2);

  thrust::device_vector<int32_t> vec3(1000);

  thrust::transform(rmm::exec_policy(0)->on(0),
                    std::cbegin(vec1),
                    std::cend(vec1),
                    std::begin(vec3),
                    cast_to_int32_fn{});

  thrust::host_vector<int32_t> vec3_host = vec3;

  EXPECT_EQ(vec2, vec3);
}

TYPED_TEST(FixedPointTestBothReps, FixedPointColumnWrapper)
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

TYPED_TEST(FixedPointTestBothReps, NoScaleOrWrongTypeID)
{
  auto const null_mask = cudf::create_null_mask(0, cudf::mask_state::ALL_NULL);

  EXPECT_THROW(cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::INT32}, 0, null_mask),
               cudf::logic_error);
}

TYPED_TEST(FixedPointTestBothReps, SimpleFixedPointColumnWrapper)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  auto const a = cudf::test::fixed_point_column_wrapper<int32_t>{{11, 22, 33}, scale_type{-1}};
  auto const b = cudf::test::fixed_point_column_wrapper<int32_t>{{110, 220, 330}, scale_type{-2}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(a, b);
}

CUDF_TEST_PROGRAM_MAIN()
