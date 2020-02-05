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

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <fixed_point/fixed_point.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

using namespace numeric;

struct FixedPointTest         : public cudf::test::BaseFixture {};

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {};

using RepresentationTypes = ::testing::Types<int32_t, int64_t>;

TYPED_TEST_CASE(FixedPointTestBothReps, RepresentationTypes);

TYPED_TEST(FixedPointTestBothReps, SimpleDecimalXXConstruction) {

    using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

    decimalXX num0{1.234567, scale_type{ 0}};
    decimalXX num1{1.234567, scale_type{-1}};
    decimalXX num2{1.234567, scale_type{-2}};
    decimalXX num3{1.234567, scale_type{-3}};
    decimalXX num4{1.234567, scale_type{-4}};
    decimalXX num5{1.234567, scale_type{-5}};
    decimalXX num6{1.234567, scale_type{-6}};

    EXPECT_EQ(1,        num0.to_double());
    EXPECT_EQ(1.2,      num1.to_double());
    EXPECT_EQ(1.23,     num2.to_double());
    EXPECT_EQ(1.234,    num3.to_double());
    EXPECT_EQ(1.2345,   num4.to_double());
    EXPECT_EQ(1.23456,  num5.to_double());
    EXPECT_EQ(1.234567, num6.to_double());

}

TYPED_TEST(FixedPointTestBothReps, SimpleBinaryFPConstruction) {

    using binary_fp = fixed_point<TypeParam, Radix::BASE_2>;

    binary_fp num0{10, scale_type{0}};
    binary_fp num1{10, scale_type{1}};
    binary_fp num2{10, scale_type{2}};
    binary_fp num3{10, scale_type{3}};
    binary_fp num4{10, scale_type{4}};

    EXPECT_EQ(10, num0.to_int32());
    EXPECT_EQ(10, num1.to_int32());
    EXPECT_EQ(8,  num2.to_int32());
    EXPECT_EQ(8,  num3.to_int32());
    EXPECT_EQ(0,  num4.to_int32());

}

TYPED_TEST(FixedPointTestBothReps, SimpleDecimalXXMath) {

    using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

    decimalXX ONE  {1, scale_type{-2}};
    decimalXX TWO  {2, scale_type{-2}};
    decimalXX THREE{3, scale_type{-2}};
    decimalXX SIX  {6, scale_type{-2}};

    EXPECT_TRUE(ONE + ONE == TWO);

    EXPECT_EQ(ONE   + ONE, TWO);
    EXPECT_EQ(ONE   * TWO, TWO);
    EXPECT_EQ(THREE * TWO, SIX);
    EXPECT_EQ(THREE - TWO, ONE);
    EXPECT_EQ(TWO   / ONE, TWO);
    EXPECT_EQ(SIX   / TWO, THREE);

}

TYPED_TEST(FixedPointTestBothReps, DecimalXXTrickyDivision) {

    using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

    decimalXX ONE_1  {1,  scale_type{1}};
    decimalXX SIX_0  {6,  scale_type{0}};
    decimalXX SIX_1  {6,  scale_type{1}};
    decimalXX TEN_0  {10, scale_type{0}};
    decimalXX TEN_1  {10, scale_type{1}};
    decimalXX SIXTY_1{60, scale_type{1}};

    decimalXX ZERO = ONE_1;

    EXPECT_EQ(ONE_1.to_int32(),    0); // 1 / 10 = 0
    EXPECT_EQ(SIX_1.to_int32(),    0); // 6 / 10 = 0
    EXPECT_EQ(TEN_0.to_int32(),   10);
    EXPECT_EQ(SIXTY_1.to_int32(), 60);

    EXPECT_EQ(SIXTY_1 / TEN_0, ZERO);
    EXPECT_EQ(SIXTY_1 / TEN_1, SIX_0);

    decimalXX A{34.56, scale_type{-2}};
    decimalXX B{1.234, scale_type{-3}};
    decimalXX C{1,     scale_type{-2}};

    EXPECT_EQ((A / B).to_int32(),       20);
    EXPECT_EQ(((A * C) / B).to_int32(), 28);

}

TYPED_TEST(FixedPointTestBothReps, DecimalXXThrust) {

   using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

    std::vector<decimalXX> vec1(1000);
    std::vector<int32_t>   vec2(1000);

    std::iota(std::begin(vec1), std::end(vec1), decimalXX{0, scale_type{-2}});
    std::iota(std::begin(vec2), std::end(vec2), 0);

    auto const res1 = thrust::reduce(
        std::cbegin(vec1),
        std::cend(vec1),
        decimalXX{0, scale_type{-2}});

    auto const res2 = std::accumulate(
        std::cbegin(vec2),
        std::cend(vec2),
        0);

    EXPECT_EQ(res1.to_int32(), res2);

    std::vector<int32_t> vec3(vec1.size());

    thrust::transform(
        std::cbegin(vec1),
        std::cend(vec1),
        std::begin(vec3),
        [] (auto const& e) { return e.to_int32(); });

    EXPECT_EQ(vec2, vec3);

}

TEST_F(FixedPointTest, OverflowDecimal32) {

    using decimal32 = fixed_point<int32_t, Radix::BASE_10>;

    #if defined(__CUDACC_DEBUG__)

    decimal32 num0{ 2, scale_type{-9}};
    decimal32 num1{-2, scale_type{-9}};

    ASSERT_NO_THROW(num0 + num0);
    ASSERT_NO_THROW(num1 - num0);

    decimal32 min{std::numeric_limits<int32_t>::min(), scale_type{0}};
    decimal32 max{std::numeric_limits<int32_t>::max(), scale_type{0}};
    decimal32 NEG_ONE{-1, scale_type{0}};
    decimal32 ONE{1, scale_type{0}};
    decimal32 TWO{2, scale_type{0}};

    ASSERT_NO_THROW(min / NEG_ONE);
    ASSERT_NO_THROW(max * TWO);
    ASSERT_NO_THROW(min * TWO);
    ASSERT_NO_THROW(max + ONE);
    ASSERT_NO_THROW(max - NEG_ONE);
    ASSERT_NO_THROW(min - ONE);
    ASSERT_NO_THROW(max - NEG_ONE);

    #endif

}

TEST_F(FixedPointTest, OverflowDecimal64) {

    using decimal64 = fixed_point<int64_t, Radix::BASE_10>;

    #if defined(__CUDACC_DEBUG__)

    decimal64 num0{ 5, scale_type{-18}};
    decimal64 num1{-5, scale_type{-18}};

    ASSERT_NO_THROW(num0 + num0);
    ASSERT_NO_THROW(num1 - num0);

    decimal64 min{std::numeric_limits<int64_t>::min(), scale_type{0}};
    decimal64 max{std::numeric_limits<int64_t>::max(), scale_type{0}};
    decimal64 NEG_ONE{-1, scale_type{0}};
    decimal64 ONE{1, scale_type{0}};
    decimal64 TWO{2, scale_type{0}};

    ASSERT_NO_THROW(min / NEG_ONE);
    ASSERT_NO_THROW(max * TWO);
    ASSERT_NO_THROW(min * TWO);
    ASSERT_NO_THROW(max + ONE);
    ASSERT_NO_THROW(max - NEG_ONE);
    ASSERT_NO_THROW(min - ONE);
    ASSERT_NO_THROW(max - NEG_ONE);

    #endif

}

template<typename ValueType, typename Binop>
void integer_vector_test(ValueType const initial_value,
                         int32_t   const size,
                         int32_t   const scale, Binop binop) {

    using decimal32 = fixed_point<int32_t, Radix::BASE_10>;

    std::vector<decimal32> vec1(size);
    std::vector<ValueType> vec2(size);

    std::iota(std::begin(vec1), std::end(vec1), decimal32{initial_value, scale_type{scale}});
    std::iota(std::begin(vec2), std::end(vec2), initial_value);

    auto const res1 = std::accumulate(
        std::cbegin(vec1),
        std::cend(vec1),
        decimal32{0, scale_type{scale}});

    auto const res2 = std::accumulate(
        std::cbegin(vec2),
        std::cend(vec2),
        static_cast<ValueType>(0));

    EXPECT_EQ(res1.to_int32(), res2);

    std::vector<ValueType> vec3(vec1.size());

    std::transform(
        std::cbegin(vec1),
        std::cend(vec1),
        std::begin(vec3),
        [] (auto const& e) { return e.to_int32(); });

    EXPECT_EQ(vec2, vec3);
}

TEST_F(FixedPointTest, Decimal32IntVector) {

    integer_vector_test(0,   10,   -2, std::plus<>());
    integer_vector_test(0,   1000, -2, std::plus<>());

    integer_vector_test(1, 10, 0, std::multiplies<>());
    integer_vector_test(2, 20, 0, std::multiplies<>());

}

template<typename ValueType, typename Binop>
void float_vector_test(ValueType const initial_value,
                       int32_t   const size,
                       int32_t   const scale, Binop binop) {

    using decimal32 = fixed_point<int32_t, Radix::BASE_10>;

    std::vector<decimal32> vec1(size);
    std::vector<ValueType> vec2(size);

    std::iota(std::begin(vec1), std::end(vec1), decimal32{initial_value, scale_type{scale}});
    std::iota(std::begin(vec2), std::end(vec2), initial_value);

    auto equal = std::equal(
        std::cbegin(vec1),
        std::cend(vec1),
        std::cbegin(vec2),
        [] (auto const& a, auto const& b) {
            return a.to_double() - b <= std::numeric_limits<ValueType>::epsilon();
        });

    EXPECT_TRUE(equal);
}

TEST_F(FixedPointTest, Decimal32FloatVector) {

    float_vector_test(0.1,  1000, -2, std::plus<>());
    float_vector_test(0.15, 1000, -2, std::plus<>());

    float_vector_test(0.1,  10, -2, std::multiplies<>());
    float_vector_test(0.15, 20, -2, std::multiplies<>());

}
