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
#include <tests/utilities/base_fixture.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <vector>
#include <algorithm> // transform, generate_n
#include <numeric>   // iota, accumulate
#include <limits>

struct FixedPointTest : public cudf::test::BaseFixture {};

using namespace cudf::fp;
using decimal32 = fixed_point<int32_t, Radix::BASE_10>;

TEST_F(FixedPointTest, SimpleDecimal32Construction) {

    decimal32 num0{1.234567, scale_type{ 0}};
    decimal32 num1{1.234567, scale_type{-1}};
    decimal32 num2{1.234567, scale_type{-2}};
    decimal32 num3{1.234567, scale_type{-3}};
    decimal32 num4{1.234567, scale_type{-4}};
    decimal32 num5{1.234567, scale_type{-5}};
    decimal32 num6{1.234567, scale_type{-6}};

    EXPECT_EQ(1,        num0.get());
    EXPECT_EQ(1.2,      num1.get());
    EXPECT_EQ(1.23,     num2.get());
    EXPECT_EQ(1.234,    num3.get());
    EXPECT_EQ(1.2345,   num4.get());
    EXPECT_EQ(1.23456,  num5.get());
    EXPECT_EQ(1.234567, num6.get());

}

TEST_F(FixedPointTest, SimpleBinaryFPConstruction) {

    using binary_fp32 = fixed_point<int32_t, Radix::BASE_2>;

    binary_fp32 num0{10, scale_type{0}};
    binary_fp32 num1{10, scale_type{1}};
    binary_fp32 num2{10, scale_type{2}};
    binary_fp32 num3{10, scale_type{3}};
    binary_fp32 num4{10, scale_type{4}};

    EXPECT_EQ(10, num0.get());
    EXPECT_EQ(10, num1.get());
    EXPECT_EQ(8,  num2.get());
    EXPECT_EQ(8,  num3.get());
    EXPECT_EQ(0,  num4.get());

}

TEST_F(FixedPointTest, SimpleDecimal32Math) {

    decimal32 ONE  {1, scale_type{-2}};
    decimal32 TWO  {2, scale_type{-2}};
    decimal32 THREE{3, scale_type{-2}};
    decimal32 SIX  {6, scale_type{-2}};

    EXPECT_TRUE(ONE + ONE == TWO);

    EXPECT_EQ(ONE   + ONE, TWO);
    EXPECT_EQ(ONE   * TWO, TWO);
    EXPECT_EQ(THREE * TWO, SIX);
    EXPECT_EQ(THREE - TWO, ONE);
    EXPECT_EQ(TWO   / ONE, TWO);
    EXPECT_EQ(SIX   / TWO, THREE);

}

TEST_F(FixedPointTest, Decimal32TrickyDivision) {

    decimal32 ONE_1  {1,  scale_type{1}};
    decimal32 SIX_0  {6,  scale_type{0}};
    decimal32 SIX_1  {6,  scale_type{1}};
    decimal32 TEN_0  {10, scale_type{0}};
    decimal32 TEN_1  {10, scale_type{1}};
    decimal32 SIXTY_1{60, scale_type{1}};

    decimal32 ZERO = ONE_1;

    EXPECT_EQ(ONE_1.get(),    0); // 1 / 10 = 0
    EXPECT_EQ(SIX_1.get(),    0); // 6 / 10 = 0
    EXPECT_EQ(TEN_0.get(),   10);
    EXPECT_EQ(SIXTY_1.get(), 60);

    EXPECT_EQ(SIXTY_1 / TEN_0, ZERO);
    EXPECT_EQ(SIXTY_1 / TEN_1, SIX_0);

    decimal32 A{34.56,  scale_type{-2}};
    decimal32 B{1.234,  scale_type{-3}};

    EXPECT_EQ((A / B).get(), 20);

}

TEST_F(FixedPointTest, OverflowDecimal32) {

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

template<typename ValueType,
         typename std::enable_if_t<std::numeric_limits<ValueType>::is_integer>* = nullptr>
void transform_test(std::vector<decimal32> const& vec1,
                    std::vector<ValueType> const& vec2) {

    std::vector<ValueType> vec3(vec1.size());

    std::transform(
        std::cbegin(vec1),
        std::cend(vec1),
        std::begin(vec3),
        [] (auto const& e) { return e.get(); });

    EXPECT_EQ(vec2, vec3);
}

template<typename ValueType,
         typename std::enable_if_t<!std::numeric_limits<ValueType>::is_integer>* = nullptr>
void transform_test(std::vector<decimal32> const& vec1,
                    std::vector<ValueType> const& vec2) {

    auto equal = std::equal(
        std::cbegin(vec1),
        std::cend(vec1),
        std::cbegin(vec2),
        [] (auto const& a, auto const& b) {
            return a.get() == b;
        });

    EXPECT_TRUE(equal);
}

template<typename ValueType, typename Binop>
void vector_test(ValueType const initial_value,
                 int32_t   const size,
                 int32_t   const scale, Binop binop) {
    using namespace cudf::fp;
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

    EXPECT_EQ(res1.get(), res2);
    EXPECT_EQ(res1.get() - res2, 0);

    transform_test(vec1, vec2);
}

TEST_F(FixedPointTest, Decimal32VectorAddition) {

    vector_test(0,   10,   -2, std::plus<>());
    vector_test(0,   1000, -2, std::plus<>());
    vector_test(0.0, 1000, -2, std::plus<>());
    // vector_test(0.1, 1000, -2, std::plus<>()); // currently FAILS

}

TEST_F(FixedPointTest, Decimal32VectorMultiplication) {

    vector_test(1, 10, 0, std::multiplies<>());

}
