/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <utilities/wrapper_types.hpp>
#include <cudf.h>

#include <gtest/gtest.h>

#include <random>


/**
 * @file types_test.cpp
 * @brief Tests the wrapper structures, their operators, and the unwrap function.
 */

template <typename T>
struct WrappersTest : public ::testing::Test {

  using UnderlyingType = typename T::value_type;

  WrappersTest()
      : dist{std::numeric_limits<UnderlyingType>::lowest(),
             std::numeric_limits<UnderlyingType>::max()} {

    // Use constant seed for deterministic results
    rng.seed(0);
  }

  std::mt19937 rng;
  std::uniform_int_distribution<UnderlyingType> dist;

  UnderlyingType rand(){
      return dist(rng);
  }
};

using WrappersTestBool8 = WrappersTest<cudf::bool8>;

// These structs enable specializing the underlying type to use in testing
// comparisons. For example, we specialize cudf::bool8 to map to bool. This
// means that it's wrapper operators are expected to act like bool even though
// its actual underlying type is a signed 8-bit integer.
template <typename T>
struct TypeToUse{
    using type = void;
};

template<>
struct TypeToUse<cudf::bool8>{
    using type = bool;
};

template <typename T, gdf_dtype dtype>
struct TypeToUse<cudf::detail::wrapper<T, dtype>>{
    using type = typename cudf::detail::wrapper<T,dtype>::value_type;
};

template <typename T>
using WrappersNoBoolTest = WrappersTest<T>;

using WrappersNoBool = ::testing::Types<cudf::category, cudf::timestamp, cudf::date32,
                                        cudf::date64>;

using Wrappers = ::testing::Types<cudf::category, cudf::timestamp, cudf::date32,
                                  cudf::date64, cudf::bool8>;

TYPED_TEST_CASE(WrappersTest, Wrappers);
TYPED_TEST_CASE(WrappersNoBoolTest, WrappersNoBool);

/**
 * @brief The number of test trials for each operator
 * 
 */
static constexpr int NUM_TRIALS{10000};


TYPED_TEST(WrappersTest, ConstructorTest)
{
    using UnderlyingType = typename TypeToUse<TypeParam>::type;

    for (int i = 0; i < NUM_TRIALS; ++i) {
      UnderlyingType t{this->rand()};
      TypeParam w{t};
      EXPECT_EQ(t, w.value);
    }
}

TYPED_TEST(WrappersTest, AssignmentTest)
{
    using UnderlyingType = typename TypeToUse<TypeParam>::type;

    for (int i = 0; i < NUM_TRIALS; ++i) {
      UnderlyingType const t0{this->rand()};
      UnderlyingType const t1{this->rand()};
      TypeParam w0{t0};
      TypeParam w1{t1};

      w0 = w1;

      EXPECT_EQ(t1, w0.value);
    }
}

TYPED_TEST(WrappersTest, UnwrapWrapperTest)
{
    using UnderlyingType = typename TypeToUse<TypeParam>::type;

    for (int i = 0; i < NUM_TRIALS; ++i) {
        UnderlyingType t0{this->rand()};
        UnderlyingType t1{this->rand()};

        TypeParam w{t0};

        EXPECT_EQ(t0, cudf::detail::unwrap(w));

        cudf::detail::unwrap(w) = t1;

        EXPECT_EQ(t1, cudf::detail::unwrap(w));
    }
}

TYPED_TEST(WrappersTest, UnwrapFundamentalTest)
{
    using UnderlyingType = typename TypeToUse<TypeParam>::type;

    for (int i = 0; i < NUM_TRIALS; ++i) {
        UnderlyingType t0{this->rand()};
        UnderlyingType t1{this->rand()};

        EXPECT_EQ(t1, cudf::detail::unwrap(t1));

        cudf::detail::unwrap(t1) = t0;
        EXPECT_EQ(t0, t1);
    }
}

TYPED_TEST(WrappersTest, ArithmeticOperators)
{
    using UnderlyingType = typename TypeToUse<TypeParam>::type;

    for(int i = 0; i < NUM_TRIALS; ++i)
    {
        UnderlyingType const t0{this->rand()};
        UnderlyingType const t1{this->rand()};

        TypeParam const w0{t0};
        TypeParam const w1{t1};

        // Types smaller than int are implicitly promoted to `int` for
        // arithmetic operations. Therefore, need to convert it back to the
        // original type
        EXPECT_EQ(static_cast<UnderlyingType>(t0 + t1),
                  static_cast<UnderlyingType>(TypeParam{w0 + w1}.value));
        EXPECT_EQ(static_cast<UnderlyingType>(t0 - t1),
                  static_cast<UnderlyingType>(TypeParam{w0 - w1}.value));
        EXPECT_EQ(static_cast<UnderlyingType>(t0 * t1),
                  static_cast<UnderlyingType>(TypeParam{w0 * w1}.value));
        if (0 != t1)
          EXPECT_EQ(static_cast<UnderlyingType>(t0 / t1),
                    static_cast<UnderlyingType>(TypeParam{w0 / w1}.value));
    }
}

TYPED_TEST(WrappersNoBoolTest, BooleanOperators)
{
    using UnderlyingType = typename TypeToUse<TypeParam>::type;

    for(int i = 0; i < NUM_TRIALS; ++i)
    {
        UnderlyingType const t0{this->rand()};
        UnderlyingType const t1{this->rand()};

        TypeParam const w0{t0};
        TypeParam const w1{t1};

        EXPECT_EQ(t0 > t1, w0 > w1);
        EXPECT_EQ(t0 < t1, w0 < w1);
        EXPECT_EQ(t0 <= t1, w0 <= w1);
        EXPECT_EQ(t0 >= t1, w0 >= w1);
        EXPECT_EQ(t0 == t1, w0 == w1);
        EXPECT_EQ(t0 != t1, w0 != w1);
    }

    TypeParam w2{42};
    TypeParam w3{43};

    EXPECT_TRUE(w2 == w2);
    EXPECT_TRUE(w2 < w3);
    EXPECT_FALSE(w2 > w3);
    EXPECT_TRUE(w2 != w3);
    EXPECT_TRUE(w2 >= w2);
    EXPECT_TRUE(w2 <= w2);
    EXPECT_FALSE(w2 >= w3);
    EXPECT_TRUE(w2 <= w3);
}

// Just for booleans
TEST_F(WrappersTestBool8, BooleanOperatorsBool8)
{
    for(int i = 0; i < NUM_TRIALS; ++i)
    {
        bool const t0{this->rand()};
        bool const t1{this->rand()};

        cudf::bool8 const w0{t0};
        cudf::bool8 const w1{t1};

        EXPECT_EQ(t0 > t1, w0 > w1);
        EXPECT_EQ(t0 < t1, w0 < w1);
        EXPECT_EQ(t0 <= t1, w0 <= w1);
        EXPECT_EQ(t0 >= t1, w0 >= w1);
        EXPECT_EQ(t0 == t1, w0 == w1);
        EXPECT_EQ(t0 != t1, w0 != w1);
    }

    cudf::bool8 w2{42};
    cudf::bool8 w3{43};

    EXPECT_TRUE(w2 == w2);
    EXPECT_TRUE(w2 == w3);
    EXPECT_FALSE(w2 < w3);
    EXPECT_FALSE(w2 > w3);
    EXPECT_FALSE(w2 != w3);
    EXPECT_TRUE(w2 >= w2);
    EXPECT_TRUE(w2 <= w2);
    EXPECT_TRUE(w2 >= w3);
    EXPECT_TRUE(w2 <= w3);

    cudf::bool8 w4{-42};
    cudf::bool8 w5{43};

    EXPECT_TRUE(w4 == w4);
    EXPECT_TRUE(w5 == w5);
    EXPECT_FALSE(w4 < w5);
    EXPECT_FALSE(w4 > w5);
    EXPECT_FALSE(w4 != w5);
    EXPECT_TRUE(w4 >= w4);
    EXPECT_TRUE(w4 <= w4);
    EXPECT_TRUE(w4 >= w5);
    EXPECT_TRUE(w4 <= w5);

    cudf::bool8 w6{0};
    cudf::bool8 w7{43};

    EXPECT_FALSE(w6 == w7);
    EXPECT_TRUE(w6 < w7);
    EXPECT_TRUE(w7 > w6);
    EXPECT_FALSE(w6 > w7);
    EXPECT_TRUE(w6 != w7);
    EXPECT_TRUE(w6 >= w6);
    EXPECT_TRUE(w6 <= w6);
    EXPECT_FALSE(w6 >= w7);
    EXPECT_TRUE(w6 <= w7);
}

// This ensures that casting cudf::bool8 to int, doing arithmetic, and casting
// the result to bool results in the right answer. If the arithmetic is done
// on random underlying values you can get the wrong answer.
TEST_F(WrappersTestBool8, CastArithmeticTest)
{
    cudf::bool8 w1{42};
    cudf::bool8 w2{-42};

    bool t1{42};
    bool t2{-42};

    EXPECT_EQ(static_cast<bool>(static_cast<int>(w1) + static_cast<int>(w2)),
              static_cast<bool>(static_cast<int>(t1) + static_cast<int>(t2)));
}

TYPED_TEST(WrappersTest, CompoundAssignmentOperators)
{
    using UnderlyingType = typename TypeToUse<TypeParam>::type;

    for(int i = 0; i < NUM_TRIALS; ++i)
    {
        UnderlyingType t0{this->rand()};
        UnderlyingType const t1{this->rand()};

        TypeParam w0{t0};
        TypeParam const w1{t1};

        t0+=t1;
        w0+=w1;
        EXPECT_EQ(t0, static_cast<UnderlyingType>(w0.value));

        t0-=t1;
        w0-=w1;
        EXPECT_EQ(t0, static_cast<UnderlyingType>(w0.value));

        t0*=t1;
        w0*=w1;
        EXPECT_EQ(t0, static_cast<UnderlyingType>(w0.value));

        if( 0 != t1)
        {
            t0/=t1;
            w0/=w1;
            EXPECT_EQ(t0, static_cast<UnderlyingType>(w0.value));
        }
    }
}

TYPED_TEST(WrappersTest, NumericLimitsTest)
{
    using UnderlyingType = typename TypeToUse<TypeParam>::type;

    EXPECT_EQ(static_cast<UnderlyingType>(std::numeric_limits<TypeParam>::max()), 
              std::numeric_limits<UnderlyingType>::max());
    EXPECT_EQ(static_cast<UnderlyingType>(std::numeric_limits<TypeParam>::min()), 
              std::numeric_limits<UnderlyingType>::min());
    EXPECT_EQ(static_cast<UnderlyingType>(std::numeric_limits<TypeParam>::lowest()), 
              std::numeric_limits<UnderlyingType>::lowest());
}
