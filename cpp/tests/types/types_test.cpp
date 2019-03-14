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

using Wrappers = ::testing::Types<cudf::category, cudf::timestamp, cudf::date32,
                                  cudf::date64>;

TYPED_TEST_CASE(WrappersTest, Wrappers);

/**
 * @brief The number of test trials for each operator
 * 
 */
static constexpr int NUM_TRIALS{10000};


TYPED_TEST(WrappersTest, ConstructorTest)
{
    using UnderlyingType = typename TypeParam::value_type;

    for (int i = 0; i < NUM_TRIALS; ++i) {
      UnderlyingType t{this->rand()};
      TypeParam w{t};
      EXPECT_EQ(t, w.value);
    }
}

TYPED_TEST(WrappersTest, AssignmentTest)
{
    using UnderlyingType = typename TypeParam::value_type;

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
    using UnderlyingType = typename TypeParam::value_type;

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
    using UnderlyingType = typename TypeParam::value_type;

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
    using UnderlyingType = typename TypeParam::value_type;

    for(int i = 0; i < NUM_TRIALS; ++i)
    {
        UnderlyingType const t0{this->rand()};
        UnderlyingType const t1{this->rand()};

        TypeParam const w0{t0};
        TypeParam const w1{t1};

        EXPECT_EQ(t0+t1, TypeParam{w0+w1}.value);
        EXPECT_EQ(t0-t1, TypeParam{w0-w1}.value);
        EXPECT_EQ(t0*t1, TypeParam{w0*w1}.value);
        if(0 != t1)
            EXPECT_EQ(t0/t1, TypeParam{w0/w1}.value);
    }
}


TYPED_TEST(WrappersTest, IncrementOperators){
    using UnderlyingType = typename TypeParam::value_type;

    for(int i = 0; i < NUM_TRIALS; ++i){
        UnderlyingType t{this->rand()};
        TypeParam w{t};
        EXPECT_EQ(t++, (w++).value);
        EXPECT_EQ(++t, (++w).value);
    }
}

TYPED_TEST(WrappersTest, DecrementOperators){
    using UnderlyingType = typename TypeParam::value_type;

    for(int i = 0; i < NUM_TRIALS; ++i){
        UnderlyingType t{this->rand()};
        TypeParam w{t};
        EXPECT_EQ(t--, (w--).value);
        EXPECT_EQ(--t, (--w).value);
    }
}

TYPED_TEST(WrappersTest, BooleanOperators)
{
    using UnderlyingType = typename TypeParam::value_type;

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

TYPED_TEST(WrappersTest, CompoundAssignmentOperators)
{
    using UnderlyingType = typename TypeParam::value_type;

    for(int i = 0; i < NUM_TRIALS; ++i)
    {
        UnderlyingType t0{this->rand()};
        UnderlyingType const t1{this->rand()};

        TypeParam w0{t0};
        TypeParam const w1{t1};

        t0+=t1;
        w0+=w1;
        EXPECT_EQ(t0, w0.value);

        t0-=t1;
        w0-=w1;
        EXPECT_EQ(t0, w0.value);

        t0*=t1;
        w0*=w1;
        EXPECT_EQ(t0, w0.value);

        if( 0 != t1)
        {
            t0/=t1;
            w0/=w1;
            EXPECT_EQ(t0, w0.value);
        }
    }
}
