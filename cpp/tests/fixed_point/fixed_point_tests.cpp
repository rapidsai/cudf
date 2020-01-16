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

struct FixedPointTest : public cudf::test::BaseFixture {};

TEST_F(FixedPointTest, SimpleDecimal32Construction) {

    using namespace cudf::fp;
    using decimal32 = fixed_point<int32_t, 10>;

    decimal32 num0{1.234567, 0};
    decimal32 num1{1.234567, 1};
    decimal32 num2{1.234567, 2};
    decimal32 num3{1.234567, 3};
    decimal32 num4{1.234567, 4};
    decimal32 num5{1.234567, 5};
    decimal32 num6{1.234567, 6};

    EXPECT_EQ(1,        num0.get());
    EXPECT_EQ(1.2,      num1.get());
    EXPECT_EQ(1.23,     num2.get());
    EXPECT_EQ(1.234,    num3.get());
    EXPECT_EQ(1.2345,   num4.get());
    EXPECT_EQ(1.23456,  num5.get());
    EXPECT_EQ(1.234567, num6.get());

}

TEST_F(FixedPointTest, SimpleBinaryFPConstruction) {

    using namespace cudf::fp;
    using binary_fp32 = fixed_point<int32_t, 2>;

    binary_fp32 num0{10,  0};
    binary_fp32 num1{10, -1};
    binary_fp32 num2{10, -2};
    binary_fp32 num3{10, -3};
    binary_fp32 num4{10, -4};

    EXPECT_EQ(10, num0.get());
    EXPECT_EQ(10, num1.get());
    EXPECT_EQ(8,  num2.get());
    EXPECT_EQ(8,  num3.get());
    EXPECT_EQ(0,  num4.get());

}
