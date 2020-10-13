/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#include <tests/binaryop/assert-binops.h>
#include <tests/binaryop/binop-fixture.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

namespace cudf {
namespace test {
namespace binop {

struct BinaryOperationIntegrationTest : public BinaryOperationTest {
};

TEST_F(BinaryOperationIntegrationTest, Add_Scalar_Vector_SI32_FP32_SI64)
{
  using TypeOut = int32_t;
  using TypeLhs = float;
  using TypeRhs = int64_t;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  auto rhs = make_random_wrapped_column<TypeRhs>(10000);

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationIntegrationTest, Add_Vector_Vector_SI32_FP32_FP32)
{
  using TypeOut = int32_t;
  using TypeLhs = float;
  using TypeRhs = float;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(10000);
  auto rhs = make_random_wrapped_column<TypeRhs>(10000);

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationIntegrationTest, Sub_Scalar_Vector_SI32_FP32_FP32)
{
  using TypeOut = int32_t;
  using TypeLhs = float;
  using TypeRhs = int64_t;

  using SUB = cudf::library::operation::Sub<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  auto rhs = make_random_wrapped_column<TypeRhs>(10000);

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::SUB, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SUB());
}

TEST_F(BinaryOperationIntegrationTest, Add_Vector_Scalar_SI08_SI16_SI32)
{
  using TypeOut = int8_t;
  using TypeLhs = int16_t;
  using TypeRhs = int32_t;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_scalar<TypeRhs>();
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationIntegrationTest, Add_Vector_Vector_SI32_FP64_SI08)
{
  using TypeOut = int32_t;
  using TypeLhs = double;
  using TypeRhs = int8_t;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationIntegrationTest, Sub_Vector_Vector_SI64)
{
  using TypeOut = int64_t;
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  using SUB = cudf::library::operation::Sub<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::SUB, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SUB());
}

TEST_F(BinaryOperationIntegrationTest, Sub_Vector_Scalar_SI64_FP64_SI32)
{
  using TypeOut = int64_t;
  using TypeLhs = double;
  using TypeRhs = int32_t;

  using SUB = cudf::library::operation::Sub<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(10000);
  auto rhs = make_random_wrapped_scalar<TypeRhs>();

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::SUB, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SUB());
}

TEST_F(BinaryOperationIntegrationTest, Sub_Vector_Vector_TimepointD_DurationS_TimepointUS)
{
  using TypeOut = cudf::timestamp_us;
  using TypeLhs = cudf::timestamp_D;
  using TypeRhs = cudf::duration_s;

  using SUB = cudf::library::operation::Sub<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::SUB, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SUB());
}

TEST_F(BinaryOperationIntegrationTest, Sub_Vector_Scalar_TimepointD_TimepointS_DurationS)
{
  using TypeOut = cudf::duration_s;
  using TypeLhs = cudf::timestamp_D;
  using TypeRhs = cudf::timestamp_s;

  using SUB = cudf::library::operation::Sub<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = cudf::scalar_type_t<TypeRhs>(typename TypeRhs::duration{34}, true);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::SUB, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SUB());
}

TEST_F(BinaryOperationIntegrationTest, Sub_Scalar_Vector_DurationS_DurationD_DurationMS)
{
  using TypeOut = cudf::duration_ms;
  using TypeLhs = cudf::duration_s;
  using TypeRhs = cudf::duration_D;

  using SUB = cudf::library::operation::Sub<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = cudf::scalar_type_t<TypeLhs>(TypeLhs{-9});
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::SUB, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SUB());
}

TEST_F(BinaryOperationIntegrationTest, Mul_Vector_Vector_SI64)
{
  using TypeOut = int64_t;
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  using MUL = cudf::library::operation::Mul<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::MUL, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MUL());
}

TEST_F(BinaryOperationIntegrationTest, Mul_Vector_Vector_SI64_FP32_FP32)
{
  using TypeOut = int64_t;
  using TypeLhs = float;
  using TypeRhs = float;

  using MUL = cudf::library::operation::Mul<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::MUL, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MUL());
}

TEST_F(BinaryOperationIntegrationTest, Mul_Scalar_Vector_SI32_DurationD_DurationMS)
{
  // Double the duration of days and convert the time interval to ms
  using TypeOut = cudf::duration_ms;
  using TypeLhs = int32_t;
  using TypeRhs = cudf::duration_D;

  using MUL = cudf::library::operation::Mul<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = cudf::scalar_type_t<TypeLhs>(2);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::MUL, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MUL());
}

TEST_F(BinaryOperationIntegrationTest, Mul_Vector_Vector_DurationS_SI32_DurationNS)
{
  // Multiple each duration with some random value and promote the result
  using TypeOut = cudf::duration_ns;
  using TypeLhs = cudf::duration_s;
  using TypeRhs = int32_t;

  using MUL = cudf::library::operation::Mul<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::MUL, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MUL());
}

TEST_F(BinaryOperationIntegrationTest, Div_Vector_Vector_SI64)
{
  using TypeOut = int64_t;
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  using DIV = cudf::library::operation::Div<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::DIV, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, DIV());
}

TEST_F(BinaryOperationIntegrationTest, Div_Vector_Vector_SI64_FP32_FP32)
{
  using TypeOut = int64_t;
  using TypeLhs = float;
  using TypeRhs = float;

  using DIV = cudf::library::operation::Div<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::DIV, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, DIV());
}

TEST_F(BinaryOperationIntegrationTest, Div_Scalar_Vector_DurationD_SI32_DurationS)
{
  using TypeOut = cudf::duration_s;
  using TypeLhs = cudf::duration_D;
  using TypeRhs = int64_t;

  using DIV = cudf::library::operation::Div<TypeOut, TypeLhs, TypeRhs>;

  // Divide 2 days by an integer and convert the ticks to seconds
  auto lhs = cudf::scalar_type_t<TypeLhs>(TypeLhs{2});
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::DIV, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, DIV());
}

TEST_F(BinaryOperationIntegrationTest, Div_Vector_Vector_DurationD_DurationS_DurationMS)
{
  using TypeOut = int64_t;
  using TypeLhs = cudf::duration_D;
  using TypeRhs = cudf::duration_s;

  using DIV = cudf::library::operation::Div<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::DIV, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, DIV());
}

TEST_F(BinaryOperationIntegrationTest, TrueDiv_Vector_Vector_SI64)
{
  using TypeOut = int64_t;
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  using TRUEDIV = cudf::library::operation::TrueDiv<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::TRUE_DIV, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, TRUEDIV());
}

TEST_F(BinaryOperationIntegrationTest, FloorDiv_Vector_Vector_SI64)
{
  using TypeOut = int64_t;
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  using FLOORDIV = cudf::library::operation::FloorDiv<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::FLOOR_DIV, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, FLOORDIV());
}

TEST_F(BinaryOperationIntegrationTest, FloorDiv_Vector_Vector_SI64_FP32_FP32)
{
  using TypeOut = int64_t;
  using TypeLhs = float;
  using TypeRhs = float;

  using FLOORDIV = cudf::library::operation::FloorDiv<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::FLOOR_DIV, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, FLOORDIV());
}

TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_SI64)
{
  using TypeOut = int64_t;
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  using MOD = cudf::library::operation::Mod<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::MOD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MOD());
}

TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP32)
{
  using TypeOut = float;
  using TypeLhs = float;
  using TypeRhs = float;

  using MOD = cudf::library::operation::Mod<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::MOD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MOD());
}

TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_SI64_FP32_FP32)
{
  using TypeOut = int64_t;
  using TypeLhs = float;
  using TypeRhs = float;

  using MOD = cudf::library::operation::Mod<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::MOD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MOD());
}

TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP64)
{
  using TypeOut = double;
  using TypeLhs = double;
  using TypeRhs = double;

  using MOD = cudf::library::operation::Mod<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::MOD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MOD());
}

TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Scalar_DurationD_SI32_DurationUS)
{
  using TypeOut = cudf::duration_us;
  using TypeLhs = cudf::duration_D;
  using TypeRhs = int64_t;

  using MOD = cudf::library::operation::Mod<TypeOut, TypeLhs, TypeRhs>;

  // Half the number of days and convert the remainder ticks to microseconds
  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = cudf::scalar_type_t<TypeRhs>(2);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::MOD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MOD());
}

TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Scalar_DurationS_DurationMS_DurationUS)
{
  using TypeOut = cudf::duration_us;
  using TypeLhs = cudf::duration_s;
  using TypeRhs = cudf::duration_ms;

  using MOD = cudf::library::operation::Mod<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::MOD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MOD());
}

TEST_F(BinaryOperationIntegrationTest, Pow_Vector_Vector_FP64_SI64_SI64)
{
  using TypeOut = double;
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  using POW = cudf::library::operation::Pow<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::POW, data_type(type_to_id<TypeOut>()));

  /**
   * According to CUDA Programming Guide, 'E.1. Standard Functions', 'Table 7 - Double-Precision
   * Mathematical Standard Library Functions with Maximum ULP Error'
   * The pow function has 2 (full range) maximum ulp error.
   */
  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, POW(), NearEqualComparator<TypeOut>{2});
}

TEST_F(BinaryOperationIntegrationTest, Pow_Vector_Vector_FP32)
{
  using TypeOut = float;
  using TypeLhs = float;
  using TypeRhs = float;

  using POW = cudf::library::operation::Pow<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::POW, data_type(type_to_id<TypeOut>()));
  /**
   * According to CUDA Programming Guide, 'E.1. Standard Functions', 'Table 7 - Double-Precision
   * Mathematical Standard Library Functions with Maximum ULP Error'
   * The pow function has 2 (full range) maximum ulp error.
   */
  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, POW(), NearEqualComparator<TypeOut>{2});
}

TEST_F(BinaryOperationIntegrationTest, And_Vector_Vector_SI16_SI64_SI32)
{
  using TypeOut = int16_t;
  using TypeLhs = int64_t;
  using TypeRhs = int32_t;

  using AND = cudf::library::operation::BitwiseAnd<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::BITWISE_AND, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, AND());
}

TEST_F(BinaryOperationIntegrationTest, Or_Vector_Vector_SI64_SI16_SI32)
{
  using TypeOut = int64_t;
  using TypeLhs = int16_t;
  using TypeRhs = int32_t;

  using OR = cudf::library::operation::BitwiseOr<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::BITWISE_OR, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, OR());
}

TEST_F(BinaryOperationIntegrationTest, Xor_Vector_Vector_SI32_SI16_SI64)
{
  using TypeOut = int32_t;
  using TypeLhs = int16_t;
  using TypeRhs = int64_t;

  using XOR = cudf::library::operation::BitwiseXor<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::BITWISE_XOR, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, XOR());
}

TEST_F(BinaryOperationIntegrationTest, Logical_And_Vector_Vector_SI16_FP64_SI8)
{
  using TypeOut = int16_t;
  using TypeLhs = double;
  using TypeRhs = int8_t;

  using AND = cudf::library::operation::LogicalAnd<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::LOGICAL_AND, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, AND());
}

TEST_F(BinaryOperationIntegrationTest, Logical_Or_Vector_Vector_B8_SI16_SI64)
{
  using TypeOut = bool;
  using TypeLhs = int16_t;
  using TypeRhs = int64_t;

  using OR = cudf::library::operation::LogicalOr<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::LOGICAL_OR, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, OR());
}

TEST_F(BinaryOperationIntegrationTest, Less_Scalar_Vector_B8_TSS_TSS)
{
  using TypeOut = bool;
  using TypeLhs = cudf::timestamp_s;
  using TypeRhs = cudf::timestamp_s;

  using LESS = cudf::library::operation::Less<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  auto rhs = make_random_wrapped_column<TypeRhs>(10);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::LESS, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LESS());
}

TEST_F(BinaryOperationIntegrationTest, Greater_Scalar_Vector_B8_TSMS_TSS)
{
  using TypeOut = bool;
  using TypeLhs = cudf::timestamp_ms;
  using TypeRhs = cudf::timestamp_s;

  using GREATER = cudf::library::operation::Greater<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::GREATER, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, GREATER());
}

TEST_F(BinaryOperationIntegrationTest, Less_Vector_Vector_B8_TSS_TSS)
{
  using TypeOut = bool;
  using TypeLhs = cudf::timestamp_s;
  using TypeRhs = cudf::timestamp_s;

  using LESS = cudf::library::operation::Less<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(10);
  auto rhs = make_random_wrapped_column<TypeRhs>(10);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::LESS, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LESS());
}

TEST_F(BinaryOperationIntegrationTest, Greater_Vector_Vector_B8_TSMS_TSS)
{
  using TypeOut = bool;
  using TypeLhs = cudf::timestamp_ms;
  using TypeRhs = cudf::timestamp_s;

  using GREATER = cudf::library::operation::Greater<TypeOut, TypeLhs, TypeRhs>;

  cudf::test::UniformRandomGenerator<long> rand_gen(1, 10);
  auto itr = cudf::test::make_counting_transform_iterator(
    0, [&rand_gen](auto row) { return rand_gen.generate() * 1000; });

  cudf::test::fixed_width_column_wrapper<TypeLhs, typename decltype(itr)::value_type> lhs(
    itr, itr + 100, make_validity_iter());

  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::GREATER, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, GREATER());
}

TEST_F(BinaryOperationIntegrationTest, Less_Scalar_Vector_B8_STR_STR)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  using LESS = cudf::library::operation::Less<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = cudf::string_scalar("eee");
  auto rhs = cudf::test::strings_column_wrapper({"ééé", "bbb", "aa", "", "<null>", "bb", "eee"});
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::LESS, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LESS());
}

TEST_F(BinaryOperationIntegrationTest, Less_Vector_Scalar_B8_STR_STR)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  using LESS = cudf::library::operation::Less<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = cudf::test::strings_column_wrapper({"ééé", "bbb", "aa", "", "<null>", "bb", "eee"});
  auto rhs = cudf::string_scalar("eee");
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::LESS, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LESS());
}

TEST_F(BinaryOperationIntegrationTest, Less_Vector_Vector_B8_STR_STR)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  using LESS = cudf::library::operation::Less<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"});
  auto rhs = cudf::test::strings_column_wrapper({"ééé", "bbb", "aa", "", "<null>", "bb", "eee"});
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::LESS, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LESS());
}

TEST_F(BinaryOperationIntegrationTest, Greater_Vector_Vector_B8_STR_STR)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  using GREATER = cudf::library::operation::Greater<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"});
  auto rhs = cudf::test::strings_column_wrapper({"ééé", "bbb", "aa", "", "<null>", "bb", "eee"});
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::GREATER, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, GREATER());
}

TEST_F(BinaryOperationIntegrationTest, Equal_Vector_Vector_B8_STR_STR)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  using EQUAL = cudf::library::operation::Equal<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"});
  auto rhs = cudf::test::strings_column_wrapper({"ééé", "bbb", "aa", "", "<null>", "bb", "eee"});
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::EQUAL, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, EQUAL());
}

TEST_F(BinaryOperationIntegrationTest, Equal_Vector_Scalar_B8_STR_STR)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  using EQUAL = cudf::library::operation::Equal<TypeOut, TypeLhs, TypeRhs>;

  auto rhs = cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"});
  auto lhs = cudf::string_scalar("");
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::EQUAL, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, EQUAL());
}

TEST_F(BinaryOperationIntegrationTest, LessEqual_Vector_Vector_B8_STR_STR)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  using LESS_EQUAL = cudf::library::operation::LessEqual<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"});
  auto rhs = cudf::test::strings_column_wrapper({"ééé", "bbb", "aa", "", "<null>", "bb", "eee"});
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::LESS_EQUAL, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LESS_EQUAL());
}

TEST_F(BinaryOperationIntegrationTest, GreaterEqual_Vector_Vector_B8_STR_STR)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  using GREATER_EQUAL = cudf::library::operation::GreaterEqual<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"});
  auto rhs = cudf::test::strings_column_wrapper({"ééé", "bbb", "aa", "", "<null>", "bb", "eee"});
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::GREATER_EQUAL, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, GREATER_EQUAL());
}

TEST_F(BinaryOperationIntegrationTest, ShiftLeft_Vector_Vector_SI32)
{
  using TypeOut = int;
  using TypeLhs = int;
  using TypeRhs = int;

  using SHIFT_LEFT = cudf::library::operation::ShiftLeft<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  // this generates values in the range 1-10 which should be reasonable for the shift
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::SHIFT_LEFT, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SHIFT_LEFT());
}

TEST_F(BinaryOperationIntegrationTest, ShiftLeft_Vector_Vector_SI32_SI16_SI64)
{
  using TypeOut = int;
  using TypeLhs = int16_t;
  using TypeRhs = int64_t;

  using SHIFT_LEFT = cudf::library::operation::ShiftLeft<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  // this generates values in the range 1-10 which should be reasonable for the shift
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::SHIFT_LEFT, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SHIFT_LEFT());
}

TEST_F(BinaryOperationIntegrationTest, ShiftLeft_Scalar_Vector_SI32)
{
  using TypeOut = int;
  using TypeLhs = int;
  using TypeRhs = int;

  using SHIFT_LEFT = cudf::library::operation::ShiftLeft<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  // this generates values in the range 1-10 which should be reasonable for the shift
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::SHIFT_LEFT, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SHIFT_LEFT());
}

TEST_F(BinaryOperationIntegrationTest, ShiftLeft_Vector_Scalar_SI32)
{
  using TypeOut = int;
  using TypeLhs = int;
  using TypeRhs = int;

  using SHIFT_LEFT = cudf::library::operation::ShiftLeft<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  // this generates values in the range 1-10 which should be reasonable for the shift
  auto rhs = make_random_wrapped_scalar<TypeRhs>();
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::SHIFT_LEFT, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SHIFT_LEFT());
}

TEST_F(BinaryOperationIntegrationTest, ShiftRight_Vector_Vector_SI32)
{
  using TypeOut = int;
  using TypeLhs = int;
  using TypeRhs = int;

  using SHIFT_RIGHT = cudf::library::operation::ShiftRight<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  // this generates values in the range 1-10 which should be reasonable for the shift
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::SHIFT_RIGHT, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SHIFT_RIGHT());
}

TEST_F(BinaryOperationIntegrationTest, ShiftRight_Vector_Vector_SI32_SI16_SI64)
{
  using TypeOut = int;
  using TypeLhs = int16_t;
  using TypeRhs = int64_t;

  using SHIFT_RIGHT = cudf::library::operation::ShiftRight<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  // this generates values in the range 1-10 which should be reasonable for the shift
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::SHIFT_RIGHT, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SHIFT_RIGHT());
}

TEST_F(BinaryOperationIntegrationTest, ShiftRight_Scalar_Vector_SI32)
{
  using TypeOut = int;
  using TypeLhs = int;
  using TypeRhs = int;

  using SHIFT_RIGHT = cudf::library::operation::ShiftRight<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  // this generates values in the range 1-10 which should be reasonable for the shift
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::SHIFT_RIGHT, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SHIFT_RIGHT());
}

TEST_F(BinaryOperationIntegrationTest, ShiftRight_Vector_Scalar_SI32)
{
  using TypeOut = int;
  using TypeLhs = int;
  using TypeRhs = int;

  using SHIFT_RIGHT = cudf::library::operation::ShiftRight<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  // this generates values in the range 1-10 which should be reasonable for the shift
  auto rhs = make_random_wrapped_scalar<TypeRhs>();
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::SHIFT_RIGHT, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SHIFT_RIGHT());
}

TEST_F(BinaryOperationIntegrationTest, ShiftRightUnsigned_Vector_Vector_SI32)
{
  using TypeOut = int;
  using TypeLhs = int;
  using TypeRhs = int;

  using SHIFT_RIGHT_UNSIGNED =
    cudf::library::operation::ShiftRightUnsigned<TypeOut, TypeLhs, TypeRhs>;

  int num_els = 4;

  TypeLhs lhs[] = {-8, 78, -93, 0, -INT_MAX};
  cudf::test::fixed_width_column_wrapper<TypeLhs> lhs_w(lhs, lhs + num_els);

  TypeRhs shift[] = {1, 1, 3, 2, 16};
  cudf::test::fixed_width_column_wrapper<TypeRhs> shift_w(shift, shift + num_els);

  TypeOut expected[] = {2147483644, 39, 536870900, 0, 32768};
  cudf::test::fixed_width_column_wrapper<TypeOut> expected_w(expected, expected + num_els);

  auto out = cudf::binary_operation(
    lhs_w, shift_w, cudf::binary_operator::SHIFT_RIGHT_UNSIGNED, data_type(type_to_id<TypeOut>()));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*out, expected_w);
}

TEST_F(BinaryOperationIntegrationTest, ShiftRightUnsigned_Vector_Vector_SI32_SI16_SI64)
{
  using TypeOut = int;
  using TypeLhs = int16_t;
  using TypeRhs = int64_t;

  using SHIFT_RIGHT_UNSIGNED =
    cudf::library::operation::ShiftRightUnsigned<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  // this generates values in the range 1-10 which should be reasonable for the shift
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::SHIFT_RIGHT_UNSIGNED, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SHIFT_RIGHT_UNSIGNED());
}

TEST_F(BinaryOperationIntegrationTest, ShiftRightUnsigned_Scalar_Vector_SI32)
{
  using TypeOut = int;
  using TypeLhs = int;
  using TypeRhs = int;

  using SHIFT_RIGHT_UNSIGNED =
    cudf::library::operation::ShiftRightUnsigned<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  // this generates values in the range 1-10 which should be reasonable for the shift
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::SHIFT_RIGHT_UNSIGNED, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SHIFT_RIGHT_UNSIGNED());
}

TEST_F(BinaryOperationIntegrationTest, ShiftRightUnsigned_Vector_Scalar_SI32)
{
  using TypeOut = int;
  using TypeLhs = int;
  using TypeRhs = int;

  using SHIFT_RIGHT_UNSIGNED =
    cudf::library::operation::ShiftRightUnsigned<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  // this generates values in the range 1-10 which should be reasonable for the shift
  auto rhs = make_random_wrapped_scalar<TypeRhs>();
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::SHIFT_RIGHT_UNSIGNED, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SHIFT_RIGHT_UNSIGNED());
}

TEST_F(BinaryOperationIntegrationTest, LogBase_Vector_Scalar_SI32_SI32_float)
{
  using TypeOut = int;      // Cast the result value to int for easy comparison
  using TypeLhs = int32_t;  // All input types get converted into doubles
  using TypeRhs = float;

  using LOG_BASE = cudf::library::operation::LogBase<TypeOut, TypeLhs, TypeRhs>;

  // Make sure there are no zeros. The log value is purposefully cast to int for easy comparison
  auto elements = make_counting_transform_iterator(1, [](auto i) { return i + 10; });
  fixed_width_column_wrapper<TypeLhs> lhs(elements, elements + 100);
  // Find log to the base 10
  auto rhs = numeric_scalar<TypeRhs>(10);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::LOG_BASE, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LOG_BASE());
}

TEST_F(BinaryOperationIntegrationTest, LogBase_Scalar_Vector_float_SI32)
{
  using TypeOut = float;
  using TypeLhs = int;
  using TypeRhs = int;  // Integral types promoted to double

  using LOG_BASE = cudf::library::operation::LogBase<TypeOut, TypeLhs, TypeRhs>;

  // Make sure there are no zeros
  auto elements = make_counting_transform_iterator(1, [](auto i) { return i + 30; });
  fixed_width_column_wrapper<TypeRhs> rhs(elements, elements + 100);
  // Find log to the base 2
  auto lhs = numeric_scalar<TypeLhs>(2);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::LOG_BASE, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LOG_BASE());
}

TEST_F(BinaryOperationIntegrationTest, LogBase_Vector_Vector_double_SI64_SI32)
{
  using TypeOut = double;
  using TypeLhs = int64_t;
  using TypeRhs = int32_t;  // Integral types promoted to double

  using LOG_BASE = cudf::library::operation::LogBase<TypeOut, TypeLhs, TypeRhs>;

  // Make sure there are no zeros
  auto elements = make_counting_transform_iterator(1, [](auto i) { return std::pow(2, i); });
  fixed_width_column_wrapper<TypeLhs> lhs(elements, elements + 50);

  // Find log to the base 7
  auto rhs_elements = make_counting_transform_iterator(0, [](auto) { return 7; });
  fixed_width_column_wrapper<TypeRhs> rhs(rhs_elements, rhs_elements + 50);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::LOG_BASE, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LOG_BASE());
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Vector_Scalar_B8_SI32_SI32)
{
  using TypeOut = bool;
  using TypeLhs = int32_t;
  using TypeRhs = int32_t;

  auto int_col =
    fixed_width_column_wrapper<TypeLhs>{{999, -37, 0, INT32_MAX}, {true, true, true, false}};
  auto int_scalar = cudf::scalar_type_t<TypeRhs>(999);

  auto op_col = cudf::binary_operation(
    int_col, int_scalar, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<bool>{{true, false, false, false}, {true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Vector_ScalarInvalid_B8_SI32_SI32)
{
  using TypeOut = bool;
  using TypeLhs = int32_t;
  using TypeRhs = int32_t;

  auto int_col    = fixed_width_column_wrapper<TypeLhs>{{-INT32_MAX, -37, 0, 499, 44, INT32_MAX},
                                                     {false, true, false, true, true, false}};
  auto int_scalar = cudf::scalar_type_t<TypeRhs>(999);
  int_scalar.set_valid(false);

  auto op_col = cudf::binary_operation(
    int_col, int_scalar, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*op_col,
                                 fixed_width_column_wrapper<bool>{
                                   {true, false, true, false, false, true},
                                   {true, true, true, true, true, true},
                                 },
                                 true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Scalar_Vector_B8_tsD_tsD)
{
  using TypeOut = bool;
  using TypeLhs = cudf::timestamp_D;
  using TypeRhs = cudf::timestamp_D;

  cudf::test::fixed_width_column_wrapper<TypeLhs, TypeLhs::rep> ts_col{
    {
      999,    // Random nullable field
      0,      // This is the UNIX epoch - 1970-01-01
      44376,  // 2091-07-01 00:00:00 GMT
      47695,  // 2100-08-02 00:00:00 GMT
      3,      // Random nullable field
      66068,  // 2150-11-21 00:00:00 GMT
      22270,  // 2030-12-22 00:00:00 GMT
      111,    // Random nullable field
    },
    {false, true, true, true, false, true, true, false}};
  auto ts_scalar = cudf::scalar_type_t<TypeRhs>(typename TypeRhs::duration{44376}, true);

  auto op_col = cudf::binary_operation(
    ts_scalar, ts_col, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*op_col,
                                 fixed_width_column_wrapper<bool>{
                                   {false, false, true, false, false, false, false, false},
                                   {true, true, true, true, true, true, true, true},
                                 },
                                 true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Vector_Scalar_B8_string_string_EmptyString)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  auto str_col = cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"},
                                                    {true, false, true, true, true, false, true});
  // Empty string
  cudf::string_scalar str_scalar("");

  auto op_col = cudf::binary_operation(
    str_col, str_scalar, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<bool>{{false, false, false, true, false, false, false},
                                     {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Scalar_Vector_B8_string_string_ValidString)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  auto str_col = cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"},
                                                    {true, false, true, true, true, false, true});
  // Match a valid string
  cudf::string_scalar str_scalar("<null>");

  auto op_col = cudf::binary_operation(
    str_scalar, str_col, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<bool>{{false, false, true, false, false, false, false},
                                     {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Vector_Scalar_B8_string_string_NoMatch)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  // Try with non nullable input
  auto str_col =
    cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"});
  // Matching a string that isn't present
  cudf::string_scalar str_scalar("foo");

  auto op_col = cudf::binary_operation(
    str_col, str_scalar, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<bool>{{false, false, false, false, false, false, false},
                                     {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Scalar_Vector_B8_string_string_NullNonNull)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  // Try with all invalid input
  auto str_col = cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"},
                                                    {true, true, true, true, true, true, true});
  // Matching a scalar that is invalid
  cudf::string_scalar str_scalar("foo");
  str_scalar.set_valid(false);

  auto op_col = cudf::binary_operation(
    str_scalar, str_col, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<bool>{{false, false, false, false, false, false, false},
                                     {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Vector_Scalar_B8_string_string_NullNonNull)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  // Try with all invalid input
  auto str_col =
    cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"},
                                       {false, false, false, false, false, false, false});
  // Matching a scalar that is valid
  cudf::string_scalar str_scalar("foo");

  auto op_col = cudf::binary_operation(
    str_scalar, str_col, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<bool>{{false, false, false, false, false, false, false},
                                     {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Scalar_Vector_B8_string_string_NullNull)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  // Try with all invalid input
  auto str_col =
    cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"},
                                       {false, false, false, false, false, false, false});
  // Matching a scalar that is invalid
  cudf::string_scalar str_scalar("foo");
  str_scalar.set_valid(false);

  auto op_col = cudf::binary_operation(
    str_scalar, str_col, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<bool>{{true, true, true, true, true, true, true},
                                     {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Scalar_Vector_B8_string_string_MatchInvalid)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  auto str_col = cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"},
                                                    {true, false, true, true, true, false, true});
  // Matching an invalid string
  cudf::string_scalar str_scalar("bb");

  auto op_col = cudf::binary_operation(
    str_scalar, str_col, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<bool>{{false, false, false, false, false, false, false},
                                     {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Vector_InvalidScalar_B8_string_string)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  auto str_col = cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"},
                                                    {true, false, true, true, true, false, true});
  // Valid string invalidated
  cudf::string_scalar str_scalar("bb");
  str_scalar.set_valid(false);

  auto op_col = cudf::binary_operation(
    str_col, str_scalar, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<bool>{{false, true, false, false, false, true, false},
                                     {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Vector_Vector_B8_tsD_tsD_NonNullable)
{
  using TypeOut = bool;
  using TypeLhs = cudf::timestamp_D;
  using TypeRhs = cudf::timestamp_D;

  cudf::test::fixed_width_column_wrapper<TypeLhs, TypeLhs::rep> lhs_col{
    0,      // This is the UNIX epoch - 1970-01-01
    44376,  // 2091-07-01 00:00:00 GMT
    47695,  // 2100-08-02 00:00:00 GMT
    66068,  // 2150-11-21 00:00:00 GMT
    22270,  // 2030-12-22 00:00:00 GMT
  };
  ASSERT_EQ(column_view{lhs_col}.nullable(), false);
  cudf::test::fixed_width_column_wrapper<TypeRhs, TypeRhs::rep> rhs_col{
    0,      // This is the UNIX epoch - 1970-01-01
    44380,  // Mismatched
    47695,  // 2100-08-02 00:00:00 GMT
    66070,  // Mismatched
    22270,  // 2030-12-22 00:00:00 GMT
  };

  auto op_col = cudf::binary_operation(
    lhs_col, rhs_col, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*op_col,
                                 fixed_width_column_wrapper<bool>{
                                   {true, false, true, false, true},
                                   {true, true, true, true, true},
                                 },
                                 true);
}

// Both vectors with mixed validity
TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Vector_Vector_B8_string_string_MixMix)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  auto lhs_col =
    cudf::test::strings_column_wrapper({"eee", "invalid", "<null>", "", "aa", "invalid", "ééé"},
                                       {true, false, true, true, true, false, true});
  auto rhs_col =
    cudf::test::strings_column_wrapper({"foo", "valid", "<null>", "", "invalid", "inv", "ééé"},
                                       {true, true, true, true, false, false, true});

  auto op_col = cudf::binary_operation(
    lhs_col, rhs_col, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<bool>{{false, false, true, true, false, true, true},
                                     {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Vector_Vector_B8_string_string_MixValid)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  auto lhs_col =
    cudf::test::strings_column_wrapper({"eee", "invalid", "<null>", "", "aa", "invalid", "ééé"},
                                       {true, false, true, true, true, false, true});
  auto rhs_col =
    cudf::test::strings_column_wrapper({"eee", "invalid", "<null>", "", "aa", "invalid", "ééé"});

  auto op_col = cudf::binary_operation(
    lhs_col, rhs_col, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<bool>{{true, false, true, true, true, false, true},
                                     {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Vector_Vector_B8_string_string_MixInvalid)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  auto lhs_col =
    cudf::test::strings_column_wrapper({"eee", "invalid", "<null>", "", "aa", "invalid", "ééé"},
                                       {true, false, true, true, true, false, true});
  auto rhs_col =
    cudf::test::strings_column_wrapper({"eee", "invalid", "<null>", "", "aa", "invalid", "ééé"},
                                       {false, false, false, false, false, false, false});

  auto op_col = cudf::binary_operation(
    lhs_col, rhs_col, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<bool>{{false, true, false, false, false, true, false},
                                     {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Vector_Vector_B8_string_string_ValidValid)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  auto lhs_col =
    cudf::test::strings_column_wrapper({"eee", "invalid", "<null>", "", "aa", "invalid", "ééé"});
  auto rhs_col =
    cudf::test::strings_column_wrapper({"eee", "invalid", "<null>", "", "aa", "invalid", "ééé"});

  auto op_col = cudf::binary_operation(
    lhs_col, rhs_col, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<bool>{{true, true, true, true, true, true, true},
                                     {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Vector_Vector_B8_string_string_ValidInvalid)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  auto lhs_col =
    cudf::test::strings_column_wrapper({"eee", "invalid", "<null>", "", "aa", "invalid", "ééé"});
  auto rhs_col =
    cudf::test::strings_column_wrapper({"eee", "invalid", "<null>", "", "aa", "invalid", "ééé"},
                                       {false, false, false, false, false, false, false});

  auto op_col = cudf::binary_operation(
    lhs_col, rhs_col, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<bool>{{false, false, false, false, false, false, false},
                                     {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Vector_Vector_B8_string_string_InvalidInvalid)
{
  using TypeOut = bool;
  using TypeLhs = std::string;
  using TypeRhs = std::string;

  auto lhs_col =
    cudf::test::strings_column_wrapper({"eee", "invalid", "<null>", "", "aa", "invalid", "ééé"},
                                       {false, false, false, false, false, false, false});
  auto rhs_col =
    cudf::test::strings_column_wrapper({"eee", "invalid", "<null>", "", "aa", "invalid", "ééé"},
                                       {false, false, false, false, false, false, false});

  auto op_col = cudf::binary_operation(
    lhs_col, rhs_col, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<bool>{{true, true, true, true, true, true, true},
                                     {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareEqual_Vector_VectorAllInvalid_B8_SI32_SI32)
{
  using TypeOut = bool;
  using TypeLhs = int32_t;
  using TypeRhs = int32_t;

  auto lhs_col = fixed_width_column_wrapper<TypeLhs>{{-INT32_MAX, -37, 0, 499, 44, INT32_MAX},
                                                     {false, false, false, false, false, false}};
  auto rhs_col = fixed_width_column_wrapper<TypeLhs>{{-47, 37, 12, 99, 4, -INT32_MAX},
                                                     {false, false, false, false, false, false}};

  auto op_col = cudf::binary_operation(
    lhs_col, rhs_col, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*op_col,
                                 fixed_width_column_wrapper<bool>{
                                   {true, true, true, true, true, true},
                                   {true, true, true, true, true, true},
                                 },
                                 true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareMin_Vector_Scalar_SI64_SI32_SI8)
{
  using TypeOut = int64_t;
  using TypeLhs = int32_t;
  using TypeRhs = int8_t;

  auto int_col = fixed_width_column_wrapper<TypeLhs>{
    {999, -37, 0, INT32_MAX},
  };
  auto int_scalar = cudf::scalar_type_t<TypeRhs>(77);

  auto op_col = cudf::binary_operation(
    int_col, int_scalar, cudf::binary_operator::NULL_MIN, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col, fixed_width_column_wrapper<TypeOut>{{77, -37, 0, 77}, {true, true, true, true}}, true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareMax_Scalar_Vector_FP64_SI32_SI64)
{
  using TypeOut = double;
  using TypeLhs = int32_t;
  using TypeRhs = int64_t;

  auto int_col =
    fixed_width_column_wrapper<TypeLhs>{{999, -37, 0, INT32_MAX, -INT32_MAX, -4379, 55},
                                        {false, true, false, true, false, true, false}};
  auto int_scalar = cudf::scalar_type_t<TypeRhs>(INT32_MAX);

  auto op_col = cudf::binary_operation(
    int_scalar, int_col, cudf::binary_operator::NULL_MAX, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<TypeOut>{
      {INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX},
      {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareMin_Vector_Scalar_SI64_SI32_FP32)
{
  using TypeOut = int64_t;
  using TypeLhs = int32_t;
  using TypeRhs = float;

  auto int_col =
    fixed_width_column_wrapper<TypeLhs>{{999, -37, 0, INT32_MAX, -INT32_MAX, -4379, 55},
                                        {false, true, false, true, false, true, false}};
  auto float_scalar = cudf::scalar_type_t<TypeRhs>(-3.14f);
  float_scalar.set_valid(false);

  auto op_col = cudf::binary_operation(
    int_col, float_scalar, cudf::binary_operator::NULL_MIN, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<TypeOut>{{0, -37, 0, INT32_MAX, 0, -4379, 0},
                                        {false, true, false, true, false, true, false}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareMax_Scalar_Vector_SI8_SI8_FP32)
{
  using TypeOut = int8_t;
  using TypeLhs = int8_t;
  using TypeRhs = float;

  auto int_col = fixed_width_column_wrapper<TypeLhs>{
    {9, -37, 0, 32, -47, -4, 55}, {false, false, false, false, false, false, false}};
  auto float_scalar = cudf::scalar_type_t<TypeRhs>(-3.14f);
  float_scalar.set_valid(false);

  auto op_col = cudf::binary_operation(
    float_scalar, int_col, cudf::binary_operator::NULL_MAX, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<TypeOut>{{0, 0, 0, 0, 0, 0, 0},
                                        {false, false, false, false, false, false, false}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareMin_Vector_Vector_SI64_SI32_SI8)
{
  using TypeOut = int64_t;
  using TypeLhs = int32_t;
  using TypeRhs = int8_t;

  auto int_col =
    fixed_width_column_wrapper<TypeLhs>{{999, -37, 0, INT32_MAX, -INT32_MAX, -4379, 55},
                                        {false, false, false, false, false, false, false}};
  auto another_int_col = fixed_width_column_wrapper<TypeLhs>{
    {9, -37, 0, 32, -47, -4, 55}, {false, false, false, false, false, false, false}};

  auto op_col = cudf::binary_operation(
    int_col, another_int_col, cudf::binary_operator::NULL_MIN, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<TypeOut>{{0, 0, 0, 0, 0, 0, 0},
                                        {false, false, false, false, false, false, false}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareMax_Vector_Vector_SI64_SI32_SI8)
{
  using TypeOut = int64_t;
  using TypeLhs = int32_t;
  using TypeRhs = int8_t;

  auto int_col = fixed_width_column_wrapper<TypeLhs>{
    {999, -37, 0, INT32_MAX, -INT32_MAX, -4379, 55}, {true, true, true, true, true, true, true}};
  auto another_int_col = fixed_width_column_wrapper<TypeLhs>{
    {9, -37, 0, 32, -47, -4, 55}, {false, false, false, false, false, false, false}};

  auto op_col = cudf::binary_operation(
    int_col, another_int_col, cudf::binary_operator::NULL_MAX, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<TypeOut>{{999, -37, 0, INT32_MAX, -INT32_MAX, -4379, 55},
                                        {true, true, true, true, true, true, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareMin_Vector_Vector_tsD_tsD_tsD)
{
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> lhs_col{
    {
      0,      // This is the UNIX epoch - 1970-01-01
      44376,  // 2091-07-01 00:00:00 GMT
      47695,  // 2100-08-02 00:00:00 GMT
      66068,  // 2150-11-21 00:00:00 GMT
      22270,  // 2030-12-22 00:00:00 GMT
    },
    {true, false, true, true, false}};
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> rhs_col{
    {
      0,      // This is the UNIX epoch - 1970-01-01
      44380,  // Mismatched
      47695,  // 2100-08-02 00:00:00 GMT
      66070,  // Mismatched
      22270,  // 2030-12-22 00:00:00 GMT
    },
    {false, true, true, true, false}};

  auto op_col = cudf::binary_operation(
    lhs_col, rhs_col, cudf::binary_operator::NULL_MIN, data_type(type_to_id<cudf::timestamp_D>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
      {0, 44380, 47695, 66068, 0}, {true, true, true, true, false}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareMax_Vector_Vector_SI32_SI64_SI8)
{
  using TypeOut = int32_t;
  using TypeLhs = int64_t;
  using TypeRhs = int8_t;

  auto int_col =
    fixed_width_column_wrapper<TypeLhs>{{999, -37, 0, INT32_MAX, -INT32_MAX, -4379, 55},
                                        {false, false, false, false, false, false, false}};
  auto another_int_col = fixed_width_column_wrapper<TypeLhs>{
    {9, -37, 0, 32, -47, -4, 55}, {true, false, true, false, true, false, true}};

  auto op_col = cudf::binary_operation(
    int_col, another_int_col, cudf::binary_operator::NULL_MAX, data_type(type_to_id<TypeOut>()));

  // Every row has a value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *op_col,
    fixed_width_column_wrapper<TypeOut>{{9, 0, 0, 0, -47, 0, 55},
                                        {true, false, true, false, true, false, true}},
    true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareMax_Vector_Vector_string_string_string_Mix)
{
  auto lhs_col = cudf::test::strings_column_wrapper(
    {"eee", "invalid", "<null>", "", "", "", "ééé", "foo", "bar", "abc", "def"},
    {false, true, true, false, true, true, true, false, false, true, true});
  auto rhs_col = cudf::test::strings_column_wrapper(
    {"eee", "goo", "<null>", "", "", "", "ééé", "bar", "foo", "def", "abc"},
    {false, true, true, true, false, true, true, false, false, true, true});

  auto op_col = cudf::binary_operation(
    lhs_col, rhs_col, cudf::binary_operator::NULL_MAX, data_type{type_id::STRING});

  auto exp_col = cudf::test::strings_column_wrapper(
    {"", "invalid", "<null>", "", "", "", "ééé", "", "", "def", "def"},
    {false, true, true, true, true, true, true, false, false, true, true});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*op_col, exp_col, true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareMin_Vector_Scalar_string_string_string_Mix)
{
  auto lhs_col = cudf::test::strings_column_wrapper(
    {"eee", "invalid", "<null>", "", "", "", "ééé", "foo", "bar", "abc", "foo"},
    {false, true, true, false, true, true, true, false, false, true, true});
  cudf::string_scalar str_scalar("foo");

  // Returns a non-nullable column as all elements are valid - it will have the scalar
  // value at the very least
  auto op_col = cudf::binary_operation(
    lhs_col, str_scalar, cudf::binary_operator::NULL_MIN, data_type{type_id::STRING});

  auto exp_col = cudf::test::strings_column_wrapper(
    {"foo", "foo", "<null>", "foo", "", "", "foo", "foo", "foo", "abc", "foo"});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*op_col, exp_col, true);
}

TEST_F(BinaryOperationIntegrationTest, NullAwareMax_Scalar_Vector_string_string_string_Mix)
{
  auto lhs_col = cudf::test::strings_column_wrapper(
    {"eee", "invalid", "<null>", "", "", "", "ééé", "foo", "bar", "abc", "foo"},
    {false, true, true, false, true, true, true, false, false, true, true});
  cudf::string_scalar str_scalar("foo");
  str_scalar.set_valid(false);

  // Returns the lhs_col
  auto op_col = cudf::binary_operation(
    str_scalar, lhs_col, cudf::binary_operator::NULL_MAX, data_type{type_id::STRING});

  auto exp_col = cudf::test::strings_column_wrapper(
    {"", "invalid", "<null>", "", "", "", "ééé", "", "", "abc", "foo"},
    {false, true, true, false, true, true, true, false, false, true, true});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*op_col, exp_col, true);
}

TEST_F(BinaryOperationIntegrationTest, CastAdd_Vector_Vector_SI32_float_float)
{
  using TypeOut = int32_t;
  using TypeLhs = float;
  using TypeRhs = float;  // Integral types promoted to double

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs      = cudf::test::fixed_width_column_wrapper<float>{1.3f, 1.6f};
  auto rhs      = cudf::test::fixed_width_column_wrapper<float>{1.3f, 1.6f};
  auto expected = cudf::test::fixed_width_column_wrapper<int>{2, 3};

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationIntegrationTest, Add_Vector_Vector_TimepointD_DurationS_TimepointUS)
{
  using TypeOut = cudf::timestamp_us;
  using TypeLhs = cudf::timestamp_D;
  using TypeRhs = cudf::duration_s;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationIntegrationTest, Add_Vector_Scalar_DurationD_TimepointS_TimepointS)
{
  using TypeOut = cudf::timestamp_s;
  using TypeLhs = cudf::duration_D;
  using TypeRhs = cudf::timestamp_s;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = cudf::scalar_type_t<TypeRhs>(typename TypeRhs::duration{34}, true);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationIntegrationTest, Add_Scalar_Vector_DurationS_DurationD_DurationMS)
{
  using TypeOut = cudf::duration_ms;
  using TypeLhs = cudf::duration_s;
  using TypeRhs = cudf::duration_D;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = cudf::scalar_type_t<TypeLhs>(TypeLhs{-9});
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationIntegrationTest, ShiftRightUnsigned_Scalar_Vector_SI64_SI64_SI32)
{
  using TypeOut = int64_t;
  using TypeLhs = int64_t;
  using TypeRhs = int32_t;

  using SHIFT_RIGHT_UNSIGNED =
    cudf::library::operation::ShiftRightUnsigned<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = cudf::scalar_type_t<TypeLhs>(-12);
  // this generates values in the range 1-10 which should be reasonable for the shift
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::SHIFT_RIGHT_UNSIGNED, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SHIFT_RIGHT_UNSIGNED());
}

TEST_F(BinaryOperationIntegrationTest, PMod_Scalar_Vector_FP32)
{
  using TypeOut = float;
  using TypeLhs = float;
  using TypeRhs = float;

  using PMOD = cudf::library::operation::PMod<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = cudf::scalar_type_t<TypeLhs>(-86099.68377);
  auto rhs = fixed_width_column_wrapper<TypeRhs>{{90770.74881, -15456.4335, 32213.22119}};

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::PMOD, data_type(type_to_id<TypeOut>()));

  auto expected_result =
    fixed_width_column_wrapper<TypeOut>{{4671.0625, -8817.51953125, 10539.974609375}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*out, expected_result);
}

TEST_F(BinaryOperationIntegrationTest, PMod_Vector_Scalar_FP64)
{
  using TypeOut = double;
  using TypeLhs = double;
  using TypeRhs = double;

  using PMOD = cudf::library::operation::PMod<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = fixed_width_column_wrapper<TypeLhs>{{90770.74881, -15456.4335, 32213.22119}};
  auto rhs = cudf::scalar_type_t<TypeRhs>(-86099.68377);

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::PMOD, data_type(type_to_id<TypeOut>()));

  auto expected_result = fixed_width_column_wrapper<TypeOut>{
    {4671.0650400000013178, -15456.433499999999185, 32213.221190000000206}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*out, expected_result);
}

TEST_F(BinaryOperationIntegrationTest, PMod_Vector_Vector_FP64_FP32_FP64)
{
  using TypeOut = double;
  using TypeLhs = float;
  using TypeRhs = double;

  using PMOD = cudf::library::operation::PMod<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = fixed_width_column_wrapper<TypeLhs>{
    {24854.55893, 79946.87288, -86099.68377, -86099.68377, 1.0, 1.0, -1.0, -1.0}};
  auto rhs = fixed_width_column_wrapper<TypeRhs>{{90770.74881,
                                                  -15456.4335,
                                                  36223.96138,
                                                  -15456.4335,
                                                  2.1336193413893147E307,
                                                  -2.1336193413893147E307,
                                                  2.1336193413893147E307,
                                                  -2.1336193413893147E307}};

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::PMOD, data_type(type_to_id<TypeOut>()));

  auto expected_result = fixed_width_column_wrapper<TypeOut>{{24854.55859375,
                                                              2664.7075000000040745,
                                                              22572.196640000001935,
                                                              -8817.5200000000040745,
                                                              1.0,
                                                              1.0,
                                                              0.0,
                                                              0.0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*out, expected_result);
}

TEST_F(BinaryOperationIntegrationTest, PMod_Vector_Vector_FP64_SI32_SI64)
{
  using TypeOut = double;
  using TypeLhs = int32_t;
  using TypeRhs = int64_t;

  using PMOD = cudf::library::operation::PMod<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(1000);
  auto rhs = make_random_wrapped_column<TypeRhs>(1000);

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::PMOD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, PMOD());
}

TEST_F(BinaryOperationIntegrationTest, PMod_Vector_Vector_SI64_SI32_SI64)
{
  using TypeOut = int64_t;
  using TypeLhs = int32_t;
  using TypeRhs = int64_t;

  using PMOD = cudf::library::operation::PMod<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(1000);
  auto rhs = make_random_wrapped_column<TypeRhs>(1000);

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::PMOD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, PMOD());
}

TEST_F(BinaryOperationIntegrationTest, PMod_Vector_Vector_SI64_FP64_FP64)
{
  using TypeOut = int64_t;
  using TypeLhs = double;
  using TypeRhs = double;

  using PMOD = cudf::library::operation::PMod<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(1000);
  auto rhs = make_random_wrapped_column<TypeRhs>(1000);

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::PMOD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, PMOD());
}

TEST_F(BinaryOperationIntegrationTest, ATan2_Scalar_Vector_FP32)
{
  using TypeOut = float;
  using TypeLhs = float;
  using TypeRhs = float;

  using ATAN2 = cudf::library::operation::ATan2<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  auto rhs = make_random_wrapped_column<TypeRhs>(10000);

  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::ATAN2, data_type(type_to_id<TypeOut>()));

  // atan2 has a max ULP error of 2 per CUDA programming guide
  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ATAN2(), NearEqualComparator<TypeOut>{2});
}

TEST_F(BinaryOperationIntegrationTest, ATan2_Vector_Scalar_FP64)
{
  using TypeOut = double;
  using TypeLhs = double;
  using TypeRhs = double;

  using ATAN2 = cudf::library::operation::ATan2<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(10000);
  auto rhs = make_random_wrapped_scalar<TypeRhs>();

  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::ATAN2, data_type(type_to_id<TypeOut>()));

  // atan2 has a max ULP error of 2 per CUDA programming guide
  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ATAN2(), NearEqualComparator<TypeOut>{2});
}

TEST_F(BinaryOperationIntegrationTest, ATan2_Vector_Vector_FP64_FP32_FP64)
{
  using TypeOut = double;
  using TypeLhs = float;
  using TypeRhs = double;

  using ATAN2 = cudf::library::operation::ATan2<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(10000);
  auto rhs = make_random_wrapped_column<TypeRhs>(10000);

  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::ATAN2, data_type(type_to_id<TypeOut>()));

  // atan2 has a max ULP error of 2 per CUDA programming guide
  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ATAN2(), NearEqualComparator<TypeOut>{2});
}

TEST_F(BinaryOperationIntegrationTest, ATan2_Vector_Vector_FP64_SI32_SI64)
{
  using TypeOut = double;
  using TypeLhs = int32_t;
  using TypeRhs = int64_t;

  using ATAN2 = cudf::library::operation::ATan2<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(10000);
  auto rhs = make_random_wrapped_column<TypeRhs>(10000);

  auto out = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::ATAN2, data_type(type_to_id<TypeOut>()));

  // atan2 has a max ULP error of 2 per CUDA programming guide
  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ATAN2(), NearEqualComparator<TypeOut>{2});
}

// template <typename T>
// struct FixedPointTestBothReps : public cudf::test::BaseFixture {
// };

// template <typename T>
// using wrapper = cudf::test::fixed_width_column_wrapper<T>;
// TYPED_TEST_CASE(FixedPointTestBothReps, cudf::test::FixedPointTypes);

// TYPED_TEST(FixedPointTestBothReps, FixedPointBinaryOpAdd)
// {
//   using namespace numeric;
//   using decimalXX = TypeParam;

//   auto const sz = std::size_t{1000};

//   auto vec1       = std::vector<decimalXX>(sz);
//   auto const vec2 = std::vector<decimalXX>(sz, decimalXX{2, scale_type{0}});
//   auto expected   = std::vector<decimalXX>(sz);

//   std::iota(std::begin(vec1), std::end(vec1), decimalXX{1, scale_type{0}});

//   std::transform(std::cbegin(vec1),
//                  std::cend(vec1),
//                  std::cbegin(vec2),
//                  std::begin(expected),
//                  std::plus<decimalXX>());

//   auto const lhs          = wrapper<decimalXX>(vec1.begin(), vec1.end());
//   auto const rhs          = wrapper<decimalXX>(vec2.begin(), vec2.end());
//   auto const expected_col = wrapper<decimalXX>(expected.begin(), expected.end());

//   auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, {});

//   CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col, result->view());
// }

// TYPED_TEST(FixedPointTestBothReps, FixedPointBinaryOpMultiply)
// {
//   using namespace numeric;
//   using decimalXX = TypeParam;

//   auto const sz = std::size_t{1000};

//   auto vec1       = std::vector<decimalXX>(sz);
//   auto const vec2 = std::vector<decimalXX>(sz, decimalXX{2, scale_type{0}});
//   auto expected   = std::vector<decimalXX>(sz);

//   std::iota(std::begin(vec1), std::end(vec1), decimalXX{1, scale_type{0}});

//   std::transform(std::cbegin(vec1),
//                  std::cend(vec1),
//                  std::cbegin(vec2),
//                  std::begin(expected),
//                  std::multiplies<decimalXX>());

//   auto const lhs          = wrapper<decimalXX>(vec1.begin(), vec1.end());
//   auto const rhs          = wrapper<decimalXX>(vec2.begin(), vec2.end());
//   auto const expected_col = wrapper<decimalXX>(expected.begin(), expected.end());

//   auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::MUL, {});

//   CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col, result->view());
// }

// template <typename T>
// using fp_wrapper = cudf::test::fixed_point_column_wrapper<T>;

// TYPED_TEST(FixedPointTestBothReps, FixedPointBinaryOpMultiply2)
// {
//   using namespace numeric;
//   using decimalXX = TypeParam;
//   using RepType   = device_storage_type_t<decimalXX>;

//   auto const lhs      = fp_wrapper<RepType>{{11, 22, 33, 44, 55}, scale_type{-1}};
//   auto const rhs      = fp_wrapper<RepType>{{10, 10, 10, 10, 10}, scale_type{0}};
//   auto const expected = fp_wrapper<RepType>{{110, 220, 330, 440, 550}, scale_type{-1}};

//   auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::MUL, {});

//   CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
// }

// TYPED_TEST(FixedPointTestBothReps, FixedPointBinaryOpDiv)
// {
//   using namespace numeric;
//   using decimalXX = TypeParam;
//   using RepType   = device_storage_type_t<decimalXX>;

//   auto const lhs      = fp_wrapper<RepType>{{10, 30, 50, 70}, scale_type{-1}};
//   auto const rhs      = fp_wrapper<RepType>{{4, 4, 4, 4}, scale_type{0}};
//   auto const expected = fp_wrapper<RepType>{{3, 8, 13, 18}, scale_type{-1}};

//   auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::DIV, {});

//   CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
// }

// TYPED_TEST(FixedPointTestBothReps, FixedPointBinaryOpAdd2)
// {
//   using namespace numeric;
//   using decimalXX = TypeParam;
//   using RepType   = device_storage_type_t<decimalXX>;

//   auto const lhs      = fp_wrapper<RepType>{{11, 22, 33, 44, 55}, scale_type{-1}};
//   auto const rhs      = fp_wrapper<RepType>{{100, 200, 300, 400, 500}, scale_type{-2}};
//   auto const expected = fp_wrapper<RepType>{{210, 420, 630, 840, 1050}, scale_type{-2}};

//   auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, {});

//   CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
// }

// TYPED_TEST(FixedPointTestBothReps, FixedPointBinaryOpAdd3)
// {
//   using namespace numeric;
//   using decimalXX = TypeParam;
//   using RepType   = device_storage_type_t<decimalXX>;

//   auto const lhs      = fp_wrapper<RepType>{{1100, 2200, 3300, 4400, 5500}, scale_type{-3}};
//   auto const rhs      = fp_wrapper<RepType>{{100, 200, 300, 400, 500}, scale_type{-2}};
//   auto const expected = fp_wrapper<RepType>{{2100, 4200, 6300, 8400, 10500}, scale_type{-3}};

//   auto const result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, {});

//   CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
// }

// TYPED_TEST(FixedPointTestBothReps, FixedPointBinaryOpMultiplyScalar)
// {
//   using namespace numeric;
//   using decimalXX = TypeParam;
//   using RepType   = device_storage_type_t<decimalXX>;

//   auto const lhs      = fp_wrapper<RepType>{{11, 22, 33, 44, 55}, scale_type{-1}};
//   auto const rhs      = make_fixed_point_scalar<decimalXX>(100, scale_type{-1});
//   auto const expected = fp_wrapper<RepType>{{1100, 2200, 3300, 4400, 5500}, scale_type{-2}};

//   auto const result = cudf::binary_operation(lhs, *rhs, cudf::binary_operator::MUL, {});

//   CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
// }

// TYPED_TEST(FixedPointTestBothReps, FixedPointBinaryOpEqualSimple)
// {
//   using namespace numeric;
//   using decimalXX = TypeParam;
//   using RepType   = device_storage_type_t<decimalXX>;

//   auto const trues    = std::vector<bool>(4, true);
//   auto const col1     = fp_wrapper<RepType>{{1, 2, 3, 4}, scale_type{0}};
//   auto const col2     = fp_wrapper<RepType>{{100, 200, 300, 400}, scale_type{-2}};
//   auto const expected = wrapper<bool>(trues.begin(), trues.end());

//   auto const result = cudf::binary_operation(col1, col2, binary_operator::EQUAL, {});

//   CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
// }

// TYPED_TEST(FixedPointTestBothReps, FixedPointBinaryOpEqualLessGreater)
// {
//   using namespace numeric;
//   using decimalXX = TypeParam;
//   using RepType   = device_storage_type_t<decimalXX>;

//   auto const sz = std::size_t{1000};

//   // TESTING binary op ADD

//   auto begin      = make_counting_transform_iterator(1, [](auto e) { return e * 1000; });
//   auto const vec1 = std::vector<RepType>(begin, begin + sz);
//   auto const vec2 = std::vector<RepType>(sz, 0);

//   auto const iota_3  = fp_wrapper<RepType>(vec1.begin(), vec1.end(), scale_type{-3});
//   auto const zeros_3 = fp_wrapper<RepType>(vec2.begin(), vec2.end(), scale_type{-1});

//   auto const iota_3_after_add = cudf::binary_operation(zeros_3, iota_3, binary_operator::ADD,
//   {});

//   CUDF_TEST_EXPECT_COLUMNS_EQUAL(iota_3, iota_3_after_add->view());

//   // TESTING binary op EQUAL, LESS, GREATER

//   auto const trues    = std::vector<bool>(sz, true);
//   auto const true_col = wrapper<bool>(trues.begin(), trues.end());

//   auto const equal_result =
//     cudf::binary_operation(iota_3, iota_3_after_add->view(), binary_operator::EQUAL, {});
//   CUDF_TEST_EXPECT_COLUMNS_EQUAL(true_col, equal_result->view());

//   auto const less_result =
//     cudf::binary_operation(zeros_3, iota_3_after_add->view(), binary_operator::LESS, {});
//   CUDF_TEST_EXPECT_COLUMNS_EQUAL(true_col, less_result->view());

//   auto const greater_result =
//     cudf::binary_operation(iota_3_after_add->view(), zeros_3, binary_operator::GREATER, {});
//   CUDF_TEST_EXPECT_COLUMNS_EQUAL(true_col, greater_result->view());
// }

}  // namespace binop
}  // namespace test
}  // namespace cudf

CUDF_TEST_PROGRAM_MAIN()
