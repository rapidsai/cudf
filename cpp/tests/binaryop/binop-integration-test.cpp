/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/binaryop.hpp>
#include <tests/binaryop/binop-fixture.hpp>

namespace cudf {
namespace test {
namespace binop {

struct BinaryOperationIntegrationTest : public BinaryOperationTest {};

TEST_F(BinaryOperationIntegrationTest, Add_Scalar_Vector_SI32_FP32_SI64) {
  using TypeOut = int32_t;
  using TypeLhs = float;
  using TypeRhs = int64_t;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  auto rhs = make_random_wrapped_column<TypeRhs>(10000);

  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::ADD,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationIntegrationTest, Sub_Scalar_Vector_SI32_FP32_SI64) {
  using TypeOut = int32_t;
  using TypeLhs = float;
  using TypeRhs = int64_t;

  using SUB = cudf::library::operation::Sub<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  auto rhs = make_random_wrapped_column<TypeRhs>(10000);

  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::SUB,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SUB());
}

TEST_F(BinaryOperationIntegrationTest, Add_Vector_Scalar_SI08_SI16_SI32) {
  using TypeOut = int8_t;
  using TypeLhs = int16_t;
  using TypeRhs = int32_t;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_scalar<TypeRhs>();
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::ADD,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationIntegrationTest, Add_Vector_Vector_SI32_FP64_SI08) {
  using TypeOut = int32_t;
  using TypeLhs = double;
  using TypeRhs = int8_t;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::ADD,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationIntegrationTest, Sub_Vector_Vector_SI64) {
  using TypeOut = int64_t;
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  using SUB = cudf::library::operation::Sub<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::SUB,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SUB());
}

TEST_F(BinaryOperationIntegrationTest, Mul_Vector_Vector_SI64) {
  using TypeOut = int64_t;
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  using MUL = cudf::library::operation::Mul<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::MUL,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MUL());
}

TEST_F(BinaryOperationIntegrationTest, Div_Vector_Vector_SI64) {
  using TypeOut = int64_t;
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  using DIV = cudf::library::operation::Div<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::DIV,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, DIV());
}

TEST_F(BinaryOperationIntegrationTest, TrueDiv_Vector_Vector_SI64) {
  using TypeOut = int64_t;
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  using TRUEDIV = cudf::library::operation::TrueDiv<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::TRUE_DIV,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, TRUEDIV());
}

TEST_F(BinaryOperationIntegrationTest, FloorDiv_Vector_Vector_SI64) {
  using TypeOut = int64_t;
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  using FLOORDIV =
      cudf::library::operation::FloorDiv<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::FLOOR_DIV,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, FLOORDIV());
}

TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_SI64) {
  using TypeOut = int64_t;
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  using MOD = cudf::library::operation::Mod<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::MOD,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MOD());
}

TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP32) {
  using TypeOut = float;
  using TypeLhs = float;
  using TypeRhs = float;

  using MOD = cudf::library::operation::Mod<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::MOD,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MOD());
}

TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP64) {
  using TypeOut = double;
  using TypeLhs = double;
  using TypeRhs = double;

  using MOD = cudf::library::operation::Mod<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::MOD,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MOD());
}

TEST_F(BinaryOperationIntegrationTest, Pow_Vector_Vector_SI64) {
  using TypeOut = int64_t;
  using TypeLhs = int64_t;
  using TypeRhs = int64_t;

  using POW = cudf::library::operation::Pow<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::POW,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, POW());
}

TEST_F(BinaryOperationIntegrationTest, And_Vector_Vector_SI16_SI64_SI32) {
  using TypeOut = int16_t;
  using TypeLhs = int64_t;
  using TypeRhs = int32_t;

  using AND = cudf::library::operation::BitwiseAnd<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::BITWISE_AND,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, AND());
}

TEST_F(BinaryOperationIntegrationTest, Or_Vector_Vector_SI64_SI16_SI32) {
  using TypeOut = int64_t;
  using TypeLhs = int16_t;
  using TypeRhs = int32_t;

  using OR = cudf::library::operation::BitwiseOr<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::BITWISE_OR,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, OR());
}

TEST_F(BinaryOperationIntegrationTest, Xor_Vector_Vector_SI32_SI16_SI64) {
  using TypeOut = int32_t;
  using TypeLhs = int16_t;
  using TypeRhs = int64_t;

  using XOR = cudf::library::operation::BitwiseXor<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::BITWISE_XOR,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, XOR());
}

TEST_F(BinaryOperationIntegrationTest,
       Logical_And_Vector_Vector_SI16_FP64_SI8) {
  using TypeOut = int16_t;
  using TypeLhs = double;
  using TypeRhs = int8_t;

  using AND = cudf::library::operation::LogicalAnd<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::LOGICAL_AND,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, AND());
}

TEST_F(BinaryOperationIntegrationTest, Logical_Or_Vector_Vector_B8_SI16_SI64) {
  using TypeOut = cudf::experimental::bool8;
  using TypeLhs = int16_t;
  using TypeRhs = int64_t;

  using OR = cudf::library::operation::LogicalOr<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100);
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::LOGICAL_OR,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, OR());
}

TEST_F(BinaryOperationIntegrationTest, Less_Scalar_Vector_B8_TSS_TSS) {
  using TypeOut = cudf::experimental::bool8;
  using TypeLhs = cudf::timestamp_s;
  using TypeRhs = cudf::timestamp_s;

  using LESS = cudf::library::operation::Less<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  auto rhs = make_random_wrapped_column<TypeRhs>(10);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::LESS,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LESS());
}

TEST_F(BinaryOperationIntegrationTest, Greater_Scalar_Vector_B8_TSMS_TSS) {
  using TypeOut = cudf::experimental::bool8;
  using TypeLhs = cudf::timestamp_ms;
  using TypeRhs = cudf::timestamp_s;

  using GREATER = cudf::library::operation::Greater<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::GREATER,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, GREATER());
}

TEST_F(BinaryOperationIntegrationTest, Less_Vector_Vector_B8_TSS_TSS) {
  using TypeOut = cudf::experimental::bool8;
  using TypeLhs = cudf::timestamp_s;
  using TypeRhs = cudf::timestamp_s;

  using LESS = cudf::library::operation::Less<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(10);
  auto rhs = make_random_wrapped_column<TypeRhs>(10);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::LESS,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LESS());
}

TEST_F(BinaryOperationIntegrationTest, Greater_Vector_Vector_B8_TSMS_TSS) {
  using TypeOut = cudf::experimental::bool8;
  using TypeLhs = cudf::timestamp_ms;
  using TypeRhs = cudf::timestamp_s;

  using GREATER = cudf::library::operation::Greater<TypeOut, TypeLhs, TypeRhs>;

  auto itr = cudf::test::make_counting_transform_iterator(
      0, [this](auto row) { return this->generate() * 1000; });

  auto lhs = cudf::test::fixed_width_column_wrapper<TypeLhs>(
      itr, itr + 100, make_validity_iter());

  auto rhs = make_random_wrapped_column<TypeRhs>(100);
  auto out = cudf::experimental::binary_operation(
      lhs, rhs, cudf::experimental::binary_operator::GREATER,
      data_type(experimental::type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, GREATER());
}

}  // namespace binop
}  // namespace test
}  // namespace cudf
