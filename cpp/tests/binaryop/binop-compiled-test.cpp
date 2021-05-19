/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/binaryop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <tests/binaryop/assert-binops.h>
#include <tests/binaryop/binop-fixture.hpp>
#include "cudf/utilities/error.hpp"

#include <type_traits>

namespace cudf {
namespace test {
namespace binop {

// combinations to test
//     n  t   d
// n n.n n.t n.d
// t t.n t.t t.d
// d d.n d.t d.d

constexpr size_type col_size = 10000;
template <typename T>
struct BinaryOperationCompiledTest : public BinaryOperationTest {
  using TypeOut = cudf::test::GetType<T, 0>;
  using TypeLhs = cudf::test::GetType<T, 1>;
  using TypeRhs = cudf::test::GetType<T, 2>;

  template <typename T1>
  auto make_random_wrapped_column(size_type size)
  {
    return BinaryOperationTest::make_random_wrapped_column<T1>();
  }
};

// using OutTypes = cudf::test::Types<bool, double, timestamp_D, timestamp_ns, duration_ms,
// duration_us>; using LhsTypes = cudf::test::Types<int32_t, float, timestamp_ms, timestamp_us,
// duration_D, duration_ns>; using RhsTypes = cudf::test::Types<uint8_t, , timestamp_ms,
// timestamp_ns, duration_D, duration_us>;
// using types = cudf::test::CrossProduct<OutTypes, LhsTypes, RhsTypes>;
// using types = cudf::test::CrossProduct<IntegralTypes, IntegralTypes, IntegralTypes>;

// ADD
//     n      t     d
// n n + n
// t      	     	t + d
// d      	d + t	d + d

using Add_types = cudf::test::Types<cudf::test::Types<bool, bool, float>,
                                    cudf::test::Types<int16_t, double, uint8_t>,
                                    cudf::test::Types<timestamp_s, timestamp_s, duration_s>,
                                    cudf::test::Types<timestamp_ns, duration_ms, timestamp_us>,
                                    cudf::test::Types<duration_us, duration_us, duration_D>,
                                    // cudf::test::Types<duration_s, int16_t, int64_t>, //valid
                                    // Extras
                                    cudf::test::Types<duration_D, duration_D, duration_D>,
                                    cudf::test::Types<timestamp_D, timestamp_D, duration_D>,
                                    cudf::test::Types<timestamp_s, timestamp_D, duration_s>,
                                    cudf::test::Types<timestamp_ms, timestamp_ms, duration_s>,
                                    cudf::test::Types<timestamp_ns, timestamp_ms, duration_ns>>;
template <typename T>
struct BinaryOperationCompiledTest_Add : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_Add, Add_types);

TYPED_TEST(BinaryOperationCompiledTest_Add, Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

// SUB
//     n      t     d
// n n - n
// t      	t - t	t - d
// d      	     	d - d

using Sub_types =
  cudf::test::Types<cudf::test::Types<int32_t, bool, float>,  // n - n
                                                              // FIXME why is t-t failing?
                    cudf::test::Types<duration_D, timestamp_D, timestamp_D>,   // t - t
                    cudf::test::Types<timestamp_s, timestamp_D, duration_s>,   // t - d
                    cudf::test::Types<duration_ns, duration_us, duration_s>,   // d - d
                    cudf::test::Types<duration_us, duration_us, duration_s>>;  // d - d
template <typename T>
struct BinaryOperationCompiledTest_Sub : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_Sub, Sub_types);

TYPED_TEST(BinaryOperationCompiledTest_Sub, Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using SUB = cudf::library::operation::Sub<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::SUB, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SUB());
}

// MUL
//     n      t     d
// n n * n	     	n * d
// t
// d d * n
using Mul_types = cudf::test::Types<cudf::test::Types<int32_t, u_int64_t, float>,
                                    cudf::test::Types<duration_s, u_int64_t, duration_s>,
                                    cudf::test::Types<duration_ms, duration_D, int16_t>,
                                    cudf::test::Types<duration_ns, duration_us, uint8_t>>;
template <typename T>
struct BinaryOperationCompiledTest_Mul : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_Mul, Mul_types);

TYPED_TEST(BinaryOperationCompiledTest_Mul, Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using MUL = cudf::library::operation::Mul<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::MUL, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MUL());
}

// DIV
//     n      t     d
// n n / n
// t
// d d / n	     	d / d
using Div_types = cudf::test::Types<cudf::test::Types<int16_t, u_int64_t, u_int64_t>,
                                    cudf::test::Types<double, int8_t, int64_t>,
                                    cudf::test::Types<duration_ms, duration_s, u_int32_t>,
                                    cudf::test::Types<duration_ns, duration_D, int16_t>,
                                    cudf::test::Types<double, duration_D, duration_ns>,
                                    cudf::test::Types<float, duration_ms, duration_ns>,
                                    cudf::test::Types<u_int64_t, duration_us, duration_ns>>;
template <typename T>
struct BinaryOperationCompiledTest_Div : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_Div, Div_types);

TYPED_TEST(BinaryOperationCompiledTest_Div, Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using DIV = cudf::library::operation::Div<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::DIV, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, DIV());
}

// TRUE-DIV
//     n      t     d
// n n / n
// t
// d
using TrueDiv_types = cudf::test::Types<cudf::test::Types<int16_t, u_int64_t, u_int64_t>,
                                        cudf::test::Types<double, int8_t, int64_t>,
                                        cudf::test::Types<int8_t, bool, u_int32_t>,
                                        cudf::test::Types<u_int64_t, float, int16_t>>;
template <typename T>
struct BinaryOperationCompiledTest_TrueDiv : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_TrueDiv, TrueDiv_types);

TYPED_TEST(BinaryOperationCompiledTest_TrueDiv, Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using TRUEDIV = cudf::library::operation::TrueDiv<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::TRUE_DIV, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, TRUEDIV());
}
// FLOOR_DIV
//     n      t     d
// n n / n
// t
// d
TYPED_TEST(BinaryOperationCompiledTest_TrueDiv, FloorDiv_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using FLOORDIV = cudf::library::operation::FloorDiv<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::FLOOR_DIV, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, FLOORDIV());
}

// MOD
//     n      t     d
// n n % n
// t
// d d % n	     	d % d
using Mod_types = cudf::test::Types<cudf::test::Types<int16_t, u_int64_t, u_int64_t>,
                                    cudf::test::Types<double, int8_t, int64_t>,
                                    cudf::test::Types<duration_ms, duration_s, u_int32_t>,
                                    cudf::test::Types<duration_D, duration_D, int16_t>,
                                    cudf::test::Types<duration_ns, duration_D, int16_t>,
                                    cudf::test::Types<duration_ns, duration_us, duration_ns>>;
template <typename T>
struct BinaryOperationCompiledTest_Mod : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_Mod, Mod_types);

TYPED_TEST(BinaryOperationCompiledTest_Mod, Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using MOD = cudf::library::operation::Mod<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::MOD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, MOD());
}
// TODO PYMOD (same types as MOD)

// POW
//     n      t     d
// n n ^ n
// t
// d

using Pow_types = cudf::test::Types<cudf::test::Types<double, int64_t, int64_t>,
                                    cudf::test::Types<float, float, float>,
                                    cudf::test::Types<int, int32_t, float>,
                                    cudf::test::Types<float, int, int>,
                                    cudf::test::Types<double, int64_t, int32_t>,
                                    cudf::test::Types<double, double, double>,
                                    cudf::test::Types<double, float, double>,
                                    cudf::test::Types<double, int32_t, int64_t>>;

template <typename T>
struct BinaryOperationCompiledTest_Pow : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_Pow, Pow_types);

TYPED_TEST(BinaryOperationCompiledTest_Pow, Pow_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using POW = cudf::library::operation::Pow<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = []() {
    // resulting value can not be represented by the target type => behavior is undefined
    // -2147483648 in host, 2147483647 in device
    if constexpr (std::is_same_v<TypeOut, int>) {
      auto elements =
        cudf::detail::make_counting_transform_iterator(1, [](auto i) { return i % 5; });
      return fixed_width_column_wrapper<TypeLhs>(elements, elements + 100);
    }
    return BinaryOperationTest::make_random_wrapped_column<TypeLhs>(100);
  }();
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(100);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::POW, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, POW(), NearEqualComparator<TypeOut>{2});
}

// LOG_BASE
//     n      t     d
// n log(n, n)
// t
// d
TYPED_TEST(BinaryOperationCompiledTest_Pow, LogBase_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using LOG_BASE = cudf::library::operation::LogBase<TypeOut, TypeLhs, TypeRhs>;

  // Make sure there are no zeros
  auto elements = cudf::detail::make_counting_transform_iterator(
    1, [](auto i) { return sizeof(TypeLhs) > 4 ? std::pow(2, i) : i + 30; });
  fixed_width_column_wrapper<TypeLhs> lhs(elements, elements + 50);

  // Find log to the base 7
  auto rhs_elements = cudf::detail::make_counting_transform_iterator(0, [](auto) { return 7; });
  fixed_width_column_wrapper<TypeRhs> rhs(rhs_elements, rhs_elements + 50);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::LOG_BASE, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LOG_BASE());
}

// ATAN2
//     n      t     d
// n ATan2(n, n)
// t
// d
TYPED_TEST(BinaryOperationCompiledTest_Pow, ATan2_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using ATAN2 = cudf::library::operation::ATan2<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::ATAN2, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ATAN2(), NearEqualComparator<TypeOut>{2});
}

// Bit Operations
//     n      t     d
// n n . n
// t
// d

// i.i, i.u, u.i, u.u -> i
// i.i, i.u, u.i, u.u -> u
using Bit_types = cudf::test::Types<cudf::test::Types<int16_t, int8_t, int16_t>,
                                    cudf::test::Types<int64_t, int32_t, uint16_t>,
                                    cudf::test::Types<int64_t, uint64_t, int64_t>,
                                    cudf::test::Types<int16_t, uint32_t, uint8_t>,
                                    // cudf::test::Types<bool, int8_t, uint8_t>, // valid
                                    cudf::test::Types<uint16_t, int8_t, int16_t>,
                                    cudf::test::Types<uint64_t, int32_t, uint16_t>,
                                    cudf::test::Types<uint64_t, uint64_t, int64_t>,
                                    cudf::test::Types<uint16_t, uint8_t, uint32_t>>;
template <typename T>
struct BinaryOperationCompiledTest_Bit : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_Bit, Bit_types);

TYPED_TEST(BinaryOperationCompiledTest_Bit, BitwiseAnd_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using BITWISE_AND = cudf::library::operation::BitwiseAnd<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::BITWISE_AND, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, BITWISE_AND());
}

TYPED_TEST(BinaryOperationCompiledTest_Bit, BitwiseOr_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using BITWISE_OR = cudf::library::operation::BitwiseOr<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::BITWISE_OR, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, BITWISE_OR());
}

TYPED_TEST(BinaryOperationCompiledTest_Bit, BitwiseXor_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using BITWISE_XOR = cudf::library::operation::BitwiseXor<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::BITWISE_XOR, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, BITWISE_XOR());
}

TYPED_TEST(BinaryOperationCompiledTest_Bit, ShiftLeft_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using SHIFT_LEFT = cudf::library::operation::ShiftLeft<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::SHIFT_LEFT, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SHIFT_LEFT());
}

TYPED_TEST(BinaryOperationCompiledTest_Bit, ShiftRight_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using SHIFT_RIGHT = cudf::library::operation::ShiftRight<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::SHIFT_RIGHT, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SHIFT_RIGHT());
}

TYPED_TEST(BinaryOperationCompiledTest_Bit, ShiftRightUnsigned_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using SHIFT_RIGHT_UNSIGNED =
    cudf::library::operation::ShiftRightUnsigned<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::SHIFT_RIGHT_UNSIGNED, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, SHIFT_RIGHT_UNSIGNED());
}

// Logical Operations
//     n      t     d
// n n . n
// t
// d
using Logical_types = cudf::test::Types<cudf::test::Types<int16_t, int8_t, int16_t>,
                                        cudf::test::Types<int64_t, int32_t, uint16_t>,
                                        cudf::test::Types<int64_t, uint64_t, double>,
                                        cudf::test::Types<uint16_t, int8_t, int16_t>,
                                        cudf::test::Types<float, int32_t, uint16_t>,
                                        cudf::test::Types<bool, uint64_t, int64_t>,
                                        cudf::test::Types<uint16_t, uint8_t, uint32_t>,
                                        cudf::test::Types<duration_ns, uint64_t, int64_t>>;
template <typename T>
struct BinaryOperationCompiledTest_Logical : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_Logical, Logical_types);

TYPED_TEST(BinaryOperationCompiledTest_Logical, LogicalAnd_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using LOGICAL_AND = cudf::library::operation::LogicalAnd<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::LOGICAL_AND, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LOGICAL_AND());
}

TYPED_TEST(BinaryOperationCompiledTest_Logical, LogicalOr_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using LOGICAL_OR = cudf::library::operation::LogicalOr<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(col_size);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(col_size);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::LOGICAL_OR, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LOGICAL_OR());
}

}  // namespace binop
}  // namespace test
}  // namespace cudf
