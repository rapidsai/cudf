/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

namespace cudf::test::binop {

template <typename T>
auto lhs_random_column(size_type size)
{
  return BinaryOperationTest::make_random_wrapped_column<T>(size);
}

template <>
auto lhs_random_column<std::string>(size_type size)
{
  return cudf::test::strings_column_wrapper({"eee", "bb", "<null>", "", "aa", "bbb", "ééé"},
                                            {1, 1, 0, 1, 1, 1, 1});
}
template <typename T>
auto rhs_random_column(size_type size)
{
  return BinaryOperationTest::make_random_wrapped_column<T>(size);
}
template <>
auto rhs_random_column<std::string>(size_type size)
{
  return cudf::test::strings_column_wrapper({"ééé", "bbb", "aa", "", "<null>", "bb", "eee"},
                                            {1, 1, 1, 1, 0, 1, 1});
}

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

  template <template <typename... Ty> class FunctorOP>
  void test(cudf::binary_operator op)
  {
    using OPERATOR = FunctorOP<TypeOut, TypeLhs, TypeRhs>;

    auto lhs = lhs_random_column<TypeLhs>(col_size);
    auto rhs = rhs_random_column<TypeRhs>(col_size);

    auto out = cudf::experimental::binary_operation(lhs, rhs, op, data_type(type_to_id<TypeOut>()));
    ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, OPERATOR());

    auto s_lhs = this->template make_random_wrapped_scalar<TypeLhs>();
    auto s_rhs = this->template make_random_wrapped_scalar<TypeRhs>();

    out = cudf::experimental::binary_operation(lhs, s_rhs, op, data_type(type_to_id<TypeOut>()));
    ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, s_rhs, OPERATOR());
    out = cudf::experimental::binary_operation(s_lhs, rhs, op, data_type(type_to_id<TypeOut>()));
    ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, s_lhs, rhs, OPERATOR());
  }
};

// ADD
//     n      t     d
// n n + n
// t      	     	t + d
// d      	d + t	d + d

using Add_types =
  cudf::test::Types<cudf::test::Types<bool, bool, float>,
                    cudf::test::Types<int16_t, double, uint8_t>,
                    cudf::test::Types<timestamp_s, timestamp_s, duration_s>,
                    cudf::test::Types<timestamp_ns, duration_ms, timestamp_us>,
                    cudf::test::Types<duration_us, duration_us, duration_D>,
                    // cudf::test::Types<duration_s, int16_t, int64_t>, //valid
                    cudf::test::Types<numeric::decimal32, numeric::decimal32, numeric::decimal32>,
                    cudf::test::Types<int, numeric::decimal32, numeric::decimal32>,
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
  this->template test<cudf::library::operation::Add>(cudf::binary_operator::ADD);
}

// SUB
//     n      t     d
// n n - n
// t      	t - t	t - d
// d      	     	d - d

using Sub_types =
  cudf::test::Types<cudf::test::Types<int32_t, bool, float>,                  // n - n
                    cudf::test::Types<duration_D, timestamp_D, timestamp_D>,  // t - t
                    cudf::test::Types<timestamp_s, timestamp_D, duration_s>,  // t - d
                    cudf::test::Types<duration_ns, duration_us, duration_s>,  // d - d
                    cudf::test::Types<duration_us, duration_us, duration_s>,  // d - d
                    cudf::test::Types<numeric::decimal32, numeric::decimal32, numeric::decimal32>,
                    cudf::test::Types<int, numeric::decimal32, numeric::decimal32>>;
template <typename T>
struct BinaryOperationCompiledTest_Sub : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_Sub, Sub_types);

TYPED_TEST(BinaryOperationCompiledTest_Sub, Vector_Vector)
{
  this->template test<cudf::library::operation::Sub>(cudf::binary_operator::SUB);
}

// MUL
//     n      t     d
// n n * n	     	n * d
// t
// d d * n
using Mul_types =
  cudf::test::Types<cudf::test::Types<int32_t, u_int64_t, float>,
                    cudf::test::Types<duration_s, u_int64_t, duration_s>,
                    cudf::test::Types<duration_ms, duration_D, int16_t>,
                    cudf::test::Types<duration_ns, duration_us, uint8_t>,
                    cudf::test::Types<numeric::decimal32, numeric::decimal32, numeric::decimal32>,
                    cudf::test::Types<int, numeric::decimal32, numeric::decimal32>,
                    cudf::test::Types<numeric::decimal32, int, int>>;
template <typename T>
struct BinaryOperationCompiledTest_Mul : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_Mul, Mul_types);

TYPED_TEST(BinaryOperationCompiledTest_Mul, Vector_Vector)
{
  this->template test<cudf::library::operation::Mul>(cudf::binary_operator::MUL);
}

// DIV
//     n      t     d
// n n / n
// t
// d d / n	     	d / d
using Div_types =
  cudf::test::Types<cudf::test::Types<int16_t, u_int64_t, u_int64_t>,
                    cudf::test::Types<double, int8_t, int64_t>,
                    cudf::test::Types<duration_ms, duration_s, u_int32_t>,
                    cudf::test::Types<duration_ns, duration_D, int16_t>,
                    cudf::test::Types<double, duration_D, duration_ns>,
                    cudf::test::Types<float, duration_ms, duration_ns>,
                    cudf::test::Types<u_int64_t, duration_us, duration_ns>,
                    cudf::test::Types<numeric::decimal32, numeric::decimal32, numeric::decimal32>,
                    cudf::test::Types<int, numeric::decimal32, numeric::decimal32>>;
template <typename T>
struct BinaryOperationCompiledTest_Div : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_Div, Div_types);

TYPED_TEST(BinaryOperationCompiledTest_Div, Vector_Vector)
{
  this->template test<cudf::library::operation::Div>(cudf::binary_operator::DIV);
}

// TRUE-DIV
//     n      t     d
// n n / n
// t
// d
using TrueDiv_types =
  cudf::test::Types<cudf::test::Types<int16_t, u_int64_t, u_int64_t>,
                    cudf::test::Types<double, int8_t, int64_t>,
                    cudf::test::Types<int8_t, bool, u_int32_t>,
                    cudf::test::Types<u_int64_t, float, int16_t>,
                    cudf::test::Types<numeric::decimal32, numeric::decimal32, numeric::decimal32>,
                    cudf::test::Types<int, numeric::decimal32, numeric::decimal32>>;
template <typename T>
struct BinaryOperationCompiledTest_TrueDiv : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_TrueDiv, TrueDiv_types);

TYPED_TEST(BinaryOperationCompiledTest_TrueDiv, Vector_Vector)
{
  this->template test<cudf::library::operation::TrueDiv>(cudf::binary_operator::TRUE_DIV);
}
// FLOOR_DIV
//     n      t     d
// n n / n
// t
// d
TYPED_TEST(BinaryOperationCompiledTest_TrueDiv, FloorDiv_Vector_Vector)
{
  this->template test<cudf::library::operation::FloorDiv>(cudf::binary_operator::FLOOR_DIV);
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
  this->template test<cudf::library::operation::Mod>(cudf::binary_operator::MOD);
}

// PYMOD
//     n      t     d
// n n % n
// t
// d      	     	d % d
using PyMod_types = cudf::test::Types<cudf::test::Types<int16_t, u_int64_t, u_int64_t>,
                                      cudf::test::Types<double, int8_t, int64_t>,
                                      cudf::test::Types<double, double, double>,
                                      cudf::test::Types<duration_ns, duration_us, duration_ns>>;
template <typename T>
struct BinaryOperationCompiledTest_PyMod : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_PyMod, PyMod_types);
TYPED_TEST(BinaryOperationCompiledTest_PyMod, Vector_Vector)
{
  this->template test<cudf::library::operation::PyMod>(cudf::binary_operator::PYMOD);
}

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
struct BinaryOperationCompiledTest_FloatOps : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_FloatOps, Pow_types);

TYPED_TEST(BinaryOperationCompiledTest_FloatOps, Pow_Vector_Vector)
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
    return lhs_random_column<TypeLhs>(100);
  }();
  auto rhs = rhs_random_column<TypeRhs>(100);

  auto out = cudf::experimental::binary_operation(
    lhs, rhs, cudf::binary_operator::POW, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, POW(), NearEqualComparator<TypeOut>{2});
}

// LOG_BASE
//     n      t     d
// n log(n, n)
// t
// d
TYPED_TEST(BinaryOperationCompiledTest_FloatOps, LogBase_Vector_Vector)
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

  auto out = cudf::experimental::binary_operation(
    lhs, rhs, cudf::binary_operator::LOG_BASE, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, LOG_BASE());
}

// ATAN2
//     n      t     d
// n ATan2(n, n)
// t
// d
TYPED_TEST(BinaryOperationCompiledTest_FloatOps, ATan2_Vector_Vector)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using ATAN2 = cudf::library::operation::ATan2<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = lhs_random_column<TypeLhs>(col_size);
  auto rhs = rhs_random_column<TypeRhs>(col_size);

  auto out = cudf::experimental::binary_operation(
    lhs, rhs, cudf::binary_operator::ATAN2, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ATAN2(), NearEqualComparator<TypeOut>{2});
}

TYPED_TEST(BinaryOperationCompiledTest_FloatOps, PMod_Vector_Vector)
{
  this->template test<cudf::library::operation::PMod>(cudf::binary_operator::PMOD);
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
  this->template test<cudf::library::operation::BitwiseAnd>(cudf::binary_operator::BITWISE_AND);
}

TYPED_TEST(BinaryOperationCompiledTest_Bit, BitwiseOr_Vector_Vector)
{
  this->template test<cudf::library::operation::BitwiseOr>(cudf::binary_operator::BITWISE_OR);
}

TYPED_TEST(BinaryOperationCompiledTest_Bit, BitwiseXor_Vector_Vector)
{
  this->template test<cudf::library::operation::BitwiseXor>(cudf::binary_operator::BITWISE_XOR);
}

TYPED_TEST(BinaryOperationCompiledTest_Bit, ShiftLeft_Vector_Vector)
{
  this->template test<cudf::library::operation::ShiftLeft>(cudf::binary_operator::SHIFT_LEFT);
}

TYPED_TEST(BinaryOperationCompiledTest_Bit, ShiftRight_Vector_Vector)
{
  this->template test<cudf::library::operation::ShiftRight>(cudf::binary_operator::SHIFT_RIGHT);
}

TYPED_TEST(BinaryOperationCompiledTest_Bit, ShiftRightUnsigned_Vector_Vector)
{
  this->template test<cudf::library::operation::ShiftRightUnsigned>(
    cudf::binary_operator::SHIFT_RIGHT_UNSIGNED);
}

// Logical Operations
//     n      t     d
// n n . n
// t
// d
using Logical_types = cudf::test::Types<cudf::test::Types<bool, int8_t, int16_t>,
                                        cudf::test::Types<bool, int32_t, uint16_t>,
                                        cudf::test::Types<bool, uint64_t, double>,
                                        cudf::test::Types<bool, int8_t, int16_t>,
                                        cudf::test::Types<bool, float, uint16_t>,
                                        cudf::test::Types<bool, uint64_t, int64_t>,
                                        cudf::test::Types<bool, uint8_t, uint32_t>,
                                        cudf::test::Types<bool, uint64_t, int64_t>>;
template <typename T>
struct BinaryOperationCompiledTest_Logical : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_Logical, Logical_types);

TYPED_TEST(BinaryOperationCompiledTest_Logical, LogicalAnd_Vector_Vector)
{
  this->template test<cudf::library::operation::LogicalAnd>(cudf::binary_operator::LOGICAL_AND);
}

TYPED_TEST(BinaryOperationCompiledTest_Logical, LogicalOr_Vector_Vector)
{
  this->template test<cudf::library::operation::LogicalOr>(cudf::binary_operator::LOGICAL_OR);
}

// Comparison Operations ==, !=, <, >, <=, >=
// n<!=>n, t<!=>t, d<!=>d, s<!=>s, dc<!=>dc
using Comparison_types =
  cudf::test::Types<cudf::test::Types<bool, int8_t, int16_t>,
                    cudf::test::Types<bool, uint32_t, uint16_t>,
                    cudf::test::Types<bool, uint64_t, double>,
                    cudf::test::Types<bool, timestamp_D, timestamp_s>,
                    cudf::test::Types<bool, timestamp_ns, timestamp_us>,
                    cudf::test::Types<bool, duration_ns, duration_ns>,
                    cudf::test::Types<bool, duration_us, duration_s>,
                    cudf::test::Types<bool, std::string, std::string>,
                    cudf::test::Types<bool, numeric::decimal32, numeric::decimal32>>;

template <typename T>
struct BinaryOperationCompiledTest_Comparison : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_Comparison, Comparison_types);

TYPED_TEST(BinaryOperationCompiledTest_Comparison, Equal_Vector_Vector)
{
  this->template test<cudf::library::operation::Equal>(cudf::binary_operator::EQUAL);
}

TYPED_TEST(BinaryOperationCompiledTest_Comparison, NotEqual_Vector_Vector)
{
  this->template test<cudf::library::operation::NotEqual>(cudf::binary_operator::NOT_EQUAL);
}

TYPED_TEST(BinaryOperationCompiledTest_Comparison, Less_Vector_Vector)
{
  this->template test<cudf::library::operation::Less>(cudf::binary_operator::LESS);
}

TYPED_TEST(BinaryOperationCompiledTest_Comparison, Greater_Vector_Vector)
{
  this->template test<cudf::library::operation::Greater>(cudf::binary_operator::GREATER);
}

TYPED_TEST(BinaryOperationCompiledTest_Comparison, LessEqual_Vector_Vector)
{
  this->template test<cudf::library::operation::LessEqual>(cudf::binary_operator::LESS_EQUAL);
}

TYPED_TEST(BinaryOperationCompiledTest_Comparison, GreaterEqual_Vector_Vector)
{
  this->template test<cudf::library::operation::GreaterEqual>(cudf::binary_operator::GREATER_EQUAL);
}

// Null Operations NullMax, NullMin
// Min(n,n) , Min(t,t), Min(d, d), Min(s, s), Min(dc, dc), Min(n,dc), Min(dc, n)
//    n   t   d  s  dc
// n  .             .
// t      .
// d          .
// s             .
// dc .             .
using Null_types =
  cudf::test::Types<cudf::test::Types<int16_t, int8_t, int16_t>,
                    cudf::test::Types<uint16_t, uint32_t, uint16_t>,
                    cudf::test::Types<double, uint64_t, double>,
                    cudf::test::Types<timestamp_s, timestamp_D, timestamp_s>,
                    cudf::test::Types<duration_ns, duration_us, duration_s>,
                    // cudf::test::Types<std::string, std::string, std::string>, // only fixed-width
                    cudf::test::Types<numeric::decimal32, numeric::decimal32, numeric::decimal32>,
                    cudf::test::Types<numeric::decimal32, uint32_t, numeric::decimal32>,
                    cudf::test::Types<int64_t, numeric::decimal64, int64_t>>;

template <typename T>
struct BinaryOperationCompiledTest_NullOps : public BinaryOperationCompiledTest<T> {
};
TYPED_TEST_CASE(BinaryOperationCompiledTest_NullOps, Null_types);

template <typename TypeOut, typename TypeLhs, typename TypeRhs, class OP>
auto NullOp_Result(column_view lhs, column_view rhs)
{
  auto [lhs_data, lhs_mask] = cudf::test::to_host<TypeLhs>(lhs);
  auto [rhs_data, rhs_mask] = cudf::test::to_host<TypeRhs>(rhs);
  std::vector<TypeOut> result(lhs.size());
  std::vector<bool> result_mask;
  std::transform(thrust::make_counting_iterator(0),
                 thrust::make_counting_iterator(lhs.size()),
                 result.begin(),
                 [&lhs_data, &lhs_mask, &rhs_data, &rhs_mask, &result_mask](auto i) -> TypeOut {
                   auto lhs_valid    = lhs_mask.data() and cudf::bit_is_set(lhs_mask.data(), i);
                   auto rhs_valid    = rhs_mask.data() and cudf::bit_is_set(rhs_mask.data(), i);
                   bool output_valid = lhs_valid or rhs_valid;
                   auto result = OP{}(lhs_data[i], rhs_data[i], lhs_valid, rhs_valid, output_valid);
                   result_mask.push_back(output_valid);
                   return result;
                 });
  return cudf::test::fixed_width_column_wrapper<TypeOut>(
    result.cbegin(), result.cend(), result_mask.cbegin());
}

TYPED_TEST(BinaryOperationCompiledTest_NullOps, NullEquals_Vector_Vector)
{
  using TypeOut     = bool;
  using TypeLhs     = typename TestFixture::TypeLhs;
  using TypeRhs     = typename TestFixture::TypeRhs;
  using NULL_EQUALS = cudf::library::operation::NullEquals<TypeOut, TypeLhs, TypeRhs>;

  auto lhs            = lhs_random_column<TypeLhs>(col_size);
  auto rhs            = rhs_random_column<TypeRhs>(col_size);
  auto const expected = NullOp_Result<TypeOut, TypeLhs, TypeRhs, NULL_EQUALS>(lhs, rhs);

  auto const result = cudf::experimental::binary_operation(
    lhs, rhs, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

using BinaryOperationCompiledTest_NullOpsString =
  BinaryOperationCompiledTest_NullOps<cudf::test::Types<std::string, std::string, std::string>>;
TEST_F(BinaryOperationCompiledTest_NullOpsString, NullEquals_Vector_Vector)
{
  using TypeOut     = bool;
  using TypeLhs     = std::string;
  using TypeRhs     = std::string;
  using NULL_EQUALS = cudf::library::operation::NullEquals<TypeOut, TypeLhs, TypeRhs>;

  auto lhs            = lhs_random_column<TypeLhs>(col_size);
  auto rhs            = rhs_random_column<TypeRhs>(col_size);
  auto const expected = NullOp_Result<TypeOut, TypeLhs, TypeRhs, NULL_EQUALS>(lhs, rhs);

  auto const result = cudf::experimental::binary_operation(
    lhs, rhs, cudf::binary_operator::NULL_EQUALS, data_type(type_to_id<TypeOut>()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(BinaryOperationCompiledTest_NullOps, NullMax_Vector_Vector)
{
  using TypeOut  = typename TestFixture::TypeOut;
  using TypeLhs  = typename TestFixture::TypeLhs;
  using TypeRhs  = typename TestFixture::TypeRhs;
  using NULL_MAX = cudf::library::operation::NullMax<TypeOut, TypeLhs, TypeRhs>;

  auto lhs            = lhs_random_column<TypeLhs>(col_size);
  auto rhs            = rhs_random_column<TypeRhs>(col_size);
  auto const expected = NullOp_Result<TypeOut, TypeLhs, TypeRhs, NULL_MAX>(lhs, rhs);

  auto const result = cudf::experimental::binary_operation(
    lhs, rhs, cudf::binary_operator::NULL_MAX, data_type(type_to_id<TypeOut>()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(BinaryOperationCompiledTest_NullOps, NullMin_Vector_Vector)
{
  using TypeOut  = typename TestFixture::TypeOut;
  using TypeLhs  = typename TestFixture::TypeLhs;
  using TypeRhs  = typename TestFixture::TypeRhs;
  using NULL_MIN = cudf::library::operation::NullMin<TypeOut, TypeLhs, TypeRhs>;

  auto lhs            = lhs_random_column<TypeLhs>(col_size);
  auto rhs            = rhs_random_column<TypeRhs>(col_size);
  auto const expected = NullOp_Result<TypeOut, TypeLhs, TypeRhs, NULL_MIN>(lhs, rhs);

  auto const result = cudf::experimental::binary_operation(
    lhs, rhs, cudf::binary_operator::NULL_MIN, data_type(type_to_id<TypeOut>()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

}  // namespace cudf::test::binop
