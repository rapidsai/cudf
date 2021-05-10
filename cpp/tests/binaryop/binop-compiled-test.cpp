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

// using MyTypes = cudf::test::Types<int8_t, int32_t, uint64_t>;
// using types = cudf::test::CrossProduct<MyTypes, MyTypes, MyTypes>;
using types = cudf::test::
  CrossProduct<cudf::test::IntegralTypes, cudf::test::IntegralTypes, cudf::test::IntegralTypes>;

TYPED_TEST_CASE(BinaryOperationCompiledTest, types);

TYPED_TEST(BinaryOperationCompiledTest, Add_Vector_Vector_Numeric)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(10000);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(10000);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

// Creating new test class because of limitation of template length.
template <typename T>
struct BinaryOperationCompiledTest2 : public BinaryOperationCompiledTest<T> {
};

// duration_D, duration_s, duration_ms, duration_us, duration_ns
using Dtypes = cudf::test::Types<cudf::test::Types<duration_D, duration_D, duration_D>,
                                 cudf::test::Types<timestamp_D, timestamp_D, duration_D>,
                                 cudf::test::Types<timestamp_s, timestamp_D, duration_s>,
                                 cudf::test::Types<timestamp_ms, timestamp_ms, duration_s>,
                                 cudf::test::Types<timestamp_ns, timestamp_ms, duration_ns>>;
// using Dtypes = cudf::test::CrossProduct<cudf::test::DurationTypes, cudf::test::DurationTypes,
// cudf::test::DurationTypes>;

TYPED_TEST_CASE(BinaryOperationCompiledTest2, Dtypes);

TYPED_TEST(BinaryOperationCompiledTest2, Add_Vector_Vector_Numeric)
{
  using TypeOut = typename TestFixture::TypeOut;
  using TypeLhs = typename TestFixture::TypeLhs;
  using TypeRhs = typename TestFixture::TypeRhs;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = BinaryOperationTest::make_random_wrapped_column<TypeLhs>(10);
  auto rhs = BinaryOperationTest::make_random_wrapped_column<TypeRhs>(10);

  auto out = cudf::binary_operation_compiled(
    lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

}  // namespace binop
}  // namespace test
}  // namespace cudf
