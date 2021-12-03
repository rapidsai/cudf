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

#include <tests/binaryop/assert-binops.h>
#include <tests/binaryop/binop-fixture.hpp>

#include <tests/binaryop/util/runtime_support.h>

namespace cudf {
namespace test {
namespace binop {
struct BinaryOperationNullTest : public BinaryOperationTest {
  template <typename T>
  auto make_random_wrapped_column(size_type size, mask_state state)
  {
    cudf::test::UniformRandomGenerator<T> rand_gen(1, 10);
    auto data_iter = make_data_iter(rand_gen);

    switch (state) {
      case mask_state::ALL_NULL: {
        auto validity_iter =
          cudf::detail::make_counting_transform_iterator(0, [](auto row) { return false; });
        return cudf::test::fixed_width_column_wrapper<T>(
          data_iter, data_iter + size, validity_iter);
      }
      case mask_state::ALL_VALID: {
        auto validity_iter =
          cudf::detail::make_counting_transform_iterator(0, [](auto row) { return true; });
        return cudf::test::fixed_width_column_wrapper<T>(
          data_iter, data_iter + size, validity_iter);
      }
      case mask_state::UNALLOCATED: {
        return cudf::test::fixed_width_column_wrapper<T>(data_iter, data_iter + size);
      }
      default: CUDF_FAIL("Unknown mask state " + std::to_string(static_cast<int64_t>(state)));
    }
  }

 protected:
  void SetUp() override
  {
    if (!can_do_runtime_jit()) { GTEST_SKIP() << "Skipping tests that require 11.5 runtime"; }
  }
};  // namespace binop

TEST_F(BinaryOperationNullTest, Scalar_Null_Vector_Valid)
{
  using TypeOut = int32_t;
  using TypeLhs = int32_t;
  using TypeRhs = int32_t;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  lhs.set_valid_async(false);
  auto rhs = make_random_wrapped_column<TypeRhs>(100, mask_state::ALL_VALID);

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationNullTest, Scalar_Valid_Vector_NonNullable)
{
  using TypeOut = int32_t;
  using TypeLhs = int32_t;
  using TypeRhs = int32_t;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  auto rhs = make_random_wrapped_column<TypeRhs>(100, mask_state::UNALLOCATED);

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationNullTest, Scalar_Null_Vector_NonNullable)
{
  using TypeOut = int32_t;
  using TypeLhs = int32_t;
  using TypeRhs = int32_t;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  lhs.set_valid_async(false);
  auto rhs = make_random_wrapped_column<TypeRhs>(100, mask_state::UNALLOCATED);

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationNullTest, Vector_Null_Scalar_Valid)
{
  using TypeOut = int32_t;
  using TypeLhs = int32_t;
  using TypeRhs = int32_t;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_scalar<TypeLhs>();
  auto rhs = make_random_wrapped_column<TypeRhs>(100, mask_state::ALL_NULL);

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationNullTest, Vector_Null_Vector_Valid)
{
  using TypeOut = int32_t;
  using TypeLhs = int32_t;
  using TypeRhs = int32_t;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100, mask_state::ALL_NULL);
  auto rhs = make_random_wrapped_column<TypeRhs>(100, mask_state::ALL_VALID);

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationNullTest, Vector_Null_Vector_NonNullable)
{
  using TypeOut = int32_t;
  using TypeLhs = int32_t;
  using TypeRhs = int32_t;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100, mask_state::ALL_NULL);
  auto rhs = make_random_wrapped_column<TypeRhs>(100, mask_state::UNALLOCATED);

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationNullTest, Vector_Valid_Vector_NonNullable)
{
  using TypeOut = int32_t;
  using TypeLhs = int32_t;
  using TypeRhs = int32_t;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100, mask_state::ALL_VALID);
  auto rhs = make_random_wrapped_column<TypeRhs>(100, mask_state::UNALLOCATED);

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

TEST_F(BinaryOperationNullTest, Vector_NonNullable_Vector_NonNullable)
{
  using TypeOut = int32_t;
  using TypeLhs = int32_t;
  using TypeRhs = int32_t;

  using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

  auto lhs = make_random_wrapped_column<TypeLhs>(100, mask_state::UNALLOCATED);
  auto rhs = make_random_wrapped_column<TypeRhs>(100, mask_state::UNALLOCATED);

  auto out =
    cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD, data_type(type_to_id<TypeOut>()));

  ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}

}  // namespace binop
}  // namespace test
}  // namespace cudf
