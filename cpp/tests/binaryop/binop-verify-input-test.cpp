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

#include <tests/binaryop/binop-fixture.hpp>
#include <cudf/binaryop.hpp>


namespace cudf {
namespace test {
namespace binop {


struct BinopVerifyInputTest : public BinaryOperationTest {};


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOutputVectorType) {
    using TypeLhs = int64_t;
    using TypeRhs = int64_t;

    auto lhs = make_random_wrapped_scalar<TypeLhs>();
    auto rhs = make_random_wrapped_column<TypeRhs>(10);

    EXPECT_THROW(
        cudf::experimental::binary_operation(lhs, rhs,
            cudf::experimental::binary_operator::ADD,
            data_type(type_id::NUM_TYPE_IDS)),
        cudf::logic_error);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorSecondOperandVectorZeroSize) {
    using TypeOut = int64_t;
    using TypeLhs = int64_t;
    using TypeRhs = int64_t;

    auto lhs = make_random_wrapped_column<TypeLhs>(1);
    auto rhs = make_random_wrapped_column<TypeRhs>(10);

    EXPECT_THROW(
        cudf::experimental::binary_operation(lhs, rhs,
            cudf::experimental::binary_operator::ADD,
            data_type(experimental::type_to_id<TypeOut>())),
        cudf::logic_error);
}


} // namespace binop
} // namespace test
} // namespace cudf
