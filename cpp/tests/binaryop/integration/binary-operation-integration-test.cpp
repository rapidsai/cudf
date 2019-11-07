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

#include <tests/binaryop/integration/assert-binops.h>
#include <tests/binaryop/integration/binop-fixture.hpp>

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/binaryop.hpp>

namespace cudf {
namespace test {
namespace binop {


TEST_F(BinaryOperationIntegrationTest, Add_Scalar_Vector_SI32_FP32_SI64) {
    using TypeOut = int32_t;
    using TypeLhs = float;
    using TypeRhs = int64_t;

    using ADD = cudf::library::operation::Add<TypeOut, TypeLhs, TypeRhs>;

    auto lhs = make_random_wrapped_scalar<TypeLhs>();
    auto rhs = make_random_wrapped_column<TypeRhs>(10000);

    auto out = cudf::experimental::binary_operation(lhs, rhs, 
                    cudf::experimental::binary_operator::ADD,
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

    auto out = cudf::experimental::binary_operation(lhs, rhs,
                    cudf::experimental::binary_operator::SUB,
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
    auto out = cudf::experimental::binary_operation(lhs, rhs, 
                    cudf::experimental::binary_operator::ADD,
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
    auto out = cudf::experimental::binary_operation(lhs, rhs, 
                    cudf::experimental::binary_operator::ADD,
                    data_type(experimental::type_to_id<TypeOut>()));

    ASSERT_BINOP<TypeOut, TypeLhs, TypeRhs>(*out, lhs, rhs, ADD());
}


    // TEST_F(BinaryOperationIntegrationTest, Sub_Vector_Vector_SI64) {
    //     using SUB = cudf::library::operation::Sub<int64_t, int64_t, int64_t>;

    //     auto lhs = cudf::test::column_wrapper<int64_t>(50000,
    //         [](cudf::size_type row) {return 100000 + row * 2;},
    //         [](cudf::size_type row) {return (row % 4 == 0);});
    //     auto rhs = cudf::test::column_wrapper<int64_t>(50000,
    //         [](cudf::size_type row) {return 50000 + row;},
    //         [](cudf::size_type row) {return (row % 3 > 0);});
    //     auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    //     CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_SUB));

    //     ASSERT_BINOP(out, lhs, rhs, SUB());
    // }


    // TEST_F(BinaryOperationIntegrationTest, Mul_Vector_Vector_SI64) {
    //     using MUL = cudf::library::operation::Mul<int64_t, int64_t, int64_t>;

    //     auto lhs = cudf::test::column_wrapper<int64_t>(50000,
    //         [](cudf::size_type row) {return 100000 + row * 2;},
    //         [](cudf::size_type row) {return (row % 3 > 0);});
    //     auto rhs = cudf::test::column_wrapper<int64_t>(50000,
    //         [](cudf::size_type row) {return 50000 + row;},
    //         [](cudf::size_type row) {return (row % 4 > 0);});
    //     auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    //     CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_MUL));

    //     ASSERT_BINOP(out, lhs, rhs, MUL());
    // }


    // TEST_F(BinaryOperationIntegrationTest, Div_Vector_Vector_SI64) {
    //     using DIV = cudf::library::operation::Div<int64_t, int64_t, int64_t>;

    //     auto lhs = cudf::test::column_wrapper<int64_t>(50000,
    //         [](cudf::size_type row) {return 100000 + row * 2;},
    //         [](cudf::size_type row) {return (row % 6 > 0);});
    //     auto rhs = cudf::test::column_wrapper<int64_t>(50000,
    //         [](cudf::size_type row) {return 50000 + row;},
    //         [](cudf::size_type row) {return (row % 8 > 0);});
    //     auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    //     CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_DIV));

    //     ASSERT_BINOP(out, lhs, rhs, DIV());
    // }


    // TEST_F(BinaryOperationIntegrationTest, TrueDiv_Vector_Vector_SI64) {
    //     using TRUEDIV = cudf::library::operation::TrueDiv<int64_t, int64_t, int64_t>;

    //     auto lhs = cudf::test::column_wrapper<int64_t>(50000,
    //         [](cudf::size_type row) {return 100000 + row * 2;},
    //         [](cudf::size_type row) {return (row % 3 == 0);});
    //     auto rhs = cudf::test::column_wrapper<int64_t>(50000,
    //         [](cudf::size_type row) {return 50000 + row;},
    //         [](cudf::size_type row) {return (row % 4 == 0);});
    //     auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    //     CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_TRUE_DIV));

    //     ASSERT_BINOP(out, lhs, rhs, TRUEDIV());
    // }


    // TEST_F(BinaryOperationIntegrationTest, FloorDiv_Vector_Vector_SI64) {
    //     using FLOORDIV = cudf::library::operation::FloorDiv<int64_t, int64_t, int64_t>;

    //     auto lhs = cudf::test::column_wrapper<int64_t>(50000,
    //         [](cudf::size_type row) {return 100000 + row * 2;},
    //         [](cudf::size_type row) {return (row % 6 > 0);});
    //     auto rhs = cudf::test::column_wrapper<int64_t>(50000,
    //         [](cudf::size_type row) {return 50000 + row;},
    //         [](cudf::size_type row) {return (row % 8 > 0);});
    //     auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    //     CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_FLOOR_DIV));

    //     ASSERT_BINOP(out, lhs, rhs, FLOORDIV());
    // }


    // TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_SI64) {
    //     using MOD = cudf::library::operation::Mod<int64_t, int64_t, int64_t>;

    //     auto lhs = cudf::test::column_wrapper<int64_t>(50,
    //         [](cudf::size_type row) {return 120 + row * 2;},
    //         [](cudf::size_type row) {return (row % 3 > 0);});
    //     auto rhs = cudf::test::column_wrapper<int64_t>(50,
    //         [](cudf::size_type row) {return 50 + row;},
    //         [](cudf::size_type row) {return (row % 5 > 0);});
    //     auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    //     CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_MOD));

    //     ASSERT_BINOP(out, lhs, rhs, MOD());
    // }


    // TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP32) {
    //     using MOD = cudf::library::operation::Mod<float, float, float>;

    //     auto lhs = cudf::test::column_wrapper<float>(50,
    //         [](cudf::size_type row) {return 120 + row * 2;},
    //         [](cudf::size_type row) {return (row % 4 > 0);});
    //     auto rhs = cudf::test::column_wrapper<float>(50,
    //         [](cudf::size_type row) {return 50 + row;},
    //         [](cudf::size_type row) {return (row % 6 > 0);});
    //     auto out = cudf::test::column_wrapper<float>(lhs.get()->size, true);

    //     CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_MOD));

    //     ASSERT_BINOP(out, lhs, rhs, MOD());
    // }


    // TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP64) {
    //     using MOD = cudf::library::operation::Mod<double, double, double>;

    //     auto lhs = cudf::test::column_wrapper<double>(50,
    //         [](cudf::size_type row) {return 120 + row * 2;},
    //         [](cudf::size_type row) {return (row % 3 == 0);});
    //     auto rhs = cudf::test::column_wrapper<double>(50,
    //         [](cudf::size_type row) {return 50 + row;},
    //         [](cudf::size_type row) {return (row % 4 > 0);});
    //     auto out = cudf::test::column_wrapper<double>(lhs.get()->size, true);

    //     CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_MOD));

    //     ASSERT_BINOP(out, lhs, rhs, MOD());
    // }


    // TEST_F(BinaryOperationIntegrationTest, Pow_Vector_Vector_SI64) {
    //     using POW = cudf::library::operation::Pow<int64_t, int64_t, int64_t>;

    //     auto lhs = cudf::test::column_wrapper<int64_t>(500,
    //         [](cudf::size_type row) {return row;},
    //         [](cudf::size_type row) {return (row % 6 > 0);});
    //     auto rhs = cudf::test::column_wrapper<int64_t>(500,
    //         [](cudf::size_type row) {return 2;},
    //         [](cudf::size_type row) {return (row % 4 > 0);});
    //     auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    //     CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_POW));

    //     ASSERT_BINOP(out, lhs, rhs, POW());
    // }


    // TEST_F(BinaryOperationIntegrationTest, And_Vector_Vector_SI16_SI64_SI32) {
    //     using AND = cudf::library::operation::BitwiseAnd<int16_t, int64_t, int32_t>;

    //     auto lhs = cudf::test::column_wrapper<int64_t>(500,
    //         [](cudf::size_type row) {return row;},
    //         [](cudf::size_type row) {return (row % 6 > 0);});
    //     auto rhs = cudf::test::column_wrapper<int64_t>(500,
    //         [](cudf::size_type row) {return 2;},
    //         [](cudf::size_type row) {return (row % 4 > 0);});
    //     auto out = cudf::test::column_wrapper<int32_t>(lhs.get()->size, true);

    //     CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_BITWISE_AND));

    //     ASSERT_BINOP(out, lhs, rhs, AND());
    // }


    // TEST_F(BinaryOperationIntegrationTest, Or_Vector_Vector_SI64_SI16_SI32) {
    //     using OR = cudf::library::operation::BitwiseOr<int64_t, int16_t, int32_t>;

    //     auto lhs = cudf::test::column_wrapper<int64_t>(500,
    //         [](cudf::size_type row) {return row;},
    //         [](cudf::size_type row) {return (row % 6 > 0);});
    //     auto rhs = cudf::test::column_wrapper<int16_t>(500,
    //         [](cudf::size_type row) {return 2;},
    //         [](cudf::size_type row) {return (row % 4 > 0);});
    //     auto out = cudf::test::column_wrapper<int32_t>(lhs.get()->size, true);

    //     CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_BITWISE_OR));

    //     ASSERT_BINOP(out, lhs, rhs, OR());
    // }


    // TEST_F(BinaryOperationIntegrationTest, Xor_Vector_Vector_SI32_SI16_SI64) {
    //     using XOR = cudf::library::operation::BitwiseXor<int32_t, int16_t, int64_t>;

    //     auto lhs = cudf::test::column_wrapper<int32_t>(500,
    //         [](cudf::size_type row) {return row;},
    //         [](cudf::size_type row) {return (row % 6 > 0);});
    //     auto rhs = cudf::test::column_wrapper<int16_t>(500,
    //         [](cudf::size_type row) {return 2;},
    //         [](cudf::size_type row) {return (row % 4 > 0);});
    //     auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    //     CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_BITWISE_XOR));

    //     ASSERT_BINOP(out, lhs, rhs, XOR());
    // }


    // TEST_F(BinaryOperationIntegrationTest, Logical_And_Vector_Vector_SI16_FP64_SI8) {
    //     using AND = cudf::library::operation::LogicalAnd<int16_t, double, int8_t>;

    //     auto lhs = cudf::test::column_wrapper<double>(500,
    //         [](cudf::size_type row) {return (row % 5);},
    //         [](cudf::size_type row) {return (row % 6 > 0);});
    //     auto rhs = cudf::test::column_wrapper<int8_t>(500,
    //         [](cudf::size_type row) {return (row % 3 > 0);},
    //         [](cudf::size_type row) {return (row % 4 > 0);});
    //     auto out = cudf::test::column_wrapper<int16_t>(lhs.get()->size, true);

    //     CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_LOGICAL_AND));

    //     ASSERT_BINOP(out, lhs, rhs, AND());
    // }

    // TEST_F(BinaryOperationIntegrationTest, Logical_Or_Vector_Vector_B8_SI16_FP32) {
    //     using OR = cudf::library::operation::LogicalOr<cudf::bool8, int16_t, float>;

    //     auto lhs = cudf::test::column_wrapper<int16_t>(500,
    //         [](cudf::size_type row) {return (row % 5);},
    //         [](cudf::size_type row) {return (row % 6 > 0);});
    //     auto rhs = cudf::test::column_wrapper<float>(500,
    //         [](cudf::size_type row) {return (row % 3 > 0);},
    //         [](cudf::size_type row) {return (row % 4 > 0);});
    //     auto out = cudf::test::column_wrapper<int8_t>(lhs.get()->size, true);

    //     CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_LOGICAL_OR));

    //     ASSERT_BINOP(out, lhs, rhs, OR());
    // }

} // namespace binop
} // namespace test
} // namespace cudf
