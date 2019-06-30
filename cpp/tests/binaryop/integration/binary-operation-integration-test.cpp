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
#include <tests/utilities/cudf_test_fixtures.h>
#include <cudf/binaryop.hpp>

namespace cudf {
namespace test {
namespace binop {

struct BinaryOperationIntegrationTest : public GdfTest {};


TEST_F(BinaryOperationIntegrationTest, Add_Scalar_Vector_SI32_FP32_SI64) {
    using ADD = cudf::library::operation::Add<int32_t, float, int64_t>;

    auto lhs = cudf::test::scalar_wrapper<float>{100};
    auto rhs = cudf::test::column_wrapper<int64_t>(100000, 
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto out = cudf::test::column_wrapper<int32_t>(rhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_ADD));

    ASSERT_BINOP(out, lhs, rhs, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Sub_Scalar_Vector_SI32_FP32_SI64) {
    using SUB = cudf::library::operation::Sub<int32_t, float, int64_t>;

    auto lhs = cudf::test::scalar_wrapper<float>{10000};
    auto rhs = cudf::test::column_wrapper<int64_t>(100000, 
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto out = cudf::test::column_wrapper<int32_t>(rhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_SUB));

    ASSERT_BINOP(out, lhs, rhs, SUB());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Scalar_SI08_SI16_SI32) {
    using ADD = cudf::library::operation::Add<int8_t, int16_t, int32_t>;

    auto lhs = cudf::test::column_wrapper<int16_t>(100,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 6 > 0);});
    auto rhs = cudf::test::scalar_wrapper<int32_t>(100);
    auto out = cudf::test::column_wrapper<int8_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_ADD));

    ASSERT_BINOP(out, lhs, rhs, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Vector_SI32_FP64_SI08) {
    using ADD = cudf::library::operation::Add<int32_t, double, int8_t>;

    auto lhs = cudf::test::column_wrapper<double>(100,
        [](gdf_size_type row) {return row * 2.0;},
        [](gdf_size_type row) {return (row % 3 > 0);});
    auto rhs = cudf::test::column_wrapper<int8_t>(100,
        [](gdf_size_type row) {return row * 1;},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto out = cudf::test::column_wrapper<int32_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_ADD));

    ASSERT_BINOP(out, lhs, rhs, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Sub_Vector_Vector_SI64) {
    using SUB = cudf::library::operation::Sub<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>(50000,
        [](gdf_size_type row) {return 100000 + row * 2;},
        [](gdf_size_type row) {return (row % 4 == 0);});
    auto rhs = cudf::test::column_wrapper<int64_t>(50000,
        [](gdf_size_type row) {return 50000 + row;},
        [](gdf_size_type row) {return (row % 3 > 0);});
    auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_SUB));

    ASSERT_BINOP(out, lhs, rhs, SUB());
}


TEST_F(BinaryOperationIntegrationTest, Mul_Vector_Vector_SI64) {
    using MUL = cudf::library::operation::Mul<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>(50000,
        [](gdf_size_type row) {return 100000 + row * 2;},
        [](gdf_size_type row) {return (row % 3 > 0);});
    auto rhs = cudf::test::column_wrapper<int64_t>(50000,
        [](gdf_size_type row) {return 50000 + row;},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_MUL));

    ASSERT_BINOP(out, lhs, rhs, MUL());
}


TEST_F(BinaryOperationIntegrationTest, Div_Vector_Vector_SI64) {
    using DIV = cudf::library::operation::Div<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>(50000,
        [](gdf_size_type row) {return 100000 + row * 2;},
        [](gdf_size_type row) {return (row % 6 > 0);});
    auto rhs = cudf::test::column_wrapper<int64_t>(50000,
        [](gdf_size_type row) {return 50000 + row;},
        [](gdf_size_type row) {return (row % 8 > 0);});
    auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_DIV));

    ASSERT_BINOP(out, lhs, rhs, DIV());
}


TEST_F(BinaryOperationIntegrationTest, TrueDiv_Vector_Vector_SI64) {
    using TRUEDIV = cudf::library::operation::TrueDiv<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>(50000,
        [](gdf_size_type row) {return 100000 + row * 2;},
        [](gdf_size_type row) {return (row % 3 == 0);});
    auto rhs = cudf::test::column_wrapper<int64_t>(50000,
        [](gdf_size_type row) {return 50000 + row;},
        [](gdf_size_type row) {return (row % 4 == 0);});
    auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_TRUE_DIV));

    ASSERT_BINOP(out, lhs, rhs, TRUEDIV());
}


TEST_F(BinaryOperationIntegrationTest, FloorDiv_Vector_Vector_SI64) {
    using FLOORDIV = cudf::library::operation::FloorDiv<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>(50000,
        [](gdf_size_type row) {return 100000 + row * 2;},
        [](gdf_size_type row) {return (row % 6 > 0);});
    auto rhs = cudf::test::column_wrapper<int64_t>(50000,
        [](gdf_size_type row) {return 50000 + row;},
        [](gdf_size_type row) {return (row % 8 > 0);});
    auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_FLOOR_DIV));

    ASSERT_BINOP(out, lhs, rhs, FLOORDIV());
}


TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_SI64) {
    using MOD = cudf::library::operation::Mod<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>(50,
        [](gdf_size_type row) {return 120 + row * 2;},
        [](gdf_size_type row) {return (row % 3 > 0);});
    auto rhs = cudf::test::column_wrapper<int64_t>(50,
        [](gdf_size_type row) {return 50 + row;},
        [](gdf_size_type row) {return (row % 5 > 0);});
    auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_MOD));

    ASSERT_BINOP(out, lhs, rhs, MOD());
}


TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP32) {
    using MOD = cudf::library::operation::Mod<float, float, float>;

    auto lhs = cudf::test::column_wrapper<float>(50,
        [](gdf_size_type row) {return 120 + row * 2;},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto rhs = cudf::test::column_wrapper<float>(50,
        [](gdf_size_type row) {return 50 + row;},
        [](gdf_size_type row) {return (row % 6 > 0);});
    auto out = cudf::test::column_wrapper<float>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_MOD));

    ASSERT_BINOP(out, lhs, rhs, MOD());
}


TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP64) {
    using MOD = cudf::library::operation::Mod<double, double, double>;

    auto lhs = cudf::test::column_wrapper<double>(50,
        [](gdf_size_type row) {return 120 + row * 2;},
        [](gdf_size_type row) {return (row % 3 == 0);});
    auto rhs = cudf::test::column_wrapper<double>(50,
        [](gdf_size_type row) {return 50 + row;},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto out = cudf::test::column_wrapper<double>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_MOD));

    ASSERT_BINOP(out, lhs, rhs, MOD());
}


TEST_F(BinaryOperationIntegrationTest, Pow_Vector_Vector_SI64) {
    using POW = cudf::library::operation::Pow<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>(500,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 6 > 0);});
    auto rhs = cudf::test::column_wrapper<int64_t>(500,
        [](gdf_size_type row) {return 2;},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_POW));

    ASSERT_BINOP(out, lhs, rhs, POW());
}


TEST_F(BinaryOperationIntegrationTest, And_Vector_Vector_SI16_SI64_SI32) {
    using AND = cudf::library::operation::BitwiseAnd<int16_t, int64_t, int32_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>(500,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 6 > 0);});
    auto rhs = cudf::test::column_wrapper<int64_t>(500,
        [](gdf_size_type row) {return 2;},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto out = cudf::test::column_wrapper<int32_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_BITWISE_AND));

    ASSERT_BINOP(out, lhs, rhs, AND());
}


TEST_F(BinaryOperationIntegrationTest, Or_Vector_Vector_SI64_SI16_SI32) {
    using OR = cudf::library::operation::BitwiseOr<int64_t, int16_t, int32_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>(500,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 6 > 0);});
    auto rhs = cudf::test::column_wrapper<int16_t>(500,
        [](gdf_size_type row) {return 2;},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto out = cudf::test::column_wrapper<int32_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_BITWISE_OR));

    ASSERT_BINOP(out, lhs, rhs, OR());
}


TEST_F(BinaryOperationIntegrationTest, Xor_Vector_Vector_SI32_SI16_SI64) {
    using XOR = cudf::library::operation::BitwiseXor<int32_t, int16_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int32_t>(500,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 6 > 0);});
    auto rhs = cudf::test::column_wrapper<int16_t>(500,
        [](gdf_size_type row) {return 2;},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_BITWISE_XOR));

    ASSERT_BINOP(out, lhs, rhs, XOR());
}


TEST_F(BinaryOperationIntegrationTest, Logical_And_Vector_Vector_SI16_FP64_SI8) {
    using AND = cudf::library::operation::LogicalAnd<int16_t, double, int8_t>;

    auto lhs = cudf::test::column_wrapper<double>(500,
        [](gdf_size_type row) {return (row % 5);},
        [](gdf_size_type row) {return (row % 6 > 0);});
    auto rhs = cudf::test::column_wrapper<int8_t>(500,
        [](gdf_size_type row) {return (row % 3 > 0);},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto out = cudf::test::column_wrapper<int16_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_LOGICAL_AND));

    ASSERT_BINOP(out, lhs, rhs, AND());
}

TEST_F(BinaryOperationIntegrationTest, Logical_Or_Vector_Vector_B8_SI16_FP32) {
    using OR = cudf::library::operation::LogicalOr<cudf::bool8, int16_t, float>;

    auto lhs = cudf::test::column_wrapper<int16_t>(500,
        [](gdf_size_type row) {return (row % 5);},
        [](gdf_size_type row) {return (row % 6 > 0);});
    auto rhs = cudf::test::column_wrapper<float>(500,
        [](gdf_size_type row) {return (row % 3 > 0);},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto out = cudf::test::column_wrapper<int8_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_LOGICAL_OR));

    ASSERT_BINOP(out, lhs, rhs, OR());
}

TEST_F(BinaryOperationIntegrationTest, CAdd_Vector_Vector_FP32_FP32_FP32) {

// c = a*a*a + b
const char* ptx =
R"***(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-26218862
// Cuda compilation tools, release 10.1, V10.1.168
// Based on LLVM 3.4svn
//

.version 6.4
.target sm_70
.address_size 64

	// .globl	_ZN8__main__7add$241Eff
.common .global .align 8 .u64 _ZN08NumbaEnv8__main__7add$241Eff;
.common .global .align 8 .u64 _ZN08NumbaEnv5numba7targets7numbers13int_power$242Efx;

.visible .func  (.param .b32 func_retval0) _ZN8__main__7add$241Eff(
	.param .b64 _ZN8__main__7add$241Eff_param_0,
	.param .b32 _ZN8__main__7add$241Eff_param_1,
	.param .b32 _ZN8__main__7add$241Eff_param_2
)
{
	.reg .f32 	%f<5>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<2>;


	ld.param.u64 	%rd1, [_ZN8__main__7add$241Eff_param_0];
	ld.param.f32 	%f1, [_ZN8__main__7add$241Eff_param_1];
	ld.param.f32 	%f2, [_ZN8__main__7add$241Eff_param_2];
	mul.f32 	%f3, %f1, %f1;
	fma.rn.f32 	%f4, %f3, %f1, %f2;
	st.f32 	[%rd1], %f4;
	mov.u32 	%r1, 0;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}
)***";

    auto CADD = [](float a, float b) {return a*a*a + b;};

    auto lhs = cudf::test::column_wrapper<float>(500,
        [](gdf_size_type row) {return (row % 3 > 0);},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto rhs = cudf::test::column_wrapper<float>(500,
        [](gdf_size_type row) {return (row % 3 > 0);},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto out = cudf::test::column_wrapper<float>(rhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), ptx));

    ASSERT_BINOP(out, lhs, rhs, CADD);
}

TEST_F(BinaryOperationIntegrationTest, CAdd_Vector_Vector_INT32_INT32_INT32) {

  using dtype = int;

// c = a*a*a + b
const char* ptx =
R"***(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-26218862
// Cuda compilation tools, release 10.1, V10.1.168
// Based on LLVM 3.4svn
//

.version 6.4
.target sm_70
.address_size 64

	// .globl	_ZN8__main__7add$241Eii
.common .global .align 8 .u64 _ZN08NumbaEnv8__main__7add$241Eii;
.common .global .align 8 .u64 _ZN08NumbaEnv5numba7targets7numbers14int_power_impl12$3clocals$3e13int_power$242Exx;

.visible .func  (.param .b32 func_retval0) _ZN8__main__7add$241Eii(
	.param .b64 _ZN8__main__7add$241Eii_param_0,
	.param .b32 _ZN8__main__7add$241Eii_param_1,
	.param .b32 _ZN8__main__7add$241Eii_param_2
)
{
	.reg .b32 	%r<3>;
	.reg .b64 	%rd<7>;


	ld.param.u64 	%rd1, [_ZN8__main__7add$241Eii_param_0];
	ld.param.u32 	%r1, [_ZN8__main__7add$241Eii_param_1];
	cvt.s64.s32	%rd2, %r1;
	mul.wide.s32 	%rd3, %r1, %r1;
	mul.lo.s64 	%rd4, %rd3, %rd2;
	ld.param.s32 	%rd5, [_ZN8__main__7add$241Eii_param_2];
	add.s64 	%rd6, %rd4, %rd5;
	st.u64 	[%rd1], %rd6;
	mov.u32 	%r2, 0;
	st.param.b32	[func_retval0+0], %r2;
	ret;
}
)***";

    auto CADD = [](dtype a,  dtype b) {return a*a*a + b;};

    auto lhs = cudf::test::column_wrapper<dtype>(500,
        [](gdf_size_type row) {return (row % 3 > 0);},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto rhs = cudf::test::column_wrapper<dtype>(500,
        [](gdf_size_type row) {return (row % 3 > 0);},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto out = cudf::test::column_wrapper<dtype>(rhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), ptx));

    ASSERT_BINOP(out, lhs, rhs, CADD);
}


} // namespace binop
} // namespace test
} // namespace cudf
