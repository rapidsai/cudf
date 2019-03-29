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

#include "tests/binary/integration/assert-binops.h"

namespace cudf {
namespace test {
namespace binop {

struct BinaryOperationIntegrationTest : public ::testing::Test {
    BinaryOperationIntegrationTest() {
    }

    virtual ~BinaryOperationIntegrationTest() {
    }

    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};


TEST_F(BinaryOperationIntegrationTest, Add_Scalar_Vector_SI32_FP32_SI64) {
    using ADD = cudf::library::operation::Add<int32_t, float, int64_t>;

    auto lhs = cudf::test::scalar_wrapper<float>{100};
    auto rhs = cudf::test::column_wrapper<int64_t>{100000, 
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto out = cudf::test::column_wrapper<int32_t>{rhs.get()->size, true};

    auto result = gdf_binary_operation_s_v(out.get(), lhs.get(), rhs.get(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Sub_Scalar_Vector_SI32_FP32_SI64) {
    using SUB = cudf::library::operation::Sub<int32_t, float, int64_t>;

    auto lhs = cudf::test::scalar_wrapper<float>{10000};
    auto rhs = cudf::test::column_wrapper<int64_t>{100000, 
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto out = cudf::test::column_wrapper<int32_t>{rhs.get()->size, true};

    auto result = gdf_binary_operation_s_v(out.get(), lhs.get(), rhs.get(), GDF_SUB);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, SUB());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Scalar_SI08_SI16_SI32) {
    using ADD = cudf::library::operation::Add<int8_t, int16_t, int32_t>;

    auto lhs = cudf::test::column_wrapper<int16_t>{100,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 6 > 0);}};
    auto rhs = cudf::test::scalar_wrapper<int32_t>{100};
    auto out = cudf::test::column_wrapper<int8_t>{lhs.get()->size, true};

    auto result = gdf_binary_operation_v_s(out.get(), lhs.get(), rhs.get(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Vector_SI32_FP64_SI08) {
    using ADD = cudf::library::operation::Add<int32_t, double, int8_t>;

    auto lhs = cudf::test::column_wrapper<double>{100,
        [](gdf_size_type row) {return row * 2.0;},
        [](gdf_size_type row) {return (row % 3 > 0);}};
    auto rhs = cudf::test::column_wrapper<int8_t>{100,
        [](gdf_size_type row) {return row * 1;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto out = cudf::test::column_wrapper<int32_t>{lhs.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), lhs.get(), rhs.get(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Sub_Vector_Vector_SI64) {
    using SUB = cudf::library::operation::Sub<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 100000 + row * 2;},
        [](gdf_size_type row) {return (row % 4 == 0);}};
    auto rhs = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 50000 + row;},
        [](gdf_size_type row) {return (row % 3 > 0);}};
    auto out = cudf::test::column_wrapper<int64_t>{lhs.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), lhs.get(), rhs.get(), GDF_SUB);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, SUB());
}


TEST_F(BinaryOperationIntegrationTest, Mul_Vector_Vector_SI64) {
    using MUL = cudf::library::operation::Mul<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 100000 + row * 2;},
        [](gdf_size_type row) {return (row % 3 > 0);}};
    auto rhs = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 50000 + row;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto out = cudf::test::column_wrapper<int64_t>{lhs.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), lhs.get(), rhs.get(), GDF_MUL);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, MUL());
}


TEST_F(BinaryOperationIntegrationTest, Div_Vector_Vector_SI64) {
    using DIV = cudf::library::operation::Div<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 100000 + row * 2;},
        [](gdf_size_type row) {return (row % 6 > 0);}};
    auto rhs = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 50000 + row;},
        [](gdf_size_type row) {return (row % 8 > 0);}};
    auto out = cudf::test::column_wrapper<int64_t>{lhs.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), lhs.get(), rhs.get(), GDF_DIV);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, DIV());
}


TEST_F(BinaryOperationIntegrationTest, TrueDiv_Vector_Vector_SI64) {
    using TRUEDIV = cudf::library::operation::TrueDiv<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 100000 + row * 2;},
        [](gdf_size_type row) {return (row % 3 == 0);}};
    auto rhs = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 50000 + row;},
        [](gdf_size_type row) {return (row % 4 == 0);}};
    auto out = cudf::test::column_wrapper<int64_t>{lhs.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), lhs.get(), rhs.get(), GDF_TRUE_DIV);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, TRUEDIV());
}


TEST_F(BinaryOperationIntegrationTest, FloorDiv_Vector_Vector_SI64) {
    using FLOORDIV = cudf::library::operation::FloorDiv<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 100000 + row * 2;},
        [](gdf_size_type row) {return (row % 6 > 0);}};
    auto rhs = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 50000 + row;},
        [](gdf_size_type row) {return (row % 8 > 0);}};
    auto out = cudf::test::column_wrapper<int64_t>{lhs.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), lhs.get(), rhs.get(), GDF_FLOOR_DIV);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, FLOORDIV());
}


TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_SI64) {
    using MOD = cudf::library::operation::Mod<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>{50,
        [](gdf_size_type row) {return 120 + row * 2;},
        [](gdf_size_type row) {return (row % 3 > 0);}};
    auto rhs = cudf::test::column_wrapper<int64_t>{50,
        [](gdf_size_type row) {return 50 + row;},
        [](gdf_size_type row) {return (row % 5 > 0);}};
    auto out = cudf::test::column_wrapper<int64_t>{lhs.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), lhs.get(), rhs.get(), GDF_MOD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, MOD());
}


TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP32) {
    using MOD = cudf::library::operation::Mod<float, float, float>;

    auto lhs = cudf::test::column_wrapper<float>{50,
        [](gdf_size_type row) {return 120 + row * 2;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto rhs = cudf::test::column_wrapper<float>{50,
        [](gdf_size_type row) {return 50 + row;},
        [](gdf_size_type row) {return (row % 6 > 0);}};
    auto out = cudf::test::column_wrapper<float>{lhs.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), lhs.get(), rhs.get(), GDF_MOD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, MOD());
}


TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP64) {
    using MOD = cudf::library::operation::Mod<double, double, double>;

    auto lhs = cudf::test::column_wrapper<double>{50,
        [](gdf_size_type row) {return 120 + row * 2;},
        [](gdf_size_type row) {return (row % 3 == 0);}};
    auto rhs = cudf::test::column_wrapper<double>{50,
        [](gdf_size_type row) {return 50 + row;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto out = cudf::test::column_wrapper<double>{lhs.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), lhs.get(), rhs.get(), GDF_MOD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, MOD());
}


TEST_F(BinaryOperationIntegrationTest, Pow_Vector_Vector_SI64) {
    using POW = cudf::library::operation::Pow<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>{500,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 6 > 0);}};
    auto rhs = cudf::test::column_wrapper<int64_t>{500,
        [](gdf_size_type row) {return 2;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto out = cudf::test::column_wrapper<int64_t>{lhs.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), lhs.get(), rhs.get(), GDF_POW);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, POW());
}


TEST_F(BinaryOperationIntegrationTest, And_Vector_Vector_SI16_SI64_SI32) {
    using AND = cudf::library::operation::BitwiseAnd<int16_t, int64_t, int32_t>;

    auto lhs = cudf::test::column_wrapper<int16_t>{500,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 6 > 0);}};
    auto rhs = cudf::test::column_wrapper<int64_t>{500,
        [](gdf_size_type row) {return 2;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto out = cudf::test::column_wrapper<int32_t>{lhs.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), lhs.get(), rhs.get(), GDF_BITWISE_AND);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, AND());
}


TEST_F(BinaryOperationIntegrationTest, Or_Vector_Vector_SI64_SI16_SI32) {
    using OR = cudf::library::operation::BitwiseOr<int64_t, int16_t, int32_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>{500,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 6 > 0);}};
    auto rhs = cudf::test::column_wrapper<int16_t>{500,
        [](gdf_size_type row) {return 2;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto out = cudf::test::column_wrapper<int32_t>{lhs.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), lhs.get(), rhs.get(), GDF_BITWISE_OR);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, OR());
}


TEST_F(BinaryOperationIntegrationTest, Xor_Vector_Vector_SI32_SI16_SI64) {
    using XOR = cudf::library::operation::BitwiseXor<int32_t, int16_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int32_t>{500,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 6 > 0);}};
    auto rhs = cudf::test::column_wrapper<int16_t>{500,
        [](gdf_size_type row) {return 2;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto out = cudf::test::column_wrapper<int64_t>{lhs.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), lhs.get(), rhs.get(), GDF_BITWISE_XOR);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, lhs, rhs, XOR());
}

} // namespace binop
} // namespace test
} // namespace cudf
