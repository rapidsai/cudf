/*
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

namespace gdf {
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
    using ADD = gdf::library::operation::Add<int32_t, float, int64_t>;

    auto vax = cudf::test::scalar_wrapper<float>{100};
    auto vay = cudf::test::column_wrapper<int64_t>{100000, 
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto out = cudf::test::column_wrapper<int32_t>{vay.get()->size, true};

    auto result = gdf_binary_operation_s_v(out.get(), vax.get(), vay.get(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, vax, vay, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Sub_Scalar_Vector_SI32_FP32_SI64) {
    using SUB = gdf::library::operation::Sub<int32_t, float, int64_t>;

    auto vax = cudf::test::scalar_wrapper<float>{10000};
    auto vay = cudf::test::column_wrapper<int64_t>{100000, 
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto out = cudf::test::column_wrapper<int32_t>{vay.get()->size, true};

    auto result = gdf_binary_operation_s_v(out.get(), vax.get(), vay.get(), GDF_SUB);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, vax, vay, SUB());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Scalar_SI08_SI16_SI32) {
    using ADD = gdf::library::operation::Add<int8_t, int16_t, int32_t>;

    auto vax = cudf::test::column_wrapper<int16_t>{100,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 6 > 0);}};
    auto vay = cudf::test::scalar_wrapper<int32_t>{100};
    auto out = cudf::test::column_wrapper<int8_t>{vax.get()->size, true};

    auto result = gdf_binary_operation_v_s(out.get(), vax.get(), vay.get(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, vax, vay, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Vector_SI32_FP64_SI08) {
    using ADD = gdf::library::operation::Add<int32_t, double, int8_t>;

    auto vax = cudf::test::column_wrapper<double>{100,
        [](gdf_size_type row) {return row * 2.0;},
        [](gdf_size_type row) {return (row % 3 > 0);}};
    auto vay = cudf::test::column_wrapper<int8_t>{100,
        [](gdf_size_type row) {return row * 1;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto out = cudf::test::column_wrapper<int32_t>{vax.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), vax.get(), vay.get(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, vax, vay, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Sub_Vector_Vector_SI64) {
    using SUB = gdf::library::operation::Sub<int64_t, int64_t, int64_t>;

    auto vax = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 100000 + row * 2;},
        [](gdf_size_type row) {return (row % 4 == 0);}};
    auto vay = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 50000 + row;},
        [](gdf_size_type row) {return (row % 3 > 0);}};
    auto out = cudf::test::column_wrapper<int64_t>{vax.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), vax.get(), vay.get(), GDF_SUB);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, vax, vay, SUB());
}


TEST_F(BinaryOperationIntegrationTest, Mul_Vector_Vector_SI64) {
    using MUL = gdf::library::operation::Mul<int64_t, int64_t, int64_t>;

    auto vax = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 100000 + row * 2;},
        [](gdf_size_type row) {return (row % 3 > 0);}};
    auto vay = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 50000 + row;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto out = cudf::test::column_wrapper<int64_t>{vax.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), vax.get(), vay.get(), GDF_MUL);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, vax, vay, MUL());
}


TEST_F(BinaryOperationIntegrationTest, Div_Vector_Vector_SI64) {
    using DIV = gdf::library::operation::Div<int64_t, int64_t, int64_t>;

    auto vax = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 100000 + row * 2;},
        [](gdf_size_type row) {return (row % 6 > 0);}};
    auto vay = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 50000 + row;},
        [](gdf_size_type row) {return (row % 8 > 0);}};
    auto out = cudf::test::column_wrapper<int64_t>{vax.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), vax.get(), vay.get(), GDF_DIV);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, vax, vay, DIV());
}


TEST_F(BinaryOperationIntegrationTest, TrueDiv_Vector_Vector_SI64) {
    using TRUEDIV = gdf::library::operation::TrueDiv<int64_t, int64_t, int64_t>;

    auto vax = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 100000 + row * 2;},
        [](gdf_size_type row) {return (row % 3 == 0);}};
    auto vay = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 50000 + row;},
        [](gdf_size_type row) {return (row % 4 == 0);}};
    auto out = cudf::test::column_wrapper<int64_t>{vax.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), vax.get(), vay.get(), GDF_TRUE_DIV);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, vax, vay, TRUEDIV());
}


TEST_F(BinaryOperationIntegrationTest, FloorDiv_Vector_Vector_SI64) {
    using FLOORDIV = gdf::library::operation::FloorDiv<int64_t, int64_t, int64_t>;

    auto vax = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 100000 + row * 2;},
        [](gdf_size_type row) {return (row % 6 > 0);}};
    auto vay = cudf::test::column_wrapper<int64_t>{50000,
        [](gdf_size_type row) {return 50000 + row;},
        [](gdf_size_type row) {return (row % 8 > 0);}};
    auto out = cudf::test::column_wrapper<int64_t>{vax.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), vax.get(), vay.get(), GDF_FLOOR_DIV);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, vax, vay, FLOORDIV());
}


TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_SI64) {
    using MOD = gdf::library::operation::Mod<int64_t, int64_t, int64_t>;

    auto vax = cudf::test::column_wrapper<int64_t>{50,
        [](gdf_size_type row) {return 120 + row * 2;},
        [](gdf_size_type row) {return (row % 3 > 0);}};
    auto vay = cudf::test::column_wrapper<int64_t>{50,
        [](gdf_size_type row) {return 50 + row;},
        [](gdf_size_type row) {return (row % 5 > 0);}};
    auto out = cudf::test::column_wrapper<int64_t>{vax.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), vax.get(), vay.get(), GDF_MOD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, vax, vay, MOD());
}


TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP32) {
    using MOD = gdf::library::operation::Mod<float, float, float>;

    auto vax = cudf::test::column_wrapper<float>{50,
        [](gdf_size_type row) {return 120 + row * 2;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto vay = cudf::test::column_wrapper<float>{50,
        [](gdf_size_type row) {return 50 + row;},
        [](gdf_size_type row) {return (row % 6 > 0);}};
    auto out = cudf::test::column_wrapper<float>{vax.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), vax.get(), vay.get(), GDF_MOD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, vax, vay, MOD());
}


TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP64) {
    using MOD = gdf::library::operation::Mod<double, double, double>;

    auto vax = cudf::test::column_wrapper<double>{50,
        [](gdf_size_type row) {return 120 + row * 2;},
        [](gdf_size_type row) {return (row % 3 == 0);}};
    auto vay = cudf::test::column_wrapper<double>{50,
        [](gdf_size_type row) {return 50 + row;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto out = cudf::test::column_wrapper<double>{vax.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), vax.get(), vay.get(), GDF_MOD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, vax, vay, MOD());
}


TEST_F(BinaryOperationIntegrationTest, Pow_Vector_Vector_SI64) {
    using POW = gdf::library::operation::Pow<int64_t, int64_t, int64_t>;

    auto vax = cudf::test::column_wrapper<int64_t>{500,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 6 > 0);}};
    auto vay = cudf::test::column_wrapper<int64_t>{500,
        [](gdf_size_type row) {return 2;},
        [](gdf_size_type row) {return (row % 4 > 0);}};
    auto out = cudf::test::column_wrapper<int64_t>{vax.get()->size, true};

    auto result = gdf_binary_operation_v_v(out.get(), vax.get(), vay.get(), GDF_POW);
    ASSERT_TRUE(result == GDF_SUCCESS);

    ASSERT_BINOP(out, vax, vay, POW());
}

} // namespace binop
} // namespace test
} // namespace gdf
