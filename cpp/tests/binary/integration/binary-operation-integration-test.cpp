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


TEST_F(BinaryOperationIntegrationTest, Add_Scalar_Vector_SI32_FP32_UI32) {
    using SI32 = gdf::library::GdfEnumType<GDF_INT32>;
    using FP32 = gdf::library::GdfEnumType<GDF_FLOAT32>;
    using UI32 = gdf::library::GdfEnumType<GDF_UINT32>;
    using ADD = gdf::library::operation::Add<SI32, FP32, UI32>;

    gdf::library::Vector<SI32> out;
    gdf::library::Scalar<FP32> vax;
    gdf::library::Vector<UI32> vay;

    vay.rangeData(0, 100000, 1)
       .rangeValid(false, 0, 4);
    vax.setValue(100);
    out.emplaceVector(vay.dataSize());

    auto result = gdf_binary_operation_v_s_v(out.column(), vax.scalar(), vay.column(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Sub_Scalar_Vector_SI32_FP32_UI32) {
    using SI32 = gdf::library::GdfEnumType<GDF_INT32>;
    using FP32 = gdf::library::GdfEnumType<GDF_FLOAT32>;
    using UI32 = gdf::library::GdfEnumType<GDF_UINT32>;
    using SUB = gdf::library::operation::Sub<SI32, FP32, UI32>;

    gdf::library::Vector<SI32> out;
    gdf::library::Scalar<FP32> vax;
    gdf::library::Vector<UI32> vay;

    vay.rangeData(0, 100000, 1)
       .rangeValid(false, 0, 4);
    vax.setValue(10000);
    out.emplaceVector(vay.dataSize());

    auto result = gdf_binary_operation_v_s_v(out.column(), vax.scalar(), vay.column(), GDF_SUB);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, SUB());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Scalar_SI08_UI16_SI16) {
    using SI08 = gdf::library::GdfEnumType<GDF_INT8>;
    using UI16 = gdf::library::GdfEnumType<GDF_UINT16>;
    using SI16 = gdf::library::GdfEnumType<GDF_INT16>;
    using ADD = gdf::library::operation::Add<SI08, UI16, SI16>;

    gdf::library::Vector<SI08> out;
    gdf::library::Vector<UI16> vax;
    gdf::library::Scalar<SI16> vay;

    vax.rangeData(0, 100, 1)
       .rangeValid(false, 0, 6);
    vay.setValue(100);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_s(out.column(), vax.column(), vay.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Vector_UI32_FP64_SI08) {
    using UI32 = gdf::library::GdfEnumType<GDF_UINT32>;
    using FP64 = gdf::library::GdfEnumType<GDF_FLOAT64>;
    using SI08 = gdf::library::GdfEnumType<GDF_INT8>;
    using ADD = gdf::library::operation::Add<UI32, FP64, SI08>;

    gdf::library::Vector<UI32> out;
    gdf::library::Vector<FP64> vax;
    gdf::library::Vector<SI08> vay;

    vax.rangeData(0.0, 200.0, 2.0)
       .rangeValid(false, 0, 3);
    vay.rangeData(0, 100, 1)
       .rangeValid(false, 0, 4);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Scalar_Vector_Default_SI32_SI16_UI64_SI64) {
    using SI32 = gdf::library::GdfEnumType<GDF_INT32>;
    using SI16 = gdf::library::GdfEnumType<GDF_INT16>;
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using SI64 = gdf::library::GdfEnumType<GDF_INT64>;
    using ADD = gdf::library::operation::Add<SI32, SI16, UI64>;

    gdf::library::Vector<SI32> out;
    gdf::library::Scalar<SI16> vax;
    gdf::library::Vector<UI64> vay;
    gdf::library::Scalar<SI64> def;

    vax.setValue(50);
    vay.rangeData(0, 10000, 2)
       .rangeValid(false, 0, 4);
    def.setValue(1000);
    out.emplaceVector(vay.dataSize());

    auto result = gdf_binary_operation_v_s_v_d(out.column(), vax.scalar(), vay.column(), def.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Sub_Scalar_Vector_Default_SI32_SI16_UI64_SI64) {
    using SI32 = gdf::library::GdfEnumType<GDF_INT32>;
    using SI16 = gdf::library::GdfEnumType<GDF_INT16>;
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using SI64 = gdf::library::GdfEnumType<GDF_INT64>;
    using SUB = gdf::library::operation::Sub<SI32, SI16, UI64>;

    gdf::library::Vector<SI32> out;
    gdf::library::Scalar<SI16> vax;
    gdf::library::Vector<UI64> vay;
    gdf::library::Scalar<SI64> def;

    vax.setValue(500);
    vay.rangeData(0, 10000, 2)
       .rangeValid(false, 0, 4);
    def.setValue(1000);
    out.emplaceVector(vay.dataSize());

    auto result = gdf_binary_operation_v_s_v_d(out.column(), vax.scalar(), vay.column(), def.scalar(), GDF_SUB);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, SUB());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Scalar_Default_FP32_SI16_UI08_UI32) {
    using FP32 = gdf::library::GdfEnumType<GDF_FLOAT32>;
    using SI16 = gdf::library::GdfEnumType<GDF_INT16>;
    using UI08 = gdf::library::GdfEnumType<GDF_UINT8>;
    using UI32 = gdf::library::GdfEnumType<GDF_UINT32>;
    using ADD = gdf::library::operation::Add<FP32, SI16, UI08>;

    gdf::library::Vector<FP32> out;
    gdf::library::Vector<SI16> vax;
    gdf::library::Scalar<UI08> vay;
    gdf::library::Scalar<UI32> def;

    vax.rangeData(0, 30000, 3)
       .rangeValid(false, 0, 4);
    vay.setValue(50);
    def.setValue(150);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_s_d(out.column(), vax.column(), vay.scalar(), def.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Add_Vector_Vector_Default_FP64_SI32_UI32_UI16) {
    using FP64 = gdf::library::GdfEnumType<GDF_FLOAT64>;
    using SI32 = gdf::library::GdfEnumType<GDF_INT32>;
    using UI32 = gdf::library::GdfEnumType<GDF_UINT32>;
    using UI16 = gdf::library::GdfEnumType<GDF_UINT16>;
    using ADD = gdf::library::operation::Add<FP64, SI32, UI32>;

    gdf::library::Vector<FP64> out;
    gdf::library::Vector<SI32> vax;
    gdf::library::Vector<UI32> vay;
    gdf::library::Scalar<UI16> def;

    vax.rangeData(0, 100000, 1)
       .rangeValid(false, 0, 3);
    vay.rangeData(0, 200000, 2)
       .rangeValid(false, 0, 4);
    def.setValue(150);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v_d(out.column(), vax.column(), vay.column(), def.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, ADD());
}


TEST_F(BinaryOperationIntegrationTest, Sub_Vector_Vector_UI64) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using SUB = gdf::library::operation::Sub<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;

    vax.rangeData(100000, 200000, 2)
       .rangeValid(true, 0, 4);
    vay.rangeData(50000, 100000, 1)
       .rangeValid(false, 0, 3);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_SUB);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, SUB());
}


TEST_F(BinaryOperationIntegrationTest, Mul_Vector_Vector_UI64) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using MUL = gdf::library::operation::Mul<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;

    vax.rangeData(100000, 200000, 2)
       .rangeValid(false, 0 , 3);
    vay.rangeData(50000, 100000, 1)
       .rangeValid(false, 0, 4);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_MUL);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, MUL());
}


TEST_F(BinaryOperationIntegrationTest, Div_Vector_Vector_UI64) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using DIV = gdf::library::operation::Div<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;

    vax.rangeData(100000, 200000, 2)
       .rangeValid(false, 0, 6);
    vay.rangeData(50000, 100000, 1)
       .rangeValid(false, 0, 8);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_DIV);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, DIV());
}


TEST_F(BinaryOperationIntegrationTest, TrueDiv_Vector_Vector_UI64) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using TRUEDIV = gdf::library::operation::TrueDiv<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;

    vax.rangeData(100000, 200000, 2)
       .rangeValid(true, 0, 3);
    vay.rangeData(50000, 100000, 1)
       .rangeValid(true, 0, 4);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_TRUE_DIV);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, TRUEDIV());
}


TEST_F(BinaryOperationIntegrationTest, FloorDiv_Vector_Vector_UI64) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using FLOORDIV = gdf::library::operation::FloorDiv<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;

    vax.rangeData(100000, 200000, 2)
       .rangeValid(false, 0, 6);
    vay.rangeData(50000, 100000, 1)
       .rangeValid(false, 0, 8);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_FLOOR_DIV);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, FLOORDIV());
}


TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_UI64) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using MOD = gdf::library::operation::Mod<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;

    vax.rangeData(120, 220, 2)
       .rangeValid(false, 0, 3);
    vay.rangeData(50, 100, 1)
       .rangeValid(false, 0, 5);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_MOD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, MOD());
}


TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP32) {
    using FP32 = gdf::library::GdfEnumType<GDF_FLOAT32>;
    using MOD = gdf::library::operation::Mod<FP32, FP32, FP32>;

    gdf::library::Vector<FP32> out;
    gdf::library::Vector<FP32> vax;
    gdf::library::Vector<FP32> vay;

    vax.rangeData(120, 220, 2)
       .rangeValid(false, 0, 4);
    vay.rangeData(50, 100, 1)
       .rangeValid(false, 0, 6);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_MOD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, MOD());
}


TEST_F(BinaryOperationIntegrationTest, Mod_Vector_Vector_FP64) {
    using FP64 = gdf::library::GdfEnumType<GDF_FLOAT64>;
    using MOD = gdf::library::operation::Mod<FP64, FP64, FP64>;

    gdf::library::Vector<FP64> out;
    gdf::library::Vector<FP64> vax;
    gdf::library::Vector<FP64> vay;

    vax.rangeData(120, 220, 2)
       .rangeValid(true, 0, 3);
    vay.rangeData(50, 100, 1)
       .rangeValid(false, 0, 4);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_MOD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, MOD());
}


TEST_F(BinaryOperationIntegrationTest, Pow_Vector_Vector_UI64) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using POW = gdf::library::operation::Pow<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;

    vax.rangeData(0, 500, 1)
       .rangeValid(false, 0, 6);
    vay.fillData(500, 2)
       .rangeValid(false, 0, 4);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_POW);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, POW());
}

} // namespace binop
} // namespace test
} // namespace gdf
