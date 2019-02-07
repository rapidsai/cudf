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

struct BinaryOperationOperandsNullTest : public ::testing::Test {
    BinaryOperationOperandsNullTest() {
    }

    virtual ~BinaryOperationOperandsNullTest() {
    }

    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

/*
 * Kernels v_v_s, using UI64
 * Output:Vector, OperandX:Vector, OperandY:Scalar
 */
TEST_F(BinaryOperationOperandsNullTest, Vector_Scalar_UI64_WithScalarOperandNull) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using ADD = gdf::library::operation::Add<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Scalar<UI64> vay;

    vax.rangeData(0, 100, 1)
       .rangeValid(false, 0, 3);
    vay.setValue(500)
       .setValid(false);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_s(out.column(), vax.column(), vay.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, ADD());
}


TEST_F(BinaryOperationOperandsNullTest, Vector_Scalar_UI64_WithScalarOperandNotNull) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using ADD = gdf::library::operation::Add<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Scalar<UI64> vay;

    vax.rangeData(0, 100, 1)
       .rangeValid(false, 0, 3);
    vay.setValue(500)
       .setValid(true);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_s(out.column(), vax.column(), vay.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, ADD());
}

/*
 * Kernels v_v_s, using FP64
 * Output:Vector, OperandX:Vector, OperandY:Scalar
 */
TEST_F(BinaryOperationOperandsNullTest, Vector_Vector_FP64_WithScalarOperandNull) {
    using FP64 = gdf::library::GdfEnumType<GDF_FLOAT64>;
    using ADD = gdf::library::operation::Add<FP64, FP64, FP64>;

    gdf::library::Vector<FP64> out;
    gdf::library::Vector<FP64> vax;
    gdf::library::Scalar<FP64> vay;

    vax.rangeData(0, 100.0, 1.0)
       .rangeValid(false, 0, 3);
    vay.setValue(500.0)
       .setValid(false);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_s(out.column(), vax.column(), vay.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, ADD());
}


TEST_F(BinaryOperationOperandsNullTest, Vector_Vector_FP64_WithScalarOperandNotNull) {
    using FP64 = gdf::library::GdfEnumType<GDF_FLOAT64>;
    using ADD = gdf::library::operation::Add<FP64, FP64, FP64>;

    gdf::library::Vector<FP64> out;
    gdf::library::Vector<FP64> vax;
    gdf::library::Scalar<FP64> vay;

    vax.rangeData(0, 100.0, 1.0)
       .rangeValid(false, 0, 3);
    vay.setValue(500.0)
       .setValid(true);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_s(out.column(), vax.column(), vay.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, ADD());
}

/*
 * Kernels v_v_v, using UI64
 * Output:Vector, OperandX:Vector, OperandY:Vector
 */
TEST_F(BinaryOperationOperandsNullTest, Vector_Vector_UI64) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using ADD = gdf::library::operation::Add<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;

    vax.rangeData(0, 100, 1)
       .rangeValid(false, 0, 3);
    vay.rangeData(0, 200, 2)
       .rangeValid(false, 0, 4);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, ADD());
}

/*
 * Kernels v_v_v, using FP64
 * Output:Vector, OperandX:Vector, OperandY:Vector
 */
TEST_F(BinaryOperationOperandsNullTest, Vector_Vector_FP64) {
    using FP64 = gdf::library::GdfEnumType<GDF_FLOAT64>;
    using ADD = gdf::library::operation::Add<FP64, FP64, FP64>;

    gdf::library::Vector<FP64> out;
    gdf::library::Vector<FP64> vax;
    gdf::library::Vector<FP64> vay;

    vax.rangeData(0, 100.0, 1.0)
       .rangeValid(false, 0, 3);
    vay.rangeData(0, 200.0, 2.0)
       .rangeValid(false, 0, 4);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v(out.column(), vax.column(), vay.column(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, ADD());
}

/*
 * Kernels v_v_s_d, using UI64
 * Output:Vector, OperandX:Vector, OperandY:Scalar, OperandDefault:Scalar
 */
TEST_F(BinaryOperationOperandsNullTest, Vector_Scalar_Default_UI64_WithAllOperandsNotNull) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using ADD = gdf::library::operation::Add<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Scalar<UI64> vay;
    gdf::library::Scalar<UI64> def;

    vax.rangeData(0, 100, 1)
       .rangeValid(false, 0, 3);
    vay.setValue(222)
       .setValid(true);
    def.setValue(555)
       .setValid(true);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_s_d(out.column(), vax.column(), vay.scalar(), def.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, ADD());
}


TEST_F(BinaryOperationOperandsNullTest, Vector_Scalar_Default_UI64_WithScalarOperandNull) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using ADD = gdf::library::operation::Add<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Scalar<UI64> vay;
    gdf::library::Scalar<UI64> def;

    vax.rangeData(0, 100, 1)
       .rangeValid(false, 0, 3);
    vay.setValue(1000)
       .setValid(false);
    def.setValue(500)
       .setValid(true);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_s_d(out.column(), vax.column(), vay.scalar(), def.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, ADD());
}


TEST_F(BinaryOperationOperandsNullTest, Vector_Scalar_Default_UI64_WithDefaultOperandNull) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using ADD = gdf::library::operation::Add<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Scalar<UI64> vay;
    gdf::library::Scalar<UI64> def;

    vax.rangeData(0, 100, 1)
       .rangeValid(false, 0, 3);
    vay.setValue(250)
       .setValid(true);
    def.setValue(750)
       .setValid(false);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_s_d(out.column(), vax.column(), vay.scalar(), def.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, ADD());
}


TEST_F(BinaryOperationOperandsNullTest, Vector_Scalar_Default_UI64_WithAllOperandsNull) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using ADD = gdf::library::operation::Add<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Scalar<UI64> vay;
    gdf::library::Scalar<UI64> def;

    vax.rangeData(0, 100, 1)
       .rangeValid(false, 0, 3);
    vay.setValue(500)
       .setValid(false);
    def.setValue(1000)
       .setValid(false);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_s_d(out.column(), vax.column(), vay.scalar(), def.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, ADD());
}

/*
 * Kernels v_v_s_d, using FP32
 * Output:Vector, OperandX:Vector, OperandY:Scalar, OperandDefault:Scalar
 */
TEST_F(BinaryOperationOperandsNullTest, Vector_Scalar_Default_FP32_WithAllOperandsNotNull) {
    using FP32 = gdf::library::GdfEnumType<GDF_FLOAT32>;
    using ADD = gdf::library::operation::Add<FP32, FP32, FP32>;

    gdf::library::Vector<FP32> out;
    gdf::library::Vector<FP32> vax;
    gdf::library::Scalar<FP32> vay;
    gdf::library::Scalar<FP32> def;

    vax.rangeData(0, 100.0, 1.0)
       .rangeValid(false, 0, 3);
    vay.setValue(222.0)
       .setValid(true);
    def.setValue(555.0)
       .setValid(true);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_s_d(out.column(), vax.column(), vay.scalar(), def.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, ADD());
}


TEST_F(BinaryOperationOperandsNullTest, Vector_Scalar_Default_FP32_WithScalarOperandNull) {
    using FP32 = gdf::library::GdfEnumType<GDF_FLOAT32>;
    using ADD = gdf::library::operation::Add<FP32, FP32, FP32>;

    gdf::library::Vector<FP32> out;
    gdf::library::Vector<FP32> vax;
    gdf::library::Scalar<FP32> vay;
    gdf::library::Scalar<FP32> def;

    vax.rangeData(0, 100.0, 1.0)
       .rangeValid(false, 0, 3);
    vay.setValue(1000.0)
       .setValid(false);
    def.setValue(500.0)
       .setValid(true);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_s_d(out.column(), vax.column(), vay.scalar(), def.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, ADD());
}


TEST_F(BinaryOperationOperandsNullTest, Vector_Scalar_Default_FP32_WithDefaultOperandNull) {
    using FP32 = gdf::library::GdfEnumType<GDF_FLOAT32>;
    using ADD = gdf::library::operation::Add<FP32, FP32, FP32>;

    gdf::library::Vector<FP32> out;
    gdf::library::Vector<FP32> vax;
    gdf::library::Scalar<FP32> vay;
    gdf::library::Scalar<FP32> def;

    vax.rangeData(0, 100.0, 1.0)
       .rangeValid(false, 0, 3);
    vay.setValue(250.0)
       .setValid(true);
    def.setValue(750.0)
       .setValid(false);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_s_d(out.column(), vax.column(), vay.scalar(), def.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, ADD());
}


TEST_F(BinaryOperationOperandsNullTest, Vector_Scalar_Default_FP32_WithAllOperandsNull) {
    using FP32 = gdf::library::GdfEnumType<GDF_FLOAT32>;
    using ADD = gdf::library::operation::Add<FP32, FP32, FP32>;

    gdf::library::Vector<FP32> out;
    gdf::library::Vector<FP32> vax;
    gdf::library::Scalar<FP32> vay;
    gdf::library::Scalar<FP32> def;

    vax.rangeData(0, 100.0, 1.0)
       .rangeValid(false, 0, 3);
    vay.setValue(500.0)
       .setValid(false);
    def.setValue(1000.0)
       .setValid(false);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_s_d(out.column(), vax.column(), vay.scalar(), def.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, ADD());
}

/*
 * Kernels v_v_v_d, using UI64
 * Output:Vector, OperandX:Vector, OperandY:Vector, OperandDefault:Scalar
 */
TEST_F(BinaryOperationOperandsNullTest, Vector_Vector_Default_UI64_WithDefaultOperandNull) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using ADD = gdf::library::operation::Add<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;
    gdf::library::Scalar<UI64> def;

    vax.rangeData(0, 100, 1)
       .rangeValid(false, 0, 3);
    vay.rangeData(0, 200, 2)
       .rangeValid(false, 0, 4);
    def.setValue(666)
       .setValid(false);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v_d(out.column(), vax.column(), vay.column(), def.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, ADD());
}


TEST_F(BinaryOperationOperandsNullTest, Vector_Vector_Default_UI64_WithDefaultOperandNotNull) {
    using UI64 = gdf::library::GdfEnumType<GDF_UINT64>;
    using ADD = gdf::library::operation::Add<UI64, UI64, UI64>;

    gdf::library::Vector<UI64> out;
    gdf::library::Vector<UI64> vax;
    gdf::library::Vector<UI64> vay;
    gdf::library::Scalar<UI64> def;

    vax.rangeData(0, 100, 1)
       .rangeValid(false, 0, 3);
    vay.rangeData(0, 200, 2)
       .rangeValid(false, 0, 4);
    def.setValue(222)
       .setValid(true);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v_d(out.column(), vax.column(), vay.column(), def.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, ADD());
}

/*
 * Kernels v_v_v_d, using FP64
 * Output:Vector, OperandX:Vector, OperandY:Vector, OperandDefault:Scalar
 */
TEST_F(BinaryOperationOperandsNullTest, Vector_Scalar_Default_FP64_WithDefaultOperandNull) {
    using FP64 = gdf::library::GdfEnumType<GDF_FLOAT64>;
    using ADD = gdf::library::operation::Add<FP64, FP64, FP64>;

    gdf::library::Vector<FP64> out;
    gdf::library::Vector<FP64> vax;
    gdf::library::Vector<FP64> vay;
    gdf::library::Scalar<FP64> def;

    vax.rangeData(0, 100.0, 1.0)
       .rangeValid(false, 0, 3);
    vay.rangeData(0, 200.0, 2.0)
       .rangeValid(false, 0, 4);
    def.setValue(555.0)
       .setValid(false);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v_d(out.column(), vax.column(), vay.column(), def.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, ADD());
}


TEST_F(BinaryOperationOperandsNullTest, Vector_Vector_Default_FP64_WithDefaultOperandNotNull) {
    using FP64 = gdf::library::GdfEnumType<GDF_FLOAT64>;
    using ADD = gdf::library::operation::Add<FP64, FP64, FP64>;

    gdf::library::Vector<FP64> out;
    gdf::library::Vector<FP64> vax;
    gdf::library::Vector<FP64> vay;
    gdf::library::Scalar<FP64> def;

    vax.rangeData(0, 100.0, 1.0)
       .rangeValid(false, 0, 3);
    vay.rangeData(0, 200.0, 2.0)
       .rangeValid(false, 0, 4);
    def.setValue(555.0)
       .setValid(true);
    out.emplaceVector(vax.dataSize());

    auto result = gdf_binary_operation_v_v_v_d(out.column(), vax.column(), vay.column(), def.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);

    out.readVector();

    ASSERT_BINOP(out, vax, vay, def, ADD());
}

} // namespace binop
} // namespace test
} // namespace gdf
