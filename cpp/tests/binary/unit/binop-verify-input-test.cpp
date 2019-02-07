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

#include "gtest/gtest.h"
#include "tests/binary/util/scalar.h"
#include "tests/binary/util/vector.h"

struct BinopVerifyInputTest : public ::testing::Test {
    BinopVerifyInputTest() {
    }

    virtual ~BinopVerifyInputTest() {
    }

    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOutputVectorZeroSize) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.fillData(0, 0);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    auto result = gdf_binary_operation_v_v_s(vector_out.column(), vector_vax.column(), scalar.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandVectorZeroSize) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.fillData(0, 0);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    auto result = gdf_binary_operation_v_v_s(vector_out.column(), vector_vax.column(), scalar.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOutputVectorNull) {
    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    auto result = gdf_binary_operation_v_v_s(nullptr, vector_vax.column(), scalar.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandVectorNull) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    auto result = gdf_binary_operation_v_v_s(vector_out.column(), nullptr, scalar.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandScalarNull) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    auto result = gdf_binary_operation_v_v_s(vector_out.column(), vector_vax.column(), nullptr, GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOutputVectorType) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);
    vector_out.column()->dtype = (gdf_dtype)100;

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    auto result = gdf_binary_operation_v_v_s(vector_out.column(), vector_vax.column(), scalar.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandVectorType) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);
    vector_vax.column()->dtype = (gdf_dtype)100;

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    auto result = gdf_binary_operation_v_v_s(vector_out.column(), vector_vax.column(), scalar.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandScalarType) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);
    scalar.scalar()->dtype = (gdf_dtype)100;

    auto result = gdf_binary_operation_v_v_s(vector_out.column(), vector_vax.column(), scalar.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorOutputVectorZeroSize) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.fillData(0, 0);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.rangeData(1, 10, 1);

    auto result = gdf_binary_operation_v_v_v(vector_out.column(), vector_vax.column(), vector_vay.column(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorFirstOperandVectorZeroSize) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.fillData(0, 0);

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.rangeData(1, 10, 1);

    auto result = gdf_binary_operation_v_v_v(vector_out.column(), vector_vax.column(), vector_vay.column(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorSecondOperandVectorZeroSize) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.fillData(0, 0);

    auto result = gdf_binary_operation_v_v_v(vector_out.column(), vector_vax.column(), vector_vay.column(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorOutputVectorNull) {
    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.rangeData(1, 10, 1);

    auto result = gdf_binary_operation_v_v_v(nullptr, vector_vax.column(), vector_vay.column(), GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorFirstOperandVectorNull) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.rangeData(1, 10, 1);

    auto result = gdf_binary_operation_v_v_v(vector_out.column(), nullptr, vector_vay.column(), GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorSecondOperandVectorNull) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    auto result = gdf_binary_operation_v_v_v(vector_out.column(), vector_vax.column(), nullptr, GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorOutputVectorType) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);
    vector_out.column()->dtype = (gdf_dtype)100;

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.rangeData(1, 10, 1);

    auto result = gdf_binary_operation_v_v_v(vector_out.column(), vector_vax.column(), vector_vay.column(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorFirstOperandVectorType) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);
    vector_vax.column()->dtype = (gdf_dtype)100;

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.rangeData(1, 10, 1);

    auto result = gdf_binary_operation_v_v_v(vector_out.column(), vector_vax.column(), vector_vay.column(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorSecondOperandVectorType) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.rangeData(1, 10, 1);
    vector_vay.column()->dtype = (gdf_dtype)100;

    auto result = gdf_binary_operation_v_v_v(vector_out.column(), vector_vax.column(), vector_vay.column(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_Default_ErrorOutputVectorZeroSize) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.fillData(0, 0);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_s_d(vector_out.column(), vector_vax.column(), scalar.scalar(), defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_Default_ErrorOperandVectorZeroSize) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.fillData(0, 0);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_s_d(vector_out.column(), vector_vax.column(), scalar.scalar(), defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_Default_ErrorOutputVectorNull) {
    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_s_d(nullptr, vector_vax.column(), scalar.scalar(), defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_Default_ErrorOperandVectorNull) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_s_d(vector_out.column(), nullptr, scalar.scalar(), defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_Default_ErrorOperandScalarNull) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_s_d(vector_out.column(), vector_vax.column(), nullptr, defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_Default_ErrorOutputVectorType) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);
    vector_out.column()->dtype = (gdf_dtype)100;

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_s_d(vector_out.column(), vector_vax.column(), scalar.scalar(), defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_Default_ErrorOperandVectorType) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);
    vector_vax.column()->dtype = (gdf_dtype)100;

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_s_d(vector_out.column(), vector_vax.column(), scalar.scalar(), defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_Default_ErrorOperandScalarType) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);
    scalar.scalar()->dtype = (gdf_dtype)100;

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_s_d(vector_out.column(), vector_vax.column(), scalar.scalar(), defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_Default_ErrorOutputVectorZeroSize) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.fillData(0, 0);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_v_d(vector_out.column(), vector_vax.column(), vector_vay.column(), defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_Default_ErrorFirstOperandVectorZeroSize) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.fillData(0, 0);

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_v_d(vector_out.column(), vector_vax.column(), vector_vay.column(), defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_Default_ErrorSecondOperandVectorZeroSize) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.fillData(0, 0);

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_v_d(vector_out.column(), vector_vax.column(), vector_vay.column(), defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_Default_ErrorOutputVectorNull) {
    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_v_d(nullptr, vector_vax.column(), vector_vay.column(), defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_Default_ErrorFirstOperandVectorNull) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_v_d(vector_out.column(), nullptr, vector_vay.column(), defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_Default_ErrorSecondOperandVectorNull) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_v_d(vector_out.column(), vector_vax.column(), nullptr, defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_Default_ErrorOutputVectorType) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);
    vector_out.column()->dtype = (gdf_dtype)100;

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_v_d(vector_out.column(), vector_vax.column(), vector_vay.column(), defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_Default_ErrorFirstOperandVectorType) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);
    vector_vax.column()->dtype = (gdf_dtype)100;

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.rangeData(1, 10, 1);

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_v_d(vector_out.column(), vector_vax.column(), vector_vay.column(), defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_Default_ErrorSecondOperandVectorType) {
    gdf::library::Vector<uint64_t> vector_out;
    vector_out.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vax;
    vector_vax.rangeData(1, 10, 1);

    gdf::library::Vector<uint64_t> vector_vay;
    vector_vay.rangeData(1, 10, 1);
    vector_vay.column()->dtype = (gdf_dtype)100;

    gdf::library::Scalar<uint64_t> defvalue;
    defvalue.setValue(100);

    auto result = gdf_binary_operation_v_v_v_d(vector_out.column(), vector_vax.column(), vector_vay.column(), defvalue.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}
