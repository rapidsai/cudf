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
#include "tests/utilities/cudf_test_utils.cuh"
#include <vector>
#include <numeric>

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
    std::vector<uint64_t> vector_out(0, 0);
    auto col_out = create_gdf_column(vector_out);

    std::vector<uint64_t> vector_vax(10);
    std::iota(vector_vax.begin(), vector_vax.end(), 1);
    auto col_vax = create_gdf_column(vector_vax);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    auto result = gdf_binary_operation_v_s(col_out.get(), col_vax.get(), scalar.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandVectorZeroSize) {
    std::vector<uint64_t> vector_out(10);
    std::iota(vector_out.begin(), vector_out.end(), 1);
    auto col_out = create_gdf_column(vector_out);

    std::vector<uint64_t> vector_vax(0, 0);
    auto col_vax = create_gdf_column(vector_vax);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    auto result = gdf_binary_operation_v_s(col_out.get(), col_vax.get(), scalar.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOutputVectorNull) {
    std::vector<uint64_t> vector_vax(10);
    std::iota(vector_vax.begin(), vector_vax.end(), 1);
    auto col_vax = create_gdf_column(vector_vax);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    auto result = gdf_binary_operation_v_s(nullptr, col_vax.get(), scalar.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandVectorNull) {
    std::vector<uint64_t> vector_out(10);
    std::iota(vector_out.begin(), vector_out.end(), 1);
    auto col_out = create_gdf_column(vector_out);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    auto result = gdf_binary_operation_v_s(col_out.get(), nullptr, scalar.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandScalarNull) {
    std::vector<uint64_t> vector_out(10);
    std::iota(vector_out.begin(), vector_out.end(), 1);
    auto col_out = create_gdf_column(vector_out);

    std::vector<uint64_t> vector_vax(10);
    std::iota(vector_vax.begin(), vector_vax.end(), 1);
    auto col_vax = create_gdf_column(vector_vax);

    auto result = gdf_binary_operation_v_s(col_out.get(), col_vax.get(), nullptr, GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOutputVectorType) {
    std::vector<uint64_t> vector_out(10);
    std::iota(vector_out.begin(), vector_out.end(), 1);
    auto col_out = create_gdf_column(vector_out);
    col_out.get()->dtype = (gdf_dtype)100;

    std::vector<uint64_t> vector_vax(10);
    std::iota(vector_vax.begin(), vector_vax.end(), 1);
    auto col_vax = create_gdf_column(vector_vax);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    auto result = gdf_binary_operation_v_s(col_out.get(), col_vax.get(), scalar.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandVectorType) {
    std::vector<uint64_t> vector_out(10);
    std::iota(vector_out.begin(), vector_out.end(), 1);
    auto col_out = create_gdf_column(vector_out);

    std::vector<uint64_t> vector_vax(10);
    std::iota(vector_vax.begin(), vector_vax.end(), 1);
    auto col_vax = create_gdf_column(vector_vax);
    col_vax.get()->dtype = (gdf_dtype)100;

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);

    auto result = gdf_binary_operation_v_s(col_out.get(), col_vax.get(), scalar.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandScalarType) {
    std::vector<uint64_t> vector_out(10);
    std::iota(vector_out.begin(), vector_out.end(), 1);
    auto col_out = create_gdf_column(vector_out);

    std::vector<uint64_t> vector_vax(10);
    std::iota(vector_vax.begin(), vector_vax.end(), 1);
    auto col_vax = create_gdf_column(vector_vax);

    gdf::library::Scalar<uint64_t> scalar;
    scalar.setValue(100);
    scalar.scalar()->dtype = (gdf_dtype)100;

    auto result = gdf_binary_operation_v_s(col_out.get(), col_vax.get(), scalar.scalar(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorOutputVectorZeroSize) {
    std::vector<uint64_t> vector_out(0, 0);
    auto col_out = create_gdf_column(vector_out);

    std::vector<uint64_t> vector_vax(10);
    std::iota(vector_vax.begin(), vector_vax.end(), 1);
    auto col_vax = create_gdf_column(vector_vax);

    std::vector<uint64_t> vector_vay(10);
    std::iota(vector_vay.begin(), vector_vay.end(), 1);
    auto col_vay = create_gdf_column(vector_vay);

    auto result = gdf_binary_operation_v_v(col_out.get(), col_vax.get(), col_vay.get(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorFirstOperandVectorZeroSize) {
    std::vector<uint64_t> vector_out(10);
    std::iota(vector_out.begin(), vector_out.end(), 1);
    auto col_out = create_gdf_column(vector_out);

    std::vector<uint64_t> vector_vax(0, 0);
    auto col_vax = create_gdf_column(vector_vax);

    std::vector<uint64_t> vector_vay(10);
    std::iota(vector_vay.begin(), vector_vay.end(), 1);
    auto col_vay = create_gdf_column(vector_vay);

    auto result = gdf_binary_operation_v_v(col_out.get(), col_vax.get(), col_vay.get(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorSecondOperandVectorZeroSize) {
    std::vector<uint64_t> vector_out(10);
    std::iota(vector_out.begin(), vector_out.end(), 1);
    auto col_out = create_gdf_column(vector_out);

    std::vector<uint64_t> vector_vax(10);
    std::iota(vector_vax.begin(), vector_vax.end(), 1);
    auto col_vax = create_gdf_column(vector_vax);

    std::vector<uint64_t> vector_vay(0, 0);
    auto col_vay = create_gdf_column(vector_vay);

    auto result = gdf_binary_operation_v_v(col_out.get(), col_vax.get(), col_vay.get(), GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorOutputVectorNull) {
    std::vector<uint64_t> vector_vax(10);
    std::iota(vector_vax.begin(), vector_vax.end(), 1);
    auto col_vax = create_gdf_column(vector_vax);

    std::vector<uint64_t> vector_vay(10);
    std::iota(vector_vay.begin(), vector_vay.end(), 1);
    auto col_vay = create_gdf_column(vector_vay);

    auto result = gdf_binary_operation_v_v(nullptr, col_vax.get(), col_vay.get(), GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorFirstOperandVectorNull) {
    std::vector<uint64_t> vector_out(10);
    std::iota(vector_out.begin(), vector_out.end(), 1);
    auto col_out = create_gdf_column(vector_out);

    std::vector<uint64_t> vector_vay(10);
    std::iota(vector_vay.begin(), vector_vay.end(), 1);
    auto col_vay = create_gdf_column(vector_vay);

    auto result = gdf_binary_operation_v_v(col_out.get(), nullptr, col_vay.get(), GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorSecondOperandVectorNull) {
    std::vector<uint64_t> vector_out(10);
    std::iota(vector_out.begin(), vector_out.end(), 1);
    auto col_out = create_gdf_column(vector_out);

    std::vector<uint64_t> vector_vax(10);
    std::iota(vector_vax.begin(), vector_vax.end(), 1);
    auto col_vax = create_gdf_column(vector_vax);

    auto result = gdf_binary_operation_v_v(col_out.get(), col_vax.get(), nullptr, GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorOutputVectorType) {
    std::vector<uint64_t> vector_out(10);
    std::iota(vector_out.begin(), vector_out.end(), 1);
    auto col_out = create_gdf_column(vector_out);
    col_out.get()->dtype = (gdf_dtype)100;

    std::vector<uint64_t> vector_vax(10);
    std::iota(vector_vax.begin(), vector_vax.end(), 1);
    auto col_vax = create_gdf_column(vector_vax);

    std::vector<uint64_t> vector_vay(10);
    std::iota(vector_vay.begin(), vector_vay.end(), 1);
    auto col_vay = create_gdf_column(vector_vay);

    auto result = gdf_binary_operation_v_v(col_out.get(), col_vax.get(), col_vay.get(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorFirstOperandVectorType) {
    std::vector<uint64_t> vector_out(10);
    std::iota(vector_out.begin(), vector_out.end(), 1);
    auto col_out = create_gdf_column(vector_out);

    std::vector<uint64_t> vector_vax(10);
    std::iota(vector_vax.begin(), vector_vax.end(), 1);
    auto col_vax = create_gdf_column(vector_vax);
    col_vax.get()->dtype = (gdf_dtype)100;

    std::vector<uint64_t> vector_vay(10);
    std::iota(vector_vay.begin(), vector_vay.end(), 1);
    auto col_vay = create_gdf_column(vector_vay);

    auto result = gdf_binary_operation_v_v(col_out.get(), col_vax.get(), col_vay.get(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorSecondOperandVectorType) {
    std::vector<uint64_t> vector_out(10);
    std::iota(vector_out.begin(), vector_out.end(), 1);
    auto col_out = create_gdf_column(vector_out);

    std::vector<uint64_t> vector_vax(10);
    std::iota(vector_vax.begin(), vector_vax.end(), 1);
    auto col_vax = create_gdf_column(vector_vax);

    std::vector<uint64_t> vector_vay(10);
    std::iota(vector_vay.begin(), vector_vay.end(), 1);
    auto col_vay = create_gdf_column(vector_vay);
    col_vay.get()->dtype = (gdf_dtype)100;

    auto result = gdf_binary_operation_v_v(col_out.get(), col_vax.get(), col_vay.get(), GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}
