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

#include "tests/utilities/column_wrapper.cuh"
#include "tests/utilities/scalar_wrapper.cuh"
#include "gtest/gtest.h"
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
    auto vector_out = cudf::test::column_wrapper<int64_t>{0};

    auto vector_lhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto scalar = cudf::test::scalar_wrapper<int64_t>{100};

    auto result = gdf_binary_operation_v_s(vector_out, vector_lhs, scalar, GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandVectorZeroSize) {
    auto vector_out = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto vector_lhs = cudf::test::column_wrapper<int64_t>{0};

    auto scalar = cudf::test::scalar_wrapper<int64_t>{100};

    auto result = gdf_binary_operation_v_s(vector_out, vector_lhs, scalar, GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOutputVectorNull) {
    auto vector_lhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto scalar = cudf::test::scalar_wrapper<int64_t>{100};

    auto result = gdf_binary_operation_v_s(nullptr, vector_lhs, scalar, GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandVectorNull) {
    auto vector_out = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto scalar = cudf::test::scalar_wrapper<int64_t>{100};

    auto result = gdf_binary_operation_v_s(vector_out, nullptr, scalar, GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandScalarNull) {
    auto vector_out = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto vector_lhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto result = gdf_binary_operation_v_s(vector_out, vector_lhs, nullptr, GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOutputVectorType) {
    auto vector_out = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};
    vector_out.get()->dtype = (gdf_dtype)100;

    auto vector_lhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto scalar = cudf::test::scalar_wrapper<int64_t>{100};

    auto result = gdf_binary_operation_v_s(vector_out, vector_lhs, scalar, GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandVectorType) {
    auto vector_out = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto vector_lhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};
    vector_lhs.get()->dtype = (gdf_dtype)100;

    auto scalar = cudf::test::scalar_wrapper<int64_t>{100};

    auto result = gdf_binary_operation_v_s(vector_out, vector_lhs, scalar, GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandScalarType) {
    auto vector_out = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto vector_lhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto scalar = cudf::test::scalar_wrapper<int64_t>{100};
    scalar.get()->dtype = (gdf_dtype)100;

    auto result = gdf_binary_operation_v_s(vector_out, vector_lhs, scalar, GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorOutputVectorZeroSize) {
    auto vector_out = cudf::test::column_wrapper<int64_t>{0};

    auto vector_lhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto vector_rhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto result = gdf_binary_operation_v_v(vector_out, vector_lhs, vector_rhs, GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorFirstOperandVectorZeroSize) {
    auto vector_out = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto vector_lhs = cudf::test::column_wrapper<int64_t>{0};

    auto vector_rhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto result = gdf_binary_operation_v_v(vector_out, vector_lhs, vector_rhs, GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorSecondOperandVectorZeroSize) {
    auto vector_out = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto vector_lhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto vector_rhs = cudf::test::column_wrapper<int64_t>{0};

    auto result = gdf_binary_operation_v_v(vector_out, vector_lhs, vector_rhs, GDF_ADD);
    ASSERT_TRUE(result == GDF_SUCCESS);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorOutputVectorNull) {
    auto vector_lhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto vector_rhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto result = gdf_binary_operation_v_v(nullptr, vector_lhs, vector_rhs, GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorFirstOperandVectorNull) {
    auto vector_out = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto vector_rhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto result = gdf_binary_operation_v_v(vector_out, nullptr, vector_rhs, GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorSecondOperandVectorNull) {
    auto vector_out = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto vector_lhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto result = gdf_binary_operation_v_v(vector_out, vector_lhs, nullptr, GDF_ADD);
    ASSERT_TRUE(result == GDF_DATASET_EMPTY);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorOutputVectorType) {
    auto vector_out = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};
    vector_out.get()->dtype = (gdf_dtype)100;

    auto vector_lhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto vector_rhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto result = gdf_binary_operation_v_v(vector_out, vector_lhs, vector_rhs, GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorFirstOperandVectorType) {
    auto vector_out = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto vector_lhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};
    vector_lhs.get()->dtype = (gdf_dtype)100;

    auto vector_rhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto result = gdf_binary_operation_v_v(vector_out, vector_lhs, vector_rhs, GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorSecondOperandVectorType) {
    auto vector_out = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto vector_lhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};

    auto vector_rhs = cudf::test::column_wrapper<int64_t>{10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;}};
    vector_rhs.get()->dtype = (gdf_dtype)100;

    auto result = gdf_binary_operation_v_v(vector_out, vector_lhs, vector_rhs, GDF_ADD);
    ASSERT_TRUE(result == GDF_UNSUPPORTED_DTYPE);
}
