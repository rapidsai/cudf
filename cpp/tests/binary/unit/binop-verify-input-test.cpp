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

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/scalar_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <cudf/binaryop.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <numeric>

struct BinopVerifyInputTest : public GdfTest {};


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOutputVectorZeroSize) {
    auto vector_out = cudf::test::column_wrapper<int64_t>(0);

    auto vector_lhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto scalar = cudf::test::scalar_wrapper<int64_t>{100};

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out.get(), vector_lhs.get(), scalar.get(), GDF_ADD),
        "Column sizes don't match");
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandVectorZeroSize) {
    auto vector_out = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto vector_lhs = cudf::test::column_wrapper<int64_t>(0);

    auto scalar = cudf::test::scalar_wrapper<int64_t>(100);

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out.get(), vector_lhs.get(), scalar.get(), GDF_ADD),
        "Column sizes don't match");
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOutputVectorNull) {
    gdf_column* vector_out = nullptr;

    auto vector_lhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto scalar = cudf::test::scalar_wrapper<int64_t>{100};

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out, vector_lhs.get(), scalar.get(), GDF_ADD),
        "Input pointers are null");
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandVectorNull) {
    auto vector_out = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    gdf_column* vector_lhs = nullptr;

    auto scalar = cudf::test::scalar_wrapper<int64_t>{100};

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out.get(), vector_lhs, scalar.get(), GDF_ADD),
        "Input pointers are null");
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandScalarNull) {
    auto vector_out = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto vector_lhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    gdf_column* scalar = nullptr;

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out.get(), vector_lhs.get(), scalar, GDF_ADD),
        "Input pointers are null");
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOutputVectorType) {
    auto vector_out = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});
    vector_out.get()->dtype = (gdf_dtype)100;

    auto vector_lhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto scalar = cudf::test::scalar_wrapper<int64_t>{100};

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out.get(), vector_lhs.get(), scalar.get(), GDF_ADD),
        "Invalid/Unsupported datatype");
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandVectorType) {
    auto vector_out = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto vector_lhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});
    vector_lhs.get()->dtype = (gdf_dtype)100;

    auto scalar = cudf::test::scalar_wrapper<int64_t>{100};

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out.get(), vector_lhs.get(), scalar.get(), GDF_ADD),
        "Invalid/Unsupported datatype");
}


TEST_F(BinopVerifyInputTest, Vector_Scalar_ErrorOperandScalarType) {
    auto vector_out = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto vector_lhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto scalar = cudf::test::scalar_wrapper<int64_t>{100};
    scalar.get()->dtype = (gdf_dtype)100;

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out.get(), vector_lhs.get(), scalar.get(), GDF_ADD),
        "Invalid/Unsupported datatype");
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorOutputVectorZeroSize) {
    auto vector_out = cudf::test::column_wrapper<int64_t>(0);

    auto vector_lhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto vector_rhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out.get(), vector_lhs.get(), vector_rhs.get(), GDF_ADD),
        "Column sizes don't match");
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorFirstOperandVectorZeroSize) {
    auto vector_out = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto vector_lhs = cudf::test::column_wrapper<int64_t>(0);

    auto vector_rhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out.get(), vector_lhs.get(), vector_rhs.get(), GDF_ADD),
        "Column sizes don't match");
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorSecondOperandVectorZeroSize) {
    auto vector_out = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto vector_lhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto vector_rhs = cudf::test::column_wrapper<int64_t>(0);

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out.get(), vector_lhs.get(), vector_rhs.get(), GDF_ADD),
        "Column sizes don't match");
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorOutputVectorNull) {
    gdf_column* vector_out = nullptr;

    auto vector_lhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto vector_rhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out, vector_lhs.get(), vector_rhs.get(), GDF_ADD),
        "Input pointers are null");
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorFirstOperandVectorNull) {
    auto vector_out = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    gdf_column* vector_lhs = nullptr;

    auto vector_rhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out.get(), vector_lhs, vector_rhs.get(), GDF_ADD),
        "Input pointers are null");
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorSecondOperandVectorNull) {
    auto vector_out = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto vector_lhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    gdf_column* vector_rhs = nullptr;

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out.get(), vector_lhs.get(), vector_rhs, GDF_ADD),
        "Input pointers are null");
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorOutputVectorType) {
    auto vector_out = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});
    vector_out.get()->dtype = (gdf_dtype)100;

    auto vector_lhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto vector_rhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out.get(), vector_lhs.get(), vector_rhs.get(), GDF_ADD),
        "Invalid/Unsupported datatype");
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorFirstOperandVectorType) {
    auto vector_out = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto vector_lhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});
    vector_lhs.get()->dtype = (gdf_dtype)100;

    auto vector_rhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out.get(), vector_lhs.get(), vector_rhs.get(), GDF_ADD),
        "Invalid/Unsupported datatype");
}


TEST_F(BinopVerifyInputTest, Vector_Vector_ErrorSecondOperandVectorType) {
    auto vector_out = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto vector_lhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});

    auto vector_rhs = cudf::test::column_wrapper<int64_t>(10,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return true;});
    vector_rhs.get()->dtype = (gdf_dtype)100;

    CUDF_EXPECT_THROW_MESSAGE(
        cudf::binary_operation(vector_out.get(), vector_lhs.get(), vector_rhs.get(), GDF_ADD),
        "Invalid/Unsupported datatype");
}
