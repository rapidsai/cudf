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

struct BinaryOperationOperandsNullTest : public GdfTest {};


/*
 * Kernels v_v_s, using int64_t
 * Output:Vector, OperandX:Vector, OperandY:Scalar
 */
TEST_F(BinaryOperationOperandsNullTest, Vector_Scalar_SI64_WithScalarOperandNull) {
    using ADD = cudf::library::operation::Add<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>(100,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 3 > 0);});
    auto rhs = cudf::test::scalar_wrapper<int64_t>(500, false);
    auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_ADD));

    ASSERT_BINOP(out, lhs, rhs, ADD());
}


TEST_F(BinaryOperationOperandsNullTest, Vector_Scalar_SI64_WithScalarOperandNotNull) {
    using ADD = cudf::library::operation::Add<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>(100,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 3 > 0);});
    auto rhs = cudf::test::scalar_wrapper<int64_t>(500, true);
    auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_ADD));

    ASSERT_BINOP(out, lhs, rhs, ADD());
}

/*
 * Kernels v_v_s, using double
 * Output:Vector, OperandX:Vector, OperandY:Scalar
 */
TEST_F(BinaryOperationOperandsNullTest, Vector_Vector_FP64_WithScalarOperandNull) {
    using ADD = cudf::library::operation::Add<double, double, double>;

    auto lhs = cudf::test::column_wrapper<double>(100,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 3 > 0);});
    auto rhs = cudf::test::scalar_wrapper<double>(500, false);
    auto out = cudf::test::column_wrapper<double>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_ADD));

    ASSERT_BINOP(out, lhs, rhs, ADD());
}


TEST_F(BinaryOperationOperandsNullTest, Vector_Vector_FP64_WithScalarOperandNotNull) {
    using ADD = cudf::library::operation::Add<double, double, double>;

    auto lhs = cudf::test::column_wrapper<double>(100,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 3 > 0);});
    auto rhs = cudf::test::scalar_wrapper<double>(500, true);
    auto out = cudf::test::column_wrapper<double>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_ADD));

    ASSERT_BINOP(out, lhs, rhs, ADD());
}

/*
 * Kernels v_v_v, using int64_t
 * Output:Vector, OperandX:Vector, OperandY:Vector
 */
TEST_F(BinaryOperationOperandsNullTest, Vector_Vector_int64_t) {
    using ADD = cudf::library::operation::Add<int64_t, int64_t, int64_t>;

    auto lhs = cudf::test::column_wrapper<int64_t>(100,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 3 > 0);});
    auto rhs = cudf::test::column_wrapper<int64_t>(100,
        [](gdf_size_type row) {return row * 2;},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto out = cudf::test::column_wrapper<int64_t>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_ADD));

    ASSERT_BINOP(out, lhs, rhs, ADD());
}

/*
 * Kernels v_v_v, using double
 * Output:Vector, OperandX:Vector, OperandY:Vector
 */
TEST_F(BinaryOperationOperandsNullTest, Vector_Vector_FP64) {
    using ADD = cudf::library::operation::Add<double, double, double>;

    auto lhs = cudf::test::column_wrapper<double>(100,
        [](gdf_size_type row) {return row;},
        [](gdf_size_type row) {return (row % 3 > 0);});
    auto rhs = cudf::test::column_wrapper<double>(100,
        [](gdf_size_type row) {return row * 2;},
        [](gdf_size_type row) {return (row % 4 > 0);});
    auto out = cudf::test::column_wrapper<double>(lhs.get()->size, true);

    CUDF_EXPECT_NO_THROW(cudf::binary_operation(out.get(), lhs.get(), rhs.get(), GDF_ADD));

    ASSERT_BINOP(out, lhs, rhs, ADD());
}

} // namespace binop
} // namespace test
} // namespace cudf
