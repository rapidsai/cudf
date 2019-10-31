/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/transpose.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

template <typename T>
class TransposeTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(TransposeTest, cudf::test::FixedWidthTypes);

TYPED_TEST(TransposeTest, NonNull)
{
    using T = TypeParam;

    cudf::test::fixed_width_column_wrapper<T> in_col1{{1, 2, 3, 4}};
    cudf::test::fixed_width_column_wrapper<T> in_col2{{5, 6, 7, 8}};
    cudf::test::fixed_width_column_wrapper<T> in_col3{{9, 10, 11, 12}};
    cudf::table_view input{{in_col1, in_col2, in_col3}};

    cudf::test::fixed_width_column_wrapper<T> out_col1{{1, 5, 9}};
    cudf::test::fixed_width_column_wrapper<T> out_col2{{2, 6, 10}};
    cudf::test::fixed_width_column_wrapper<T> out_col3{{3, 7, 11}};
    cudf::test::fixed_width_column_wrapper<T> out_col4{{4, 8, 12}};
    cudf::table_view expected{{out_col1, out_col2, out_col3, out_col4}};

    auto result = transpose(input);
    auto result_view = result->view();
    CUDF_EXPECTS(result_view.num_columns() == expected.num_columns(), "Expected same number of columns");
    for (cudf::size_type i = 0; i < result_view.num_columns(); ++i) {
        cudf::test::expect_columns_equal(result_view.column(i), expected.column(i));
    }
}

TYPED_TEST(TransposeTest, HasNulls)
{
    using T = TypeParam;

    cudf::test::fixed_width_column_wrapper<T> in_col1{{1, 2, 3}, {1, 0, 1}};
    cudf::test::fixed_width_column_wrapper<T> in_col2{{4, 5, 6}, {0, 1, 1}};
    cudf::test::fixed_width_column_wrapper<T> in_col3{{7, 8, 9}, {1, 0, 0}};
    cudf::test::fixed_width_column_wrapper<T> in_col4{{10, 11, 12}, {1, 1, 1}};
    cudf::table_view input{{in_col1, in_col2, in_col3, in_col4}};

    cudf::test::fixed_width_column_wrapper<T> out_col1{{1, 4, 7, 10}, {1, 0, 1, 1}};
    cudf::test::fixed_width_column_wrapper<T> out_col2{{2, 5, 8, 11}, {0, 1, 0, 1}};
    cudf::test::fixed_width_column_wrapper<T> out_col3{{3, 6, 9, 12}, {1, 1, 0, 1}};
    cudf::table_view expected{{out_col1, out_col2, out_col3}};

    auto result = transpose(input);
    auto result_view = result->view();
    CUDF_EXPECTS(result_view.num_columns() == expected.num_columns(), "Expected same number of columns");
    for (cudf::size_type i = 0; i < result_view.num_columns(); ++i) {
        cudf::test::expect_columns_equal(result_view.column(i), expected.column(i));
    }
}

TYPED_TEST(TransposeTest, EmptyTable)
{
    using T = TypeParam;

    cudf::table_view input{{}};

    cudf::table_view expected{{}};

    auto result = transpose(input);
    auto result_view = result->view();
    CUDF_EXPECTS(result_view.num_columns() == expected.num_columns(), "Expected same number of columns");
    for (cudf::size_type i = 0; i < result_view.num_columns(); ++i) {
        cudf::test::expect_columns_equal(result_view.column(i), expected.column(i));
    }
}

TYPED_TEST(TransposeTest, EmptyColumns)
{
    using T = TypeParam;

    cudf::test::fixed_width_column_wrapper<T> in_col1{};
    cudf::test::fixed_width_column_wrapper<T> in_col2{};
    cudf::test::fixed_width_column_wrapper<T> in_col3{};
    cudf::table_view input{{in_col1, in_col2, in_col3}};

    // Expect to get empty columns of same dimension as input
    cudf::table_view expected{{in_col1, in_col2, in_col3}};

    auto result = transpose(input);
    auto result_view = result->view();
    CUDF_EXPECTS(result_view.num_columns() == expected.num_columns(), "Expected same number of columns");
    for (cudf::size_type i = 0; i < result_view.num_columns(); ++i) {
        cudf::test::expect_columns_equal(result_view.column(i), expected.column(i));
    }
}
