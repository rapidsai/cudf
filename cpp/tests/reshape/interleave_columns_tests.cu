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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_list_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/reshape.hpp>

using namespace cudf::test;

template <typename T>
struct InterleaveColumnsTest : public BaseFixture {};

TYPED_TEST_CASE(InterleaveColumnsTest, cudf::test::AllTypes);

TYPED_TEST(InterleaveColumnsTest, NoColumns)
{
    using T = TypeParam;

    cudf::table_view in (std::vector<cudf::column_view>{ });

    EXPECT_THROW(cudf::experimental::interleave_columns(in), cudf::logic_error);
}

TYPED_TEST(InterleaveColumnsTest, OneColumn)
{
    using T = TypeParam;

    cudf::table_view in (std::vector<cudf::column_view>{
        fixed_width_column_wrapper<T>({ -1, 0, 1}),
    });

    auto expected = fixed_width_column_wrapper<T>({ -1, 0, 1});
    auto actual = cudf::experimental::interleave_columns(in);

    cudf::test::expect_columns_equal(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, TwoColumns)
{
    using T = TypeParam;

    auto a = fixed_width_column_wrapper<T>({ 0, 2 });
    auto b = fixed_width_column_wrapper<T>({ 1, 3 });

    cudf::table_view in (std::vector<cudf::column_view>{
        a,
        b,
    });

    auto expected = fixed_width_column_wrapper<T>({ 0, 1, 2, 3 });
    auto actual = cudf::experimental::interleave_columns(in);

    cudf::test::expect_columns_equal(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, ThreeColumns)
{
    using T = TypeParam;

    auto a = fixed_width_column_wrapper<T>({ 0, 3, 6 });
    auto b = fixed_width_column_wrapper<T>({ 1, 4, 7 });
    auto c = fixed_width_column_wrapper<T>({ 2, 5, 8 });

    cudf::table_view in (std::vector<cudf::column_view>{ a, b, c });

    auto expected = fixed_width_column_wrapper<T>({ 0, 1, 2, 3, 4, 5, 6, 7, 8 });
    auto actual = cudf::experimental::interleave_columns(in);

    cudf::test::expect_columns_equal(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, OneColumnEmpty)
{
    using T = TypeParam;

    cudf::table_view in (std::vector<cudf::column_view>{
        fixed_width_column_wrapper<T>({ })
    });

    auto expected = fixed_width_column_wrapper<T>({ });
    auto actual = cudf::experimental::interleave_columns(in);

    cudf::test::expect_columns_equal(expected, actual->view());
}
