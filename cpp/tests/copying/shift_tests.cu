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

#include <memory>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <cudf/copying.hpp>
#include <gtest/gtest-typed-test.h>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <cudf/scalar/scalar.hpp>
#include "cudf/scalar/scalar_device_view.cuh"
#include "cudf/scalar/scalar_factories.hpp"
#include "cudf/types.hpp"

using cudf::test::fixed_width_column_wrapper;
using TestTypes = cudf::test::Types<int32_t>;

template <typename T>
struct ShiftTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(ShiftTest, TestTypes);

// TYPED_TEST(ShiftTest, OneColumnEmpty)
// {
//     using T =  TypeParam;

//     fixed_width_column_wrapper<T> a{};

//     cudf::table_view input{ { a } };
//     cudf::experimental::shift(input, 5, {});
// }

// TYPED_TEST(ShiftTest, TwoColumnsEmpty)
// {
//     using T =  TypeParam;

//     fixed_width_column_wrapper<T> a{};
//     fixed_width_column_wrapper<T> b{};

//     cudf::table_view input{ { a, b } };
//     cudf::experimental::shift(input, 5, {});
// }

TYPED_TEST(ShiftTest, OneColumn)
{
    using T =  TypeParam;

    auto input_a = fixed_width_column_wrapper<T>{ 1, 2, 3, 4, 5 };
    auto input = cudf::table_view { { input_a } };

    auto expected_a = fixed_width_column_wrapper<T>{ 7, 7, 1, 2, 3 };
    auto expected = cudf::table_view{ { expected_a } };
    
    auto fill = std::make_unique<cudf::numeric_scalar<T>>(7, true);
    auto fills = std::vector<cudf::scalar> { *fill };
    // auto actual = cudf::experimental::shift(input, 2, fills);

    // cudf::test::expect_tables_equal(expected, *actual);
}
