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

#include <cudf/cudf.h>
#include <tests/utilities/base_fixture.hpp>
#include <cudf/unary.hpp>
#include <cudf/column/column_factories.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <cudf/legacy/interop.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <vector>


template <typename T>
cudf::test::fixed_width_column_wrapper<T> create_fixed_columns(cudf::size_type start, cudf::size_type size, bool nullable) {
    auto iter = cudf::test::make_counting_transform_iterator(start, [](auto i) { return T(i);});

    if(not nullable) {
        return cudf::test::fixed_width_column_wrapper<T> (iter, iter + size);
    } else {
        auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i%2==0? true:false; });
        return  cudf::test::fixed_width_column_wrapper<T> (iter, iter + size, valids);
    }

}

template <typename T>
cudf::test::fixed_width_column_wrapper<T> create_expected_columns(cudf::size_type size, bool nullable, bool nulls_to_be) {

    if(not nullable) {
        auto iter = cudf::test::make_counting_transform_iterator(0, [nulls_to_be](auto i) { return not nulls_to_be;});
        return cudf::test::fixed_width_column_wrapper<T> (iter, iter + size);
    } else {
        auto iter = cudf::test::make_counting_transform_iterator(0, [nulls_to_be](auto i) { return i%2==0? not nulls_to_be: nulls_to_be; });
        return cudf::test::fixed_width_column_wrapper<T> (iter, iter + size);
    }
}

template <typename T>
struct cudf_logical_test : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(cudf_logical_test, cudf::test::NumericTypes);

TYPED_TEST(cudf_logical_test, LogicalNot)
{
    cudf::size_type colSize = 1000;
    std::vector<TypeParam> h_input_v(colSize, false);
    std::vector<cudf::experimental::bool8> h_expect_v(colSize);

    std::transform(
        std::cbegin(h_input_v),
        std::cend(h_input_v),
        std::begin(h_expect_v),
        [] (TypeParam e) -> cudf::bool8 {
            return static_cast<cudf::bool8>(!e);
        });

    cudf::test::fixed_width_column_wrapper<TypeParam>                 input    (std::cbegin(h_input_v),  std::cend(h_input_v));
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected (std::cbegin(h_expect_v), std::cend(h_expect_v));

    auto output = cudf::experimental::unary_operation(input, cudf::experimental::unary_op::NOT);

    cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_logical_test, SimpleLogicalNot)
{
    cudf::test::fixed_width_column_wrapper<TypeParam>                 input    {{ true,  true,  true,  true  }};
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected {{ false, false, false, false }};
    auto output = cudf::experimental::unary_operation(input, cudf::experimental::unary_op::NOT);
    cudf::test::expect_columns_equal(expected, output->view());
}

template <typename T>
struct cudf_math_test : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(cudf_math_test, cudf::test::NumericTypes);

TYPED_TEST(cudf_math_test, ABS)
{
    using T = TypeParam;

    cudf::size_type colSize = 1000;
    std::vector<T> h_input_v(colSize);
    std::vector<T> h_expect_v(colSize);

    std::iota(
        std::begin(h_input_v),
        std::end(h_input_v),
        -1 * colSize);

    std::transform(
        std::cbegin(h_input_v),
        std::cend(h_input_v),
        std::begin(h_expect_v),
        [] (auto e) { return std::abs(e); });

    cudf::test::fixed_width_column_wrapper<T> input    (std::cbegin(h_input_v),  std::cend(h_input_v));
    cudf::test::fixed_width_column_wrapper<T> expected (std::cbegin(h_expect_v), std::cend(h_expect_v));

    auto output = cudf::experimental::unary_operation(input, cudf::experimental::unary_op::ABS);

    cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_test, SQRT)
{
    using T = TypeParam;

    cudf::size_type colSize = 175;
    std::vector<T> h_input_v(colSize);
    std::vector<T> h_expect_v(colSize);

    std::generate(
        std::begin(h_input_v),
        std::end(h_input_v),
        [i = 0] () mutable { ++i; return i * i; });

    std::transform(
        std::cbegin(h_input_v),
        std::cend(h_input_v),
        std::begin(h_expect_v),
        [] (auto e) { return std::sqrt(static_cast<float>(e)); });

    cudf::test::fixed_width_column_wrapper<T> input    (std::cbegin(h_input_v),  std::cend(h_input_v));
    cudf::test::fixed_width_column_wrapper<T> expected (std::cbegin(h_expect_v), std::cend(h_expect_v));

    auto output = cudf::experimental::unary_operation(input, cudf::experimental::unary_op::SQRT);

    cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_test, SimpleABS)
{
    cudf::test::fixed_width_column_wrapper<TypeParam> input    {{ -2, -1, 1, 2 }};
    cudf::test::fixed_width_column_wrapper<TypeParam> expected {{  2,  1, 1, 2 }};
    auto output = cudf::experimental::unary_operation(input, cudf::experimental::unary_op::ABS);
    cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_test, SimpleSQRT)
{
    cudf::test::fixed_width_column_wrapper<TypeParam> input    {{ 1, 4, 9, 16 }};
    cudf::test::fixed_width_column_wrapper<TypeParam> expected {{ 1, 2, 3, 4  }};
    auto output = cudf::experimental::unary_operation(input, cudf::experimental::unary_op::SQRT);
    cudf::test::expect_columns_equal(expected, output->view());
}

template <typename T>
struct cudf_math_with_floating_point_test : public cudf::test::BaseFixture {};

using floating_point_type_list = ::testing::Types<float, double>;

TYPED_TEST_CASE(cudf_math_with_floating_point_test, floating_point_type_list);

TYPED_TEST(cudf_math_with_floating_point_test, SimpleSIN)
{
    cudf::test::fixed_width_column_wrapper<TypeParam> input    {{ 0.0 }};
    cudf::test::fixed_width_column_wrapper<TypeParam> expected {{ 0.0 }};
    auto output = cudf::experimental::unary_operation(input, cudf::experimental::unary_op::SIN);
    cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_with_floating_point_test, SimpleCOS)
{
    cudf::test::fixed_width_column_wrapper<TypeParam> input    {{ 0.0 }};
    cudf::test::fixed_width_column_wrapper<TypeParam> expected {{ 1.0 }};
    auto output = cudf::experimental::unary_operation(input, cudf::experimental::unary_op::COS);
    cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_with_floating_point_test, SimpleFLOOR)
{
    cudf::test::fixed_width_column_wrapper<TypeParam> input    {{ 1.1, 3.3, 5.5, 7.7 }};
    cudf::test::fixed_width_column_wrapper<TypeParam> expected {{ 1.0, 3.0, 5.0, 7.0 }};
    auto output = cudf::experimental::unary_operation(input, cudf::experimental::unary_op::FLOOR);
    cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_with_floating_point_test, SimpleCEIL)
{
    cudf::test::fixed_width_column_wrapper<TypeParam> input    {{ 1.1, 3.3, 5.5, 7.7 }};
    cudf::test::fixed_width_column_wrapper<TypeParam> expected {{ 2.0, 4.0, 6.0, 8.0 }};
    auto output = cudf::experimental::unary_operation(input, cudf::experimental::unary_op::CEIL);
    cudf::test::expect_columns_equal(expected, output->view());
}

template <typename T>
struct IsNull : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(IsNull, cudf::test::NumericTypes);

TYPED_TEST(IsNull, AllValid)
{
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 10;
    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, false);
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = create_expected_columns<cudf::experimental::bool8>(size, false, true);

    std::unique_ptr<cudf::column> got = cudf::experimental::is_null(col);

    cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNull, WithInvalids)
{
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 10;
    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, true);
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = create_expected_columns<cudf::experimental::bool8>(size, true, true);

    std::unique_ptr<cudf::column> got = cudf::experimental::is_null(col);

    cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNull, EmptyColumns)
{
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 0;
    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, true);
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = create_expected_columns<cudf::experimental::bool8>(size, true, true);

    std::unique_ptr<cudf::column> got = cudf::experimental::is_null(col);

    cudf::test::expect_columns_equal(expected, got->view());
}

template <typename T>
struct IsNotNull : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(IsNotNull, cudf::test::NumericTypes);

TYPED_TEST(IsNotNull, AllValid)
{
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 10;
    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, false);
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = create_expected_columns<cudf::experimental::bool8>(size, false, false);

    std::unique_ptr<cudf::column> got = cudf::experimental::is_valid(col);

    cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNotNull, WithInvalids)
{
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 10;
    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, true);
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = create_expected_columns<cudf::experimental::bool8>(size, true, false);

    std::unique_ptr<cudf::column> got = cudf::experimental::is_valid(col);

    cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNotNull, EmptyColumns)
{
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 0;
    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, true);
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = create_expected_columns<cudf::experimental::bool8>(size, true, false);

    std::unique_ptr<cudf::column> got = cudf::experimental::is_valid(col);

    cudf::test::expect_columns_equal(expected, got->view());
}

