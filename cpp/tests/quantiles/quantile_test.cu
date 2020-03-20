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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/quantiles.hpp>
#include <limits>
#include <memory>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_list_utilities.hpp>
#include <tests/utilities/type_lists.hpp>
#include <type_traits>

using namespace cudf::test;

using std::vector;
using cudf::experimental::bool8;
using cudf::null_order;
using cudf::order;

namespace {

struct q_res
{
    double value;
    bool valid = true;
};

// ----- test data -------------------------------------------------------------

namespace testdata {

    struct q_expect
    {
        q_expect(double quantile):
            quantile(quantile),
            higher{0, false}, lower{0, false}, linear{0, false}, midpoint{0, false}, nearest{0, false} { }

        q_expect(double quantile,
                 double higher, double lower, double linear, double midpoint, double nearest):
            quantile{quantile},
            higher{higher}, lower{lower}, linear{linear}, midpoint{midpoint}, nearest{nearest} { }

        double quantile;
        q_res higher;
        q_res lower;
        q_res linear;
        q_res midpoint;
        q_res nearest;
    };

template<typename T>
struct test_case {
    fixed_width_column_wrapper<T> column;
    vector<q_expect> expectations;
    fixed_width_column_wrapper<cudf::size_type> sortmap = {};
};

// interpolate_center

template<typename T>
test_case<T>
interpolate_center() {
    auto low = std::numeric_limits<T>::lowest();
    auto max = std::numeric_limits<T>::max();
    auto mid_d = std::is_floating_point<T>::value ? 0.0 : -0.5;

    // int64_t is internally casted to a double, meaning the lerp center point
    // is float-like.
    auto lin_d = std::is_floating_point<T>::value ||
                 std::is_same<T, int64_t>::value ? 0.0 : -0.5;
    auto max_d = static_cast<double>(max);
    auto low_d = static_cast<double>(low);
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ low, max }),
        {
            q_expect{ 0.50, max_d, low_d, lin_d, mid_d, low_d }
        }
    };
}

template<>
test_case<bool8>
interpolate_center() {
    auto low = std::numeric_limits<bool8>::lowest();
    auto max = std::numeric_limits<bool8>::max();
    auto mid_d = 0.5;
    auto low_d = static_cast<double>(low);
    auto max_d = static_cast<double>(max);
    return test_case<bool8> {
        fixed_width_column_wrapper<bool8> ({ low, max }),
        {
            q_expect{ 0.5, max_d, low_d, mid_d, mid_d, low_d }
        }
    };
}

// interpolate_extrema_high

template<typename T>
test_case<T>
interpolate_extrema_high() {
    T max = std::numeric_limits<T>::max();
    T low = max - 2;
    auto low_d = static_cast<double>(low);
    auto max_d = static_cast<double>(max);
    auto exact_d = static_cast<double>(max - 1);
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ low, max }),
        {
            q_expect{ 0.50, max_d, low_d, exact_d, exact_d, low_d }
        }
    };
}

template<>
test_case<bool8>
interpolate_extrema_high<bool8>() {
    return interpolate_center<bool8>();
}

// interpolate_extrema_low

template<typename T>
test_case<T>
interpolate_extrema_low() {
    T lowest = std::numeric_limits<T>::lowest();
    T a = lowest;
    T b = lowest + 2;
    auto a_d = static_cast<double>(a);
    auto b_d = static_cast<double>(b);
    auto exact_d = static_cast<double>(a + 1);
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ a, b }),
        {
            q_expect{ 0.50, b_d, a_d, exact_d, exact_d, a_d }
        }
    };
}

template<>
test_case<bool8>
interpolate_extrema_low<bool8>() {
    return interpolate_center<bool8>();
}

// single

template<typename T>
std::enable_if_t<std::is_floating_point<T>::value, test_case<T>>
single() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 7.309999942779541 }),
        {
            q_expect{ -1.0, 7.309999942779541, 7.309999942779541, 7.309999942779541, 7.309999942779541, 7.309999942779541, },
            q_expect{  0.0, 7.309999942779541, 7.309999942779541, 7.309999942779541, 7.309999942779541, 7.309999942779541, },
            q_expect{  1.0, 7.309999942779541, 7.309999942779541, 7.309999942779541, 7.309999942779541, 7.309999942779541, },
        }
    };
}

template<typename T>
std::enable_if_t<std::is_integral<T>::value and not cudf::is_boolean<T>(), test_case<T>>
single() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 1 }),
        {
            q_expect{ 0.7, 1, 1, 1, 1, 1 }
        }
    };
}

template<typename T>
std::enable_if_t<cudf::is_boolean<T>(), test_case<T>>
single() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 1 }),
        {
            q_expect{ 0.7, 1.0, 1.0, 1.0, 1.0, 1.0 }
        }
    };
}

// all_invalid

template<typename T>
std::enable_if_t<std::is_floating_point<T>::value, test_case<T>>
all_invalid() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 0.375, 0.15, 3.4, 4.17, 2.13, 1.11, -1.01, 0.8, 5.7 },
                                       { 0,     0,    0,   0,    0,    0,     0,    0,   0 }),
        {
            q_expect{ -1.0 },
            q_expect{  0.0 },
            q_expect{  0.5 },
            q_expect{  1.0 },
            q_expect{  2.0 }
        }
    };
}

template<typename T>
std::enable_if_t<std::is_integral<T>::value and not cudf::is_boolean<T>(), test_case<T>>
all_invalid() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 6, 0, 3, 4, 2, 1, -1, 1, 6 },
                                       { 0, 0, 0, 0, 0, 0,  0, 0, 0}),
        {
            q_expect{ 0.7 }
        }
    };
}

template<typename T>
std::enable_if_t<cudf::is_boolean<T>(), test_case<T>>
all_invalid() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 1, 0, 1, 1, 0, 1, 0, 1, 1 },
                                       { 0, 0, 0, 0, 0, 0, 0, 0, 0 }),
        {
            q_expect{ 0.7 }
        }
    };
}

// unsorted without sortmap

template<typename T>
std::enable_if_t<std::is_floating_point<T>::value, test_case<T>>
unsorted_no_sortmap() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 0.375, 0.15, 3.4, 4.17, 2.13, 1.11, -1.00, 0.8, 5.7 }),
        {
            q_expect{ 0.0, 0.375, 0.375, 0.375, 0.375, 0.375 },
        }
    };
}

template<typename T>
std::enable_if_t<std::is_integral<T>::value and not cudf::is_boolean<T>(), test_case<T>>
unsorted_no_sortmap() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 6, 0, 3, 4, 2, 1, -1, 1, 6 }),
        {
            q_expect{ 0.0, 6, 6, 6, 6, 6 }
        }
    };
}

template<typename T>
std::enable_if_t<cudf::is_boolean<T>(), test_case<T>>
unsorted_no_sortmap() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 1, 0, 1, 1, 0, 1, 1, 0, 1 }),
        {
            q_expect{ 0.0, 1.0, 1.0, 1.0, 1.0, 1.0 }
        }
    };
}

// unsorted with sortmap

template<typename T>
std::enable_if_t<std::is_floating_point<T>::value, test_case<T>>
unsorted_with_sortmap() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ .375, 1, -1, 2 }),
        {
            q_expect{ 0.0, -1.00, -1.00, -1.00, -1.00, -1.00 },
            q_expect{ 2.0,  2.00,  2.00,  2.00,  2.00,  2.00 }
        },
        fixed_width_column_wrapper<cudf::size_type>({ 2, 0, 1, 3 })
    };
}

template<typename T>
std::enable_if_t<std::is_integral<T>::value and not cudf::is_boolean<T>(), test_case<T>>
unsorted_with_sortmap() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 6, 0, 3, 4, 2, 1, -1, 1, 6 }),
        {
            q_expect{ 0.0, -1, -1, -1, -1, -1 },
            q_expect{ 1.0,  6,  6,  6,  6,  6 }
        },
        fixed_width_column_wrapper<cudf::size_type> ({ 6, 1, 5, 7, 4, 2, 3, 0, 8 })
    };
}

template<typename T>
std::enable_if_t<cudf::is_boolean<T>(), test_case<T>>
unsorted_with_sortmap() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 1, 0, 1, 1, 0, 1, 1, 0, 1 }),
        {
            q_expect{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
            q_expect{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }
        },
        fixed_width_column_wrapper<cudf::size_type> ({ 1, 4, 7, 0, 2, 3, 5, 6, 8 })
    };
}

} // namespace testdata

// =============================================================================
// ----- helper functions ------------------------------------------------------

template<typename T>
void test(testdata::test_case<T> test_case) {
    using namespace cudf::experimental;

    for (testdata::q_expect & expected : test_case.expectations) {
        auto input = cudf::table_view({ test_case.column });

        auto actual_higher = quantiles(input, { expected.quantile }, interpolation::HIGHER, test_case.sortmap);
        auto expected_higher_a = expected.higher.valid
            ? fixed_width_column_wrapper<double>({ expected.higher.value })
            : fixed_width_column_wrapper<double>({ expected.higher.value }, { 0 });
        auto expected_higher = cudf::table_view({ expected_higher_a });
        expect_tables_equal(expected_higher, *actual_higher);

        auto actual_lower = quantiles(input, { expected.quantile }, interpolation::LOWER, test_case.sortmap);
        auto expected_lower_a = expected.higher.valid
            ? fixed_width_column_wrapper<double>({ expected.lower.value })
            : fixed_width_column_wrapper<double>({ expected.lower.value }, { 0 });
        auto expected_lower = cudf::table_view({ expected_lower_a });
        expect_tables_equal(expected_lower, *actual_lower);

        auto actual_linear = quantiles(input, { expected.quantile }, interpolation::LINEAR, test_case.sortmap);
        auto expected_linear_a = expected.linear.valid
            ? fixed_width_column_wrapper<double>({ expected.linear.value })
            : fixed_width_column_wrapper<double>({ expected.linear.value }, { 0 });
        auto expected_linear = cudf::table_view({ expected_linear_a });
        expect_tables_equal(expected_linear, *actual_linear);

        auto actual_midpoint = quantiles(input, { expected.quantile }, interpolation::MIDPOINT, test_case.sortmap);
        auto expected_midpoint_a = expected.midpoint.valid
            ? fixed_width_column_wrapper<double>({ expected.midpoint.value })
            : fixed_width_column_wrapper<double>({ expected.midpoint.value }, { 0 });
        auto expected_midpoint = cudf::table_view({ expected_midpoint_a });
        expect_tables_equal(expected_midpoint, *actual_midpoint);

        auto actual_nearest = quantiles(input, { expected.quantile }, interpolation::NEAREST, test_case.sortmap);
        auto expected_nearest_a = expected.nearest.valid
            ? fixed_width_column_wrapper<double>({ expected.nearest.value })
            : fixed_width_column_wrapper<double>({ expected.nearest.value }, { 0 });
        auto expected_nearest = cudf::table_view({ expected_nearest_a });
        expect_tables_equal(expected_nearest, *actual_nearest);
    }
}

// =============================================================================
// ----- tests -----------------------------------------------------------------

template <typename T>
struct QuantileTest : public BaseFixture {
};

using TestTypes = NumericTypes;
// using TestTypes = cudf::test::Types<int32_t>;
TYPED_TEST_CASE(QuantileTest, TestTypes);

TYPED_TEST(QuantileTest, TestEmpty)
{
    auto input_a = fixed_width_column_wrapper<TypeParam>({ });
    auto input = cudf::table_view({ input_a });

    EXPECT_THROW(cudf::experimental::quantiles(input, { 0 }),
                 cudf::logic_error);
}

TYPED_TEST(QuantileTest, TestSingle)
{
    test(testdata::single<TypeParam>());
}

TYPED_TEST(QuantileTest, TestAllElementsInvalid)
{
    test(testdata::all_invalid<TypeParam>());
}

TYPED_TEST(QuantileTest, TestInterpolateInvalids)
{
    using T = TypeParam;
    auto input_a = fixed_width_column_wrapper<T>({ 0, 1, 0, 1 }, { 0, 1, 1, 0 });
    auto input = cudf::table_view({ input_a });

    auto actual_higher   = cudf::experimental::quantiles(input, { 0.25, 0.50, 0.75 });
    auto expected_higher_col = fixed_width_column_wrapper<double>({ 1.0, 0.0, 1.0 }, { 1, 1, 0 });
    auto expected_higher = cudf::table_view({ expected_higher_col });

    auto actual_lower    = cudf::experimental::quantiles(input, { 0.25, 0.50, 0.75 });
    auto expected_lower_col = fixed_width_column_wrapper<double>({ 0.0, 1.0, 0.0 }, { 0, 1, 1 });
    auto expected_lower = cudf::table_view({ expected_lower_col });

    auto actual_linear   = cudf::experimental::quantiles(input, { 0.25, 0.50, 0.75 });
    auto expected_linear_col = fixed_width_column_wrapper<double>({ 0.5, 0.5, 0.5 }, { 0, 1, 0 });
    auto expected_linear = cudf::table_view({ expected_linear_col });

    auto actual_midpoint = cudf::experimental::quantiles(input, { 0.25, 0.50, 0.75 });
    auto expected_midpoint_col = fixed_width_column_wrapper<double>({ 0.5, 0.5, 0.5 }, { 0, 1, 0 });
    auto expected_midpoint = cudf::table_view({ expected_midpoint_col });

    auto actual_nearest  = cudf::experimental::quantiles(input, { 0.25, 0.50, 0.75 });
    auto expected_neareset_col = fixed_width_column_wrapper<double>({ 1.0, 0.0, 1.0 }, { 1, 1, 0 });
    auto expected_neareset = cudf::table_view({ expected_neareset_col });
}

TYPED_TEST(QuantileTest, TestUnsortedNoSortmap)
{
    test(testdata::unsorted_no_sortmap<TypeParam>());
}

TYPED_TEST(QuantileTest, TestUnsortedWithSortmap)
{
    test(testdata::unsorted_with_sortmap<TypeParam>());
}

TYPED_TEST(QuantileTest, TestInterpolateCenter)
{
    test(testdata::interpolate_center<TypeParam>());
}

TYPED_TEST(QuantileTest, TestInterpolateExtremaHigh)
{
    test(testdata::interpolate_extrema_high<TypeParam>());
}

TYPED_TEST(QuantileTest, TestInterpolateExtremaLow)
{
    test(testdata::interpolate_extrema_low<TypeParam>());
}

TYPED_TEST(QuantileTest, TestUnsortedInputWithoutSortmap)
{
    auto a_val = std::numeric_limits<TypeParam>::lowest();
    auto b_val = std::numeric_limits<TypeParam>::max();

    auto input_a = fixed_width_column_wrapper<TypeParam>({ b_val, a_val });
    auto input = cudf::table_view({ input_a });

    std::unique_ptr<cudf::experimental::table> actual;
    EXPECT_NO_THROW(actual = cudf::experimental::quantiles(input, { 0 }));
    auto expected_q = fixed_width_column_wrapper<double>({ static_cast<double>(b_val) });
    auto expected = cudf::table_view({ expected_q });
    expect_tables_equal(expected, actual->view());
}

template <typename T>
struct QuantileUnsupportedTypesTest : public BaseFixture {
};

using UnsupportedTestTypes = RemoveIf<ContainedIn<TestTypes>, AllTypes>;
TYPED_TEST_CASE(QuantileUnsupportedTypesTest, UnsupportedTestTypes);

TYPED_TEST(QuantileUnsupportedTypesTest, TestZeroElements)
{
    auto input_a = fixed_width_column_wrapper<TypeParam>({ });
    auto input = cudf::table_view({ input_a });

    EXPECT_THROW(cudf::experimental::quantiles(input, { 0 }),
                 cudf::logic_error);
}

TYPED_TEST(QuantileUnsupportedTypesTest, TestOneElements)
{
    auto input_a = fixed_width_column_wrapper<TypeParam>({ 0 });
    auto input = cudf::table_view({ input_a });

    EXPECT_THROW(cudf::experimental::quantiles(input, { 0 }),
                 cudf::logic_error);
}

TYPED_TEST(QuantileUnsupportedTypesTest, TestMultipleElements)
{
    auto input_a = fixed_width_column_wrapper<TypeParam>({ 0, 1, 2 });
    auto input = cudf::table_view({ input_a });

    EXPECT_THROW(cudf::experimental::quantiles(input, { 0 }),
                 cudf::logic_error);
}

} // anonymous namespace
