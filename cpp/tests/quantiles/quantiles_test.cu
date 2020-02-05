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

#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/quantiles.hpp>
#include <limits>
#include <memory>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/scalar_utilities.hpp>
#include <tests/utilities/type_list_utilities.hpp>
#include <tests/utilities/type_lists.hpp>
#include <type_traits>

using namespace cudf::test;

using std::vector;
using cudf::experimental::bool8;
using cudf::null_order;
using cudf::order;

namespace {

using q_res = cudf::numeric_scalar<double>;

// ----- test data -------------------------------------------------------------

namespace testdata {

    struct q_expect
    {
        q_expect(double quantile):
            quantile(quantile),
            higher(0, false), lower(0, false), linear(0, false), midpoint(0, false), nearest(0, false) { }

        q_expect(double quantile,
                 double higher, double lower, double linear, double midpoint, double nearest):
            quantile(quantile),
            higher(higher), lower(lower), linear(linear), midpoint(midpoint), nearest(nearest) { }

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
    cudf::order_info column_order;
};

// empty

template<typename T>
test_case<T>
empty() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ }),
        {
            q_expect{ -1.0 },
            q_expect{  0.0 },
            q_expect{  0.5 },
            q_expect{  1.0 },
            q_expect{  2.0 }
        }
    };
}

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

// sorted_ascending_null_before

template<typename T>
std::enable_if_t<std::is_floating_point<T>::value, test_case<T>>
sorted_ascending_null_before() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 1, 2, 3, 4, 5, 6, 7, 8, 9 },
                                       { 0, 0, 0, 0, 0, 1, 1, 1, 1 }),
        {
            q_expect{ 0.00, 6, 6, 6, 6, 6 },
            q_expect{ 0.75, 9, 8, 8.25, 8.5, 8 },
            q_expect{ 1.00, 9, 9, 9, 9, 9 }
        },
        { true, cudf::order::ASCENDING, cudf::null_order::BEFORE }
    };
}

template<typename T>
std::enable_if_t<std::is_integral<T>::value and not cudf::is_boolean<T>(), test_case<T>>
sorted_ascending_null_before() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 1, 2, 3, 4, 5, 6, 7, 8, 9 },
                                       { 0, 0, 0, 0, 0, 1, 1, 1, 1 }),
        {
            q_expect{ 0.00, 6, 6, 6, 6, 6 },
            q_expect{ 0.50, 8, 7, 7.5, 7.5, 8 },
            q_expect{ 1.00, 9, 9, 9, 9, 9 }
        },
        { true, cudf::order::ASCENDING, cudf::null_order::BEFORE }
    };
}

template<typename T>
std::enable_if_t<cudf::is_boolean<T>(), test_case<T>>
sorted_ascending_null_before() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 1, 0, 1, },
                                       { 0, 1, 1, }),
        {
            q_expect{ 0.00, 0, 0, 0, 0, 0 },
            q_expect{ 0.50, 1, 0, 0.5, 0.5, 0 },
            q_expect{ 1.50, 1, 1, 1, 1, 1 }
        },
        { true, cudf::order::ASCENDING, cudf::null_order::BEFORE }
    };
}

// sorted_descending_null_after

template<typename T>
std::enable_if_t<std::is_floating_point<T>::value, test_case<T>>
sorted_descending_null_after() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 9, 8, 7, 6, 5, 4, 3, 2, 1 },
                                       { 1, 1, 1, 1, 0, 0, 0, 0, 0 }),
        {
            q_expect{ 0.00, 6, 6, 6, 6, 6 },
            q_expect{ 0.75, 9, 8, 8.25, 8.5, 8 },
            q_expect{ 1.00, 9, 9, 9, 9, 9 }
        },
        { true, cudf::order::DESCENDING, cudf::null_order::AFTER }
    };
}

template<typename T>
std::enable_if_t<std::is_integral<T>::value and not cudf::is_boolean<T>(), test_case<T>>
sorted_descending_null_after() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 9, 8, 7, 6, 5, 4, 3, 2, 1 },
                                       { 1, 1, 1, 1, 0, 0, 0, 0, 0 }),
        {
            q_expect{ 0.00, 6, 6, 6, 6, 6 },
            q_expect{ 0.50, 8, 7, 7.5, 7.5, 8 },
            q_expect{ 1.00, 9, 9, 9, 9, 9 }
        },
        { true, cudf::order::DESCENDING, cudf::null_order::AFTER }
    };
}

template<typename T>
std::enable_if_t<cudf::is_boolean<T>(), test_case<T>>
sorted_descending_null_after() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 1, 0, 1, },
                                       { 1, 1, 0, }),
        {
            q_expect{ 0.50, 1, 0, 0.5, 0.5, 0 }
        },
        { true, cudf::order::DESCENDING, cudf::null_order::AFTER }
    };
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
        fixed_width_column_wrapper<T> ({ 6.8, 0.15, 3.4, 4.17, 2.13, 1.11, -1.01, 0.8, 5.7 },
                                       { 0,      0,   0,    0,    0,    0,     0,   0,   0 }),
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

// some invalid

template<typename T>
std::enable_if_t<std::is_same<T, double>::value, test_case<T>>
some_invalid() {
    T high = 0.16;
    T low = -1.024;
    T mid = -0.432;
    T lin = -0.432;
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 6.8, high, 3.4, 4.17, 2.13, 1.11, low, 0.8, 5.7 },
                                       { 0,      1,   0,    0,    0,    0,   1,   0,   0 }),
        {
            q_expect{ -1.0, low,  low,  low,  low,  low },
            q_expect{  0.0, low,  low,  low,  low,  low },
            q_expect{  0.5, high, low,  lin,  mid,  low },
            q_expect{  1.0, high, high, high, high, high },
            q_expect{  2.0, high, high, high, high, high }
        }
    };
}

template<typename T>
std::enable_if_t<std::is_same<T, float>::value, test_case<T>>
some_invalid() {
    T high = 0.16;
    T low = -1.024;
    double mid = -0.43200002610683441;
    double lin = -0.43200002610683441;
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 6.8, high, 3.4, 4.17, 2.13, 1.11, low, 0.8, 5.7 },
                                       { 0,      1,   0,    0,    0,    0,   1,   0,   0 }),
        {
            q_expect{ -1.0, low,  low,  low,  low,  low },
            q_expect{  0.0, low,  low,  low,  low,  low },
            q_expect{  0.5, high, low,  lin,  mid,  low },
            q_expect{  1.0, high, high, high, high, high },
            q_expect{  2.0, high, high, high, high, high }
        }
    };
}

template<typename T>
std::enable_if_t<std::is_integral<T>::value and not cudf::is_boolean<T>(), test_case<T>>
some_invalid() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 6, 0, 3, 4, 2, 1, -1, 1, 6 },
                                       { 0, 0, 1, 0, 0, 0,  0, 0, 1}),
        {
            q_expect{ 0.0, 3.0, 3.0, 3.0, 3.0, 3.0 },
            q_expect{ 0.5, 6.0, 3.0, 4.5, 4.5, 3.0 },
            q_expect{ 1.0, 6.0, 6.0, 6.0, 6.0, 6.0 }
        }
    };
}

template<typename T>
std::enable_if_t<cudf::is_boolean<T>(), test_case<T>>
some_invalid() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 1, 0, 1, 1, 0, 1, 0, 1, 1 },
                                       { 0, 0, 1, 0, 1, 0, 0, 0, 0}),
        {
            q_expect{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
            q_expect{ 0.5, 1.0, 0.0, 0.5, 0.5, 0.0 },
            q_expect{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }
        }
    };
}

// unsorted

template<typename T>
std::enable_if_t<std::is_floating_point<T>::value, test_case<T>>
unsorted() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 6.8, 0.15, 3.4, 4.17, 2.13, 1.11, -1.00, 0.8, 5.7 }),
        {
            q_expect{ 0.0, -1.00, -1.00, -1.00, -1.00, -1.00 },
        }
    };
}

template<typename T>
std::enable_if_t<std::is_integral<T>::value and not cudf::is_boolean<T>(), test_case<T>>
unsorted() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 6, 0, 3, 4, 2, 1, -1, 1, 6 }),
        {
            q_expect{ 0.0, -1, -1, -1, -1, -1 }
        }
    };
}

template<typename T>
std::enable_if_t<cudf::is_boolean<T>(), test_case<T>>
unsorted() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 0, 0, 1, 1, 0, 1, 1, 0, 1 }),
        {
            q_expect{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,}
        }
    };
}

} // namespace testdata

// =============================================================================
// ----- helper functions ------------------------------------------------------

template<typename T>
void test(testdata::test_case<T> test_case) {
    using namespace cudf::experimental;

    for (auto & expected : test_case.expectations) {

        auto actual_higher = quantile(test_case.column, expected.quantile, interpolation::HIGHER, test_case.column_order);
        expect_scalars_equal(expected.higher, *actual_higher);

        auto actual_lower = quantile(test_case.column, expected.quantile, interpolation::LOWER, test_case.column_order);
        expect_scalars_equal(expected.lower, *actual_lower);

        auto actual_linear = quantile(test_case.column, expected.quantile, interpolation::LINEAR, test_case.column_order);
        expect_scalars_equal(expected.linear, *actual_linear);

        auto actual_midpoint = quantile(test_case.column, expected.quantile, interpolation::MIDPOINT, test_case.column_order);
        expect_scalars_equal(expected.midpoint, *actual_midpoint);

        auto actual_nearest = quantile(test_case.column, expected.quantile, interpolation::NEAREST, test_case.column_order);
        expect_scalars_equal(expected.nearest, *actual_nearest);
    }
}

// =============================================================================
// ----- tests -----------------------------------------------------------------

template <typename T>
struct QuantileTest : public BaseFixture {
};

using TestTypes = NumericTypes;
TYPED_TEST_CASE(QuantileTest, TestTypes);

TYPED_TEST(QuantileTest, TestEmpty)
{
    test(testdata::empty<TypeParam>());
}

TYPED_TEST(QuantileTest, TestSingle)
{
    test(testdata::single<TypeParam>());
}

TYPED_TEST(QuantileTest, TestSomeElementsInvalid)
{
    test(testdata::some_invalid<TypeParam>());
}

TYPED_TEST(QuantileTest, TestAllElementsInvalid)
{
    test(testdata::all_invalid<TypeParam>());
}

TYPED_TEST(QuantileTest, TestUnsorted)
{
    test(testdata::unsorted<TypeParam>());
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

TYPED_TEST(QuantileTest, TestSortedAscendingNullBefore)
{
    test(testdata::sorted_ascending_null_before<TypeParam>());
}

TYPED_TEST(QuantileTest, TestSortedDescendingNullAfter)
{
    test(testdata::sorted_descending_null_after<TypeParam>());
}

TYPED_TEST(QuantileTest, TestImplicitlyUnsortedInputs)
{
    auto a_val = std::numeric_limits<TypeParam>::lowest();
    auto b_val = std::numeric_limits<TypeParam>::max();

    fixed_width_column_wrapper<TypeParam> input ({ b_val, a_val });

    std::unique_ptr<cudf::scalar> q_actual;
    EXPECT_NO_THROW(q_actual = cudf::experimental::quantile(input, 0));
    auto q_expected = q_res(a_val);
    expect_scalars_equal(q_expected, *q_actual);
}

template <typename T>
struct QuantileUnsupportedTypesTest : public BaseFixture {
};

using UnsupportedTestTypes = RemoveIf<ContainedIn<TestTypes>, AllTypes>;
TYPED_TEST_CASE(QuantileUnsupportedTypesTest, UnsupportedTestTypes);

TYPED_TEST(QuantileUnsupportedTypesTest, TestZeroElements)
{
    fixed_width_column_wrapper<TypeParam> input ({ });

    EXPECT_THROW(cudf::experimental::quantile(input, 0),
                 cudf::logic_error);
}

TYPED_TEST(QuantileUnsupportedTypesTest, TestOneElements)
{
    fixed_width_column_wrapper<TypeParam> input ({ 0 });

    EXPECT_THROW(cudf::experimental::quantile(input, 0),
                 cudf::logic_error);
}

TYPED_TEST(QuantileUnsupportedTypesTest, TestMultipleElements)
{
    fixed_width_column_wrapper<TypeParam> input ({ 0, 1, 2 });

    EXPECT_THROW(cudf::experimental::quantile(input, 0),
                 cudf::logic_error);
}

} // anonymous namespace
