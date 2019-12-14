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
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/scalar_utilities.hpp>
#include <tests/utilities/type_list_utilities.hpp>
#include <tests/utilities/type_lists.hpp>
#include <type_traits>
#include "cudf/scalar/scalar_factories.hpp"
#include "cudf/utilities/error.hpp"
#include "cudf/utilities/legacy/wrapper_types.hpp"
#include "cudf/utilities/traits.hpp"
#include "cudf/wrappers/bool.hpp"
#include "cudf/wrappers/timestamps.hpp"

using std::vector;
using cudf::experimental::bool8;
using cudf::null_order;
using cudf::order;
using cudf::test::expect_scalars_equal;
using cudf::test::fixed_width_column_wrapper;

using q_res = cudf::numeric_scalar<double>;

// ----- test precision --------------------------------------------------------
// Used when calling `expect_scalar_equals`

template<typename T>
struct precision {
    constexpr static double tolerance = std::numeric_limits<T>::epsilon();
};

template<>
struct precision<bool8> {
    constexpr static double tolerance = 0;
};

// template<>
// struct precision<float> {
//     constexpr static double tolerance = 1.0e-7;
// };

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
    bool is_sorted;
    cudf::order order;
    cudf::null_order null_order;
};

// most numerics

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

template<typename T>
test_case<T>
interpolate_center() {
    auto low = std::numeric_limits<T>::lowest();
    auto max = std::numeric_limits<T>::max();
    auto mid_d = std::is_floating_point<T>::value ? 0.0 : -0.5;

    // int64_t is internally casted to a double, meaning the center point is float-like.
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

// floating point

template<typename T>
typename std::enable_if_t<std::is_floating_point<T>::value, test_case<T>>
single() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 7.31 }),
        {
            q_expect{ -1.0, 7.31, 7.31, 7.31, 7.31, 7.31, },
            q_expect{  0.0, 7.31, 7.31, 7.31, 7.31, 7.31, },
            q_expect{  1.0, 7.31, 7.31, 7.31, 7.31, 7.31, },
        }
    };
}

template<typename T>
typename std::enable_if_t<std::is_floating_point<T>::value, test_case<T>>
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
typename std::enable_if_t<std::is_floating_point<T>::value, test_case<T>>
some_invalid() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 6.8, 0.15, 3.4, 4.17, 2.13, 1.11, -1.01, 0.8, 5.7 },
                                       { 0,      1,   0,    0,    0,    0,     1,   0,   0 }),
        {
            q_expect{ -1.0,  -1.01,  -1.01,  -1.01,  -1.01,  -1.01 },
            q_expect{  0.0,  -1.01,  -1.01,  -1.01,  -1.01,  -1.01 },
            q_expect{  0.5,   0.15,  -1.01,  -0.43,  -0.43,  -1.01 },
            q_expect{  1.0,   0.15,   0.15,   0.15,   0.15,   0.15 },
            q_expect{  2.0,   0.15,   0.15,   0.15,   0.15,   0.15 }
        }
    };
}

template<typename T>
typename std::enable_if_t<std::is_floating_point<T>::value, test_case<T>>
unsorted() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 6.8, 0.15, 3.4, 4.17, 2.13, 1.11, -1.01, 0.8, 5.7 }),
        {
            q_expect{ 0.0, -1.01, -1.01, -1.01, -1.01, -1.01 },
        }
    };
}

// integral

template<typename T>
typename std::enable_if_t<std::is_integral<T>::value and not cudf::is_boolean<T>(), test_case<T>>
single() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 1 }),
        {
            q_expect{ 0.7, 1, 1, 1, 1, 1 }
        }
    };
}

template<typename T>
typename std::enable_if_t<std::is_integral<T>::value and not cudf::is_boolean<T>(), test_case<T>>
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
typename std::enable_if_t<std::is_integral<T>::value and not cudf::is_boolean<T>(), test_case<T>>
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
typename std::enable_if_t<std::is_integral<T>::value and not cudf::is_boolean<T>(), test_case<T>>
unsorted() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 6, 0, 3, 4, 2, 1, -1, 1, 6 }),
        {
            q_expect{ 0.0, -1, -1, -1, -1, -1 }
        }
    };
}

// boolean

template<typename T>
typename std::enable_if_t<cudf::is_boolean<T>(), test_case<T>>
single() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 1 }),
        {
            q_expect{ 0.7, 1.0, 1.0, 1.0, 1.0, 1.0 }
        }
    };
}

template<typename T>
typename std::enable_if_t<cudf::is_boolean<T>(), test_case<T>>
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

template<typename T>
typename std::enable_if_t<cudf::is_boolean<T>(), test_case<T>>
all_invalid() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 1, 0, 1, 1, 0, 1, 0, 1, 1 },
                                       { 0, 0, 0, 0, 0, 0, 0, 0, 0}),
        {
            q_expect{ 0.7 }
        }
    };
}

template<typename T>
typename std::enable_if_t<cudf::is_boolean<T>(), test_case<T>>
unsorted() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 0, 0, 1, 1, 0, 1, 1, 0, 1 }),
        {
            q_expect{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,}
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

template<>
test_case<bool8>
interpolate_extrema_high<bool8>() {
    return interpolate_center<bool8>();
}

template<>
test_case<bool8>
interpolate_extrema_low<bool8>() {
    return interpolate_center<bool8>();
}

} // namespace testdata

// =============================================================================
// ----- helper functions ------------------------------------------------------

template<typename T>
void test(testdata::test_case<T> test_case) {
    using namespace cudf::experimental;

    cudf::table_view in_table { { test_case.column } };

    for (auto & expected : test_case.expectations) {

        auto actual_higher = quantiles(in_table, expected.quantile, interpolation::HIGHER, test_case.is_sorted, { order::ASCENDING }, { null_order::AFTER });
        expect_scalars_equal(expected.higher, *actual_higher[0], precision<T>::tolerance);

        auto actual_lower = quantiles(in_table, expected.quantile, interpolation::LOWER, test_case.is_sorted, { order::ASCENDING }, { null_order::AFTER });
        expect_scalars_equal(expected.lower, *actual_lower[0], precision<T>::tolerance);

        auto actual_linear = quantiles(in_table, expected.quantile, interpolation::LINEAR, test_case.is_sorted, { order::ASCENDING }, { null_order::AFTER });
        expect_scalars_equal(expected.linear, *actual_linear[0], precision<T>::tolerance);

        auto actual_midpoint = quantiles(in_table, expected.quantile, interpolation::MIDPOINT, test_case.is_sorted, { order::ASCENDING }, { null_order::AFTER });
        expect_scalars_equal(expected.midpoint, *actual_midpoint[0], precision<T>::tolerance);

        auto actual_nearest = quantiles(in_table, expected.quantile, interpolation::NEAREST, test_case.is_sorted, { order::ASCENDING }, { null_order::AFTER });
        expect_scalars_equal(expected.nearest, *actual_nearest[0], precision<T>::tolerance);
    }
}

// =============================================================================
// ----- tests -----------------------------------------------------------------

template <typename T>
struct QuantilesTest : public cudf::test::BaseFixture {
};

using TestTypes = cudf::test::NumericTypes;

TYPED_TEST_CASE(QuantilesTest, TestTypes);

TYPED_TEST(QuantilesTest, TestEmpty)
{
    test(testdata::empty<TypeParam>());
}

TYPED_TEST(QuantilesTest, TestSingle)
{
    test(testdata::single<TypeParam>());
}

TYPED_TEST(QuantilesTest, TestSomeElementsInvalid)
{
    test(testdata::some_invalid<TypeParam>());
}

TYPED_TEST(QuantilesTest, TestAllElementsInvalid)
{
    test(testdata::all_invalid<TypeParam>());
}

TYPED_TEST(QuantilesTest, TestUnsorted)
{
    test(testdata::unsorted<TypeParam>());
}

TYPED_TEST(QuantilesTest, TestInterpolateCenter)
{
    test(testdata::interpolate_center<TypeParam>());
}

TYPED_TEST(QuantilesTest, TestInterpolateExtremaHigh)
{
    test(testdata::interpolate_extrema_high<TypeParam>());
}

TYPED_TEST(QuantilesTest, TestInterpolateExtremaLow)
{
    test(testdata::interpolate_extrema_low<TypeParam>());
}
