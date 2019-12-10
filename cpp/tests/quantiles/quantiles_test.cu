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

#include <cudf/wrappers/bool.hpp>
#include <limits>
#include <algorithm>
#include <tests/utilities/cudf_gtest.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <memory>
#include <tests/utilities/base_fixture.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/scalar_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_list_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <cudf/quantiles.hpp>
#include <type_traits>


using namespace std;
using namespace cudf;
using namespace test;
using bool8 = experimental::bool8;

using q_res = numeric_scalar<double>;

namespace {

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
    order order;
    null_order null_order;
};

// all numerics

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

// floating point

template<typename T>
typename std::enable_if_t<is_floating_point<T>::value, test_case<T>>
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
typename std::enable_if_t<is_floating_point<T>::value, test_case<T>>
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
typename std::enable_if_t<is_floating_point<T>::value, test_case<T>>
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
typename std::enable_if_t<is_integral<T>::value and not is_boolean<T>(), test_case<T>>
single() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 7 }),
        {
            q_expect{ 0.7, 7, 7, 7, 7, 7 }
        }
    };
}

template<typename T>
typename std::enable_if_t<is_integral<T>::value and not is_boolean<T>(), test_case<T>>
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
typename std::enable_if_t<is_integral<T>::value and not is_boolean<T>(), test_case<T>>
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
typename std::enable_if_t<is_boolean<T>(), test_case<T>>
single() {
    return test_case<T> {
        fixed_width_column_wrapper<T> ({ 1 }),
        {
            q_expect{ 0.7, 1.0, 1.0, 1.0, 1.0, 1.0 }
        }
    };
}

template<typename T>
typename std::enable_if_t<is_boolean<T>(), test_case<T>>
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
typename std::enable_if_t<is_boolean<T>(), test_case<T>>
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
    for (auto & expected : test_case.expectations) {
        using namespace experimental;
        auto actual_higher   = quantile(test_case.column, expected.quantile, interpolation::HIGHER);
        auto actual_lower    = quantile(test_case.column, expected.quantile, interpolation::LOWER);
        auto actual_nearest  = quantile(test_case.column, expected.quantile, interpolation::NEAREST);
        auto actual_midpoint = quantile(test_case.column, expected.quantile, interpolation::MIDPOINT);
        auto actual_linear   = quantile(test_case.column, expected.quantile, interpolation::LINEAR);
        expect_scalars_equal(expected.higher,   *actual_higher);
        expect_scalars_equal(expected.lower,    *actual_lower);
        expect_scalars_equal(expected.linear,   *actual_nearest);
        expect_scalars_equal(expected.midpoint, *actual_midpoint);
        expect_scalars_equal(expected.nearest,  *actual_linear);
    }
}

// =============================================================================
// ----- tests -----------------------------------------------------------------

template <typename T>
struct QuantilesTest : public BaseFixture {
};

// using TestTypes = test::Types<int, float, double, bool8>;
// using TestTypes = AllTypes;
using TestTypes = NumericTypes;

TYPED_TEST_CASE(QuantilesTest, TestTypes);

TYPED_TEST(QuantilesTest, TestEmpty)
{
    test(testdata::empty<TypeParam>());
}

TYPED_TEST(QuantilesTest, TestSingle)
{
    test(testdata::single<TypeParam>());
}

TYPED_TEST(QuantilesTest, TestAllElementsInvalid)
{
    test(testdata::all_invalid<TypeParam>());
}

TYPED_TEST(QuantilesTest, TestUnsorted)
{
    test(testdata::unsorted<TypeParam>());
}

// TYPED_TEST(QuantilesTest, TestUnsortedWithInvalids)
// {
//     test(testdata::unsorted_with_invalids<TypeParam>());
// }

} // anonymous namespace
