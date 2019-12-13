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
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/scalar_utilities.hpp>
#include <tests/utilities/type_list_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

using std::vector;
using cudf::experimental::bool8;
using cudf::null_order;
using cudf::order;
using cudf::test::expect_scalars_equal;
using cudf::test::fixed_width_column_wrapper;

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
    bool is_sorted;
    cudf::order order;
    cudf::null_order null_order;
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
        fixed_width_column_wrapper<T> ({ 7 }),
        {
            q_expect{ 0.7, 7, 7, 7, 7, 7 }
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

} // namespace testdata

// =============================================================================
// ----- helper functions ------------------------------------------------------

template<typename T>
void test(testdata::test_case<T> test_case) {
    using namespace cudf::experimental;
    for (auto & expected : test_case.expectations) {
        cudf::table_view in_table { { test_case.column } };
        auto actual_higher   = quantiles(in_table, expected.quantile, interpolation::HIGHER,   test_case.is_sorted, { order::ASCENDING }, { null_order::AFTER });
        auto actual_lower    = quantiles(in_table, expected.quantile, interpolation::LOWER,    test_case.is_sorted, { order::ASCENDING }, { null_order::AFTER });
        auto actual_nearest  = quantiles(in_table, expected.quantile, interpolation::NEAREST,  test_case.is_sorted, { order::ASCENDING }, { null_order::AFTER });
        auto actual_midpoint = quantiles(in_table, expected.quantile, interpolation::MIDPOINT, test_case.is_sorted, { order::ASCENDING }, { null_order::AFTER });
        auto actual_linear   = quantiles(in_table, expected.quantile, interpolation::LINEAR,   test_case.is_sorted, { order::ASCENDING }, { null_order::AFTER });
        expect_scalars_equal(expected.higher,   *actual_higher[0]);
        expect_scalars_equal(expected.lower,    *actual_lower[0]);
        expect_scalars_equal(expected.linear,   *actual_nearest[0]);
        expect_scalars_equal(expected.midpoint, *actual_midpoint[0]);
        expect_scalars_equal(expected.nearest,  *actual_linear[0]);
    }
}

// =============================================================================
// ----- tests -----------------------------------------------------------------

template <typename T>
struct QuantilesTest : public cudf::test::BaseFixture {
};

// using TestTypes = test::Types<int, float, double, bool8>;
// using TestTypes = AllTypes;
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
