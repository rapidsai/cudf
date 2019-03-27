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

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <type_traits>
#include <stdlib.h>

#include "gtest/gtest.h"

#include <cudf.h>
#include <cudf/functions.h>

#include <thrust/device_vector.h>

#include "tests/utilities/cudf_test_fixtures.h"
#include "tests/utilities/cudf_test_utils.cuh"
#include "tests/utilities/column_wrapper.cuh"
#include "tests/utilities/scalar_wrapper.cuh"

// This is the main test feature
template <typename T>
struct ReductionTest : public GdfTest
{
    // Sum/Prod/SumOfSquare never support non arithmetics
    static constexpr bool ret_non_arithmetic =
        (std::is_arithmetic<T>::value) ? true : false;

    ReductionTest(){}

    ~ReductionTest(){}

    void reduction_test(std::vector<int>& int_values,
        T expected_value, bool succeeded_condition,
        gdf_reduction_op op)
    {
        gdf_size_type col_size = int_values.size();
        std::vector<T> input_values(col_size);

        std::transform(int_values.begin(), int_values.end(),
            input_values.begin(),
           [](int x) { T t(x) ; return t; } );

        cudf::test::column_wrapper<T> const col(input_values);

        const gdf_column * underlying_column = col.get();
        thrust::device_vector<T> dev_result(1);

        auto statement = [&]() {
            cudf::test::scalar_wrapper<T> result 
		= gdf_reduction(underlying_column, op,
                                underlying_column->dtype);
              EXPECT_EQ(expected_value, result.value());
        };

        if( succeeded_condition ){
            CUDF_EXPECT_NO_THROW(statement());
        }else{
            EXPECT_ANY_THROW(statement());
        }
    }
};

using Types = testing::Types<
    int8_t,int16_t, int32_t, int64_t, float, double,
    cudf::date32, cudf::date64, cudf::timestamp, cudf::category>;

TYPED_TEST_CASE(ReductionTest, Types);

// ------------------------------------------------------------------------
TYPED_TEST(ReductionTest, MinMax)
{
    std::vector<int> v({5, 0, -120, -111, 64, 63, 99, 123, -16});
    int expected_min_result = *( std::min_element(v.begin(), v.end()) );
    int expected_max_result = *( std::max_element(v.begin(), v.end()) );

    // Min/Max succeeds for any gdf types including
    // non-arithmetic types (date32, date64, timestamp, category)
    bool result_error(true);

    this->reduction_test(v, TypeParam(expected_min_result),
        result_error, GDF_REDUCTION_MIN);
    this->reduction_test(v, TypeParam(expected_max_result),
        result_error, GDF_REDUCTION_MAX);
}

TYPED_TEST(ReductionTest, Product)
{
    std::vector<int> v({5, -1, 1, 3, 2, 4});
    int expected_value = std::accumulate(v.begin(), v.end(), 1,
        [](int acc, int i) { return acc * i; });

    this->reduction_test(v, TypeParam(expected_value),
        this->ret_non_arithmetic, GDF_REDUCTION_PRODUCTION);
}

TYPED_TEST(ReductionTest, Sum)
{
    std::vector<int> v({6, -14, 13, 64, -13, -20, 45});
    int expected_value = std::accumulate(v.begin(), v.end(), 0);

    this->reduction_test(v, TypeParam(expected_value),
        this->ret_non_arithmetic, GDF_REDUCTION_SUM);
}

TYPED_TEST(ReductionTest, SumOfSquare)
{
    std::vector<int> v({-3, 2,  1, 5, -3, -2});
    int expected_value = std::accumulate(v.begin(), v.end(),  0,
        [](int acc, int i) { return acc + i * i; });

    this->reduction_test(v, TypeParam(expected_value),
        this->ret_non_arithmetic, GDF_REDUCTION_SUMOFSQUARES);
}

// ----------------------------------------------------------------------------

struct ReductionDtypeTest : public GdfTest
{
    template <typename T_in, typename T_out>
    void reduction_test(std::vector<int>& int_values,
        T_out expected_value, bool succeeded_condition,
        gdf_reduction_op op, gdf_dtype out_dtype,
        bool expected_overflow = false)
    {
        gdf_size_type col_size = int_values.size();
        std::vector<T_in> input_values(col_size);

        std::transform(int_values.begin(), int_values.end(),
            input_values.begin(),
           [](int x) { T_in t(x) ; return t; } );

        cudf::test::column_wrapper<T_in> const col(input_values);
        const gdf_column * underlying_column = col.get();

        auto statement = [&]() {
            cudf::test::scalar_wrapper<T_out> result =
            	gdf_reduction(underlying_column, op, out_dtype);
            if( result.is_valid() && ! expected_overflow){
                EXPECT_EQ(expected_value, result.value());
                std::cout << "the value = <" << expected_value
                    << ", " << result.value() << ">" << std::endl;
            }
        };

        if( succeeded_condition ){
            CUDF_EXPECT_NO_THROW(statement());
        }else{
            EXPECT_ANY_THROW(statement());
        }
    }
};

// test case for different output precision
TEST_F(ReductionDtypeTest, different_precision)
{
    constexpr bool expected_overflow = true;
    std::vector<int> v({6, -14, 13, 109, -13, -20, 45, 98, 122, 123});
    int expected_value = std::accumulate(v.begin(), v.end(), 0);

    std::cout << "expected = " << expected_value << std::endl;

    // over flow
    this->reduction_test<int8_t, int8_t>
        (v, int8_t(expected_value), true, GDF_REDUCTION_SUM, GDF_INT8,
            expected_overflow);

    this->reduction_test<int8_t, int64_t>
        (v, int64_t(expected_value), true, GDF_REDUCTION_SUM, GDF_INT64);

    this->reduction_test<int8_t, double>
        (v, double(expected_value), true, GDF_REDUCTION_SUM, GDF_FLOAT64);

    // down cast (over flow)
    this->reduction_test<double, int8_t>
        (v, int8_t(expected_value), true, GDF_REDUCTION_SUM, GDF_INT8,
            expected_overflow);

    // down cast (no over flow)
    this->reduction_test<double, int16_t>
        (v, int16_t(expected_value), true, GDF_REDUCTION_SUM, GDF_INT16);

    // not supported case:
    // any of wrapper classes is not convertible
    this->reduction_test<cudf::date64, cudf::timestamp>
        (v, cudf::timestamp(expected_value), false, GDF_REDUCTION_SUM, GDF_TIMESTAMP);

    this->reduction_test<cudf::date32, cudf::category>
        (v, cudf::category(expected_value), false, GDF_REDUCTION_SUM, GDF_CATEGORY);

    this->reduction_test<cudf::date32, cudf::date64>
        (v, cudf::date64(expected_value), false, GDF_REDUCTION_SUM, GDF_DATE64);

    // Though the underlying type of cudf::date64 is int64_t,
    // they are not convertible types.
    this->reduction_test<cudf::date64, int64_t>
        (v, int64_t(expected_value), false, GDF_REDUCTION_SUM, GDF_INT64);

}
