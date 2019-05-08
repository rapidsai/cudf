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
#include <reduction.hpp>

#include <thrust/device_vector.h>

#include "tests/utilities/cudf_test_fixtures.h"
#include "tests/utilities/cudf_test_utils.cuh"
#include "tests/utilities/column_wrapper.cuh"
#include "tests/utilities/scalar_wrapper.cuh"

template <typename T>
std::vector<T> convert_values(std::vector<int> const & int_values)
{
    std::vector<T> v(int_values.size());
    std::transform(int_values.begin(), int_values.end(), v.begin(),
        [](int x) { return static_cast<T>(x); } );
    return v;
}

// This is the main test feature
template <typename T>
struct ReductionTest : public GdfTest
{
    // Sum/Prod/SumOfSquare never support non arithmetics
    static constexpr bool ret_non_arithmetic =
        (std::is_arithmetic<T>::value || std::is_same<T, cudf::bool8>::value)
            ? true : false;

    ReductionTest(){}

    ~ReductionTest(){}

    void reduction_test(std::vector<T>& input_values,
        T expected_value, bool succeeded_condition,
        gdf_reduction_op op)
    {
        cudf::test::column_wrapper<T> const col(input_values);

        const gdf_column * underlying_column = col.get();
        thrust::device_vector<T> dev_result(1);

        auto statement = [&]() {
            cudf::test::scalar_wrapper<T> result 
                = cudf::reduction(underlying_column, op,
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
    cudf::bool8, cudf::date32, cudf::date64, cudf::timestamp, cudf::category>;

TYPED_TEST_CASE(ReductionTest, Types);

// ------------------------------------------------------------------------
TYPED_TEST(ReductionTest, MinMax)
{
   std::vector<int> int_values({5, 0, -120, -111, 0, 64, 63, 99, 123, -16});
   std::vector<TypeParam> v = convert_values<TypeParam>(int_values);

   TypeParam expected_min_result = *( std::min_element(v.begin(), v.end()) );
   TypeParam expected_max_result = *( std::max_element(v.begin(), v.end()) );

   // Min/Max succeeds for any gdf types including
   // non-arithmetic types (date32, date64, timestamp, category)
   bool result_error(true);

   this->reduction_test(v, expected_min_result, result_error, GDF_REDUCTION_MIN);
   this->reduction_test(v, expected_max_result, result_error, GDF_REDUCTION_MAX);
}

TYPED_TEST(ReductionTest, Product)
{
    std::vector<int> int_values({5, -1, 1, 0, 3, 2, 4});
    std::vector<TypeParam> v = convert_values<TypeParam>(int_values);

    TypeParam expected_value = std::accumulate(v.begin(), v.end(), TypeParam{1},
        [](TypeParam acc, TypeParam i) { return acc * i; });

    this->reduction_test(v, expected_value, this->ret_non_arithmetic, 
                         GDF_REDUCTION_PRODUCT);
}

TYPED_TEST(ReductionTest, Sum)
{
    std::vector<int> int_values({6, -14, 13, 64, 0, -13, -20, 45});
    std::vector<TypeParam> v = convert_values<TypeParam>(int_values);

    TypeParam expected_value = std::accumulate(v.begin(), v.end(), TypeParam{0});

    this->reduction_test(v, expected_value, this->ret_non_arithmetic, GDF_REDUCTION_SUM);
}

TYPED_TEST(ReductionTest, SumOfSquare)
{
    std::vector<int> int_values({-3, 2,  1, 0, 5, -3, -2});
    std::vector<TypeParam> v = convert_values<TypeParam>(int_values);

    TypeParam expected_value = std::accumulate(v.begin(), v.end(), TypeParam{0},
        [](TypeParam acc, TypeParam i) { return acc + i * i; });

    this->reduction_test(v, expected_value, this->ret_non_arithmetic,
                         GDF_REDUCTION_SUMOFSQUARES);
}

// ----------------------------------------------------------------------------

struct ReductionDtypeTest : public GdfTest
{
    template <typename T_in, typename T_out>
    void reduction_test(std::vector<int> & int_values,
        T_out expected_value, bool succeeded_condition,
        gdf_reduction_op op, gdf_dtype out_dtype,
        bool expected_overflow = false)
    {
        std::vector<T_in> input_values = convert_values<T_in>(int_values);
        cudf::test::column_wrapper<T_in> const col(input_values);

        auto statement = [&]() {
            cudf::test::scalar_wrapper<T_out> result =
                cudf::reduction(col.get(), op, out_dtype);
            if( result.is_valid() && ! expected_overflow){
                EXPECT_EQ(expected_value, result.value());
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
    std::vector<int> int_values({6, -14, 13, 109, -13, -20, 0, 98, 122, 123});
    int expected_value = std::accumulate(int_values.begin(), int_values.end(), 0);

    // over flow
    this->reduction_test<int8_t, int8_t>
        (int_values, static_cast<int8_t>(expected_value), true,
         GDF_REDUCTION_SUM, GDF_INT8, expected_overflow);

    this->reduction_test<int8_t, int64_t>
        (int_values, static_cast<int64_t>(expected_value), true,
         GDF_REDUCTION_SUM, GDF_INT64);

    this->reduction_test<int8_t, double>
        (int_values, static_cast<double>(expected_value), true,
         GDF_REDUCTION_SUM, GDF_FLOAT64);

    // down cast (over flow)
    this->reduction_test<double, int8_t>
        (int_values, static_cast<int8_t>(expected_value), true,
         GDF_REDUCTION_SUM, GDF_INT8, expected_overflow);

    // down cast (no over flow)
    this->reduction_test<double, int16_t>
        (int_values, static_cast<int16_t>(expected_value), true,
         GDF_REDUCTION_SUM, GDF_INT16);

    // not supported case:
    // wrapper classes other than cudf::bool8 are not convertible
    this->reduction_test<cudf::date64, cudf::timestamp>
        (int_values, static_cast<cudf::timestamp>(expected_value), false,
         GDF_REDUCTION_SUM, GDF_TIMESTAMP);

    this->reduction_test<cudf::date32, cudf::category>
        (int_values, static_cast<cudf::category>(expected_value), false,
         GDF_REDUCTION_SUM, GDF_CATEGORY);

    this->reduction_test<cudf::date32, cudf::date64>
        (int_values, static_cast<cudf::date64>(expected_value), false,
         GDF_REDUCTION_SUM, GDF_DATE64);

    // supported case: cudf::bool8
    std::vector<bool> v = convert_values<bool>(int_values);

    // When summing bool8 values into an non-bool arithmetic type,
    // it's an integer/float sum of ones and zeros.
    int expected_bool8 = std::accumulate(v.begin(), v.end(), int{0});

    this->reduction_test<cudf::bool8, int8_t>
        (int_values, static_cast<int8_t>(expected_bool8), true,
         GDF_REDUCTION_SUM, GDF_INT8);
    this->reduction_test<cudf::bool8, int16_t>
        (int_values, static_cast<int16_t>(expected_bool8), true,
         GDF_REDUCTION_SUM, GDF_INT16);
    this->reduction_test<cudf::bool8, int32_t>
        (int_values, static_cast<int32_t>(expected_bool8), true,
         GDF_REDUCTION_SUM, GDF_INT32);
    this->reduction_test<cudf::bool8, int64_t>
        (int_values, static_cast<int64_t>(expected_bool8), true,
         GDF_REDUCTION_SUM, GDF_INT64);
    this->reduction_test<cudf::bool8, float>
        (int_values, static_cast<float>(expected_bool8), true,
         GDF_REDUCTION_SUM, GDF_FLOAT32);
    this->reduction_test<cudf::bool8, double>
        (int_values, static_cast<double>(expected_bool8), true,
         GDF_REDUCTION_SUM, GDF_FLOAT64);

    // make sure boolean arithmetic semantics are obeyed when
    // reducing to a bool
    this->reduction_test<cudf::bool8, cudf::bool8>
        (int_values, cudf::true_v, true, GDF_REDUCTION_SUM, GDF_BOOL8);

    // TODO: should this work? Currently "input type not convertible to output"
    /*this->reduction_test<int32_t, cudf::bool8>
        (int_values, cudf::true_v, false, GDF_REDUCTION_SUM, GDF_BOOL8);*/

    // Though the underlying type of cudf::date64 is int64_t,
    // they are not convertible types.
    this->reduction_test<cudf::date64, int64_t>
        (int_values, static_cast<int64_t>(expected_value), false,
         GDF_REDUCTION_SUM, GDF_INT64);

}

// test case for empty input cases
TEST(ReductionErrorTest, empty_column)
{
    using T = int32_t;
    auto statement = [](const gdf_column* col) {
        cudf::test::scalar_wrapper<T> result =
            cudf::reduction(col, GDF_REDUCTION_SUM, GDF_INT64);
        EXPECT_EQ( result.is_valid(), false );
    };

    // test input column is nullptr, reduction throws an error if input is nullptr
    EXPECT_ANY_THROW(statement(nullptr));

    // test if the size of input column is zero
    // expect result.is_valid() is false
    cudf::test::column_wrapper<T> const col0(0);
    CUDF_EXPECT_NO_THROW(statement(col0.get()));

    // test if null count is equal or greater than size of input 
    // expect result.is_valid() is false
    int col_size = 5;
    std::vector<T> col_data(col_size);
    std::vector<gdf_valid_type> valids(gdf_valid_allocation_size(col_size));
    std::fill(valids.begin(), valids.end(), 0);
    
    cudf::test::column_wrapper<T> col_empty(col_data, valids);
    CUDF_EXPECT_NO_THROW(statement(col_empty.get()));
}


