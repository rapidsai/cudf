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

#include "gtest/gtest.h"

#include <cudf.h>
#include <cudf/functions.h>

#include <thrust/device_vector.h>

#include "tests/utilities/cudf_test_fixtures.h"
#include "tests/utilities/cudf_test_utils.cuh"
#include "tests/utilities/column_wrapper.cuh"

// This is the main test feature
template <typename T>
struct ReductionTest : public GdfTest
{
    // Sum/Prod/SumOfSquare never support non arithmetics
    static constexpr gdf_error ret_non_arithmetic =
        (std::is_arithmetic<T>::value)
         ? GDF_SUCCESS : GDF_UNSUPPORTED_DTYPE;

    ReductionTest(){}

    ~ReductionTest(){}

    void examin(std::vector<int>& int_values,
        T exact_value, gdf_error reduction_error_code,
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

        gdf_dtype out_dtype = underlying_column->dtype;
        gdf_error result_error{GDF_SUCCESS};

        result_error = gdf_reduction( underlying_column, op,
            thrust::raw_pointer_cast( dev_result.data() ), out_dtype);

        EXPECT_EQ(reduction_error_code, result_error);

        if( result_error == GDF_SUCCESS){
            thrust::host_vector<T> host_result(dev_result);
            EXPECT_EQ(exact_value, host_result[0]);
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
    int exact_min_result = *( std::min_element(v.begin(), v.end()) );
    int exact_max_result = *( std::max_element(v.begin(), v.end()) );

    // Min/Max succeeds for any gdf types including
    // non-arithmetic types (date32, date64, timestamp, category)
    gdf_error result_error(GDF_SUCCESS);

    this->examin(v, TypeParam(exact_min_result),
        result_error, GDF_REDUCTION_MIN);
    this->examin(v, TypeParam(exact_max_result),
        result_error, GDF_REDUCTION_MAX);
}

TYPED_TEST(ReductionTest, Product)
{
    std::vector<int> v({5, -1, 1, 3, 2, 4});
    int exact = std::accumulate(v.begin(), v.end(), 1,
        [](int acc, int i) { return acc * i; });

    this->examin(v, TypeParam(exact),
        this->ret_non_arithmetic, GDF_REDUCTION_PRODUCTION);
}

TYPED_TEST(ReductionTest, Sum)
{
    std::vector<int> v({6, -14, 13, 64, -13, -20, 45});
    int exact = std::accumulate(v.begin(), v.end(), 0);

    this->examin(v, TypeParam(exact),
        this->ret_non_arithmetic, GDF_REDUCTION_SUM);
}

TYPED_TEST(ReductionTest, SumOfSquare)
{
    std::vector<int> v({-3, 2,  1, 5, -3, -2});
    int exact = std::accumulate(v.begin(), v.end(),  0,
        [](int acc, int i) { return acc + i * i; });

    this->examin(v, TypeParam(exact),
        this->ret_non_arithmetic, GDF_REDUCTION_SUMOFSQUARES);
}

// ----------------------------------------------------------------------------

struct ReductionDtypeTest : public GdfTest
{
    template <typename T_in, typename T_out>
    void examin(std::vector<int>& int_values,
        T_out exact_value, gdf_error reduction_error_code,
        gdf_reduction_op op, gdf_dtype out_dtype)
    {
        gdf_size_type col_size = int_values.size();
        std::vector<T_in> input_values(col_size);

        std::transform(int_values.begin(), int_values.end(),
            input_values.begin(),
           [](int x) { T_in t(x) ; return t; } );

        cudf::test::column_wrapper<T_in> const col(input_values);

        const gdf_column * underlying_column = col.get();
        thrust::device_vector<T_out> dev_result(1);

        gdf_error result_error{GDF_SUCCESS};

        result_error = gdf_reduction( underlying_column, op,
            thrust::raw_pointer_cast( dev_result.data() ), out_dtype);

        EXPECT_EQ(reduction_error_code, result_error);

        if( result_error == GDF_SUCCESS){
            thrust::host_vector<T_out> host_result(dev_result);
            EXPECT_EQ(exact_value, host_result[0]);
            std::cout << "the value = <" << exact_value
                << ", " << host_result[0] << ">" << std::endl;
        }
    }
};

TEST_F(ReductionDtypeTest, int8_to_others)
{
    std::vector<int> v({6, -14, 13, 109, -13, -20, 45, 98, 122, 123});
    int exact = std::accumulate(v.begin(), v.end(), 0);

    printf("exact = %d\n", exact);

    // over flow
    this->examin<int8_t, int8_t>
        (v, int8_t(exact), GDF_SUCCESS,
        GDF_REDUCTION_SUM, GDF_INT8);

    this->examin<int8_t, int64_t>
        (v, int64_t(exact), GDF_SUCCESS,
        GDF_REDUCTION_SUM, GDF_INT64);


    // not supported case
    this->examin<int8_t, int64_t>
        (v, int64_t(exact), GDF_UNSUPPORTED_DTYPE,
        GDF_REDUCTION_SUM, GDF_DATE64);
}


