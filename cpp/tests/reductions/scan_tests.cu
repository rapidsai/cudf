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
struct ScanTest : public GdfTest
{
    void examin(std::vector<int>& int_values,
        std::vector<int>& exact_values,
        gdf_error reduction_error_code, gdf_scan_op op, bool inclusive)
    {

        this->val_check(int_values, "input = ");
        this->val_check(exact_values, "exact = ");

        gdf_size_type col_size = int_values.size();
        std::vector<T> input_values(col_size);

        std::transform(int_values.begin(), int_values.end(),
            input_values.begin(),
           [](int x) { T t(x) ; return t; } );

        cudf::test::column_wrapper<T> const col_in(input_values);
        const gdf_column * raw_input = col_in.get();

        cudf::test::column_wrapper<T> col_out(col_size);
        gdf_column * raw_output = col_out.get();

        gdf_error result_error{GDF_SUCCESS};
        result_error = gdf_scan(raw_input, raw_output, op, inclusive);

        EXPECT_EQ(reduction_error_code, result_error);

        using UnderlyingType = T;

        if( result_error == GDF_SUCCESS){
            auto tuple_host_result = col_out.to_host();
            auto host_result = std::get<0>(tuple_host_result);

            this->val_check(host_result, "result = ");

            std::equal(exact_values.begin(), exact_values.end(),
                host_result.begin(), host_result.end(),
                [](int x, UnderlyingType y) {
                    EXPECT_EQ(UnderlyingType(x), y); return true; });
        }
    }

    template <typename Ti>
    void val_check(std::vector<Ti> & v, const char* msg = nullptr){
#if 0   // for debugging
        printf("%s {", msg);
        std::for_each(v.begin(), v.end(), [](Ti i){ printf("%d, ", i);});
        printf("}\n");
#endif
#if 1
        range_check(v);
#endif
    }

    // make sure all elements in the range of sint8([-128, 127])
    template <typename Ti>
    void range_check(std::vector<Ti> & v){
        std::for_each(v.begin(), v.end(),
            [](Ti i){
                ASSERT_GE(i, -128);
                ASSERT_LT(i,  128);
            });
    }

};

using Types = testing::Types<
#if 1
    int8_t,int16_t, int32_t, int64_t, float, double>;
#else
    int8_t>;
#endif

TYPED_TEST_CASE(ScanTest, Types);

// ------------------------------------------------------------------------
TYPED_TEST(ScanTest, Min)
{
    std::vector<int> v({123, 64, 63, 99, -5, 123, -16, -120, -111});
    std::vector<int> exact;
    int acc(v[0]);

    std::for_each(v.begin(), v.end(),
        [&acc, &exact](int i){
            acc = std::min(acc, i); exact.push_back(acc);
        }
    );

    this->examin(v, exact, GDF_SUCCESS, GDF_SCAN_MIN, true);
}

TYPED_TEST(ScanTest, Max)
{
    std::vector<int> v({-120, 5, 0, -120, -111, 64, 63, 99, 123, -16});

    std::vector<int> exact;
    int acc(v[0]);

    std::for_each(v.begin(), v.end(),
        [&acc, &exact](int i){
            acc = std::max(acc, i); exact.push_back(acc);
        }
    );

    this->examin(v, exact, GDF_SUCCESS, GDF_SCAN_MAX, true);
}


TYPED_TEST(ScanTest, Product)
{
    std::vector<int> v({5, -1, 1, 3, -2, 4});

    std::vector<int> exact;
    int acc(1);
    std::for_each(v.begin(), v.end(),
        [&acc, &exact](int i){ acc *= i; exact.push_back(acc); });

    this->examin(v, exact, GDF_SUCCESS, GDF_SCAN_PRODUCTION, true);
}

TYPED_TEST(ScanTest, Sum)
{
    std::vector<int> v({-120, 5, 6, 113, -111, 64, -63, 9, 34, -16});

    std::vector<int> exact;
    int acc(0);
    std::for_each(v.begin(), v.end(),
        [&acc, &exact](int i){ acc += i; exact.push_back(acc); });

    this->examin(v, exact, GDF_SUCCESS, GDF_SCAN_SUM, true);
}


