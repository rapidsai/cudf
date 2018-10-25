/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
 *     Copyright 2018 Felipe Aramburu <felipe@blazingdb.com>
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

#include "gtest/gtest.h"
#include <iostream>
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <tuple>
#include "helper/utils.cuh"
#include "gdf_test_fixtures.h"

/*
 ============================================================================
 Description : Compute gpu_comparison and apply_stencil of gdf_columns using Thrust on GPU
 ============================================================================
 */

struct FilterOperationsTest : public GdfTest {};

TEST_F(FilterOperationsTest, usage_example) {

    using LeftValueType = int16_t;
    using RightValueType = int16_t;
    int column_size = 10;
    int init_value = 10;
    int max_size = 4;
    gdf_comparison_operator gdf_operator = GDF_EQUALS;

    gdf_column lhs = gen_gdb_column<LeftValueType>(column_size, init_value); // 4, 2, 0
    
    gdf_column rhs = gen_gdb_column<RightValueType>(column_size, 0.01 + max_size - init_value); // 0, 2, 4

    gdf_column output = gen_gdb_column<int8_t>(column_size, 0);

    gdf_error error = gpu_comparison(&lhs, &rhs, &output, gdf_operator);
    EXPECT_TRUE(error == GDF_SUCCESS);

    std::cout << "Left" << std::endl;
    print_column<LeftValueType>(&lhs);

    std::cout << "Right" << std::endl;
    print_column<RightValueType>(&rhs);

    std::cout << "Output" << std::endl;
    print_column<int8_t>(&output);

    check_column_for_comparison_operation<LeftValueType, RightValueType>(&lhs, &rhs, &output, gdf_operator);

    /// lhs.dtype === rhs.dtype
    gpu_apply_stencil(&lhs, &output, &rhs);

    check_column_for_stencil_operation<LeftValueType, RightValueType>(&lhs, &output, &rhs);

    delete_gdf_column(&lhs);
    delete_gdf_column(&rhs);
    delete_gdf_column(&output);
}


template <typename LeftValueType, typename RightValueType>
void test_filterops_using_templates(gdf_comparison_operator gdf_operator = GDF_EQUALS)
{
    //0, ..., 100,
    //100, 10000, 10000, 100000
    for (int column_size = 0; column_size < 10; column_size += 1)
    {
        const int max_size = 8;
        for (int init_value = 0; init_value <= 1; init_value++)
        {
            gdf_column lhs = gen_gdb_column<LeftValueType>(column_size, init_value); // 4, 2, 0
            // lhs.null_count = 2;

            gdf_column rhs = gen_gdb_column<RightValueType>(column_size, 0.01 + max_size - init_value); // 0, 2, 4
            // rhs.null_count = 1;

            gdf_column output = gen_gdb_column<int8_t>(column_size, 0);

            gdf_error error = gpu_comparison(&lhs, &rhs, &output, gdf_operator);
            EXPECT_TRUE(error == GDF_SUCCESS);

            check_column_for_comparison_operation<LeftValueType, RightValueType>(&lhs, &rhs, &output, gdf_operator);

            if (lhs.dtype == rhs.dtype ) {
                gpu_apply_stencil(&lhs, &output, &rhs);
                check_column_for_stencil_operation<LeftValueType, RightValueType>(&lhs, &output, &rhs);
            }

            delete_gdf_column(&lhs);
            delete_gdf_column(&rhs);
            delete_gdf_column(&output);
        }
    }
}

TEST_F(FilterOperationsTest, WithInt8AndOthers)
{
    test_filterops_using_templates<int8_t, int8_t>();
    test_filterops_using_templates<int8_t, int16_t>();
    
    test_filterops_using_templates<int8_t, int32_t>();
    test_filterops_using_templates<int8_t, int64_t>();
    test_filterops_using_templates<int8_t, float>(); 
    test_filterops_using_templates<int8_t, double>();
}

TEST_F(FilterOperationsTest, WithInt16AndOthers)
{
    test_filterops_using_templates<int16_t, int8_t>();
    test_filterops_using_templates<int16_t, int16_t>();
    test_filterops_using_templates<int16_t, int32_t>();
    test_filterops_using_templates<int16_t, int64_t>();
    test_filterops_using_templates<int16_t, float>();
    test_filterops_using_templates<int16_t, double>();
   
}

TEST_F(FilterOperationsTest, WithInt32AndOthers)
{
    test_filterops_using_templates<int32_t, int8_t>();
    test_filterops_using_templates<int32_t, int16_t>();
    test_filterops_using_templates<int32_t, int32_t>();
    test_filterops_using_templates<int32_t, int64_t>();
    test_filterops_using_templates<int32_t, float>();
    test_filterops_using_templates<int32_t, double>();
   
}

TEST_F(FilterOperationsTest, WithInt64AndOthers)
{
    test_filterops_using_templates<int64_t, int8_t>();
    test_filterops_using_templates<int64_t, int16_t>();
    test_filterops_using_templates<int64_t, int32_t>();
    test_filterops_using_templates<int64_t, int64_t>();
    test_filterops_using_templates<int64_t, float>();
    test_filterops_using_templates<int64_t, double>();
   
}

TEST_F(FilterOperationsTest, WithFloat32AndOthers)
{
    test_filterops_using_templates<float, int8_t>();
    test_filterops_using_templates<float, int16_t>();
    test_filterops_using_templates<float, int32_t>();
    test_filterops_using_templates<float, int64_t>();
    test_filterops_using_templates<float, float>();
    test_filterops_using_templates<float, double>();
   
}

TEST_F(FilterOperationsTest, WithFloat64AndOthers)
{
    test_filterops_using_templates<double, int8_t>();
    test_filterops_using_templates<double, int16_t>();
    test_filterops_using_templates<double, int32_t>();
    test_filterops_using_templates<double, int64_t>();
    test_filterops_using_templates<double, float>();
    test_filterops_using_templates<double, double>();
   
}
