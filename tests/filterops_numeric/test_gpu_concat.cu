/*
 ============================================================================
 Name        : testing-libgdf.cu
 Author      : felipe
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */

#include "gtest/gtest.h"

#include <iostream>
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include "helper/utils.cuh"

TEST(GpuConcatTest, WithDifferentColumnSizes)
{
    for (int lhs_size = 0; lhs_size < 100; lhs_size += 1)
    {
        for (int rhs_size = 0; rhs_size < 100; rhs_size += 1)
        {
            gdf_column lhs = gen_gdb_column(lhs_size, 1);

            gdf_column rhs = gen_gdb_column(rhs_size, 2);

            gdf_column output = gen_gdb_column(lhs_size + rhs_size, 0);
            gdf_error error = gpu_concat(&lhs, &rhs, &output);

            // std::cout << "Left" << std::endl;
            // print_column(&lhs);

            // std::cout << "Right" << std::endl;
            // print_column(&rhs);

            // std::cout << "Output" << std::endl;
            // print_column(&output);

            check_column(&lhs, &rhs, &output);

            delete_gdf_column(&lhs);
            delete_gdf_column(&rhs);
            delete_gdf_column(&output);
        }
    }
}