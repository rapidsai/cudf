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

using ValueType = int16_t;

TEST(GdfConcat, CaseWithZeroLeft)
{
    //  0 + 2
    //  2
    const size_t lhs_size = 0;
    const size_t rhs_size = 2;
    gdf_column lhs = gen_gdb_column<ValueType>(lhs_size, 2);
    gdf_column rhs = gen_gdb_column<ValueType>(rhs_size, 3);
    std::cout << "*****left**************\n";
    print_column(&lhs);
    std::cout << "*****right**************\n";
    print_column(&rhs);
    std::cout << "*******************\n";
    gdf_column output = gen_gdb_column<ValueType>(lhs_size + rhs_size, 0);

    gpu_concat(&lhs, &rhs, &output);
    check_column_for_concat_operation<ValueType>(&lhs, &rhs, &output);
    delete_gdf_column(&lhs);
    delete_gdf_column(&rhs);
    delete_gdf_column(&output);
}

TEST(GdfConcat, CaseWithZeroRight)
{
    //  2 + 0
    //  2
    const size_t lhs_size = 2;
    const size_t rhs_size = 0;
    gdf_column lhs = gen_gdb_column<ValueType>(lhs_size, 2);
    gdf_column rhs = gen_gdb_column<ValueType>(rhs_size, 3);
    std::cout << "*****left**************\n";
    print_column(&lhs);
    std::cout << "*****right**************\n";
    print_column(&rhs);
    std::cout << "*******************\n";
    gdf_column output = gen_gdb_column<ValueType>(lhs_size + rhs_size, 0);

    gpu_concat(&lhs, &rhs, &output);
    check_column_for_concat_operation<ValueType>(&lhs, &rhs, &output);
    delete_gdf_column(&lhs);
    delete_gdf_column(&rhs);
    delete_gdf_column(&output);
}

TEST(GdfConcat, CaseWithOutputOfOneByte)
{
    //  3 + 4
    //  3|4
    const size_t lhs_size = 3;
    const size_t rhs_size = 4;
    gdf_column lhs = gen_gdb_column<ValueType>(lhs_size, 2);
    gdf_column rhs = gen_gdb_column<ValueType>(rhs_size, 3);
    std::cout << "*****left**************\n";
    print_column(&lhs);
    std::cout << "*****right**************\n";
    print_column(&rhs);
    std::cout << "*******************\n";
    gdf_column output = gen_gdb_column<ValueType>(lhs_size + rhs_size, 0);

    gpu_concat(&lhs, &rhs, &output);
    check_column_for_concat_operation<ValueType>(&lhs, &rhs, &output);
    delete_gdf_column(&lhs);
    delete_gdf_column(&rhs);
    delete_gdf_column(&output);
}

TEST(GdfConcat, CaseWithOutputOfTwoBytes)
{
    //  3 + 7  // caso especial
    //  3|5, 3
    const size_t lhs_size = 3;
    const size_t rhs_size = 7;
    gdf_column lhs = gen_gdb_column<ValueType>(lhs_size, 2);
    gdf_column rhs = gen_gdb_column<ValueType>(rhs_size, 3);
    std::cout << "*****left**************\n";
    print_column(&lhs);
    std::cout << "*****right**************\n";
    print_column(&rhs);
    std::cout << "*******************\n";
    gdf_column output = gen_gdb_column<ValueType>(lhs_size + rhs_size, 0);

    gpu_concat(&lhs, &rhs, &output);
    check_column_for_concat_operation<ValueType>(&lhs, &rhs, &output);
    delete_gdf_column(&lhs);
    delete_gdf_column(&rhs);
    delete_gdf_column(&output);
}

TEST(GdfConcat, CaseWithInput_2_2_Output3)
{
    //  8, 3 + 8, 1
    //  8, 3|5, 3|1

    const size_t lhs_size = 8 + 3;
    const size_t rhs_size = 8 + 1;

    gdf_column lhs = gen_gdb_column<ValueType>(lhs_size, 2);
    gdf_column rhs = gen_gdb_column<ValueType>(rhs_size, 3);
    gdf_column output = gen_gdb_column<ValueType>(lhs_size + rhs_size, 0);

    gpu_concat(&lhs, &rhs, &output);
    check_column_for_concat_operation<ValueType>(&lhs, &rhs, &output);
    delete_gdf_column(&lhs);
    delete_gdf_column(&rhs);
    delete_gdf_column(&output);
}

TEST(GdfConcat, CaseWithInput_2_5_Output5)
{
    //  8, 2 + 8, 8, 8, 8, 5
    //  8, 2|6, 2|6, 2|6, 2|5

    const size_t lhs_size = 8 + 2;
    const size_t rhs_size = 8 + 8 + 8 + 8 + 5;

    gdf_column lhs = gen_gdb_column<ValueType>(lhs_size, 2);
    gdf_column rhs = gen_gdb_column<ValueType>(rhs_size, 3);
    gdf_column output = gen_gdb_column<ValueType>(lhs_size + rhs_size, 0);

    gpu_concat(&lhs, &rhs, &output);
    check_column_for_concat_operation<ValueType>(&lhs, &rhs, &output);
    delete_gdf_column(&lhs);
    delete_gdf_column(&rhs);
    delete_gdf_column(&output);
}

TEST(GdfConcat, CaseWithInput_1_4_Output5)
{
    //  3 + 8, 8, 8, 7      // caso especial
    //  3|5, 3|5, 3|5, 3|5, 2

    // 100
    //  10101111 10101111 10101111 10000 00
    // 100 10101
    //
    //      11110101
    //            11110101
    //                11110000
    //                        00

    const size_t lhs_size = 3;
    const size_t rhs_size = 8 + 8 + 8 + 7;

    gdf_column lhs = gen_gdb_column<ValueType>(lhs_size, 2);
    gdf_column rhs = gen_gdb_column<ValueType>(rhs_size, 3);
    gdf_column output = gen_gdb_column<ValueType>(lhs_size + rhs_size, 0);
    std::cout << "*****left**************\n";
    print_column(&lhs);
    std::cout << "*****right**************\n";
    print_column(&rhs);
    std::cout << "*******************\n";

    gpu_concat(&lhs, &rhs, &output);
    check_column_for_concat_operation<ValueType>(&lhs, &rhs, &output);

    delete_gdf_column(&lhs);
    delete_gdf_column(&rhs);
    delete_gdf_column(&output);
}


TEST(GdfConcat, CaseWithInput_0_9_Output2)
{
    //  3 + 8, 8, 8, 7      // caso especial
    //  3|5, 3|5, 3|5, 3|5, 2

    // 100
    //  10101111 10101111 10101111 10000 00
    // 100 10101
    //
    //      11110101
    //            11110101
    //                11110000
    //                        00

    const size_t lhs_size = 0;
    const size_t rhs_size = 8 + 1;

    gdf_column lhs = gen_gdb_column<ValueType>(lhs_size, 2);
    gdf_column rhs = gen_gdb_column<ValueType>(rhs_size, 3);
    gdf_column output = gen_gdb_column<ValueType>(lhs_size + rhs_size, 0);
    std::cout << "*****left**************\n";
    print_column(&lhs);
    std::cout << "*****right**************\n";
    print_column(&rhs);
    std::cout << "*******************\n";

    gpu_concat(&lhs, &rhs, &output);
    check_column_for_concat_operation<ValueType>(&lhs, &rhs, &output);

    delete_gdf_column(&lhs);
    delete_gdf_column(&rhs);
    delete_gdf_column(&output);
}

TEST(GpuConcatTest, WithDifferentColumnSizes)
{
    using ValueType = int16_t;
    for (int lhs_size = 0; lhs_size < 10; lhs_size += 1)
    {
        for (int rhs_size = 0; rhs_size < 10; rhs_size += 3)
        {
            gdf_column lhs = gen_gdb_column<ValueType>(lhs_size, 2);

            gdf_column rhs = gen_gdb_column<ValueType>(rhs_size, 3);

            gdf_column output = gen_gdb_column<ValueType>(lhs_size + rhs_size, 0);
            gdf_error error = gpu_concat(&lhs, &rhs, &output);

            // std::cout << "Left" << std::endl;
            // print_column<ValueType>(&lhs);

            // std::cout << "Right" << std::endl;
            // print_column<ValueType>(&rhs);

            // std::cout << "Output" << std::endl;
            // print_column<ValueType>(&output);

            check_column_for_concat_operation<ValueType>(&lhs, &rhs, &output);

            delete_gdf_column(&lhs);
            delete_gdf_column(&rhs);
            delete_gdf_column(&output);
        }
    }
}