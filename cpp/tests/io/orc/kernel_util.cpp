/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

// unit_test.cpp : Defines the entry point for the console application.
//

#include "tests_common.h"
#include "io/orc/orc_util.hpp"
#include "io/orc/kernel_orc.cuh"
#include "io/orc/kernel_util.cuh"

TEST(KernelUtil, get_bit_count)
{
    EXPECT_EQ(0, getBitCount(0x00));
    EXPECT_EQ(8, getBitCount(0xff));
    EXPECT_EQ(5, getBitCount(0xc7));
    EXPECT_EQ(3, getBitCount(0xa8));
    EXPECT_EQ(1, getBitCount(0x40));
}

size_t calc_binary_search_result(int* expects, size_t size_of_expects, const int* arr, size_t size_of_array)
{
    int last = 0;
    size_t count_of_expects = 1;
    for (int index = 0; index < size_of_array; index++)
    {
        if (arr[index] == last)continue;

        for (; count_of_expects <= arr[index]; count_of_expects++) {
            expects[count_of_expects-1] = index;
        }
    }

    return count_of_expects -1;
}

TEST(KernelUtil, binary_search_range)
{
    const int total_count = 20;
    int arr[] = { 1, 3, 8, 9, 13, 20 };
    int size_of_array = sizeof(arr) / sizeof(int);
    
    int expects[total_count];
    EXPECT_EQ(total_count, calc_binary_search_result(expects, total_count, arr, size_of_array) );

    for (int i = 0; i < total_count; i++) {
        EXPECT_EQ(expects[i], binarySearchRange(arr, 0, size_of_array-1, i+1));
    }
}

TEST(KernelUtil, binary_search_range_with_null)
{
    const int total_count = 20;
    int arr[] = { 0, 0, 0, 3, 8, 9, 9, 9, 13, 20 };
    int size_of_array = sizeof(arr) / sizeof(int);

    EXPECT_EQ(3, binarySearchRange(arr, 0, size_of_array - 1, 1));
    EXPECT_EQ(5, binarySearchRange(arr, 0, size_of_array - 1, 9));
    EXPECT_EQ(8, binarySearchRange(arr, 0, size_of_array - 1, 10));

    int expects[total_count];
    EXPECT_EQ(total_count, calc_binary_search_result(expects, total_count, arr, size_of_array));

    for (int i = 0; i < total_count; i++) {
        EXPECT_EQ(expects[i], binarySearchRange(arr, 0, size_of_array-1, i + 1));
    }
}

TEST(KernelUtil, convert_DateTime)
{
    const int day = 24 * 60 * 60 * 1000;

    EXPECT_EQ(0 * day, convertDateToGdfDate64(1970, 1, 1));
    EXPECT_EQ(-1 * day, convertDateToGdfDate64(1969, 12, 31));

    EXPECT_EQ(0, convertDateToUnixEpoch(1970, 1, 1));
    EXPECT_EQ(-1, convertDateToUnixEpoch(1969, 12, 31));

    EXPECT_EQ(0, convertDateToOrcTimestampDate(2015, 1, 1));
    EXPECT_EQ((13 * 60 + 15) * 60 + 40, convertDateToOrcTimestampDate(2015, 1, 1, 13, 15, 40));

    EXPECT_EQ(0, convertDateToOrcTimestampDate(2015, 1, 1));

    EXPECT_EQ(convertDateToOrcTimestampTime(10, 0, 0), (1 << 3) + 7);
    EXPECT_EQ(convertDateToOrcTimestampTime(123, 0, 0), (123 << 3) + 6);
    EXPECT_EQ(convertDateToOrcTimestampTime(0, 999, 999), (orc_uint64(999999) << 3));
    EXPECT_EQ(convertDateToOrcTimestampTime(999, 999, 999), (orc_uint64(999999999) << 3));


}

