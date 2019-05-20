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

/* Proof the concept for iterator driven aggregations to reuse the logic

   The concepts:
   1. computes the aggregation by given iterators
   2. computes by using cub and thrust with same function parameters
   3. accepts nulls and group_by with same function parameters

    CUB:  https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html#aa4adabeb841b852a7a5ecf4f99a2daeb
    Thrust: https://thrust.github.io/doc/group__reductions.html#ga43eea9a000f912716189687306884fc7
*/

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <bitset>
#include <cstdint>
#include <iostream>

#include <utilities/device_atomics.cuh> // need for device operators.

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

#include <cub/device/device_reduce.cuh>
#include <thrust/device_vector.h>
#include <thrust/transform.h>


template <typename T>
struct IteratorTest : public GdfTest
{
    // iterator test case which uses cub
    template <typename InputIterator, typename T_output>
    void iterator_test_cub(T_output expected, InputIterator d_in, int num_items)
    {
        T init = T{0};
        thrust::device_vector<T> dev_result(1);

        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;

        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, dev_result.begin(), num_items,
            cudf::DeviceSum{}, init);
        // Allocate temporary storage
        RMM_TRY(RMM_ALLOC(&d_temp_storage, temp_storage_bytes, 0));

        // Run reduction
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, dev_result.begin(), num_items,
            cudf::DeviceSum{}, init);

        evaluate(expected, dev_result, "cub test");
    }

    // iterator test case which uses thrust
    template <typename InputIterator, typename T_output>
    void iterator_test_thrust(T_output expected, InputIterator d_in, int num_items)
    {
        T init = T{0};

        InputIterator d_in_last =  d_in + num_items;

        T result = thrust::reduce(d_in, d_in_last, init, cudf::DeviceSum{});

        EXPECT_EQ(expected, result) << "thrust test";
    }

    void evaluate(T expected, thrust::device_vector<T> &dev_result, const char* msg=nullptr)
    {
        thrust::host_vector<T>  hos_result(dev_result);

        EXPECT_EQ(expected, dev_result[0]) << msg ;
//        std::cout << "expected <" << msg << "> = " << expected << std::endl;
    }
};

using TestingTypes = ::testing::Types<
    int32_t
>;

TYPED_TEST_CASE(IteratorTest, TestingTypes);


// tests for non-null iterator (pointer of device array)
TYPED_TEST(IteratorTest, non_null_iterator)
{
    using T = int32_t;
    std::vector<T> hos_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
    thrust::device_vector<T> dev_array(hos_array);

    T expected_value = std::accumulate(hos_array.begin(), hos_array.end(), T{0});

    this->iterator_test_cub(expected_value, dev_array.begin(), dev_array.size());

    this->iterator_test_thrust(expected_value, dev_array.begin(), dev_array.size());
}

// tests for null iterator (column with null bitmap)
TYPED_TEST(IteratorTest, null_iterator)
{
    // TBD.
}

// tests for group_by iterator
TYPED_TEST(IteratorTest, group_by_iterator)
{
    // TBD.
}


// tests for group_by iterator
TYPED_TEST(IteratorTest, group_by_iterator_null)
{
    // Discussion: how to do if all of values are nulls ?
    // maybe need to exclude null values first ? (it also gives `count` of a column value in the group)

    // TBD.
}