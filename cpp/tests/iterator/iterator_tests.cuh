/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS,  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

#include <cudf/detail/iterator.cuh>                             // include iterator header
#include <cudf/detail/utilities/transform_unary_functions.cuh>  //for meanvar

#include <bitset>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_lists.hpp>

#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

// for reduction tests
#include <thrust/device_vector.h>
#include <cub/device/device_reduce.cuh>

// Base Typed test fixture for iterator test

template <typename T>
struct IteratorTest : public cudf::test::BaseFixture {
  // iterator test case which uses cub
  template <typename InputIterator, typename T_output>
  void iterator_test_cub(T_output expected, InputIterator d_in, int num_items)
  {
    T_output init{0};
    thrust::device_vector<T_output> dev_result(1, init);

    // Get temporary storage size
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce(nullptr,
                              temp_storage_bytes,
                              d_in,
                              dev_result.begin(),
                              num_items,
                              thrust::minimum<T_output>{},
                              init);

    // Allocate temporary storage
    rmm::device_buffer d_temp_storage(temp_storage_bytes);

    // Run reduction
    cub::DeviceReduce::Reduce(d_temp_storage.data(),
                              temp_storage_bytes,
                              d_in,
                              dev_result.begin(),
                              num_items,
                              thrust::minimum<T_output>{},
                              init);

    evaluate(expected, dev_result, "cub test");
  }

  // iterator test case which uses thrust
  template <typename InputIterator, typename T_output>
  void iterator_test_thrust(thrust::host_vector<T_output>& expected,
                            InputIterator d_in,
                            int num_items)
  {
    InputIterator d_in_last = d_in + num_items;
    EXPECT_EQ(thrust::distance(d_in, d_in_last), num_items);
    thrust::device_vector<T_output> dev_expected(expected);

    // Can't use this because time_point make_pair bug in libcudacxx
    // bool result = thrust::equal(thrust::device, d_in, d_in_last, dev_expected.begin());
    bool result = thrust::transform_reduce(
      thrust::device,
      thrust::make_zip_iterator(thrust::make_tuple(d_in, dev_expected.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(d_in_last, dev_expected.end())),
      [] __device__(auto it) {
        return static_cast<typename InputIterator::value_type>(thrust::get<0>(it)) ==
               T_output(thrust::get<1>(it));
      },
      true,
      thrust::logical_and<bool>());
#ifndef NDEBUG
    thrust::device_vector<bool> vec(expected.size(), false);
    thrust::transform(
      thrust::device,
      thrust::make_zip_iterator(thrust::make_tuple(d_in, dev_expected.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(d_in_last, dev_expected.end())),
      vec.begin(),
      [] __device__(auto it) { return (thrust::get<0>(it)) == T_output(thrust::get<1>(it)); });
    thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<bool>(std::cout, " "));
    std::cout << std::endl;
#endif

    EXPECT_TRUE(result) << "thrust test";
  }

  template <typename T_output>
  void evaluate(T_output expected,
                thrust::device_vector<T_output>& dev_result,
                const char* msg = nullptr)
  {
    thrust::host_vector<T_output> hos_result(dev_result);

    EXPECT_EQ(expected, hos_result[0]) << msg;
    std::cout << "Done: expected <" << msg
              << "> = "
              //<< hos_result[0] //TODO uncomment after time_point ostream operator<<
              << std::endl;
  }

  template <typename T_output>
  void values_equal_test(thrust::host_vector<T_output>& expected,
                         const cudf::column_device_view& col)
  {
    if (col.nullable()) {
      auto it_dev = cudf::detail::make_null_replacement_iterator(col, T_output{0});
      iterator_test_thrust(expected, it_dev, col.size());
    } else {
      auto it_dev = col.begin<T_output>();
      iterator_test_thrust(expected, it_dev, col.size());
    }
  }
};
