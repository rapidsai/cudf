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

#include <cstdlib>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>

#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

#include "gtest/gtest.h"
#include <../../src/groupby/hash/groupby_kernels.cuh>
#include <../../src/groupby/hash/groupby_compute_api.h>
#include <../../src/groupby/hash/aggregation_operations.cuh>



/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  This file is for unit testing all functions and kernels that are below the
 * public libgdf groupby API.
 */
/* ----------------------------------------------------------------------------*/



// This is necessary to do a parametrized typed-test over multiple template arguments
template <typename Key, typename Value, template <typename T> class Aggregation_Operator>
struct KeyValueTypes
{
  using key_type = Key;
  using value_type = Value;
  using op_type = Aggregation_Operator<value_type>;
};

// A new instance of this class will be created for each *TEST(GroupByTest, ...)
// Put all repeated stuff for each test here
template <class T>
struct GroupByTest : public testing::Test 
{
  using key_type = typename T::key_type;
  using value_type = typename T::value_type;
  using op_type = typename T::op_type;
  using map_type = concurrent_unordered_map<key_type, value_type, std::numeric_limits<key_type>::max()>;
  using size_type = typename map_type::size_type;

  std::unique_ptr<map_type> the_map;

  const key_type unused_key = std::numeric_limits<key_type>::max();
  const value_type unused_value = op_type::IDENTITY;

  size_type hash_table_size;
  size_type input_size;

  const int THREAD_BLOCK_SIZE{256};

  std::vector<key_type> groupby_column;
  std::vector<value_type> aggregation_column;

  thrust::device_vector<key_type> d_groupby_column;
  thrust::device_vector<value_type> d_aggregation_column;

  key_type *d_groupby_result{nullptr};
  value_type *d_aggregation_result{nullptr};


  GroupByTest(const size_type _hash_table_size = 10000)
    : hash_table_size(_hash_table_size), the_map(new map_type(_hash_table_size, op_type::IDENTITY))
  {
  }

  ~GroupByTest()
  {
    cudaFree(d_groupby_result);
    cudaFree(d_aggregation_result);
  }

  std::pair<key_type*, value_type*>
  create_input(const int num_keys, const int num_values_per_key, const int max_key = RAND_MAX, const int max_value = RAND_MAX, bool print = false, const int ratio = 1) 
  {

    input_size = num_keys * num_values_per_key;

    hash_table_size = ratio * input_size;

    this->the_map.reset(new map_type(hash_table_size, unused_value));

    groupby_column.reserve(input_size);
    aggregation_column.reserve(input_size);

    // Always use the same seed so the random sequence is the same each time
    std::srand(0);

    for(int i = 0; i < num_keys; ++i )
    {
      // Create random key
      key_type current_key = std::rand() % max_key;

      // Don't use unused_key
      while(current_key == this->unused_key)
      {
        current_key = std::rand();
      }

      // For the current key, generate random values
      for(int j = 0; j < num_values_per_key; ++j)
      {
        value_type current_value = std::rand() % max_value;

        // Don't use unused_value
        while(current_value == this->unused_value)
        {
          current_value = std::rand() % max_value;
        }

        // Store current key and value
        groupby_column.push_back(current_key);
        aggregation_column.push_back(current_value);

      }
    }

    d_groupby_column = groupby_column;
    d_aggregation_column = aggregation_column;

    return std::make_pair(thrust::raw_pointer_cast(d_groupby_column.data()), 
                          thrust::raw_pointer_cast(d_aggregation_column.data()));
  }

  template <class aggregation_operation>
  std::map<key_type, value_type> compute_reference_solution(bool print = false)
  {
    std::map<key_type, value_type> expected_values;

    aggregation_operation op;

    for(size_t i = 0; i < groupby_column.size(); ++i){

      key_type current_key = groupby_column[i];
      value_type current_value = aggregation_column[i];

      // Use a STL map to keep track of the aggregation for each key
      auto found = expected_values.find(current_key);

      // Key doesn't exist yet, insert it
      if(found == expected_values.end())
      {
        // To support operations like `count`, on the first insert, perform the
        // operation on the new value and the operation's identity value and store the result
        current_value = op(current_value, aggregation_operation::IDENTITY);

        expected_values.insert(std::make_pair(current_key,current_value)); 

        if(print)
          std::cout << "First Insert of Key: " << current_key << " value: " << current_value << std::endl;
      }
      // Key exists, update the value with the operator
      else
      {
        value_type new_value = op(current_value, found->second);
        if(print)
          std::cout << "Insert of Key: " << current_key << " inserting value: " << current_value 
            << " storing: " << new_value << std::endl;
        found->second = new_value;
      }
    }

    return expected_values;
  }

  void build_aggregation_table_device(std::pair<key_type*, value_type*> input)
  {

    const dim3 grid_size ((this->input_size + this->THREAD_BLOCK_SIZE - 1) / this->THREAD_BLOCK_SIZE, 1, 1);
    const dim3 block_size (this->THREAD_BLOCK_SIZE, 1, 1);

    key_type * d_group = input.first;
    value_type * d_agg = input.second;

    cudaDeviceSynchronize();
    build_aggregation_table<<<grid_size, block_size>>>((this->the_map).get(), d_group, d_agg, this->input_size, op_type());
    cudaDeviceSynchronize(); 

  }

  void verify_aggregation_table(std::map<key_type, value_type> const & expected_values){

    for(auto const &expected : expected_values)
    {
      key_type test_key = expected.first;

      value_type expected_value = expected.second;

      auto found = this->the_map->find(test_key);

      ASSERT_NE(this->the_map->end(), found) << "Key is: " << test_key;

      value_type test_value = found->second;

      if(std::is_integral<value_type>::value){
        EXPECT_EQ(expected_value, test_value) << "Key is: " << test_key;
      }
      else if(std::is_same<value_type, float>::value){
        EXPECT_FLOAT_EQ(expected_value, test_value) << "Key is: " << test_key;
      }
      else if(std::is_same<value_type, double>::value){
        EXPECT_DOUBLE_EQ(expected_value, test_value) << "Key is: " << test_key;
      }
      else{
        std::cout << "Unhandled value type.\n";
      }
    }
  }

  size_t extract_groupby_result_device()
  {

    const dim3 grid_size ((this->hash_table_size + this->THREAD_BLOCK_SIZE - 1) / this->THREAD_BLOCK_SIZE, 1, 1);
    const dim3 block_size (this->THREAD_BLOCK_SIZE, 1, 1);

    // TODO: Find a more efficient way to size the output buffer.
    // In general, input_size is going to be larger than the actual
    // size of the result.
    cudaMallocManaged(&d_groupby_result, input_size * sizeof(key_type));
    cudaMallocManaged(&d_aggregation_result, input_size * sizeof(value_type));

    // This variable is used by the threads to coordinate where they should write 
    // to the output buffers
    unsigned int * global_write_index{nullptr}; 
    cudaMallocManaged(&global_write_index, sizeof(unsigned int));
    *global_write_index = 0;

    cudaDeviceSynchronize();
    extract_groupby_result<<<grid_size, block_size>>>((this->the_map).get(), 
                                                      (this->the_map)->size(), 
                                                      d_groupby_result, 
                                                      d_aggregation_result, 
                                                      global_write_index );
    cudaDeviceSynchronize();

    size_t result_size = *global_write_index;

    // Return the actual size of the result
    return result_size;

  }

  void verify_groupby_result(size_t computed_result_size, std::map<key_type, value_type> const & expected_values )
  {

    ASSERT_NE(nullptr, d_groupby_result);
    ASSERT_NE(nullptr, d_aggregation_result);

    // The size of the result should be equal to the number of unique keys
    const auto begin = this->d_groupby_column.begin();
    const auto end = this->d_groupby_column.end();
    thrust::sort(begin, end);
    size_t unique_count = thrust::unique(begin, end) - begin;
    ASSERT_EQ(unique_count, computed_result_size);

    // Prefetch groupby and aggregation result to host to improve performance
    cudaMemPrefetchAsync(d_groupby_result, input_size * sizeof(key_type), cudaCpuDeviceId);
    cudaMemPrefetchAsync(d_aggregation_result, input_size * sizeof(value_type), cudaCpuDeviceId);

    // Verify that every <key,value> in the computed result is present in the reference solution
    for(size_type i = 0; i < expected_values.size(); ++i)
    {
      key_type groupby_key = d_groupby_result[i];
      value_type aggregation_value = d_aggregation_result[i];

      auto found = expected_values.find(groupby_key);

      ASSERT_NE(expected_values.end(), found) << "key: " << groupby_key;

      EXPECT_EQ(found->first, groupby_key) << "index: " << i;

      if(std::is_integral<value_type>::value){
        EXPECT_EQ(found->second, aggregation_value) << "key: " << groupby_key << " index: " << i;
      }
      else if(std::is_same<value_type, float>::value){
        EXPECT_FLOAT_EQ(found->second, aggregation_value) << "key: " << groupby_key << " index: " << i;
      }
      else if(std::is_same<value_type, double>::value){
        EXPECT_DOUBLE_EQ(found->second, aggregation_value) << "key: " << groupby_key << " index: " << i;
      }
      else{
        std::cout << "Unhandled value type.\n";
      }
    }

  }

  unsigned int groupby(const key_type * const groupby_column, const value_type * const aggregation_column)
  {

    // TODO: Find a more efficient way to size the output buffer.
    // In general, input_size is going to be larger than the actual
    // size of the result.
    cudaMallocManaged(&d_groupby_result, input_size * sizeof(key_type));
    cudaMallocManaged(&d_aggregation_result, input_size * sizeof(value_type));

    size_type result_size{0};

    GroupbyHash(groupby_column,
                aggregation_column,
                input_size,
                d_groupby_result,
                d_aggregation_result,
                &result_size,
                op_type());

    return result_size;
  }

};

// Google Test can only do a parameterized typed-test over a single type, so we have
// to nest multiple types inside of the KeyValueTypes struct above
// KeyValueTypes<type1, type2> implies key_type = type1, value_type = type2
// This list is the types across which Google Test will run our tests
typedef ::testing::Types< 
                            KeyValueTypes<int32_t, int32_t, max_op>,
                            KeyValueTypes<int32_t, float, max_op>,
                            KeyValueTypes<int32_t, double, max_op>,
                            KeyValueTypes<int32_t, int64_t, max_op>,
                            KeyValueTypes<int32_t, uint64_t, max_op>,
                            KeyValueTypes<int64_t, int32_t, max_op>,
                            KeyValueTypes<int64_t, float, max_op>,
                            KeyValueTypes<int64_t, double, max_op>,
                            KeyValueTypes<int64_t, int64_t, max_op>,
                            KeyValueTypes<int64_t, uint64_t, max_op>,
                            KeyValueTypes<uint64_t, int32_t, max_op>,
                            KeyValueTypes<uint64_t, float, max_op>,
                            KeyValueTypes<uint64_t, double, max_op>,
                            KeyValueTypes<uint64_t, int64_t, max_op>,
                            KeyValueTypes<uint64_t, uint64_t, max_op>,
                            KeyValueTypes<int32_t, int32_t, min_op>,
                            KeyValueTypes<int32_t, float, min_op>,
                            KeyValueTypes<int32_t, double, min_op>,
                            KeyValueTypes<int32_t, int64_t, min_op>,
                            KeyValueTypes<int32_t, uint64_t, min_op>,
                            KeyValueTypes<uint64_t, int32_t, min_op>,
                            KeyValueTypes<uint64_t, float, min_op>,
                            KeyValueTypes<uint64_t, double, min_op>,
                            KeyValueTypes<uint64_t, int64_t, min_op>,
                            KeyValueTypes<uint64_t, uint64_t, min_op>,
                            KeyValueTypes<int32_t, int32_t, count_op>,
                            KeyValueTypes<int32_t, float, count_op>,
                            KeyValueTypes<int32_t, double, count_op>,
                            KeyValueTypes<int32_t, int64_t, count_op>,
                            KeyValueTypes<int32_t, uint64_t, count_op>,
                            KeyValueTypes<uint64_t, int32_t, count_op>,
                            KeyValueTypes<uint64_t, float, count_op>,
                            KeyValueTypes<uint64_t, double, count_op>,
                            KeyValueTypes<uint64_t, int64_t, count_op>,
                            KeyValueTypes<uint64_t, uint64_t, count_op>,
                            KeyValueTypes<int32_t, int32_t, sum_op>,
                            //KeyValueTypes<int32_t, float, sum_op>, // TODO: Tests for SUM on single precision floats currently fail due to numerical stability issues
                            KeyValueTypes<int32_t, double, sum_op>,
                            KeyValueTypes<int32_t, int64_t, sum_op>,
                            KeyValueTypes<int32_t, uint64_t, sum_op>,
                            KeyValueTypes<uint64_t, double, sum_op>,
                            KeyValueTypes<uint64_t, double, sum_op>,
                            KeyValueTypes<uint64_t, int64_t, sum_op>,
                            KeyValueTypes<uint64_t, uint64_t, sum_op>
                            > Implementations;

  TYPED_TEST_CASE(GroupByTest, Implementations);




TYPED_TEST(GroupByTest, AggregationTestDeviceAllSame)
{
  const int num_keys = 1;
  const int num_values_per_key = 1<<12;

  auto input = this->create_input(num_keys, num_values_per_key);

  // When you have a templated member function of a templated class, the preceeding 'template' keyword is required
  // See: https://stackoverflow.com/questions/16508743/error-expected-expression-in-this-template-code
  using aggregation_op = typename GroupByTest<TypeParam>::op_type;
  auto expected_values = this->template compute_reference_solution<aggregation_op>();

  this->build_aggregation_table_device(input);
  this->verify_aggregation_table(expected_values);

  size_t computed_result_size = this->extract_groupby_result_device();
  this->verify_groupby_result(computed_result_size, expected_values);
}

// TODO Update the create_input function to ensure all keys are actually unique
TYPED_TEST(GroupByTest, AggregationTestDeviceAllUnique)
{
  const int num_keys = 1<<12;
  const int num_values_per_key = 1;
  auto input = this->create_input(num_keys, num_values_per_key);
  // When you have a templated member function of a templated class, the preceeding 'template' keyword is required
  // See: https://stackoverflow.com/questions/16508743/error-expected-expression-in-this-template-code
  using aggregation_op = typename GroupByTest<TypeParam>::op_type;
  auto expected_values = this->template compute_reference_solution<aggregation_op>();

  this->build_aggregation_table_device(input);
  this->verify_aggregation_table(expected_values);

  size_t computed_result_size = this->extract_groupby_result_device();
  this->verify_groupby_result(computed_result_size,expected_values);
}

TYPED_TEST(GroupByTest, AggregationTestDeviceWarpSame)
{
  const int num_keys = 1<<12;
  const int num_values_per_key = 32;

  auto input = this->create_input(num_keys, num_values_per_key);
  // When you have a templated member function of a templated class, the preceeding 'template' keyword is required
  // See: https://stackoverflow.com/questions/16508743/error-expected-expression-in-this-template-code
  using aggregation_op = typename GroupByTest<TypeParam>::op_type;
  auto expected_values = this->template compute_reference_solution<aggregation_op>();

  this->build_aggregation_table_device(input);
  this->verify_aggregation_table(expected_values);

  size_t computed_result_size = this->extract_groupby_result_device();
  this->verify_groupby_result(computed_result_size,expected_values);
}

TYPED_TEST(GroupByTest, AggregationTestDeviceBlockSame)
{
  const int num_keys = 1<<8;
  const int num_values_per_key = this->THREAD_BLOCK_SIZE;
  auto input = this->create_input(num_keys, num_values_per_key);
  // When you have a templated member function of a templated class, the preceeding 'template' keyword is required
  // See: https://stackoverflow.com/questions/16508743/error-expected-expression-in-this-template-code
  using aggregation_op = typename GroupByTest<TypeParam>::op_type;
  auto expected_values = this->template compute_reference_solution<aggregation_op>();

  this->build_aggregation_table_device(input);
  this->verify_aggregation_table(expected_values);

  size_t computed_result_size = this->extract_groupby_result_device();
  this->verify_groupby_result(computed_result_size, expected_values);
}

TYPED_TEST(GroupByTest, GroupByHash)
{
  const int num_keys = 1<<12;
  const int num_values_per_key = 1;

  auto input = this->create_input(num_keys, num_values_per_key);
  // When you have a templated member function of a templated class, the preceeding 'template' keyword is required
  // See: https://stackoverflow.com/questions/16508743/error-expected-expression-in-this-template-code
  using aggregation_op = typename GroupByTest<TypeParam>::op_type;
  auto expected_values = this->template compute_reference_solution<aggregation_op>();

  const size_t computed_result_size = this->groupby(input.first, input.second);
  this->verify_groupby_result(computed_result_size,expected_values);
}


