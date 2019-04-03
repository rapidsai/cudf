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

#include <tests/utilities/cudf_test_fixtures.h>

#include <hash/concurrent_unordered_map.cuh>
#include <groupby/aggregation_operations.hpp>

#include <cudf.h>

#include <thrust/device_vector.h>

#include <rmm/thrust_rmm_allocator.h>

#include <gtest/gtest.h>


#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>

#include <cstdlib>


// This is necessary to do a parametrized typed-test over multiple template arguments
template <typename Key, typename Value, template <typename> typename Aggregation_Operator>
struct KeyValueTypes
{
  using key_type = Key;
  using value_type = Value;
  using op_type = Aggregation_Operator<value_type>;
};

// A new instance of this class will be created for each *TEST(MapTest, ...)
// Put all repeated stuff for each test here
template <class T>
struct MapTest : public GdfTest
{
  using key_type = typename T::key_type;
  using value_type = typename T::value_type;
  using op_type = typename T::op_type;
  using map_type = concurrent_unordered_map<key_type, value_type, std::numeric_limits<key_type>::max()>;
  using pair_type = thrust::pair<key_type, value_type>;

  std::unique_ptr<map_type> the_map;

  const key_type unused_key = std::numeric_limits<key_type>::max();
  const value_type unused_value = op_type::IDENTITY;

  const int size;

  const int THREAD_BLOCK_SIZE{256};

  std::vector<thrust::pair<key_type,value_type>> pairs;

  rmm::device_vector<pair_type> d_pairs;

  std::unordered_map<key_type, value_type> expected_values;

  MapTest(const int hash_table_size = 10000)
    : size(hash_table_size), the_map(new map_type(hash_table_size, op_type::IDENTITY))
  {
  }

  pair_type * create_input(const int num_unique_keys, const int num_values_per_key, const int ratio = 2, const int max_key = RAND_MAX, const int max_value = RAND_MAX, bool shuffle = false)
  {

    const int TOTAL_PAIRS = num_unique_keys * num_values_per_key;

    this->the_map.reset(new map_type(ratio*TOTAL_PAIRS, unused_value));

    pairs.reserve(TOTAL_PAIRS);

    // Always use the same seed so the random sequence is the same each time
    std::srand(0);

    for(int i = 0; i < num_unique_keys; ++i )
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
          current_value = std::rand();
        }

        // Store current key and value
        pairs.push_back(std::make_pair(current_key, current_value));

        // Use a STL map to keep track of the max value for each key
        auto found = expected_values.find(current_key);

        // Key doesn't exist yet, insert it
        if(found == expected_values.end())
        {
          expected_values.insert(std::make_pair(current_key,current_value));
        }
        // Key exists, update the value with the operator
        else
        {
          op_type op;
          value_type new_value = op(found->second, current_value);
          found->second = new_value;
        }
      }
    }

    if(shuffle == true)
      std::random_shuffle(pairs.begin(), pairs.end());

    d_pairs = pairs;

    return thrust::raw_pointer_cast(d_pairs.data());
  }

  void check_answer(){

    for(auto const &k : this->expected_values)
    {
      key_type test_key = k.first;

      value_type expected_value = k.second;

      auto found = this->the_map->find(test_key);

      ASSERT_NE(this->the_map->end(), found);

      value_type test_value = found->second;

      EXPECT_EQ(expected_value, test_value) << "Key is: " << test_key;
    }
  }

  ~MapTest(){
  }


};

// Google Test can only do a parameterized typed-test over a single type, so we have
// to nest multiple types inside of the KeyValueTypes struct above
// KeyValueTypes<type1, type2> implies key_type = type1, value_type = type2
// This list is the types across which Google Test will run our tests
typedef ::testing::Types< KeyValueTypes<int, int, max_op>,
                          KeyValueTypes<int, float, max_op>,
                          KeyValueTypes<int, double, max_op>,
                          KeyValueTypes<int, long long int, max_op>,
                          KeyValueTypes<int, unsigned long long int, max_op>,
                          KeyValueTypes<unsigned long long int, int, max_op>,
                          KeyValueTypes<unsigned long long int, float, max_op>,
                          KeyValueTypes<unsigned long long int, double, max_op>,
                          KeyValueTypes<unsigned long long int, long long int, max_op>,
                          KeyValueTypes<unsigned long long int, unsigned long long int, max_op>,
                          KeyValueTypes<int, int, min_op>,
                          KeyValueTypes<int, float, min_op>,
                          KeyValueTypes<int, double, min_op>,
                          KeyValueTypes<int, long long int, min_op>,
                          KeyValueTypes<int, unsigned long long int, min_op>,
                          KeyValueTypes<unsigned long long int, int, min_op>,
                          KeyValueTypes<unsigned long long int, float, min_op>,
                          KeyValueTypes<unsigned long long int, double, min_op>,
                          KeyValueTypes<unsigned long long int, long long int, min_op>,
                          KeyValueTypes<unsigned long long int, unsigned long long int, min_op>
                          > Implementations;

TYPED_TEST_CASE(MapTest, Implementations);

TYPED_TEST(MapTest, InitialState)
{
  using key_type = typename TypeParam::key_type;
  using value_type = typename TypeParam::value_type;

  auto begin = this->the_map->begin();
  auto end = this->the_map->end();
  EXPECT_NE(begin,end);

}

TYPED_TEST(MapTest, CheckUnusedValues){

  EXPECT_EQ(this->the_map->get_unused_key(), this->unused_key);

  auto begin = this->the_map->begin();
  EXPECT_EQ(begin->first, this->unused_key);
  EXPECT_EQ(begin->second, this->unused_value);
}


template<typename map_type, typename Aggregation_Operator>
__global__ void build_table(map_type * const the_map,
                            const typename map_type::value_type * const input_pairs,
                            const typename map_type::size_type input_size,
                            Aggregation_Operator op)
{

  using size_type = typename map_type::size_type;

  size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  while( i < input_size ){
    the_map->insert(input_pairs[i], op);
    i += blockDim.x * gridDim.x;
  }

}



TYPED_TEST(MapTest, AggregationTestDeviceAllSame)
{
  using value_type = typename TypeParam::value_type;
  using pair_type = typename MapTest<TypeParam>::pair_type;
  using op_type = typename MapTest<TypeParam>::op_type;

  pair_type * d_pairs = this->create_input(1, 1<<20);

  const dim3 grid_size ((this->d_pairs.size() + this->THREAD_BLOCK_SIZE -1) / this->THREAD_BLOCK_SIZE,1,1);
  const dim3 block_size (this->THREAD_BLOCK_SIZE, 1, 1);

  cudaDeviceSynchronize();
  build_table<<<grid_size, block_size>>>((this->the_map).get(), d_pairs, this->d_pairs.size(), op_type());
  cudaDeviceSynchronize(); 

  this->check_answer();

}

TYPED_TEST(MapTest, AggregationTestDeviceAllUnique)
{
  using value_type = typename TypeParam::value_type;
  using pair_type = typename MapTest<TypeParam>::pair_type;
  using op_type = typename MapTest<TypeParam>::op_type;
  

  pair_type * d_pairs = this->create_input(1<<18, 1);

  const dim3 grid_size ((this->d_pairs.size() + this->THREAD_BLOCK_SIZE -1) / this->THREAD_BLOCK_SIZE,1,1);
  const dim3 block_size (this->THREAD_BLOCK_SIZE, 1, 1);

  cudaDeviceSynchronize();
  build_table<<<grid_size, block_size>>>((this->the_map).get(), d_pairs, this->d_pairs.size(), op_type());
  cudaDeviceSynchronize(); 

  this->check_answer();
}

TYPED_TEST(MapTest, AggregationTestDeviceWarpSame)
{
  using value_type = typename TypeParam::value_type;
  using pair_type = typename MapTest<TypeParam>::pair_type;
  using op_type = typename MapTest<TypeParam>::op_type;

  pair_type * d_pairs = this->create_input(1<<15, 32);

  const dim3 grid_size ((this->d_pairs.size() + this->THREAD_BLOCK_SIZE -1) / this->THREAD_BLOCK_SIZE,1,1);
  const dim3 block_size (this->THREAD_BLOCK_SIZE, 1, 1);

  cudaDeviceSynchronize();
  build_table<<<grid_size, block_size>>>((this->the_map).get(), d_pairs, this->d_pairs.size(), op_type());
  cudaDeviceSynchronize(); 

  this->check_answer();
}

TYPED_TEST(MapTest, AggregationTestDeviceBlockSame)
{
  using value_type = typename TypeParam::value_type;
  using pair_type = typename MapTest<TypeParam>::pair_type;
  using op_type = typename MapTest<TypeParam>::op_type;

  pair_type * d_pairs = this->create_input(1<<12, this->THREAD_BLOCK_SIZE);

  const dim3 grid_size ((this->d_pairs.size() + this->THREAD_BLOCK_SIZE -1) / this->THREAD_BLOCK_SIZE,1,1);
  const dim3 block_size (this->THREAD_BLOCK_SIZE, 1, 1);

  cudaDeviceSynchronize();
  build_table<<<grid_size, block_size>>>((this->the_map).get(), d_pairs, this->d_pairs.size(), op_type());
  cudaDeviceSynchronize(); 

  this->check_answer();
}


int main(int argc, char * argv[]){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
