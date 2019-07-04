/*
 * Copyright (c) 2018-19, NVIDIA CORPORATION.
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

#include <cudf/cudf.h>
#include <tests/utilities/cudf_test_fixtures.h>
#include <hash/concurrent_unordered_map.cuh>
#include <groupby/aggregation_operations.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <thrust/device_vector.h>
#include <cstdlib>
#include <limits>
#include <thrust/logical.h>


// This is necessary to do a parametrized typed-test over multiple template arguments
template <typename Key, typename Value, template <typename> typename Aggregation_Operator>
struct KeyValueTypes
{
  using key_type = Key;
  using value_type = Value;
  using op_type = Aggregation_Operator<value_type>;
};

template <typename Map>
struct key_finder{

  using key_type = typename Map::key_type;
  using value_type = typename Map::mapped_type;
  key_finder(Map* _map) : map{_map} {}

  __device__ bool operator()(thrust::pair<key_type, value_type> const& p) {
    auto found = map->find(p.first);

    if (found == map->end()) {
      return false;
    }
    return found->second == p.second;
  }
  Map* map;
};

// A new instance of this class will be created for each *TEST(MapTest, ...)
// Put all repeated stuff for each test here
template <class T>
struct MapTest : public GdfTest
{
  using key_type = typename T::key_type;
  using value_type = typename T::value_type;
  using op_type = typename T::op_type;
  using map_type = concurrent_unordered_map < key_type, value_type,
        default_hash<key_type>, equal_to<key_type>,
        managed_allocator<thrust::pair<key_type, value_type>>>;
  using pair_type = thrust::pair<key_type, value_type>;

  std::unique_ptr<map_type> the_map;

  const key_type unused_key{std::numeric_limits<key_type>::max()};
  const value_type unused_element{op_type::IDENTITY};

  const std::size_t size;

  const int THREAD_BLOCK_SIZE{256};

  std::vector<thrust::pair<key_type,value_type>> pairs;

  rmm::device_vector<pair_type> d_pairs;

  std::unordered_map<key_type, value_type> expected_values;

  MapTest(const std::size_t hash_table_size = 10000)
      : size(hash_table_size),
        the_map(new map_type(hash_table_size, unused_element, unused_key)) {
    EXPECT_EQ(hash_table_size, the_map->capacity());
  }

  pair_type * create_input(const int num_unique_keys, const int num_values_per_key, const int ratio = 2, const int max_key = RAND_MAX, const int max_value = RAND_MAX, bool shuffle = false)
  {

    const int TOTAL_PAIRS = num_unique_keys * num_values_per_key;

    this->the_map.reset(new map_type(ratio*TOTAL_PAIRS, unused_element, unused_key));
    EXPECT_EQ(static_cast<std::size_t>(ratio * TOTAL_PAIRS), the_map->capacity());
    EXPECT_EQ(unused_key, the_map->get_unused_key());
    EXPECT_EQ(unused_element, the_map->get_unused_element());

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

        // Don't use unused_element
        while(current_value == this->unused_element)
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
    std::vector<thrust::pair<key_type, value_type>> host_pairs(
        expected_values.begin(), expected_values.end());

    thrust::device_vector<thrust::pair<key_type, value_type>> expected_pairs(
        host_pairs);

    EXPECT_TRUE(thrust::all_of(expected_pairs.begin(), expected_pairs.end(),
                               key_finder<map_type>(this->the_map.get())));
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

  pair_type * d_pairs = this->create_input(1, 10000);

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
  

  pair_type * d_pairs = this->create_input(10000, 1);

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

  pair_type * d_pairs = this->create_input(5000, 32);

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

  pair_type * d_pairs = this->create_input(1000, this->THREAD_BLOCK_SIZE);

  const dim3 grid_size ((this->d_pairs.size() + this->THREAD_BLOCK_SIZE -1) / this->THREAD_BLOCK_SIZE,1,1);
  const dim3 block_size (this->THREAD_BLOCK_SIZE, 1, 1);

  cudaDeviceSynchronize();
  build_table<<<grid_size, block_size>>>((this->the_map).get(), d_pairs, this->d_pairs.size(), op_type());
  cudaDeviceSynchronize(); 

  this->check_answer();
}

template <typename K, typename V>
struct key_value_types{
  using key_type = K;
  using value_type = V;
  using pair_type = thrust::pair<K, V>;
  using map_type = concurrent_unordered_map<key_type, value_type>;
};

template <typename T>
struct InsertTest : public GdfTest {
  using key_type = typename T::key_type;
  using value_type = typename T::value_type;
  using pair_type = typename T::pair_type;
  using map_type = typename T::map_type;

  InsertTest(){

      // prevent overflow of small types
      const size_t input_size = std::min(static_cast<key_type>(size), std::numeric_limits<key_type>::max());

      pairs.resize(input_size);
      map.reset(new map_type(compute_hash_table_size(size)));
    }

  const gdf_size_type size{10000};
  rmm::device_vector<pair_type> pairs;
  std::unique_ptr<map_type> map;
};

using TestTypes = ::testing::Types<
    key_value_types<int32_t, int32_t>, key_value_types<int64_t, int64_t>,
    key_value_types<int8_t, int8_t>, key_value_types<int16_t, int16_t>,
    key_value_types<int8_t, float>, key_value_types<int16_t, double>,
    key_value_types<int32_t, float>, key_value_types<int64_t, double>>;

TYPED_TEST_CASE(InsertTest, TestTypes);

template <typename map_type, typename pair_type>
struct insert_pair {
  insert_pair(map_type* _map) : map{_map} {}

  __device__ bool operator()(pair_type const& pair) {

    auto result = map->insert(pair);
    if (result.first == map->end()) {
      return false;
    }
    return result.second;
  }

  map_type* map;
};

template <typename map_type, typename pair_type>
struct find_pair {
  find_pair(map_type* _map) : map{_map} {}

  __device__ bool operator()(pair_type const& pair) {
    auto result = map->find(pair.first);
    if (result == map->end()) {
      return false;
    }
    return *result == pair;
  }
  map_type* map;
};

template <typename pair_type,
          typename key_type = typename pair_type::first_type,
          typename value_type = typename pair_type::second_type>
struct unique_pair_generator{
  __device__ pair_type operator()(gdf_size_type i) {
    return thrust::make_pair(key_type(i), value_type(i));
  }
};

template <typename pair_type,
          typename key_type = typename pair_type::first_type,
          typename value_type = typename pair_type::second_type>
struct identical_pair_generator {
  identical_pair_generator(key_type k = 42, value_type v = 42)
      : key{k}, value{v} {}
  __device__ pair_type operator()(gdf_size_type i) {
    return thrust::make_pair(key, value);
  }
  key_type key;
  value_type value;
};

template <typename pair_type,
          typename key_type = typename pair_type::first_type,
          typename value_type = typename pair_type::second_type>
struct identical_key_generator {
  identical_key_generator(key_type k = 42)
      : key{k}{}
  __device__ pair_type operator()(gdf_size_type i) {
    return thrust::make_pair(key, value_type(i));
  }
  key_type key;
};

TYPED_TEST(InsertTest, UniqueKeysUniqueValues) {
  using map_type = typename TypeParam::map_type;
  using pair_type = typename TypeParam::pair_type;
  thrust::tabulate(this->pairs.begin(), this->pairs.end(),
                   unique_pair_generator<pair_type>{});
  // All pairs should be new inserts
  EXPECT_TRUE(
      thrust::all_of(this->pairs.begin(), this->pairs.end(),
                     insert_pair<map_type, pair_type>{this->map.get()}));

  // All pairs should be present in the map
  EXPECT_TRUE(thrust::all_of(this->pairs.begin(), this->pairs.end(),
                             find_pair<map_type, pair_type>{this->map.get()}));
}

TYPED_TEST(InsertTest, IdenticalKeysIdenticalValues) {
  using map_type = typename TypeParam::map_type;
  using pair_type = typename TypeParam::pair_type;
  thrust::tabulate(this->pairs.begin(), this->pairs.end(),
                   identical_pair_generator<pair_type>{});
  // Insert a single pair
  EXPECT_TRUE(
      thrust::all_of(this->pairs.begin(), this->pairs.begin() + 1,
                     insert_pair<map_type, pair_type>{this->map.get()}));
  // Identical inserts should all return false (no new insert)
  EXPECT_FALSE(
      thrust::all_of(this->pairs.begin(), this->pairs.end(),
                     insert_pair<map_type, pair_type>{this->map.get()}));

  // All pairs should be present in the map
  EXPECT_TRUE(thrust::all_of(this->pairs.begin(), this->pairs.end(),
                             find_pair<map_type, pair_type>{this->map.get()}));
}

TYPED_TEST(InsertTest, IdenticalKeysUniqueValues) {
  using map_type = typename TypeParam::map_type;
  using pair_type = typename TypeParam::pair_type;
  thrust::tabulate(this->pairs.begin(), this->pairs.end(),
                   identical_key_generator<pair_type>{});

  // Insert a single pair
  EXPECT_TRUE(
      thrust::all_of(this->pairs.begin(), this->pairs.begin() + 1,
                     insert_pair<map_type, pair_type>{this->map.get()}));

  // Identical key inserts should all return false (no new insert)
  EXPECT_FALSE(
      thrust::all_of(this->pairs.begin() + 1, this->pairs.end(),
                     insert_pair<map_type, pair_type>{this->map.get()}));

  // Only first pair is present in map
  EXPECT_TRUE(thrust::all_of(this->pairs.begin(), this->pairs.begin() + 1,
                             find_pair<map_type, pair_type>{this->map.get()}));

  EXPECT_FALSE(thrust::all_of(this->pairs.begin() + 1, this->pairs.end(),
                              find_pair<map_type, pair_type>{this->map.get()}));
}