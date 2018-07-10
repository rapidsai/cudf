#include <cstdlib>
#include <iostream>
#include <vector>
#include <unordered_map>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <../../src/hashmap/concurrent_unordered_map.cuh>


// This is necessary to do a parametrized typed-test over multiple template arguments
template <typename Key, typename Value>
struct KeyValueTypes
{
  using key_type = Key;
  using value_type = Value;
};

// Have to use a functor instead of a device lambda because
// you can't create a device lambda inside of a Google Test
// because the macro expands into a private member function
// and you can't have a device lambda inside a private member
// function
template<typename value_type>
  struct max_op
  {
    __host__ __device__
    value_type operator()(value_type a, value_type b)
    {
      return (a > b? a : b);
    }
  };


// A new instance of this class will be created for each *TEST(MapTest, ...)
// Put all repeated stuff for each test here
template <class T>
struct MapTest : public testing::Test 
{
  using key_type = typename T::key_type;
  using value_type = typename T::value_type;
  using map_type = concurrent_unordered_map<key_type, value_type, std::numeric_limits<key_type>::max()>;

  std::unique_ptr<map_type> the_map;

  const key_type unused_key = std::numeric_limits<key_type>::max();
  const value_type unused_value = std::numeric_limits<value_type>::max();

  const int size;

  const int THREAD_BLOCK_SIZE{256};

  std::vector<key_type> keys;
  std::vector<value_type> values;
  thrust::device_vector<key_type> d_keys;
  thrust::device_vector<value_type> d_values;

  std::unordered_map<key_type, value_type> expected_values;

  MapTest(const int hash_table_size = 10000)
    : size(hash_table_size), the_map(new map_type(hash_table_size))
  {
  }

  int create_input(const int num_unique_keys, const int num_values_per_key, const int max_key = RAND_MAX, const int max_value = RAND_MAX)
  {


    const int TOTAL_PAIRS = num_unique_keys * num_values_per_key;

    this->the_map.reset(new map_type(2*TOTAL_PAIRS));

    keys.reserve(TOTAL_PAIRS);
    values.reserve(TOTAL_PAIRS);

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
        keys.push_back(current_key);
        values.push_back(current_value);

        // Use a STL map to keep track of the max value for each key
        auto found = expected_values.find(current_key);

        // Key doesn't exist yet, insert it
        if(found == expected_values.end())
        {
          expected_values.insert(std::make_pair(current_key,current_value));
        }
        // Key exists, update the value with the max
        else
        {
          max_op<value_type> op;
          value_type new_value = op(found->second, current_value);
          found->second = new_value;
        }
      }
    }

    d_keys = keys;
    d_values = values;
    return TOTAL_PAIRS;
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
typedef ::testing::Types< KeyValueTypes<int,int>
                         // KeyValueTypes<int,float>,
                         // KeyValueTypes<int,double>,
                         // KeyValueTypes<int,long long int>,
                         // KeyValueTypes<int,unsigned long long int>,
                         // KeyValueTypes<unsigned long long int, int>,
                         // KeyValueTypes<unsigned long long int, float>,
                         // KeyValueTypes<unsigned long long int, double>,
                         // KeyValueTypes<unsigned long long int, long long int>,
                         // KeyValueTypes<unsigned long long int, unsigned long long int>
                          > Implementations;

TYPED_TEST_CASE(MapTest, Implementations);

/*
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

TYPED_TEST(MapTest, Insert)
{
  using key_type = typename TypeParam::key_type;
  using value_type = typename TypeParam::value_type;

  const int NUM_PAIRS{this->size};

  // Generate a list of pairs (key, value) to insert into map
  std::vector<thrust::pair<key_type, value_type>> pairs(NUM_PAIRS);
  std::generate(pairs.begin(), pairs.end(), 
                [] () {static int i = 0; return thrust::make_pair(i,(i++)*10);});

  // Insert every pair into the map
  for(const auto& it : pairs){
    this->the_map->insert(it);
  }

  // Make sure all the pairs are in the map
  for(const auto& it : pairs){
    auto found = this->the_map->find(it.first);
    ASSERT_NE(found, this->the_map->end());
    EXPECT_EQ(found->first, it.first);
    EXPECT_EQ(found->second, it.second);
  }

}

TYPED_TEST(MapTest, MaxAggregationTestHost)
{

  using key_type = typename TypeParam::key_type;
  using value_type = typename TypeParam::value_type;

  thrust::pair<key_type, value_type> first_pair{0,0};
  thrust::pair<key_type, value_type> second_pair{0,10};
  thrust::pair<key_type, value_type> third_pair{0,5};

  auto max = [](value_type a, value_type b) { return (a >= b ? a : b); };

  this->the_map->insert(first_pair, max);
  auto found = this->the_map->find(0);
  EXPECT_EQ(0, found->second);

  this->the_map->insert(second_pair, max);
  found = this->the_map->find(0);
  EXPECT_EQ(10, found->second);

  this->the_map->insert(third_pair, max);
  found = this->the_map->find(0);
  EXPECT_EQ(10, found->second);

  this->the_map->insert(thrust::make_pair(0,11), max);
  found = this->the_map->find(0);
  EXPECT_EQ(11, found->second);

  this->the_map->insert(thrust::make_pair(7, 42), max);
  found = this->the_map->find(7);
  EXPECT_EQ(42, found->second);

  this->the_map->insert(thrust::make_pair(7, 62), max);
  found = this->the_map->find(7);
  EXPECT_EQ(62, found->second);

  this->the_map->insert(thrust::make_pair(7, 42), max);
  found = this->the_map->find(7);
  EXPECT_EQ(62, found->second);

  found = this->the_map->find(0);
  EXPECT_EQ(11, found->second);

}
*/


template<typename map_type, typename Aggregation_Operator>
__global__ void build_table(map_type * const the_map,
                            const typename map_type::key_type * const input_keys,
                            const typename map_type::mapped_type * const input_values,
                            const typename map_type::size_type input_size,
                            Aggregation_Operator op)
{

  using size_type = typename map_type::size_type;

  size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  while( i < input_size ){
    const auto p = thrust::make_pair(input_keys[i], input_values[i]);
    the_map->insert(p, op);
    i += blockDim.x * gridDim.x;
  }

}



TYPED_TEST(MapTest, MaxAggregationTestDevice)
{
  using key_type = typename TypeParam::key_type;
  using value_type = typename TypeParam::value_type;
  using size_type = typename MapTest<TypeParam>::map_type::size_type;

  const size_type input_size = this->create_input(512, 256*256);

  key_type *k = thrust::raw_pointer_cast(this->d_keys.data());
  value_type *v = thrust::raw_pointer_cast(this->d_values.data());

  const dim3 grid_size ((input_size + this->THREAD_BLOCK_SIZE -1) / this->THREAD_BLOCK_SIZE,1,1);
  const dim3 block_size (this->THREAD_BLOCK_SIZE, 1, 1);

  std::cout << "Input Size: " << input_size << " Grid Size: " << grid_size.x << " Block Size: " << block_size.x << std::endl;

  cudaDeviceSynchronize();
  build_table<<<grid_size, block_size>>>((this->the_map).get(), k, v, input_size, max_op<value_type>());
  cudaDeviceSynchronize(); 

  this->check_answer();

}


int main(int argc, char * argv[]){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
