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

// This is necessary to do a parametrized typed-test over multiple template arguments
template <typename Key, typename Value, template <typename> typename Aggregation_Operator>
struct KeyValueTypes
{
  using key_type = Key;
  using value_type = Value;
  using op_type = Aggregation_Operator<value_type>;
};

// Have to use a functor instead of a device lambda because
// you can't create a device lambda inside of a Google Test
// because the macro expands into a private member function
// and you can't have a device lambda inside a private member
// function
template<typename value_type>
struct max_op
{
  constexpr static value_type IDENTITY{std::numeric_limits<value_type>::min()};

  __host__ __device__
    value_type operator()(value_type a, value_type b)
    {
      return (a > b? a : b);
    }
};

template<typename value_type>
struct min_op
{
  constexpr static value_type IDENTITY{std::numeric_limits<value_type>::max()};

  __host__ __device__
    value_type operator()(value_type a, value_type b)
    {
      return (a < b? a : b);
    }
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

  std::unordered_map<key_type, value_type> expected_values;

  GroupByTest(const size_type _hash_table_size = 10000)
    : hash_table_size(_hash_table_size), the_map(new map_type(_hash_table_size, op_type::IDENTITY))
  {
  }

  std::pair<key_type*, value_type*>
  create_input(const int num_keys, const int num_values_per_key, const int ratio = 1, const int max_key = RAND_MAX, const int max_value = RAND_MAX)
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

        // Use a STL map to keep track of the aggregation for each key
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

    d_groupby_column = groupby_column;
    d_aggregation_column = aggregation_column;

    return std::make_pair(thrust::raw_pointer_cast(d_groupby_column.data()), 
                          thrust::raw_pointer_cast(d_aggregation_column.data()));
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

  void verify_aggregation_table(){

    for(auto const &k : this->expected_values)
    {
      key_type test_key = k.first;

      value_type expected_value = k.second;

      auto found = this->the_map->find(test_key);

      ASSERT_NE(this->the_map->end(), found) << "Key is: " << test_key;

      value_type test_value = found->second;

      EXPECT_EQ(expected_value, test_value) << "Key is: " << test_key;
    }
  }

  unsigned int extract_groupby_result_device()
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

    unsigned int result_size = *global_write_index;

    // Return the actual size of the result
    return result_size;

  }

  void verify_groupby_result(const unsigned int result_size) const
  {

    ASSERT_NE(nullptr, d_groupby_result);
    ASSERT_NE(nullptr, d_aggregation_result);

    for(size_type i = 0; i < result_size; ++i)
    {
      key_type groupby_key = d_groupby_result[i];
      value_type aggregation_value = d_aggregation_result[i];

      auto found = expected_values.find(groupby_key);

      ASSERT_NE(expected_values.end(), found) << "key: " << groupby_key;

      EXPECT_EQ(found->first, groupby_key);
      EXPECT_EQ(found->second, aggregation_value);
    }

  }


  ~GroupByTest(){
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

TYPED_TEST_CASE(GroupByTest, Implementations);

TYPED_TEST(GroupByTest, AggregationTestHost)
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


TYPED_TEST(GroupByTest, AggregationTestDeviceAllSame)
{
  const int num_keys = 1;
  const int num_values_per_key = 1<<16;
  auto input = this->create_input(num_keys, num_values_per_key);
  this->build_aggregation_table_device(input);
  this->verify_aggregation_table();
  unsigned int result_size = this->extract_groupby_result_device();
  this->verify_groupby_result(result_size);

  // The size of the result should be equal to the number of unique keys
  auto begin = this->d_groupby_column.begin();
  auto end = this->d_groupby_column.end();
  thrust::sort(begin, end);
  unsigned int unique_count = thrust::unique(begin, end) - begin;
  EXPECT_EQ(unique_count, result_size);
}

TYPED_TEST(GroupByTest, AggregationTestDeviceAllUnique)
{
  const int num_keys = 1<<16;
  const int num_values_per_key = 1;
  auto input = this->create_input(num_keys, num_values_per_key);
  this->build_aggregation_table_device(input);
  this->verify_aggregation_table();
  unsigned int result_size = this->extract_groupby_result_device();
  this->verify_groupby_result(result_size);

  // The size of the result should be equal to the number of unique keys
  auto begin = this->d_groupby_column.begin();
  auto end = this->d_groupby_column.end();
  thrust::sort(begin, end);
  unsigned int unique_count = thrust::unique(begin, end) - begin;
  EXPECT_EQ(unique_count, result_size);
}

TYPED_TEST(GroupByTest, AggregationTestDeviceWarpSame)
{
  const int num_keys = 1<<15;
  const int num_values_per_key = 32;
  auto input = this->create_input(num_keys, num_values_per_key);
  this->build_aggregation_table_device(input);
  this->verify_aggregation_table();
  unsigned int result_size = this->extract_groupby_result_device();
  this->verify_groupby_result(result_size);

  // The size of the result should be equal to the number of unique keys
  auto begin = this->d_groupby_column.begin();
  auto end = this->d_groupby_column.end();
  thrust::sort(begin, end);
  unsigned int unique_count = thrust::unique(begin, end) - begin;
  EXPECT_EQ(unique_count, result_size);
}

TYPED_TEST(GroupByTest, AggregationTestDeviceBlockSame)
{
  const int num_keys = 1<<12;
  const int num_values_per_key = this->THREAD_BLOCK_SIZE;
  auto input = this->create_input(num_keys, num_values_per_key);
  this->build_aggregation_table_device(input);
  this->verify_aggregation_table();
  unsigned int result_size = this->extract_groupby_result_device();
  this->verify_groupby_result(result_size);

  // The size of the result should be equal to the number of unique keys
  auto begin = this->d_groupby_column.begin();
  auto end = this->d_groupby_column.end();
  thrust::sort(begin, end);
  unsigned int unique_count = thrust::unique(begin, end) - begin;
  EXPECT_EQ(unique_count, result_size);
}


int main(int argc, char * argv[]){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
