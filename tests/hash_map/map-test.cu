#include <cstdlib>
#include <iostream>
#include <vector>

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


// A new instance of this class will be created for each *TEST(MapTest, ...)
// Put all repeated stuff for each test here
template <class T>
class MapTest : public testing::Test 
{
public:
  using key_type = typename T::key_type;
  using value_type = typename T::value_type;


  concurrent_unordered_map<key_type, 
                                value_type, 
                                std::numeric_limits<key_type>::max() > the_map;

  const key_type unused_key = std::numeric_limits<key_type>::max();
  const value_type unused_value = std::numeric_limits<value_type>::max();

  const int size;


  MapTest(const int hash_table_size = 100)
    : the_map(hash_table_size), size(hash_table_size)
  {


  }

  ~MapTest(){
  }


};

// Google Test can only do a parameterized typed-test over a single type, so we have
// to nest multiple types inside of the KeyValueTypes struct above
// KeyValueTypes<type1, type2> implies key_type = type1, value_type = type2
// This list is the types across which Google Test will run our tests
typedef ::testing::Types< KeyValueTypes<int,int>, 
                          KeyValueTypes<int,float>, 
                          KeyValueTypes<int,double>,
                          KeyValueTypes<int,long long int>,
                          KeyValueTypes<int,unsigned long long int>,
                          KeyValueTypes<unsigned long long int, int>,
                          KeyValueTypes<unsigned long long int, float>,
                          KeyValueTypes<unsigned long long int, double>,
                          KeyValueTypes<unsigned long long int, long long int>,
                          KeyValueTypes<unsigned long long int, unsigned long long int>
                          > Implementations;

TYPED_TEST_CASE(MapTest, Implementations);

TYPED_TEST(MapTest, InitialState)
{
  using key_type = typename TypeParam::key_type;
  using value_type = typename TypeParam::value_type;

  auto begin = this->the_map.begin();
  auto end = this->the_map.end();
  EXPECT_NE(begin,end);

}

TYPED_TEST(MapTest, CheckUnusedValues){

  EXPECT_EQ(this->the_map.get_unused_key(), this->unused_key);

  auto begin = this->the_map.begin();
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
    this->the_map.insert(it);
  }

  // Make sure all the pairs are in the map
  for(const auto& it : pairs){
    auto found = this->the_map.find(it.first);
    EXPECT_NE(found, this->the_map.end());
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

  auto max = [](value_type a, value_type b) { return std::max<value_type>(a,b); };

  this->the_map.insert(first_pair, max);
  auto found = this->the_map.find(0);
  EXPECT_EQ(0, found->second);

  this->the_map.insert(second_pair, max);
  found = this->the_map.find(0);
  EXPECT_EQ(10, found->second);

  this->the_map.insert(third_pair, max);
  found = this->the_map.find(0);
  EXPECT_EQ(10, found->second);

  this->the_map.insert(thrust::make_pair(0,11), max);
  found = this->the_map.find(0);
  EXPECT_EQ(11, found->second);


  this->the_map.insert(thrust::make_pair(7, 42), max);
  found = this->the_map.find(7);
  EXPECT_EQ(42, found->second);

  this->the_map.insert(thrust::make_pair(7, 62), max);
  found = this->the_map.find(7);
  EXPECT_EQ(62, found->second);

  this->the_map.insert(thrust::make_pair(7, 42), max);
  found = this->the_map.find(7);
  EXPECT_EQ(62, found->second);

  found = this->the_map.find(0);
  EXPECT_EQ(11, found->second);

}


int main(int argc, char * argv[]){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
