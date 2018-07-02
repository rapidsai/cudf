#include <cstdlib>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include </home/jhemstad/RAPIDS/libgdf/src/hash-join/concurrent_unordered_multimap.cuh>

// This is necessary to do a parametrized typed-test over multiple template arguments
template <typename Key, typename Value>
struct KeyValueTypes
{
  using key_type = Key;
  using value_type = Value;
};


template <class T>
class HashTest : public testing::Test 
{
  using key_type = typename T::key_type;
  using value_type = typename T::value_type;


  concurrent_unordered_multimap<key_type, 
                                value_type, 
                                std::numeric_limits<key_type>::max(), 
                                std::numeric_limits<value_type>::max() > the_map;


public:
  HashTest(const int hash_table_size = 100)
    : the_map(hash_table_size)
  {


  }

  ~HashTest(){
  }


};

// The list of types we want to test.
typedef ::testing::Types< KeyValueTypes<int,int> > Implementations;

TYPED_TEST_CASE(HashTest, Implementations);

TYPED_TEST(HashTest, jake)
{

  using key_type = typename TypeParam::key_type;
  using value_type = typename TypeParam::value_type;

  key_type six = 6;

  ASSERT_EQ(6,six);

}


int main(int argc, char * argv[]){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
