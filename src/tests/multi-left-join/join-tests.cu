#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>

#include "../../joining.h"


// This is necessary to do a parametrized typed-test over multiple template arguments
template<typename T0, typename T1, typename T2>
struct InputTypes
{
  using col0_type = T0;
  using col1_type = T1;
  using col2_type = T2;
};

// A new instance of this class will be created for each *TEST(JoinTest, ...)
// Put all repeated setup and validation stuff here
template <class T>
struct JoinTest : public testing::Test
{

protected:

  // Extract the types for the input columns 
  using col0_type = typename T::col0_type;
  using col1_type = typename T::col1_type;
  using col2_type = typename T::col2_type;

  // Use the first column as the keys
  using key_type = col0_type;
  using value_type = size_t;

  std::multimap<key_type, value_type> the_map;

  JoinTest()
  {

  }

};
