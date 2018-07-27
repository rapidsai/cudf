#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include <type_traits>

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

// Wrapper class for a gdf_column
template <typename col_type>
struct gdf_column_wrapper : gdf_column
{
  thrust::device_vector<col_type> device_vector;

  gdf_column_wrapper(size_t column_size, size_t column_range=RAND_MAX)
  {
    // Generate random vector on host and copy it to device
    std::vector<col_type> host_vector;
    std::generate(host_vector.begin(), host_vector.end(), [column_range](){return std::rand() % column_range;});
    device_vector(host_vector);

    // Deduce the type and set the gdf_dtype accordingly
    gdf_dtype gdf_col_type;
    if(std::is_same<col_type,int8_t>::value) gdf_col_type = GDF_INT8;
    else if(std::is_same<col_type,uint8_t>::value) gdf_col_type = GDF_INT8;
    else if(std::is_same<col_type,int16_t>::value) gdf_col_type = GDF_INT16;
    else if(std::is_same<col_type,uint16_t>::value) gdf_col_type = GDF_INT16;
    else if(std::is_same<col_type,int32_t>::value) gdf_col_type = GDF_INT32;
    else if(std::is_same<col_type,uint32_t>::value) gdf_col_type = GDF_INT32;
    else if(std::is_same<col_type,int64_t>::value) gdf_col_type = GDF_INT64;
    else if(std::is_same<col_type,uint64_t>::value) gdf_col_type = GDF_INT64;
    else if(std::is_same<col_type,float>::value) gdf_col_type = GDF_FLOAT32;
    else if(std::is_same<col_type,double>::value) gdf_col_type = GDF_FLOAT64;

    // Fill the base gdf_column structure
    this->data = device_vector.data().get();
    this->valid = nullptr;
    this->size = device_vector.size();
    this->dtype = gdf_col_type;
    gdf_dtype_extra_info extra_info;
    extra_info.time_unit = TIME_UNIT_NONE;
    this->dtype_info = extra_info;
  }

  ~gdf_column_wrapper()
  {

  }
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


  // Array of columns that will be inputs to the gdf_join function
  std::vector<gdf_column*> left_columns;
  std::vector<gdf_column*> right_columns;

  JoinTest()
  {
    // Use constant seed so the psuedo-random order is the same each time
    std::srand(0);
  }

  // TODO: Support more than 3 columns
  void create_input(size_t col0_size, size_t col0_range=RAND_MAX,
                    size_t col1_size=0, size_t col1_range=RAND_MAX,
                    size_t col2_size=0, size_t col2_range=RAND_MAX)
  {
    left_columns.emplace_back(col0_size, col0_range);
    right_columns.emplace_back(col0_size, col0_range);

    if(col1_size > 0){ 
      left_columns.emplace_back(col1_size, col1_range);
      right_columns.emplace_back(col1_size, col1_range);
    }
    if(col2_size > 0){ 
      left_columns.emplace_back(col2_size, col2_range);
      right_columns.emplace_back(col2_size, col2_range);
    }
  }

  void compute_reference_solution()
  {
    // Multimap used to compute the reference solution
    std::multimap<key_type, value_type> the_map;

  }
  
};
