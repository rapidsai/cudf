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

template <typename col_type>
gdf_column create_gdf_column(std::vector<col_type> host_vector)
{
  // Copy host vector to device
  thrust::device_vector<col_type> device_vector(host_vector);

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

  // Fill gdf_column structure
  gdf_column the_column;
  the_column.data = device_vector.data().get();
  the_column.valid = nullptr;
  the_column.size = device_vector.size();
  the_column.dtype = gdf_col_type;
  gdf_dtype_extra_info extra_info;
  extra_info.time_unit = TIME_UNIT_NONE;
  the_column.dtype_info = extra_info;

  return the_column;
}



// A new instance of this class will be created for each *TEST(JoinTest, ...)
// Put all repeated setup and validation stuff here
template <class T>
struct LeftJoinTest : public testing::Test
{

  // Extract the types for the input columns 
  using col0_type = typename T::col0_type;
  using col1_type = typename T::col1_type;
  using col2_type = typename T::col2_type;

  // Container of columns of different types
  // No way to create a dynamically sized container of vectors of different types
  // without type erasure... let's not do that here
  // Therefore, the maximum number of columns will have to be hardcoded as the size
  // of the tuple
  using multi_column_t = typename std::tuple<std::vector<col0_type>,
                                             std::vector<col1_type>,
                                             std::vector<col2_type>>;
  multi_column_t left_columns;

  multi_column_t right_columns;

  LeftJoinTest()
  {
    static size_t number_of_instantiations{0};

    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    std::srand(number_of_instantiations++);

  }

  multi_column_t create_random_columns(size_t col0_size, size_t col0_range=RAND_MAX,
                                       size_t col1_size=0, size_t col1_range=RAND_MAX,
                                       size_t col2_size=0, size_t col2_range=RAND_MAX)
  {

    multi_column_t the_columns;

    // Allocate storage in each vector 
    std::get<0>(the_columns).reserve(col0_size);
    std::get<1>(the_columns).reserve(col1_size);
    std::get<2>(the_columns).reserve(col2_size);

    // Fill each vector with random values
    std::generate(std::get<0>(the_columns).begin(), std::get<0>(the_columns).end(), [col0_range](){return std::rand() % col0_range;});
    std::generate(std::get<1>(the_columns).begin(), std::get<1>(the_columns).end(), [col1_range](){return std::rand() % col1_range;});
    std::generate(std::get<2>(the_columns).begin(), std::get<2>(the_columns).end(), [col2_range](){return std::rand() % col2_range;});

    return the_columns;

  }

  // TODO: Support more than 3 columns
  void create_input(size_t col0_size, size_t col0_range=RAND_MAX,
                    size_t col1_size=0, size_t col1_range=RAND_MAX,
                    size_t col2_size=0, size_t col2_range=RAND_MAX)
  {

    left_columns = create_random_columns(col0_size, col0_range, 
                                         col1_size, col1_range,
                                         col2_size, col2_range);

    right_columns = create_random_columns(col0_size, col0_range, 
                                          col1_size, col1_range,
                                          col2_size, col2_range);
  }

  void compute_reference_solution()
  {
    // Use the first column as the keys
    using key_type = col0_type;
    using value_type = size_t;

    // Multimap used to compute the reference solution
    std::multimap<key_type, value_type> the_map;

    // Use first right column as the build column
    std::vector<col0_type> * build_column = std::get<0>(right_columns);

    // Build hash table
    for(size_t i = 0; i < (*build_column).size(); ++i)
    {
      the_map.insert(std::make_pair((*build_column)[i], i));
    }

    // Probe hash table
    std::vector<col0_type> * probe_column = std::get<0>(left_columns);
    for(size_t i = 0; i < (*probe_column).size(); ++i)
    {
      auto found = the_map.find((*probe_column)[i]);

      // First column matches, check the rest
      if(found != the_map.end()){

      }

    }

  }

  //gdf_error gdf_multi_left_join_generic(int num_cols, gdf_column **leftcol, gdf_column **rightcol, gdf_join_result_type **out_result)


  
};
