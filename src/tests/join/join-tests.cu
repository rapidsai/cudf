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

// See this header for all of the recursive handling of tuples of vectors
#include "tuple_vectors.h"

enum struct join_kind
{
  INNER,
  LEFT
};

// Creates a gdf_column from a std::vector
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


// Each element of the result will be an index into the left and right columns where
// left_columns[left_index] == right_columns[right_index]
struct result_type 
{
  size_t left_index{};
  size_t right_index{};

  result_type(size_t _l, size_t _r) : 
    left_index{_l}, right_index{_r} {}

  // Overload comparison so the result vector can be sorted
  bool operator <(result_type const& rhs){
    return( std::tie(left_index, right_index) < std::tie(rhs.left_index, rhs.right_index) );
  }

  friend std::ostream& operator<<(std::ostream& os, const result_type& result);

};

// Overload the stream operator to make it easier to print a result 
std::ostream& operator<<(std::ostream& os, const result_type& result)
{
  os << result.left_index << ", " << result.right_index << std::endl;
  return os;
}

// A new instance of this class will be created for each *TEST(InnerJoinTest, ...)
// Put all repeated setup and validation stuff here
template <typename multi_column_t>
struct InnerJoinTest : public testing::Test
{
  multi_column_t left_columns;

  multi_column_t right_columns;

  InnerJoinTest()
  {
    static size_t number_of_instantiations{0};

    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    std::srand(number_of_instantiations++);
  }

  void create_input( size_t left_column_length, size_t left_column_range,
                     size_t right_column_length, size_t right_column_range,
                     bool print = false)
  {

    initialize_tuple(left_columns, left_column_length, left_column_range); 

    initialize_tuple(right_columns, right_column_length, right_column_range); 

    if(print)
    {
      std::cout << "Left column(s) created. Size: " << std::get<0>(left_columns).size() << std::endl;
      print_tuple(left_columns);

      std::cout << "Right column(s) created. Size: " << std::get<0>(right_columns).size() << std::endl;
      print_tuple(right_columns);
    }
  }

  std::vector<result_type> compute_reference_solution(join_kind join_method, bool print = false, bool sort = true)
  {

    // Use the type of the first vector as the key_type
    using key_type = typename std::tuple_element<0, multi_column_t>::type::value_type;
    using value_type = size_t;

    // Multimap used to compute the reference solution
    std::multimap<key_type, value_type> the_map;

    // Use first right column as the build column
    std::vector<key_type> const & build_column = std::get<0>(right_columns);

    // Build hash table that maps a value to its index in the column
    for(size_t right_index = 0; right_index < build_column.size(); ++right_index)
    {
      the_map.insert(std::make_pair(build_column[right_index], right_index));
    }

    std::vector<result_type> result;

    // Probe hash table with first left column
    std::vector<key_type> const & probe_column = std::get<0>(left_columns);

    for(size_t left_index = 0; left_index < probe_column.size(); ++left_index)
    {
      // Find all keys that match probe_key
      const auto probe_key = probe_column[left_index];
      auto range = the_map.equal_range(probe_key);

      // Every element in the range identifies a row in the right columns where
      // the first column matches. Need to check if all other columns also match
      bool match{false};
      for(auto i = range.first; i != range.second; ++i)
      {
        const auto right_index = i->second;

        // If all of the columns in right_columns[right_index] == all of the columns in left_columns[left_index]
        // Then this index pair is added to the result as a matching pair of row indices
        if( true == rows_equal(left_columns, right_columns, left_index, right_index)){
          result.emplace_back(left_index, right_index);
          match = true;
        }
      }

      // For left joins, insert a NULL if no match is found
      if((false == match) && (join_method == join_kind::LEFT)){
        constexpr int JoinNullValue{-1};
        result.emplace_back(left_index, JoinNullValue);
      }
    }

    // Sort the result
    if(sort)
    {
      std::sort(result.begin(), result.end());
    }

    if(print)
    {
      std::cout << "Result size: " << result.size() << std::endl;
      std::cout << "left index, right index" << std::endl;
      std::copy(result.begin(), result.end(), std::ostream_iterator<result_type>(std::cout, ""));
      std::cout << "\n";
    }

    return result;
  }
  //gdf_error gdf_multi_left_join_generic(int num_cols, gdf_column **leftcol, gdf_column **rightcol, gdf_join_result_type **out_result)
};


typedef ::testing::Types< std::tuple< std::vector<int> >,
                          std::tuple< std::vector<int>, std::vector<double>, std::vector<float>, std::vector<long long int> >
                          > Implementations;

TYPED_TEST_CASE(InnerJoinTest, Implementations);

TYPED_TEST(InnerJoinTest, debug)
{
  this->create_input(5,2,5,2,true);
  std::vector<result_type> result = this->compute_reference_solution(join_kind::LEFT, true);
}
