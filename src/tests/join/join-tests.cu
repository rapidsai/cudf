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

// A new instance of this class will be created for each *TEST(JoinTest, ...)
// Put all repeated setup and validation stuff here
template <class T>
struct InnerJoinTest : public testing::Test
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


  InnerJoinTest()
  {
    static size_t number_of_instantiations{0};

    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    std::srand(number_of_instantiations++);

  }

  multi_column_t create_random_columns(size_t num_columns,
                                       size_t column_length, size_t column_range)
  {
    assert(num_columns > 0);
    assert(num_columns <= std::tuple_size<multi_column_t>::value);

    multi_column_t the_columns;

    // Allocate storage in each vector 
    // Fill each vector with random values
    if(num_columns >= 1){
      std::get<0>(the_columns).resize(column_length);
      std::generate(std::get<0>(the_columns).begin(), std::get<0>(the_columns).end(), [column_range](){return std::rand() % column_range;});
    }
    if(num_columns >= 2){
      std::get<1>(the_columns).resize(column_length);
      std::generate(std::get<1>(the_columns).begin(), std::get<1>(the_columns).end(), [column_range](){return std::rand() % column_range;});
    }
    if(num_columns >= 3){
      std::get<2>(the_columns).resize(column_length);
      std::generate(std::get<2>(the_columns).begin(), std::get<2>(the_columns).end(), [column_range](){return std::rand() % column_range;});
    }

    return the_columns;
  }

  // TODO: Support more than 3 columns
  void create_input(size_t num_columns, 
                    size_t left_column_length, size_t left_column_range,
                    size_t right_column_length, size_t right_column_range,
                    bool print = false)
  {
    assert(num_columns > 0);
    assert(num_columns <= std::tuple_size<multi_column_t>::value);

    left_columns = create_random_columns(num_columns, left_column_length, left_column_range);


    right_columns = create_random_columns(num_columns, right_column_length, right_column_range);


    if(print)
    {
      std::cout << "Left column(s) created. Size: " << std::get<0>(left_columns).size() << std::endl;
      std::copy(std::get<0>(left_columns).begin(), std::get<0>(left_columns).end(), std::ostream_iterator<col0_type>(std::cout, ", "));
      std::cout << "\n";
      std::copy(std::get<1>(left_columns).begin(), std::get<1>(left_columns).end(), std::ostream_iterator<col1_type>(std::cout, ", "));
      std::cout << "\n";
      std::copy(std::get<2>(left_columns).begin(), std::get<2>(left_columns).end(), std::ostream_iterator<col2_type>(std::cout, ", "));
      std::cout << "\n";

      std::cout << "Right column(s) created. Size: " << std::get<0>(right_columns).size() << std::endl;
      std::copy(std::get<0>(right_columns).begin(), std::get<0>(right_columns).end(), std::ostream_iterator<col0_type>(std::cout, ", "));
      std::cout << "\n";
      std::copy(std::get<1>(right_columns).begin(), std::get<1>(right_columns).end(), std::ostream_iterator<col1_type>(std::cout, ", "));
      std::cout << "\n";
      std::copy(std::get<2>(right_columns).begin(), std::get<2>(right_columns).end(), std::ostream_iterator<col2_type>(std::cout, ", "));
      std::cout << "\n";
    }
  }

  bool rows_match(size_t num_columns, size_t left_index, size_t right_index){
    
    assert(num_columns > 0);
    assert(num_columns <= std::tuple_size<multi_column_t>::value);

    bool match{false};

    // Technically this is redudant as the hash table already told us the first column matches,
    // but for completeness, we'll just check again
    if(num_columns >= 1){
      auto const & first_left_column = std::get<0>(left_columns);
      auto const & first_right_column = std::get<0>(right_columns);
      if(first_left_column[left_index] == first_right_column[right_index]){
        match = true;
      } 
    }

    if(num_columns >= 2){
      auto const & second_left_column = std::get<1>(left_columns);
      auto const & second_right_column = std::get<1>(right_columns);
      if(second_left_column[left_index] != second_right_column[right_index]){
        match = false;
      }
    }

    if(num_columns >= 3){
      auto const & third_left_column = std::get<2>(left_columns);
      auto const & third_right_column = std::get<2>(right_columns);
      if(third_left_column[left_index] != third_right_column[right_index])
      {
        match = false;
      }
    }

    return match;
  }

  std::vector<result_type> compute_reference_solution(size_t num_columns, bool print = false, bool sort = true)
  {
    assert(num_columns > 0);
    assert(num_columns <= std::tuple_size<multi_column_t>::value);

    // Use the first column as the keys
    using key_type = col0_type;
    using value_type = size_t;

    // Multimap used to compute the reference solution
    std::multimap<key_type, value_type> the_map;

    // Use first right column as the build column
    std::vector<col0_type> const & build_column = std::get<0>(right_columns);

    // Build hash table that maps a value to its index in the column
    for(size_t right_index = 0; right_index < build_column.size(); ++right_index)
    {
      the_map.insert(std::make_pair(build_column[right_index], right_index));
    }

    std::vector<result_type> result;

    // Probe hash table with first left column
    std::vector<col0_type> const & probe_column = std::get<0>(left_columns);
    for(size_t left_index = 0; left_index < probe_column.size(); ++left_index)
    {

      // Find all keys that match probe_key
      const auto probe_key = probe_column[left_index];
      auto range = the_map.equal_range(probe_key);

      // Every element in the range identifies a row in the right columns where
      // the first column matches. Need to check if all other columns also match
      for(auto i =  range.first; i != range.second; ++i)
      {
        const auto right_index = i->second;

        // If all of the columns in right_columns[right_index] == all of the columns in left_columns[left_index]
        // Then this index pair is added to the result as a matching pair of row indices
        if( true == rows_match(num_columns, left_index, right_index))
          result.emplace_back(left_index, right_index);
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


typedef ::testing::Types<InputTypes<int,int,int>> Implementations;

TYPED_TEST_CASE(InnerJoinTest, Implementations);

TYPED_TEST(InnerJoinTest, debug)
{
  this->create_input(3,5,2,5,2, true);
  std::vector<result_type> result = this->compute_reference_solution(3, true);
}
