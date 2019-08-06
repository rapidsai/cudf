/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

// See this header for all of the recursive handling of tuples of vectors
#include <tests/utilities/tuple_vectors.h>

// See this header for all of the handling of valids' vectors
#include <tests/utilities/valid_vectors.h>
#include <tests/utilities/cudf_test_fixtures.h>

#include <join/joining.h>
#include <join/join_compute_api.h>
#include <utilities/bit_util.cuh>

#include <cudf/cudf.h>
#include <cudf/join.hpp>

#include <rmm/rmm.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <iostream>
#include <vector>
#include <map>
#include <type_traits>
#include <memory>

#include <cstdlib>

// Selects the kind of join operation that is performed
enum struct join_op
{
  INNER,
  LEFT,
  FULL
};

// Each element of the result will be an index into the left and right columns where
// left_columns[left_index] == right_columns[right_index]
using result_type = typename std::pair<int, int>;

// Define stream operator for a std::pair for conveinience of printing results.
// Needs to be in the std namespace to work with std::copy
namespace std{
  template <typename first_t, typename second_t>
  std::ostream& operator<<(std::ostream& os, std::pair<first_t, second_t> const & p)
  {
    os << p.first << ", " << p.second;
    std::cout << "\n";
    return os;
  }
}

// A new instance of this class will be created for each *TEST(JoinTest, ...)
// Put all repeated setup and validation stuff here
template <class test_parameters>
struct JoinTest : public GdfTest
{
  // The join type is passed via a member of the template argument class
  const join_op op = test_parameters::op;

  gdf_context ctxt = {
    test_parameters::join_type == gdf_method::GDF_SORT,
    test_parameters::join_type, 
    0
  };
  // multi_column_t is a tuple of vectors. The number of vectors in the tuple
  // determines the number of columns to be joined, and the value_type of each
  // vector determiens the data type of the column
  using multi_column_t = typename test_parameters::multi_column_t;
  multi_column_t left_columns;
  multi_column_t right_columns;

  // valids for multi_columns
  std::vector<host_valid_pointer> left_valids;
  std::vector<host_valid_pointer> right_valids;

  // Type for a unique_ptr to a gdf_column with a custom deleter
  // Custom deleter is defined at construction
  using gdf_col_pointer = 
    typename std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>;

  // Containers for unique_ptrs to gdf_columns that will be used in the gdf_join
  // functions. unique_ptrs are used to automate freeing device memory
  std::vector<gdf_col_pointer> gdf_left_columns;
  std::vector<gdf_col_pointer> gdf_right_columns;

  // Containers for the raw pointers to the gdf_columns that will be used as
  // input to the gdf_join functions
  std::vector<gdf_column*> gdf_raw_left_columns;
  std::vector<gdf_column*> gdf_raw_right_columns;

  JoinTest()
  {
    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    static size_t number_of_instantiations{0};
    std::srand(number_of_instantiations++);
  }

  ~JoinTest()
  {
  }

  /* --------------------------------------------------------------------------*
  * @brief Creates a unique_ptr that wraps a gdf_column structure 
  *           intialized with a host vector
  *
  * @param host_vector vector containing data to be transfered to device side column
  * @param host_valid  vector containing valid masks associated with the supplied vector
  * @param n_count     null_count to be set for the generated column
  *
  * @returns A unique_ptr wrapping the new gdf_column
  * --------------------------------------------------------------------------*/
  template <typename col_type>
  gdf_col_pointer create_gdf_column(std::vector<col_type> const & host_vector, gdf_valid_type* host_valid,
          const gdf_size_type n_count)
  {
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

    // Create a new instance of a gdf_column with a custom deleter that will
    //  free the associated device memory when it eventually goes out of scope
    auto deleter = [](gdf_column* col) {
      col->size = 0; 
      RMM_FREE(col->data, 0); 
      RMM_FREE(col->valid, 0); 
    };
    gdf_col_pointer the_column{new gdf_column{}, deleter};

    // Allocate device storage for gdf_column and copy contents from host_vector
    EXPECT_EQ(RMM_ALLOC(&(the_column->data), host_vector.size() * sizeof(col_type), 0), RMM_SUCCESS);
    EXPECT_EQ(cudaMemcpy(the_column->data, host_vector.data(), host_vector.size() * sizeof(col_type), cudaMemcpyHostToDevice), cudaSuccess);

    // Allocate device storage for gdf_column.valid
    if (host_valid != nullptr) {
      EXPECT_EQ(RMM_ALLOC((void**)&(the_column->valid), gdf_valid_allocation_size(host_vector.size()), 0), RMM_SUCCESS);
      EXPECT_EQ(cudaMemcpy(the_column->valid, host_valid, gdf_num_bitmask_elements(host_vector.size()), cudaMemcpyHostToDevice), cudaSuccess);
      the_column->null_count = n_count;
    } else {
        the_column->valid = nullptr;
        the_column->null_count = 0;
    }

    // Fill the gdf_column members
    the_column->size = host_vector.size();
    the_column->dtype = gdf_col_type;
    gdf_dtype_extra_info extra_info{TIME_UNIT_NONE};
    the_column->dtype_info = extra_info;

    return the_column;
  }

  // Compile time recursion to convert each vector in a tuple of vectors into
  // a gdf_column and append it to a vector of gdf_columns
  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  convert_tuple_to_gdf_columns(std::vector<gdf_col_pointer> &gdf_columns,std::tuple<std::vector<Tp>...>& t, std::vector<host_valid_pointer>& valids, const gdf_size_type n_count)
  {
    //bottom of compile-time recursion
    //purposely empty...
  }
  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type
  convert_tuple_to_gdf_columns(std::vector<gdf_col_pointer> &gdf_columns,std::tuple<std::vector<Tp>...>& t, std::vector<host_valid_pointer>& valids, const gdf_size_type n_count)
  {
    // Creates a gdf_column for the current vector and pushes it onto
    // the vector of gdf_columns
    if (valids.size() != 0) {
      gdf_columns.push_back(create_gdf_column(std::get<I>(t), valids[I].get(), n_count));
    } else {
      gdf_columns.push_back(create_gdf_column(std::get<I>(t), nullptr, n_count));
    }

    //recurse to next vector in tuple
    convert_tuple_to_gdf_columns<I + 1, Tp...>(gdf_columns, t, valids, n_count);
  }

  // Converts a tuple of host vectors into a vector of gdf_columns
  std::vector<gdf_col_pointer>
  initialize_gdf_columns(multi_column_t host_columns, std::vector<host_valid_pointer>& valids,
          const gdf_size_type n_count)
  {
    std::vector<gdf_col_pointer> gdf_columns;
    convert_tuple_to_gdf_columns(gdf_columns, host_columns, valids, n_count);
    return gdf_columns;
  }

  /* --------------------------------------------------------------------------*
   * @brief  Initializes two sets of columns, left and right, with random 
   *            values for the join operation.
   *
   * @param left_column_length The length of the left set of columns
   * @param left_column_range The upper bound of random values for the left 
   *                          columns. Values are [0, left_column_range)
   * @param right_column_length The length of the right set of columns
   * @param right_column_range The upper bound of random values for the right 
   *                           columns. Values are [0, right_column_range)
   * @param print Optionally print the left and right set of columns for debug
   * -------------------------------------------------------------------------*/
  void create_input( size_t left_column_length, size_t left_column_range,
                     size_t right_column_length, size_t right_column_range,
                     bool print = false, const gdf_size_type n_count = 0)
  {
    initialize_tuple(left_columns, left_column_length, left_column_range, static_cast<size_t>(ctxt.flag_sorted));
    initialize_tuple(right_columns, right_column_length, right_column_range, static_cast<size_t>(ctxt.flag_sorted));

    auto n_columns = std::tuple_size<multi_column_t>::value;
    initialize_valids(left_valids, n_columns, left_column_length, 0);
    initialize_valids(right_valids, n_columns, right_column_length, 0);

    gdf_left_columns = initialize_gdf_columns(left_columns, left_valids, n_count);
    gdf_right_columns = initialize_gdf_columns(right_columns, right_valids, n_count);

    // Fill vector of raw pointers to gdf_columns
    gdf_raw_left_columns.clear();
    gdf_raw_right_columns.clear();
    for(auto const& c : gdf_left_columns){
      gdf_raw_left_columns.push_back(c.get());
    }

    for(auto const& c : gdf_right_columns){
      gdf_raw_right_columns.push_back(c.get());
    }

    if(print)
    {
      std::cout << "Left column(s) created. Size: " << std::get<0>(left_columns).size() << std::endl;
      print_tuples_and_valids(left_columns, left_valids);

      std::cout << "Right column(s) created. Size: " << std::get<0>(right_columns).size() << std::endl;
      print_tuples_and_valids(right_columns, right_valids);
    }
  }

  /* --------------------------------------------------------------------------*
   * @brief  Creates two gdf_columns with size 1 data buffer allocations, but
   * with a specified `size` attributed
   *
   * @param left_column_length The length of the left column
   * @param right_column_length The length of the right column
   * -------------------------------------------------------------------------*/
  void create_dummy_input( gdf_size_type const left_column_length, 
                           gdf_size_type const right_column_length)
  {
    using col_type = typename std::tuple_element<0, multi_column_t>::type::value_type;
    
    // Only allocate a single element
    std::vector<col_type> dummy_vector_left(1, static_cast<col_type>(0));
    std::vector<col_type> dummy_vector_right(1, static_cast<col_type>(0));
    gdf_left_columns.push_back(create_gdf_column<col_type>(dummy_vector_left, nullptr, 0));
    gdf_right_columns.push_back(create_gdf_column<col_type>(dummy_vector_right, nullptr, 0));

    
    // Fill vector of raw pointers to gdf_columns
    for (auto const& c : gdf_left_columns) {
      c->size = left_column_length;
      gdf_raw_left_columns.push_back(c.get());
    }

    for (auto const& c : gdf_right_columns) {
      c->size = right_column_length;
      gdf_raw_right_columns.push_back(c.get());
    }
  }

  /* --------------------------------------------------------------------------*/
  /**
   * @brief  Computes a reference solution for joining the left and right sets of columns
   *
   * @param print Option to print the solution for debug
   * @param sort Option to sort the solution. This is necessary for comparison against the gdf solution
   *
   * @returns A vector of 'result_type' where result_type is a structure with a left_index, right_index
   * where left_columns[left_index] == right_columns[right_index]
   */
  /* ----------------------------------------------------------------------------*/
  std::vector<result_type> compute_reference_solution(bool print = false, bool sort = true)
  {

    // Use the type of the first vector as the key_type
    using key_type = typename std::tuple_element<0, multi_column_t>::type::value_type;
    using value_type = size_t;

    // Multimap used to compute the reference solution
    std::multimap<key_type, value_type> the_map;

    // Build hash table that maps the first right columns' values to their row index in the column
    std::vector<key_type> const & build_column = std::get<0>(right_columns);
    auto build_valid = right_valids[0].get();

    for(size_t right_index = 0; right_index < build_column.size(); ++right_index)
    {
      if (gdf_is_valid(build_valid, right_index)) {
        the_map.insert(std::make_pair(build_column[right_index], right_index));
      }
    }

    std::vector<result_type> reference_result;

    // Probe hash table with first left column
    std::vector<key_type> const & probe_column = std::get<0>(left_columns);
    auto probe_valid = left_valids[0].get();

    for(size_t left_index = 0; left_index < probe_column.size(); ++left_index)
    {
      bool match{false};
      if (gdf_is_valid(probe_valid, left_index)) {
        // Find all keys that match probe_key
        const auto probe_key = probe_column[left_index];
        auto range = the_map.equal_range(probe_key);

        // Every element in the returned range identifies a row in the first right column that
        // matches the probe_key. Need to check if all other columns also match
        for(auto i = range.first; i != range.second; ++i)
        {
          const auto right_index = i->second;

          // If all of the columns in right_columns[right_index] == all of the columns in left_columns[left_index]
          // Then this index pair is added to the result as a matching pair of row indices
          if( true == rows_equal_using_valids(left_columns, right_columns, left_valids, right_valids, left_index, right_index)){
            reference_result.emplace_back(left_index, right_index);
            match = true;
          }
        }
      }
      // For left joins, insert a NULL if no match is found
      if((false == match) &&
              ((op == join_op::LEFT) || (op == join_op::FULL))){
        constexpr int JoinNullValue{-1};
        reference_result.emplace_back(left_index, JoinNullValue);
      }
    }

    if (op == join_op::FULL)
    {
        the_map.clear();
        // Build hash table that maps the first left columns' values to their row index in the column
        for(size_t left_index = 0; left_index < probe_column.size(); ++left_index)
        {
          if (gdf_is_valid(probe_valid, left_index)) {
            the_map.insert(std::make_pair(probe_column[left_index], left_index));
          }
        }
        // Probe the hash table with first right column
        // Add rows where a match for the right column does not exist
        for(size_t right_index = 0; right_index < build_column.size(); ++right_index)
        {
          const auto probe_key = build_column[right_index];
          auto search = the_map.find(probe_key);
          if ((search == the_map.end()) || (!gdf_is_valid(build_valid, right_index)))
          {
              constexpr int JoinNullValue{-1};
              reference_result.emplace_back(JoinNullValue, right_index);
          }
        }
    }

    // Sort the result
    if(sort)
    {
      std::sort(reference_result.begin(), reference_result.end());
    }

    if(print)
    {
      std::cout << "Reference result size: " << reference_result.size() << std::endl;
      std::cout << "left index, right index" << std::endl;
      std::copy(reference_result.begin(), reference_result.end(), std::ostream_iterator<result_type>(std::cout, ""));
      std::cout << "\n";
    }

    return reference_result;
  }

  /* --------------------------------------------------------------------------*/
  /**
   * @brief  Computes the result of joining the left and right sets of columns with the libgdf functions
   *
   * @param gdf_result A vector of result_type that holds the result of the libgdf join function
   * @param print Option to print the result computed by the libgdf function
   * @param sort Option to sort the result. This is required to compare the result against the reference solution
   */
  /* ----------------------------------------------------------------------------*/
  std::vector<result_type> compute_gdf_result(bool print = false, bool sort = true, gdf_error expected_result = GDF_SUCCESS)
  {
    const int num_columns = std::tuple_size<multi_column_t>::value;

    gdf_column left_result{};
    gdf_column right_result{};
    left_result.size = 0;
    right_result.size = 0;

    cudf::table left_gdf_columns(gdf_raw_left_columns);
    cudf::table right_gdf_columns(gdf_raw_right_columns);
    std::pair <cudf::table, cudf::table> result;
    std::vector<int> range;
    for (int i = 0; i < num_columns; ++i) {range.push_back(i);}
    switch(op)
    {
      case join_op::LEFT:
        {
          result = cudf::gdf_left_join(
                                       left_gdf_columns, range,
                                       right_gdf_columns, range,
                                       &left_result, &right_result,
                                       &ctxt, range, range);
          break;
        }
      case join_op::INNER:
        {
          result =  cudf::gdf_inner_join(
                                       left_gdf_columns, range,
                                       right_gdf_columns, range,
                                       &left_result, &right_result,
                                       &ctxt, range, range);
          break;
        }
      case join_op::FULL:
        {
          result =  cudf::gdf_full_join(
                                       left_gdf_columns, range,
                                       right_gdf_columns, range,
                                       &left_result, &right_result,
                                       &ctxt, range, range);
          break;
        }
      default:
        std::cout << "Invalid join method" << std::endl;
        EXPECT_TRUE(false);
    }
   
    EXPECT_EQ(left_result.size, right_result.size) << "Join output size mismatch";
    // The output is an array of size `n` where the first n/2 elements are the
    // left_indices and the last n/2 elements are the right indices
    size_t total_pairs = left_result.size;
    size_t output_size = total_pairs*2;

    int * l_join_output = static_cast<int*>(left_result.data);
    int * r_join_output = static_cast<int*>(right_result.data);

    // Host vector to hold gdf join output
    std::vector<int> host_result(output_size);

    // Copy result of gdf join to the host
    EXPECT_EQ(cudaMemcpy(host_result.data(),
               l_join_output, total_pairs * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(host_result.data() + total_pairs,
               r_join_output, total_pairs * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    // Free the original join result
    if(output_size > 0){
      gdf_column_free(&left_result);
      gdf_column_free(&right_result);
    }

    // Host vector of result_type pairs to hold final result for comparison to reference solution
    std::vector<result_type> host_pair_result(total_pairs);

    // Copy raw output into corresponding result_type pair
    for(size_t i = 0; i < total_pairs; ++i){
      host_pair_result[i].first = host_result[i];
      host_pair_result[i].second = host_result[i + total_pairs];
    }

    // Sort the output for comparison to reference solution
    if(sort){
      std::sort(host_pair_result.begin(), host_pair_result.end());
    }

    if(print){
      std::cout << "GDF result size: " << host_pair_result.size() << std::endl;
      std::cout << "left index, right index" << std::endl;
      std::copy(host_pair_result.begin(), host_pair_result.end(), std::ostream_iterator<result_type>(std::cout, ""));
      std::cout << "\n";
    }

    return host_pair_result;
  }
};

// This structure is used to nest the join operations, join method and
// number/types of columns for use with Google Test type-parameterized
// tests .Here join_operation refers to the type of join eg. INNER,
// LEFT, FULL and join_method refers to the underlying join algorithm
//that performs it eg. GDF_HASH or GDF_SORT.
template<join_op join_operation, 
         gdf_method join_method, 
         typename tuple_of_vectors,
         bool keys_are_unique = false>
struct TestParameters
{
  // The method to use for the join
  const static join_op op{join_operation};

  // The method to use for the join
  const static gdf_method join_type{join_method};

  // The tuple of vectors that determines the number and types of the columns to join
  using multi_column_t = tuple_of_vectors;

  const static bool unique_keys{keys_are_unique};
};

const static gdf_method HASH = gdf_method::GDF_HASH;
const static gdf_method SORT = gdf_method::GDF_SORT;

template <typename... T>
using VTuple = std::tuple<std::vector<T>...>;

// Using Google Tests "Type Parameterized Tests"
// Every test defined as TYPED_TEST(JoinTest, *) will be run once for every instance of
// TestParameters defined below
// The kind of join is determined by the first template argument to TestParameters
// The number and types of columns used in both the left and right sets of columns are
// determined by the number and types of vectors in the std::tuple<...> that is the second
// template argument to TestParameters
typedef ::testing::Types<
                          // Single column inner join tests for all types
                          TestParameters< join_op::INNER, HASH, VTuple<int32_t > >,
                          TestParameters< join_op::INNER, HASH, VTuple<int64_t > >,
                          TestParameters< join_op::INNER, HASH, VTuple<float   > >,
                          TestParameters< join_op::INNER, HASH, VTuple<double  > >,
                          TestParameters< join_op::INNER, HASH, VTuple<uint32_t> >,
                          TestParameters< join_op::INNER, HASH, VTuple<uint64_t> >,
                          TestParameters< join_op::INNER, SORT, VTuple<int32_t > >,
                          TestParameters< join_op::INNER, SORT, VTuple<int64_t > >,
                          TestParameters< join_op::INNER, SORT, VTuple<float   > >,
                          TestParameters< join_op::INNER, SORT, VTuple<double  > >,
                          TestParameters< join_op::INNER, SORT, VTuple<uint32_t> >,
                          TestParameters< join_op::INNER, SORT, VTuple<uint64_t> >,
                          // Single column left join tests for all types
                          TestParameters< join_op::LEFT,  HASH, VTuple<int32_t > >,
                          TestParameters< join_op::LEFT,  HASH, VTuple<int64_t > >,
                          TestParameters< join_op::LEFT,  HASH, VTuple<float   > >,
                          TestParameters< join_op::LEFT,  HASH, VTuple<double  > >,
                          TestParameters< join_op::LEFT,  HASH, VTuple<uint32_t> >,
                          TestParameters< join_op::LEFT,  HASH, VTuple<uint64_t> >,
                          TestParameters< join_op::LEFT,  SORT, VTuple<int32_t > >,
                          TestParameters< join_op::LEFT,  SORT, VTuple<int64_t > >,
                          TestParameters< join_op::LEFT,  SORT, VTuple<float   > >,
                          TestParameters< join_op::LEFT,  SORT, VTuple<double  > >,
                          TestParameters< join_op::LEFT,  SORT, VTuple<uint32_t> >,
                          TestParameters< join_op::LEFT,  SORT, VTuple<uint64_t> >,
                          // Single column full join tests for all types
                          TestParameters< join_op::FULL, HASH, VTuple<int32_t > >,
                          TestParameters< join_op::FULL, HASH, VTuple<int64_t > >,
                          TestParameters< join_op::FULL, HASH, VTuple<float   > >,
                          TestParameters< join_op::FULL, HASH, VTuple<double  > >,
                          TestParameters< join_op::FULL, HASH, VTuple<uint32_t> >,
                          TestParameters< join_op::FULL, HASH, VTuple<uint64_t> >,
                          // Two Column Left Join tests for some combination of types
                          TestParameters< join_op::LEFT,  HASH, VTuple<int32_t , int32_t> >,
                          TestParameters< join_op::LEFT,  HASH, VTuple<uint32_t, int32_t> >,
                          // Three Column Left Join tests for some combination of types
                          TestParameters< join_op::LEFT,  HASH, VTuple<int32_t , uint32_t, float  > >,
                          TestParameters< join_op::LEFT,  HASH, VTuple<double  , uint32_t, int64_t> >,
                          // Two Column Inner Join tests for some combination of types
                          TestParameters< join_op::INNER, HASH, VTuple<int32_t , int32_t> >,
                          TestParameters< join_op::INNER, HASH, VTuple<uint32_t, int32_t> >,
                          // Three Column Inner Join tests for some combination of types
                          TestParameters< join_op::INNER, HASH, VTuple<int32_t , uint32_t, float  > >,
                          TestParameters< join_op::INNER, HASH, VTuple<double  , uint32_t, int64_t> >,
                          // Four column test for Left Joins
                          TestParameters< join_op::LEFT, HASH, VTuple<double, int32_t, int64_t, int32_t> >,
                          TestParameters< join_op::LEFT, HASH, VTuple<float, uint32_t, double, int32_t> >,
                          // Four column test for Inner Joins
                          TestParameters< join_op::INNER, HASH, VTuple<uint32_t, float, int64_t, int32_t> >,
                          TestParameters< join_op::INNER, HASH, VTuple<double, float, int64_t, double> >,
                          // Five column test for Left Joins
                          TestParameters< join_op::LEFT, HASH, VTuple<double, int32_t, int64_t, int32_t, int32_t> >,
                          // Five column test for Inner Joins
                          TestParameters< join_op::INNER, HASH, VTuple<uint32_t, float, int64_t, int32_t, float> >
                          > Implementations;

TYPED_TEST_CASE(JoinTest, Implementations);

// This test is used for debugging purposes and is disabled by default.
// The input sizes are small and has a large amount of debug printing enabled.
TYPED_TEST(JoinTest, DISABLED_DebugTest)
{
  this->create_input(5, 2,
                     5, 2,
                     true);

  std::vector<result_type> reference_result = this->compute_reference_solution(true);

  std::vector<result_type> gdf_result = this->compute_gdf_result(true);

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}


TYPED_TEST(JoinTest, EqualValues)
{
  this->create_input(100,1,
                     1000,1);

  std::vector<result_type> reference_result = this->compute_reference_solution();

  std::vector<result_type> gdf_result = this->compute_gdf_result();

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

TYPED_TEST(JoinTest, MaxRandomValues)
{
  this->create_input(10000,RAND_MAX,
                     10000,RAND_MAX);

  std::vector<result_type> reference_result = this->compute_reference_solution();

  std::vector<result_type> gdf_result = this->compute_gdf_result();

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

TYPED_TEST(JoinTest, LeftColumnsBigger)
{
  this->create_input(10000,100,
                     100,100);

  std::vector<result_type> reference_result = this->compute_reference_solution();

  std::vector<result_type> gdf_result = this->compute_gdf_result();

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

TYPED_TEST(JoinTest, RightColumnsBigger)
{
  this->create_input(100,100,
                     10000,100);

  std::vector<result_type> reference_result = this->compute_reference_solution();

  std::vector<result_type> gdf_result = this->compute_gdf_result();

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

TYPED_TEST(JoinTest, EmptyLeftFrame)
{
  this->create_input(0,100,
                     1000,100);

  std::vector<result_type> reference_result = this->compute_reference_solution();

  std::vector<result_type> gdf_result = this->compute_gdf_result();

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

TYPED_TEST(JoinTest, EmptyRightFrame)
{
  this->create_input(1000,100,
                     0,100);

  std::vector<result_type> reference_result = this->compute_reference_solution();

  std::vector<result_type> gdf_result = this->compute_gdf_result();

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

TYPED_TEST(JoinTest, BothFramesEmpty)
{
  this->create_input(0,100,
                     0,100);

  std::vector<result_type> reference_result = this->compute_reference_solution();

  std::vector<result_type> gdf_result = this->compute_gdf_result();

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

// The below tests check correct reporting of missing valid pointer

// Create a new derived class from JoinTest so we can do a new Typed Test set of tests
template <class test_parameters>
struct JoinValidTest : public JoinTest<test_parameters>
{ };

using ValidTestImplementation = testing::Types< TestParameters< join_op::INNER, SORT, VTuple<int32_t >>,
                                                TestParameters< join_op::LEFT , SORT, VTuple<int32_t >>,
                                                TestParameters< join_op::FULL , SORT, VTuple<int32_t >> >;

TYPED_TEST_CASE(JoinValidTest, ValidTestImplementation);

TYPED_TEST(JoinValidTest, ReportValidMaskError)
{
  this->create_input(1000,100,
                     100,100,
                     false, 1);

  std::vector<result_type> gdf_result = this->compute_gdf_result(false, true, GDF_VALIDITY_UNSUPPORTED);
}


// The below tests are for testing inputs that are at or above the maximum input size possible

// Create a new derived class from JoinTest so we can do a new Typed Test set of tests
template <class test_parameters>
struct MaxJoinTest : public JoinTest<test_parameters>
{ };

// Only test for single column inputs for Inner and Left joins because these tests take a long time
using MaxImplementations = testing::Types< TestParameters< join_op::INNER, HASH, VTuple<int32_t >>,
                                           TestParameters< join_op::LEFT, HASH, VTuple<int32_t >> >;

TYPED_TEST_CASE(MaxJoinTest, MaxImplementations);

TYPED_TEST(MaxJoinTest, InputTooLarge)
{   
    const gdf_size_type left_table_size = 100;  
    const gdf_size_type right_table_size = 
      static_cast<gdf_size_type>(std::numeric_limits<int>::max());

    this->create_dummy_input(left_table_size, right_table_size);

    const bool print_result{false};
    const bool sort_result{false};

    // We expect the function to fail when the input is this large
    const gdf_error expected_error{GDF_COLUMN_SIZE_TOO_BIG};

    std::vector<result_type> gdf_result = this->compute_gdf_result(print_result, 
                                                                   sort_result, 
                                                                   expected_error);
}

// These tests will only fail on a non-release build where `assert`s are enabled
#ifndef NDEBUG
TEST(HashTableSizeDeathTest, ZeroOccupancyTest){
    int const num_insertions{100};
    uint32_t occupancy{0};
    EXPECT_DEATH(compute_hash_table_size(num_insertions,occupancy),"");
}

TEST(HashTableSizeDeathTest, TooLargeOccupancyTest){
    int const num_insertions{100};
    uint32_t occupancy{101};
    EXPECT_DEATH(compute_hash_table_size(num_insertions,occupancy),"");
}
#endif

TEST(HashTableSizeTest, OverflowTest){
    int const num_insertions{std::numeric_limits<int>::max()};
    uint32_t occupancy{50};
    size_t hash_table_size = compute_hash_table_size(num_insertions, occupancy);
    size_t expected_size{ size_t{2} * std::numeric_limits<int>::max()};
    ASSERT_TRUE(hash_table_size > num_insertions);
    EXPECT_EQ(expected_size, hash_table_size);
}
