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
#include "order_by_type_vectors.h"

#include <tests/utilities/cudf_test_fixtures.h>

// See this header for all of the handling of valids' vectors
#include <tests/utilities/valid_vectors.h>

// See this header for all of the recursive handling of tuples of vectors
#include <tests/utilities/tuple_vectors.h>

#include <utilities/bit_util.cuh>
#include <utilities/wrapper_types.hpp>
#include <bitmask/legacy_bitmask.hpp>
#include <cudf.h>

#include <rmm/rmm.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdlib>
#include <iostream>
#include <vector>
#include <type_traits>
#include <memory>
#include <numeric>

// A new instance of this class will be created for each *TEST(OrderbyTest, ...)
// Put all repeated setup and validation stuff here
template <class test_parameters>
struct OrderByTest : public GdfTest
{
  const bool nulls_are_smallest = test_parameters::nulls_are_smallest;

  // The sorting order for each column is passed via a member of the template argument class
  std::vector<int8_t> sort_order_types;

  // multi_column_t is a tuple of vectors. The number of vectors in the tuple
  // determines the number of columns to be ordered by, and the value_type of each
  // vector determines the data type of the column
  using multi_column_t = typename test_parameters::multi_column_t;
  multi_column_t orderby_columns;

  size_t numberOfColumns = std::tuple_size<multi_column_t>::value;

  // valids for multi_columns
  std::vector<host_valid_pointer> orderby_valids;

  // Type for a unique_ptr to a gdf_column with a custom deleter
  // Custom deleter is defined at construction
  using gdf_col_pointer =
    typename std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>;

  // Containers for unique_ptrs to gdf_columns that will be used in the orderby
  // functions. unique_ptrs are used to automate freeing device memory
  std::vector<gdf_col_pointer> gdf_orderby_columns;
  gdf_col_pointer gdf_sort_order_types;
  gdf_col_pointer gdf_output_indices_column;

  // Containers for the raw pointers to the gdf_columns that will be used as
  // input to the orderby functions
  std::vector<gdf_column*> gdf_raw_orderby_columns;
  gdf_column* gdf_raw_sort_order_types;
  gdf_column* gdf_raw_output_indices_column;

  OrderByTest()
  {
    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    static size_t number_of_instantiations{0};
    std::srand(number_of_instantiations++);
  }

  ~OrderByTest()
  {
  }

  /* --------------------------------------------------------------------------*
  * @brief Creates a unique_ptr that wraps a gdf_column structure 
  *           initialized with a host vector
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
    gdf_dtype gdf_col_type = GDF_INT8;
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
    gdf_col_pointer the_column{new gdf_column, deleter};

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
    gdf_dtype_extra_info extra_info;
    extra_info.time_unit = TIME_UNIT_NONE;
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
   * @brief  Initializes a set of columns with random values for the order by
   *            operation.
   *
   * @param orderby_column_length The length of the orderby set of columns
   * @param orderby_column_range The upper bound of random values for the orderby
   *                          columns. Values are [0, orderby_column_range)
   * @param n_count The null count in the columns
   * @param random_order_type_values Randomly initialize the sort type for each
   *                                 column.
   * @param print Optionally print the set of columns for debug
   * -------------------------------------------------------------------------*/
  void create_input( size_t orderby_column_length, size_t orderby_column_range,
                     const gdf_size_type n_count = 0, bool random_order_type_values = true, bool print = false)
  {
    initialize_tuple(orderby_columns, orderby_column_length, orderby_column_range);

    auto n_columns = std::tuple_size<multi_column_t>::value;
    initialize_valids(orderby_valids, n_columns, orderby_column_length, n_count);

    gdf_orderby_columns = initialize_gdf_columns(orderby_columns, orderby_valids, n_count);

    // Fill vector of raw pointers to gdf_columns
    gdf_raw_orderby_columns.clear();
    for(auto const& c : gdf_orderby_columns){
      gdf_raw_orderby_columns.push_back(c.get());
    }

    initialize_order_by_types(sort_order_types, n_columns, random_order_type_values);
    gdf_sort_order_types = create_gdf_column(sort_order_types, nullptr, 0);
    gdf_raw_sort_order_types = gdf_sort_order_types.get();

    if(print)
    {
      std::cout << "orderby column(s) created. Size: " << std::get<0>(orderby_columns).size() << std::endl;
      print_tuples_valids_and_order_by_types(orderby_columns, orderby_valids, sort_order_types);
    }
  }

  void create_gdf_output_buffers(const size_t orderby_column_length) {
    std::vector<int> temp(orderby_column_length, 0);
    gdf_output_indices_column = create_gdf_column(temp, nullptr, 0);
    gdf_raw_output_indices_column = gdf_output_indices_column.get();
  }

  // Compile time recursion to sort an array of indices by each vector in a tuple of vectors
  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  sort_multi_column(std::tuple<std::vector<Tp>...>& t, std::vector<host_valid_pointer>& valids, std::vector<int8_t>& asc_desc, std::vector<size_t>& indices)
  {
    //bottom of compile-time recursion
    //purposely empty...
  }
  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type
  sort_multi_column(std::tuple<std::vector<Tp>...>& t, std::vector<host_valid_pointer>& valids, std::vector<int8_t>& asc_desc, std::vector<size_t>& indices)
  {
    const size_t col_index = sizeof...(Tp)-I-1;
    
    // First column have higher priority so we sort back to front 
    auto column = std::get<col_index>(t);
    auto column_valids = valids[col_index].get();

    // Group the invalid rows together at the beginning or the end
    bool nulls_at_front = (nulls_are_smallest && asc_desc[col_index] == GDF_ORDER_ASC) || 
                          (!nulls_are_smallest && asc_desc[col_index] == GDF_ORDER_DESC);
    size_t invalid_count = 0;
    for(size_t i = 0; i < column.size(); ++i)
    {
      size_t j = (nulls_at_front ? i : column.size()-i-1);
      if (!gdf_is_valid(column_valids, indices[j])) {
        if (nulls_at_front) {
          std::rotate(indices.begin()+invalid_count, indices.begin()+i, indices.begin()+i+1);
        }
        else {
          std::rotate(indices.rbegin()+invalid_count, indices.rbegin()+i, indices.rbegin()+i+1);
        }
        ++invalid_count;
      }
    }    

    auto cmp = [&](size_t i1, size_t i2) {
        return (asc_desc[col_index] == GDF_ORDER_ASC ? column[i1] < column[i2] : column[i1] > column[i2]);
      };

    if (nulls_at_front) {
      std::stable_sort(indices.begin() + invalid_count, indices.end(), cmp);
    }
    else {
      std::stable_sort(indices.begin(), indices.end() - invalid_count, cmp);
    }
    
    //recurse to next vector in tuple
    sort_multi_column<I + 1, Tp...>(t, valids, asc_desc, indices);
  }

  /* --------------------------------------------------------------------------*/
  /**
   * @brief  Computes a reference solution
   *
   * @param print Option to print the solution for debug
   *
   * @returns A vector of 'size_t' sorted indices
   */
  /* ----------------------------------------------------------------------------*/
  std::vector<size_t> compute_reference_solution(bool print = false)
  {
    const size_t colums_size = std::get<0>(orderby_columns).size();

    std::vector<size_t> reference_result(colums_size);
    std::iota(std::begin(reference_result), std::end(reference_result), 0);

    sort_multi_column(orderby_columns, orderby_valids, sort_order_types, reference_result);

    if(print)
    {
      std::cout << "Reference result size: " << reference_result.size() << std::endl;
      std::cout << "Indices:" << std::endl;
      std::copy(reference_result.begin(), reference_result.end(), std::ostream_iterator<size_t>(std::cout, ", "));
      std::cout << "\n";
    }

    return reference_result;
  }

  /* --------------------------------------------------------------------------*/
  /**
   * @brief  Computes the result of sorting the set of columns with the libgdf functions
   *
   * @param use_default_sort_order Whether or not to sort using the default ascending order 
   * @param print Option to print the result computed by the libgdf function
   */
  /* ----------------------------------------------------------------------------*/
  std::vector<size_t> compute_gdf_result(bool use_default_sort_order = false, bool print = false, gdf_error expected_result = GDF_SUCCESS)
  {
    const int num_columns = std::tuple_size<multi_column_t>::value;

    gdf_error result_error{GDF_SUCCESS};

    gdf_column** columns_to_sort = gdf_raw_orderby_columns.data();
    gdf_column* sort_order_types = gdf_raw_sort_order_types;
    gdf_column* sorted_indices_output = gdf_raw_output_indices_column;

    result_error = gdf_order_by(columns_to_sort,
                                (use_default_sort_order ? nullptr : (int8_t*)(sort_order_types->data)),
                                num_columns,
                                sorted_indices_output,
                                nulls_are_smallest);

    EXPECT_EQ(expected_result, result_error) << "The gdf order by function did not complete successfully";

    // If the expected result was not GDF_SUCCESS, then this test was testing for a
    // specific error condition, in which case we return imediately and do not do
    // any further work on the output
    if(GDF_SUCCESS != expected_result){
      return std::vector<size_t>();
    }

    size_t output_size = sorted_indices_output->size;
    int* device_result = static_cast<int*>(sorted_indices_output->data);

    // Host vector to hold gdf sort output
    std::vector<int> host_result(output_size);

    // Copy result of gdf sorted_indices_output the host
    EXPECT_EQ(cudaMemcpy(host_result.data(),
               device_result, output_size * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    if(print){
      std::cout << "GDF result size: " << host_result.size() << std::endl;
      std::cout << "Indices:" << std::endl;
      std::copy(host_result.begin(), host_result.end(), std::ostream_iterator<size_t>(std::cout, ", "));
      std::cout << "\n";
    }

    return std::vector<size_t>(host_result.begin(), host_result.end());
  }
};

// This structure is used to nest the number/types of columns and
// the nulls_are_smallest flag for use with Google Test type-parameterized
// tests.
template<typename tuple_of_vectors,
         bool smaller_nulls = true>
struct TestParameters
{
  // The tuple of vectors that determines the number and types of the columns to sort
  using multi_column_t = tuple_of_vectors;

  // nulls are first
   const static bool nulls_are_smallest{smaller_nulls};
};

template <typename... T>
using VTuple = std::tuple<std::vector<T>...>;

// Using Google Tests "Type Parameterized Tests"
// Every test defined as TYPED_TEST(OrderByTest, *) will be run once for every instance of
// TestParameters defined below
typedef ::testing::Types<
                          // Single column Order by Tests for some types
                          TestParameters< VTuple<int32_t>, false >,
                          TestParameters< VTuple<uint64_t>, false >,
                          TestParameters< VTuple<float>, false >,
                          TestParameters< VTuple<int64_t>, true >,
                          TestParameters< VTuple<uint32_t>, true >,
                          TestParameters< VTuple<double>, true >,
                          // Two Column Order by Tests for some combination of types
                          TestParameters< VTuple<int32_t, int32_t>, false >,
                          TestParameters< VTuple<int64_t, uint32_t>, false >,
                          TestParameters< VTuple<uint32_t, double>, false >,
                          TestParameters< VTuple<float, float>, true >,
                          TestParameters< VTuple<uint64_t, float>, true >,
                          TestParameters< VTuple<double, int32_t>, true >,
                          // Three Column Order by Tests for some combination of types
                          TestParameters< VTuple<int32_t, double, uint32_t>, false >,
                          TestParameters< VTuple<float, int32_t, float>, true >

                          // TODO: enable and fix sorting tests for GDF_BOOL8
                          //TestParameters< VTuple<cudf::bool8>, true >,
                          //TestParameters< VTuple<int32_t, cudf::bool8>, false >,
                          //TestParameters< VTuple<double, cudf::bool8>, true >,
                          //TestParameters< VTuple<float, int32_t, cudf::bool8>, true >
                          > Implementations;

TYPED_TEST_CASE(OrderByTest, Implementations);

// This test is used for debugging purposes and is disabled by default.
// The input sizes are small and has a large amount of debug printing enabled.
TYPED_TEST(OrderByTest, DISABLED_DebugTest)
{
  this->create_input(5, 2, 1, true, true);
  this->create_gdf_output_buffers(5);

  std::vector<size_t> reference_result = this->compute_reference_solution(true);

  std::vector<size_t> gdf_result = this->compute_gdf_result(false, true);

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}


TYPED_TEST(OrderByTest, EqualValues)
{
  this->create_input(100, 1);
  this->create_gdf_output_buffers(100);

  std::vector<size_t> reference_result = this->compute_reference_solution();

  std::vector<size_t> gdf_result = this->compute_gdf_result();

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

TYPED_TEST(OrderByTest, EqualValuesNull)
{
  this->create_input(100, 1, 100);
  this->create_gdf_output_buffers(100);

  std::vector<size_t> reference_result = this->compute_reference_solution();

  std::vector<size_t> gdf_result = this->compute_gdf_result();

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

TYPED_TEST(OrderByTest, MaxRandomValues)
{
  this->create_input(10000, RAND_MAX);
  this->create_gdf_output_buffers(10000);

  std::vector<size_t> reference_result = this->compute_reference_solution();

  std::vector<size_t> gdf_result = this->compute_gdf_result();

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

TYPED_TEST(OrderByTest, MaxRandomValuesAndNulls)
{
  this->create_input(10000, RAND_MAX, 2000);
  this->create_gdf_output_buffers(10000);

  std::vector<size_t> reference_result = this->compute_reference_solution();

  std::vector<size_t> gdf_result = this->compute_gdf_result();

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

TYPED_TEST(OrderByTest, EmptyColumns)
{
  this->create_input(0,100);
  this->create_gdf_output_buffers(0);

  std::vector<size_t> reference_result = this->compute_reference_solution();

  std::vector<size_t> gdf_result = this->compute_gdf_result();

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

/*
 * Below group of test are for testing the gdf_order_by method which always
 * sort in ascendig.
 **/

TYPED_TEST(OrderByTest, EqualValuesDefaultSort)
{
  this->create_input(100, 1, 0, false);
  this->create_gdf_output_buffers(100);

  std::vector<size_t> reference_result = this->compute_reference_solution();

  std::vector<size_t> gdf_result = this->compute_gdf_result(true);

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

TYPED_TEST(OrderByTest, EqualValuesNullDefaultSort)
{
  this->create_input(100, 1, 100, false);
  this->create_gdf_output_buffers(100);

  std::vector<size_t> reference_result = this->compute_reference_solution();

  std::vector<size_t> gdf_result = this->compute_gdf_result(true);

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

TYPED_TEST(OrderByTest, MaxRandomValuesDefaultSort)
{
  this->create_input(10000, RAND_MAX, 0, false);
  this->create_gdf_output_buffers(10000);

  std::vector<size_t> reference_result = this->compute_reference_solution();

  std::vector<size_t> gdf_result = this->compute_gdf_result(true);

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

TYPED_TEST(OrderByTest, MaxRandomValuesAndNullsDefaultSort)
{
  this->create_input(10000, RAND_MAX, 2000, false);
  this->create_gdf_output_buffers(10000);

  std::vector<size_t> reference_result = this->compute_reference_solution();

  std::vector<size_t> gdf_result = this->compute_gdf_result(true);

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}

TYPED_TEST(OrderByTest, EmptyColumnsDefaultSort)
{
  this->create_input(0,100, 0, false);
  this->create_gdf_output_buffers(0);

  std::vector<size_t> reference_result = this->compute_reference_solution();

  std::vector<size_t> gdf_result = this->compute_gdf_result(true);

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}
