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
#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include <type_traits>
#include <memory>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>

#include "gdf_test_utils.cuh"

// Put all repeated setup and validation stuff here
template <class test_parameters>
struct HashPartitionTest : public testing::Test
{

  // multi_column_t is a tuple of vectors. The number of vectors in the tuple
  // determines the number of columns, and the value_type of each
  // vector determines the data type of the column
  using multi_column_t = typename test_parameters::multi_column_t;
  multi_column_t the_columns;

  // Containers for unique_ptrs to gdf_columns 
  // unique_ptrs are used to automate freeing device memory
  std::vector<gdf_col_pointer> gdf_columns;

  // Containers for the raw pointers to the gdf_columns
  std::vector<gdf_column*> raw_gdf_columns;

  HashPartitionTest()
  {
    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    static size_t number_of_instantiations{0};
    std::srand(number_of_instantiations++);
  }

  ~HashPartitionTest()
  {
  }

  void create_input( size_t num_rows, size_t max_value,
                     bool print = false)
  {
    initialize_tuple(the_columns, num_rows, max_value);

    gdf_columns = initialize_gdf_columns(the_columns);

    // Fill vector of raw pointers to gdf_columns
    for(auto const& c : gdf_columns){
      this->raw_gdf_columns.push_back(c.get());
    }

    if(print)
    {
      std::cout << "Column(s) created. Size: " << std::get<0>(the_columns).size() << std::endl;
      print_tuple(the_columns);
    }
  }

  void compute_reference_solution(bool print = false, bool sort = true)
  {

  }

  void compute_gdf_result(bool print = false, bool sort = true)
  {
    const int num_columns = std::tuple_size<multi_column_t>::value;

    gdf_error result_error{GDF_SUCCESS};

    gdf_column ** gdf_input_columns = raw_gdf_columns.data();

    EXPECT_EQ(GDF_SUCCESS, result_error);

    if(print){
    }
  }
};

template< typename tuple_of_vectors >
struct TestParameters
{
  // The tuple of vectors that determines the number and types of the columns 
  using multi_column_t = tuple_of_vectors;
};


// Using Google Tests "Type Parameterized Tests"
// Every test defined as TYPED_TEST(HashPartitionTest, *) will be run once for every instance of
// TestParameters defined below
// The number and types of columns determined by the number and types of vectors 
// in the VTuple<...> 
typedef ::testing::Types< TestParameters< VTuple<int32_t> > >Implementations;

TYPED_TEST_CASE(HashPartitionTest, Implementations);

TYPED_TEST(HashPartitionTest, ExampleTest)
{
  /*
  this->create_input(10, 2,
                     10, 2);

  std::vector<result_type> reference_result = this->compute_reference_solution();

  std::vector<result_type> gdf_result = this->compute_gdf_result();

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
  */
}
