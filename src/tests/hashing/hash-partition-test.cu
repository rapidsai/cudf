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
  const size_t num_cols_to_hash = test_parameters::num_cols_to_hash;
  std::array<int, test_parameters::num_cols_to_hash> cols_to_hash = test_parameters::cols_to_hash;

  // multi_column_t is a tuple of vectors. The number of vectors in the tuple
  // determines the number of columns, and the value_type of each
  // vector determines the data type of the column
  using multi_column_t = typename test_parameters::multi_column_t;
  multi_column_t input_columns;
  multi_column_t output_columns;

  // Containers for unique_ptrs to gdf_columns 
  // unique_ptrs are used to automate freeing device memory
  std::vector<gdf_col_pointer> gdf_input_columns;
  std::vector<gdf_col_pointer> gdf_output_columns;


  // Containers for the raw pointers to the gdf_columns
  std::vector<gdf_column*> raw_gdf_input_columns;
  std::vector<gdf_column*> raw_gdf_output_columns;

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
    initialize_tuple(input_columns, num_rows, max_value);
    initialize_tuple(output_columns, num_rows, max_value);

    gdf_input_columns = initialize_gdf_columns(input_columns);
    gdf_output_columns = initialize_gdf_columns(output_columns);

    // Fill vector of raw pointers to gdf_columns
    for(auto const& c : gdf_input_columns){
      this->raw_gdf_input_columns.push_back(c.get());
      this->raw_gdf_output_columns.push_back(c.get());
    }

    if(print)
    {
      std::cout << "Input column(s) created. Size: " 
                << std::get<0>(input_columns).size() << std::endl;
      print_tuple(input_columns);
    }
  }

  void compute_gdf_result(const int num_partitions, bool print = false)
  {
    const int num_columns = std::tuple_size<multi_column_t>::value;

    gdf_error result_error{GDF_SUCCESS};

    gdf_column ** gdf_input_columns = raw_gdf_input_columns.data();
    gdf_column ** gdf_output_columns = raw_gdf_output_columns.data();

    std::vector<int> partition_offsets(num_partitions,0);

    result_error = gdf_hash_partition(num_columns, 
                                      gdf_input_columns,
                                      this->cols_to_hash.data(),
                                      this->num_cols_to_hash,
                                      num_partitions,
                                      gdf_output_columns,
                                      partition_offsets.data(),
                                      GDF_HASH_MURMUR3);

    EXPECT_EQ(GDF_SUCCESS, result_error);

    if(print){
    }
  }
};

template< typename tuple_of_vectors, 
          int... cols>
struct TestParameters
{

  static_assert((std::tuple_size<tuple_of_vectors>::value >= sizeof...(cols)), 
      "The number of columns to hash must be less than or equal to the total number of columns.");

  // The tuple of vectors that determines the number and types of the columns 
  using multi_column_t = tuple_of_vectors;

  constexpr static const int num_cols_to_hash{sizeof...(cols)};

  // The columns that will be hashed to determine the partitions
  constexpr static const std::array<int, sizeof...(cols)> cols_to_hash{{cols...}};
  //constexpr static const std::vector<size_t> cols_to_hash{{cols...}};
};


// Using Google Tests "Type Parameterized Tests"
// Every test defined as TYPED_TEST(HashPartitionTest, *) will be run once for every instance of
// TestParameters defined below
// The number and types of columns determined by the number and types of vectors 
// in the VTuple<...> 
// The columns to be hashed to determine the partition assignment are the last N integer template
// arguments, where N <= the number of columns specified in the VTuple
typedef ::testing::Types< TestParameters< VTuple<int32_t>, 0 >,
                          TestParameters< VTuple<int32_t, int32_t>, 0, 1> >Implementations;

TYPED_TEST_CASE(HashPartitionTest, Implementations);

TYPED_TEST(HashPartitionTest, ExampleTest)
{

  const int num_partitions = 2;

  this->create_input(10, 2, true);

  this->compute_gdf_result(num_partitions);

  /*

  std::vector<result_type> reference_result = this->compute_reference_solution();

  std::vector<result_type> gdf_result = this->compute_gdf_result();

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
  */
}
