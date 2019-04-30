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

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>

#include <utilities/int_fastdiv.h>
#include <table/device_table.cuh>
#include <hash/hash_functions.cuh>

#include <cudf.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

#include <rmm/thrust_rmm_allocator.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <iostream>
#include <vector>
#include <map>
#include <type_traits>
#include <memory>

#include <cstdlib>

template <template <typename> class hash_function>
struct row_partition_mapper
{
  __device__
  row_partition_mapper(device_table table_to_hash, const gdf_size_type _num_partitions)
    : the_table{table_to_hash}, num_partitions{_num_partitions}
  {}

  __device__
  hash_value_type operator()(gdf_size_type row_index) const
  {
    return hash_row<hash_function>(the_table, row_index) % num_partitions;
  }

  device_table the_table;

  // Using int_fastdiv can return results different from using the normal modulus
  // operation, therefore we need to use it in result verfication as well
  gdf_size_type num_partitions;
};

// Put all repeated setup and validation stuff here
template <class test_parameters>
struct HashPartitionTest : public GdfTest
{

  constexpr static gdf_hash_func gdf_hash_function = test_parameters::gdf_hash_function;

  const int num_cols_to_hash = test_parameters::num_cols_to_hash;
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
    }
    for(auto const& c : gdf_output_columns){
      this->raw_gdf_output_columns.push_back(c.get());
    }

    if(print)
    {
      std::cout << "Input column(s) created. Size: " 
                << std::get<0>(input_columns).size() << std::endl;
      print_tuple(input_columns);
    }
  }

  std::vector<int> compute_gdf_result(const int num_partitions, bool print = false)
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
                                      gdf_hash_function);

    EXPECT_EQ(GDF_SUCCESS, result_error);

    if(print)
    {
      std::cout << "Partition offsets: ";
      for(int i = 0; i < num_partitions; ++i)
      {
        std::cout << partition_offsets[i] << " ";
      }
      std::cout << std::endl;
    }

    return partition_offsets;
  } 


  void verify_gdf_result(int num_partitions, std::vector<int> partition_offsets, bool print = false)
  {


    std::vector<gdf_column*> gdf_cols_to_hash;

    for(int i = 0; i < num_cols_to_hash; ++i)
    {
      gdf_cols_to_hash.push_back(raw_gdf_output_columns[cols_to_hash[i]]);
    }

    // Create a table from the gdf output of only the columns that were hashed
    auto table_to_hash = device_table::create(num_cols_to_hash, gdf_cols_to_hash.data());

    rmm::device_vector<int> row_partition_numbers(table_to_hash->num_rows());

    // Compute the partition number for every row in the result
    switch(gdf_hash_function)
    {
      case GDF_HASH_MURMUR3:
        {
          thrust::tabulate(thrust::device,
                           row_partition_numbers.begin(),
                           row_partition_numbers.end(),
                           row_partition_mapper<MurmurHash3_32>(*table_to_hash,num_partitions));
          break;
        }
      case GDF_HASH_IDENTITY:
        {
          thrust::tabulate(thrust::device,
                           row_partition_numbers.begin(),
                           row_partition_numbers.end(),
                           row_partition_mapper<IdentityHash>(*table_to_hash,num_partitions));

          break;
        }
      default:
        std::cerr << "Invalid GDF hash function.\n";
    }


    std::vector<int> host_row_partition_numbers(table_to_hash->num_rows());

    cudaMemcpy(host_row_partition_numbers.data(), 
               row_partition_numbers.data().get(),
               table_to_hash->num_rows() * sizeof(int),
               cudaMemcpyDeviceToHost);

    if(print)
    {
      std::cout << "Row partition numbers:\n";
      std::copy(host_row_partition_numbers.begin(), 
                host_row_partition_numbers.end(), 
                std::ostream_iterator<int>(std::cout, ", "));
      std::cout << std::endl;
    }

    // Check that the partition number for every row is correct
    for(int partition_number = 0; partition_number < num_partitions; ++partition_number)
    {
      const int partition_start = partition_offsets[partition_number];
      int partition_stop{0};

      if(partition_number < (num_partitions - 1))
      {
        partition_stop = partition_offsets[partition_number + 1];
      }
      // The end of the last partition is the end of the table
      else
      {
        partition_stop = table_to_hash->num_rows();
      }

      // Everything in the current partition should have the same partition
      // number
      for(int i = partition_start; i < partition_stop; ++i)
      {
        EXPECT_EQ(partition_number, host_row_partition_numbers[i]) << "Partition number for row: " << i << " doesn't match!";
      }
    }
  }
};

template< typename tuple_of_vectors, 
          gdf_hash_func hash,
          int... cols>
struct TestParameters
{
  static_assert((std::tuple_size<tuple_of_vectors>::value >= sizeof...(cols)), 
      "The number of columns to hash must be less than or equal to the total number of columns.");

  // The tuple of vectors that determines the number and types of the columns 
  using multi_column_t = tuple_of_vectors;

  // The hash function to use
  constexpr static const gdf_hash_func gdf_hash_function = hash;

  // The number of columns to hash
  constexpr static const int num_cols_to_hash{sizeof...(cols)};

  // The indices of the columns that will be hashed to determine the partitions
  constexpr static const std::array<int, sizeof...(cols)> cols_to_hash{{cols...}};
};
// Some compilers require this extra declaration outside the class to avoid an
// `undefined reference` link error with the array
template< typename tuple_of_vectors,
          gdf_hash_func hash,
          int... cols>
constexpr const std::array<int, sizeof...(cols)> TestParameters<tuple_of_vectors, hash, cols...>::cols_to_hash;

// Using Google Tests "Type Parameterized Tests"
// Every test defined as TYPED_TEST(HashPartitionTest, *) will be run once for every instance of
// TestParameters defined below
// The number and types of columns determined by the number and types of vectors 
// in the VTuple<...> 
// The hash function to be used is determined by the gdf_hash_func enum
// The columns to be hashed to determine the partition assignment are the last N integer template
// arguments, where N <= the number of columns specified in the VTuple
typedef ::testing::Types< TestParameters< VTuple<int32_t>, GDF_HASH_IDENTITY, 0 >,
                          TestParameters< VTuple<int32_t, int32_t>, GDF_HASH_MURMUR3, 0, 1>,
                          TestParameters< VTuple<float, double>, GDF_HASH_MURMUR3, 1>,
                          TestParameters< VTuple<int64_t, int32_t>, GDF_HASH_MURMUR3, 1>,
                          TestParameters< VTuple<int64_t, int64_t>, GDF_HASH_MURMUR3, 0, 1>,
                          TestParameters< VTuple<int64_t, int64_t, float, double>, GDF_HASH_IDENTITY, 2, 3>,
                          TestParameters< VTuple<int32_t, double, int32_t, double>, GDF_HASH_MURMUR3, 0, 2, 3>,
                          TestParameters< VTuple<int64_t, int64_t, float, double>, GDF_HASH_MURMUR3, 1, 3>,
                          TestParameters< VTuple<int64_t, int64_t>, GDF_HASH_MURMUR3, 0, 1>,
                          TestParameters< VTuple<float, int32_t>, GDF_HASH_MURMUR3, 0>
                         >Implementations;

TYPED_TEST_CASE(HashPartitionTest, Implementations);


TYPED_TEST(HashPartitionTest, ExampleTest)
{
  const int num_partitions = 5;

  this->create_input(100, 100);

  std::vector<int> partition_offsets = this->compute_gdf_result(num_partitions);

  this->verify_gdf_result(num_partitions, partition_offsets);
}



TYPED_TEST(HashPartitionTest, OnePartition)
{
  const int num_partitions = 1;

  this->create_input(100000, 1000);

  std::vector<int> partition_offsets = this->compute_gdf_result(num_partitions);

  this->verify_gdf_result(num_partitions, partition_offsets);
}


TYPED_TEST(HashPartitionTest, TenPartitions)
{
  const int num_partitions = 10;

  this->create_input(1000000, 1000);

  std::vector<int> partition_offsets = this->compute_gdf_result(num_partitions);

  this->verify_gdf_result(num_partitions, partition_offsets);
}

TYPED_TEST(HashPartitionTest, EightPartitions)
{
  const int num_partitions = 8;

  this->create_input(1000000, 1000);

  std::vector<int> partition_offsets = this->compute_gdf_result(num_partitions);

  this->verify_gdf_result(num_partitions, partition_offsets);
}

TYPED_TEST(HashPartitionTest, 257Partitions)
{
  const int num_partitions = 257;

  this->create_input(1000000, 1000);

  std::vector<int> partition_offsets = this->compute_gdf_result(num_partitions);

  this->verify_gdf_result(num_partitions, partition_offsets);
}

