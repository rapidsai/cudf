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
#include <unordered_map>
#include <random>
#include <limits>
#include <memory>
#include <functional>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <../../src/groupby/hash/aggregation_operations.cuh>

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  This file is for unit testing the top level libgdf hash based groupby API
 */
/* ----------------------------------------------------------------------------*/

enum struct agg_op
{
  MIN,
  MAX,
  SUM,
  COUNT,
  AVG
};

// unique_ptr wrappers for gdf_columns that will free their data/
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call ) 									   \
{                                                                                                  \
    cudaError_t cudaStatus = call;                                                                 \
    if ( cudaSuccess != cudaStatus ) {                                                             \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);    \
        exit(1);										   \
    }												   \
}
#endif
std::function<void(gdf_column*)> gdf_col_deleter = [](gdf_column* col){col->size = 0; CUDA_RT_CALL(cudaFree(col->data));};
using gdf_col_pointer = typename std::unique_ptr<gdf_column, decltype(gdf_col_deleter)>;

// Creates a gdf_column from a std::vector
template <typename col_type>
gdf_col_pointer create_gdf_column(std::vector<col_type> const & host_vector)
{
  // Create a new instance of a gdf_column with a custom deleter that will free
  // the associated device memory when it eventually goes out of scope
  gdf_col_pointer the_column{new gdf_column, gdf_col_deleter};
  // Allocate device storage for gdf_column and copy contents from host_vector
  const size_t input_size_bytes = host_vector.size() * sizeof(col_type);
  cudaMallocManaged(&(the_column->data), input_size_bytes);
  cudaMemcpy(the_column->data, host_vector.data(), input_size_bytes, cudaMemcpyHostToDevice);

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
  // Fill the gdf_column members
  the_column->valid = nullptr;
  the_column->size = host_vector.size();
  the_column->dtype = gdf_col_type;
  gdf_dtype_extra_info extra_info;
  extra_info.time_unit = TIME_UNIT_NONE;
  the_column->dtype_info = extra_info;
  return the_column;
}

// A new instance of this class will be created for each *TEST(GroupByTest, ...)
// Put all repeated stuff for each test here
template <typename test_parameters>
struct GDFGroupByTest : public testing::Test 
{
  using key_type = typename test_parameters::key_type;
  using value_type = typename test_parameters::value_type;
  using agg_output_type = typename test_parameters::agg_output_type;

  const agg_op aggregation_operation{test_parameters::the_aggregator};

  const key_type unused_key{std::numeric_limits<key_type>::max()};
  const value_type unused_value{std::numeric_limits<value_type>::max()};

  std::vector<key_type> groupby_column;
  std::vector<value_type> aggregation_column;

  gdf_col_pointer gdf_groupby_column;
  gdf_col_pointer gdf_aggregation_column; 
  gdf_col_pointer gdf_groupby_output;
  gdf_col_pointer gdf_agg_output;

  GDFGroupByTest() 
  {
    // Use constant seed so the psuedo-random order is the same each time
    std::srand(0);
  }

  ~GDFGroupByTest() {} 

  std::pair<std::vector<key_type>, std::vector<value_type>>
    create_reference_input(const size_t num_keys, const size_t num_values_per_key, const int max_key = RAND_MAX, const int max_value = RAND_MAX, bool print = false) 
    {
      const size_t input_size = num_keys * num_values_per_key;

      std::vector<key_type> groupby_column;
      std::vector<value_type> aggregation_column;

      groupby_column.reserve(input_size);
      aggregation_column.reserve(input_size);

      for(size_t i = 0; i < num_keys; ++i )
      {
        // Create random key
        key_type current_key = std::rand() % max_key;

        // Add a decimal to floating point key types
        if(std::is_floating_point<key_type>::value)
        {
          current_key += current_key / std::rand();
        }

        // Don't use unused_key
        while(current_key == this->unused_key)
        {
          current_key = std::rand();
        }

        // For the current key, generate random values
        for(size_t j = 0; j < num_values_per_key; ++j)
        {
          value_type current_value = std::rand() % max_value;

          // Add a decimal to floating point aggregation types
          if(std::is_floating_point<value_type>::value)
          {
            current_value += current_value / std::rand();
          }

          // Don't use unused_value
          while(current_value == this->unused_value)
          {
            current_value = std::rand() % max_value;
          }

          // Store current key and value
          groupby_column.push_back(current_key);
          aggregation_column.push_back(current_value);
        }
      }

      if(print)
      {
        std::cout << "Number of unique keys: " << num_keys 
          << " Values per key: " << num_values_per_key << "\n";

        std::cout << "Group By Column. Size: " << groupby_column.size() << " \n";
        std::copy(groupby_column.begin(), groupby_column.end(), std::ostream_iterator<key_type>(std::cout, " "));
        std::cout << "\n";

        std::cout << "Aggregation Column. Size: " << aggregation_column.size() << "\n";
        std::copy(aggregation_column.begin(), aggregation_column.end(), std::ostream_iterator<value_type>(std::cout, " "));
        std::cout << "\n";
      }

      return std::make_pair(groupby_column, aggregation_column);
    }


  template <class aggregation_operation, typename accumulation_type = agg_output_type>
  std::map<key_type, accumulation_type> 
  dispatch_reference_solution(std::vector<key_type> const & groupby_column,
                              std::vector<value_type> const & aggregation_column,
                              bool print = false)
    {
      std::map<key_type, accumulation_type> expected_values;

      // Computing the reference solution for AVG has to be handled uniquely
      if(std::is_same<aggregation_operation, avg_op<value_type>>::value)
      {

        // For each unique key, compute the SUM and COUNT aggregation
        std::map<key_type, size_t> counts = dispatch_reference_solution<count_op<size_t>, size_t>(groupby_column, aggregation_column);
        std::map<key_type, accumulation_type> sums = dispatch_reference_solution<sum_op<accumulation_type>, accumulation_type>(groupby_column, aggregation_column);

        // For each unique key, compute it's AVG as SUM / COUNT
        for(auto const & sum: sums)
        {
          const auto current_key = sum.first;

          auto count = counts.find(current_key);

          EXPECT_NE(count, counts.end()) << "Failed to find match for key " << current_key << " from the SUM solution in the COUNT solution";

          // Store the average for the current key into expected_values map
          expected_values.insert(std::make_pair(current_key, sum.second / static_cast<accumulation_type>(count->second)));
        }

      }
      else
      {

        aggregation_operation op;

        for(size_t i = 0; i < groupby_column.size(); ++i){

          key_type current_key = groupby_column[i];
          value_type current_value = aggregation_column[i];

          // Use a STL map to keep track of the aggregation for each key
          auto found = expected_values.find(current_key);

          // Key doesn't exist yet, insert it
          if(found == expected_values.end())
          {
            // To support operations like `count`, on the first insert, perform the
            // operation on the new value and the operation's identity value and store the result
            const value_type new_value = op(current_value, aggregation_operation::IDENTITY);

            if(print)
              std::cout << "key: " << current_key << " insert value: " << current_value << " new value: " << new_value << std::endl;

            expected_values.insert(std::make_pair(current_key,new_value)); 

          }
          // Key exists, update the value with the operator
          else
          {
            const value_type new_value = op(current_value, found->second);

            if(print)
              std::cout << "key: " << current_key << " insert value: " << current_value << " new value: " << new_value << std::endl;

            found->second = new_value;
          }
        }
      }

      if(print)
      {
        std::cout << "Reference solution. Size: " << expected_values.size() << std::endl;
        for(auto const & a : expected_values)
        {
          std::cout << a.first << ", " << a.second << std::endl;
        }
      }
      return expected_values;
    }

  // Dispatches computing the reference solution based on which aggregation is to be performed
  // determined by the test_parameters class template argument
  std::map<key_type, agg_output_type> 
  compute_reference_solution(std::vector<key_type> const & groupby_column, 
                             std::vector<value_type> const & aggregation_column,
                             bool print = false)
    {

      std::map<key_type, agg_output_type> reference_solution{};
      switch(test_parameters::the_aggregator)
      {
        case agg_op::MIN:   
          {
            reference_solution = dispatch_reference_solution<min_op<value_type>>(groupby_column,aggregation_column, print);
            break;
          }
        case agg_op::MAX:   
          {
            reference_solution = dispatch_reference_solution<max_op<agg_output_type>>(groupby_column,aggregation_column, print);
            break;
          }
        case agg_op::SUM:   
          {
            reference_solution = dispatch_reference_solution<sum_op<value_type>>(groupby_column,aggregation_column, print);
            break;
          }
        case agg_op::COUNT: 
          {
            reference_solution = dispatch_reference_solution<count_op<value_type>>(groupby_column,aggregation_column, print);
            break;
          }
        case agg_op::AVG:
          {
            reference_solution = dispatch_reference_solution<avg_op<value_type>>(groupby_column,aggregation_column, print);
            break;
          }
        default: 
          {
            std::cout << "Invalid aggregation operation.\n";
            break;
          }
      }

      return reference_solution;
    }

  gdf_error compute_gdf_result(gdf_column * groupby_input, 
                               gdf_column * aggregation_input, 
                               gdf_column * groupby_output,
                               gdf_column * aggregation_output,
                               bool print = false)
  {
    gdf_error error{GDF_SUCCESS};

    gdf_context the_context{0, GDF_HASH, 0, 1};

    switch(aggregation_operation)
    {
      case agg_op::MIN:
        {
          error = gdf_group_by_min(1,
                                   &groupby_input,
                                   aggregation_input,
                                   nullptr,
                                   &groupby_output,
                                   aggregation_output,
                                   &the_context);
          break;
        }
      case agg_op::MAX:
        {
          error = gdf_group_by_max(1,
                                   &groupby_input,
                                   aggregation_input,
                                   nullptr,
                                   &groupby_output,
                                   aggregation_output,
                                   &the_context);
          break;
        }
      case agg_op::SUM:
        {
          error = gdf_group_by_sum(1,
                                   &groupby_input,
                                   aggregation_input,
                                   nullptr,
                                   &groupby_output,
                                   aggregation_output,
                                   &the_context);
          break;
        }
      case agg_op::COUNT:
        {
          error = gdf_group_by_count(1,
                                     &groupby_input,
                                     aggregation_input,
                                     nullptr,
                                     &groupby_output,
                                     aggregation_output,
                                     &the_context);
          break;
        }
      case agg_op::AVG:
        {
          error = gdf_group_by_avg(1,
                                   &groupby_input,
                                   aggregation_input,
                                   nullptr,
                                   &groupby_output,
                                   aggregation_output,
                                   &the_context);
          break;
        }
      default:
        error = GDF_INVALID_AGGREGATOR;
    }

    if(print)
    {
      const size_t output_size = groupby_output->size;
      std::cout << "GDF Output. Size: " << output_size << "\n";
      for(size_t i = 0; i < output_size; ++i)
      {
        std::cout << static_cast<key_type*>(groupby_output->data)[i] << ", " 
                  << static_cast<agg_output_type*>(aggregation_output->data)[i]
                  << std::endl;
      }
    }

    EXPECT_EQ(error, GDF_SUCCESS) << "GDF GroupBy Failed!\n";

    return error;
  }

  void verify_gdf_result(std::map<key_type, agg_output_type> const & expected_values, 
                         gdf_column const * const __restrict__ gdf_groupby_output,
                         gdf_column const * const __restrict__ gdf_agg_output)
  {
    ASSERT_EQ(expected_values.size(), gdf_groupby_output->size) << "Size of GDF Group By output does not match reference solution";
    ASSERT_EQ(expected_values.size(), gdf_agg_output->size) << "Size of GDF Aggregation output does not match reference solution";

    size_t i{0};
    key_type * p_gdf_groupby_output = static_cast<key_type*>(gdf_groupby_output->data);
    agg_output_type * p_gdf_aggregation_output = static_cast<agg_output_type*>(gdf_agg_output->data);
    for(auto const & expected : expected_values)
    {
      if(std::is_floating_point<value_type>::value)
      {
        EXPECT_FLOAT_EQ(expected.first, p_gdf_groupby_output[i]);
        EXPECT_FLOAT_EQ(expected.second, p_gdf_aggregation_output[i]);
      }
      else
      {
        EXPECT_EQ(expected.first, p_gdf_groupby_output[i]);
        EXPECT_EQ(expected.second, p_gdf_aggregation_output[i]);
      }
      ++i;
    }
  }

};

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Google Test can only do a parameterized typed-test over a single type, 
 * so we have to nest multiple types inside of the TestParameters struct.
 * @tparam groupby_type The type used for the single group by column
 * @tparam aggregation_type The type used for the single aggregation column
 * @tparam the_agg An enum that specifies which aggregation is to be performed
 * @tparam accumulation_type The type that is used for the aggregation output. 
 * This defaults to the same type as the aggregation input type. e.g., for AVG with an
 * input of ints, you probably want to use floating point outputs.
 */
/* ----------------------------------------------------------------------------*/
template <typename groupby_type, typename aggregation_type, agg_op the_agg, typename accumulation_type = aggregation_type>
struct TestParameters
{
  using key_type = groupby_type;
  using value_type = aggregation_type;
  using agg_output_type = accumulation_type;
  const static agg_op the_aggregator{the_agg};
};
using TestCases = ::testing::Types< 
                                    //Tests for MAX
                                    TestParameters<int32_t, int32_t, agg_op::MAX>,
                                    TestParameters<int32_t, float, agg_op::MAX>,
                                    TestParameters<int32_t, double, agg_op::MAX>,
                                    TestParameters<int32_t, int64_t, agg_op::MAX>,
                                    TestParameters<int64_t, int32_t, agg_op::MAX>,
                                    TestParameters<int64_t, float, agg_op::MAX>,
                                    TestParameters<int64_t, double, agg_op::MAX>,
                                    TestParameters<int64_t, uint64_t, agg_op::MAX>,
                                    TestParameters<uint64_t, int32_t, agg_op::MAX>,
                                    TestParameters<uint64_t, float, agg_op::MAX>,
                                    TestParameters<uint64_t, double, agg_op::MAX>,
                                    TestParameters<uint64_t, int64_t, agg_op::MAX>,
                                    TestParameters<float, float, agg_op::MAX>,
                                    TestParameters<double, uint64_t, agg_op::MAX>,
                                    // Tests for MIN
                                    TestParameters<int32_t, int32_t, agg_op::MIN>,
                                    TestParameters<int32_t, float, agg_op::MIN>,
                                    TestParameters<int32_t, double, agg_op::MIN>,
                                    TestParameters<int32_t, int64_t, agg_op::MIN>,
                                    TestParameters<int32_t, uint64_t, agg_op::MIN>,
                                    TestParameters<uint64_t, int32_t, agg_op::MIN>,
                                    TestParameters<uint64_t, float, agg_op::MIN>,
                                    TestParameters<uint64_t, double, agg_op::MIN>,
                                    TestParameters<uint64_t, int64_t, agg_op::MIN>,
                                    TestParameters<uint64_t, uint64_t, agg_op::MIN>,
                                    // Tests for COUNT
                                    TestParameters<int32_t, int32_t, agg_op::COUNT>,
                                    TestParameters<int32_t, float, agg_op::COUNT>,
                                    TestParameters<int32_t, double, agg_op::COUNT>,
                                    TestParameters<int32_t, int64_t, agg_op::COUNT>,
                                    TestParameters<int32_t, uint64_t, agg_op::COUNT>,
                                    TestParameters<uint64_t, int32_t, agg_op::COUNT>,
                                    TestParameters<uint64_t, float, agg_op::COUNT>,
                                    TestParameters<uint64_t, double, agg_op::COUNT>,
                                    TestParameters<uint64_t, int64_t, agg_op::COUNT>,
                                    TestParameters<uint64_t, uint64_t, agg_op::COUNT>,
                                    // Tests for SUM 
                                    TestParameters<int32_t, float, agg_op::SUM>, 
                                    TestParameters<int32_t, double, agg_op::SUM>,
                                    TestParameters<int32_t, int64_t, agg_op::SUM>,
                                    TestParameters<int32_t, uint64_t, agg_op::SUM>,
                                    TestParameters<uint64_t, double, agg_op::SUM>,
                                    TestParameters<uint64_t, double, agg_op::SUM>,
                                    TestParameters<uint64_t, int64_t, agg_op::SUM>,
                                    TestParameters<uint64_t, uint64_t, agg_op::SUM>,
                                    // Tests for AVG 
                                    TestParameters<int32_t, int32_t, agg_op::AVG, double>,
                                    TestParameters<uint32_t, uint32_t, agg_op::AVG, double>,
                                    TestParameters<uint64_t, int32_t, agg_op::AVG, double>,
                                    TestParameters<int64_t, int64_t, agg_op::AVG, double>,
                                    TestParameters<int32_t, float, agg_op::AVG, double>,
                                    TestParameters<int32_t, double, agg_op::AVG, double>,
                                    TestParameters<float, double, agg_op::AVG, double>,
                                    TestParameters<double, double, agg_op::AVG, double>
                                    >;

TYPED_TEST_CASE(GDFGroupByTest, TestCases);

TYPED_TEST(GDFGroupByTest, ExampleTest)
{
  const int num_keys = 1<<14;
  const int num_values_per_key = 32;
  const int max_key = 1000;
  const int max_value = 1000;

  // Create reference input columns
  // Note: If the maximum possible value for the aggregation column is large, it is very likely
  // you'll overflow an int when doing SUM or AVG
  std::tie(this->groupby_column, 
           this->aggregation_column) = this->create_reference_input(num_keys, 
                                                                    num_values_per_key,max_key,max_value);

  auto expected_values = this->compute_reference_solution(this->groupby_column, 
                                                          this->aggregation_column);

  // Create gdf_columns with same data as the reference input
  this->gdf_groupby_column = create_gdf_column(this->groupby_column);
  this->gdf_aggregation_column = create_gdf_column(this->aggregation_column);

  // Allocate buffers for output
  const size_t output_size = this->gdf_groupby_column->size;
  this->gdf_groupby_output = create_gdf_column(std::vector<typename TestFixture::key_type>(output_size));
  this->gdf_agg_output = create_gdf_column(std::vector<typename TestFixture::agg_output_type>(output_size));

  this->compute_gdf_result(this->gdf_groupby_column.get(), 
                           this->gdf_aggregation_column.get(), 
                           this->gdf_groupby_output.get(),
                           this->gdf_agg_output.get());
  
  this->verify_gdf_result(expected_values, 
                          this->gdf_groupby_output.get(),
                          this->gdf_agg_output.get());

}

TYPED_TEST(GDFGroupByTest, AllKeysSame)
{
  const int num_keys = 1;
  const int num_values_per_key = 1<<14;
  // Note: If the maximum possible value for the aggregation column is large, it is very likely
  // you'll overflow an int when doing SUM or AVG
  const int max_key = 1000;
  const int max_value = 1000;

  // Create reference input columns
  std::tie(this->groupby_column, 
           this->aggregation_column) = this->create_reference_input(num_keys, 
                                                                    num_values_per_key,
                                                                    max_key,
                                                                    max_value);

  auto expected_values = this->compute_reference_solution(this->groupby_column, 
                                                          this->aggregation_column);

  // Create gdf_columns with same data as the reference input
  this->gdf_groupby_column = create_gdf_column(this->groupby_column);
  this->gdf_aggregation_column = create_gdf_column(this->aggregation_column);

  // Allocate buffers for output
  const size_t output_size = this->gdf_groupby_column->size;
  this->gdf_groupby_output = create_gdf_column(std::vector<typename TestFixture::key_type>(output_size));
  this->gdf_agg_output = create_gdf_column(std::vector<typename TestFixture::agg_output_type>(output_size));

  this->compute_gdf_result(this->gdf_groupby_column.get(), 
                           this->gdf_aggregation_column.get(), 
                           this->gdf_groupby_output.get(),
                           this->gdf_agg_output.get());
  
  this->verify_gdf_result(expected_values, 
                          this->gdf_groupby_output.get(),
                          this->gdf_agg_output.get());

}

// TODO Update so that all the keys are guaranteed to be unique
TYPED_TEST(GDFGroupByTest, AllKeysDifferent)
{
  const int num_keys = 1;
  const int num_values_per_key = 1<<14;
  // Note: If the maximum possible value for the aggregation column is large, it is very likely
  // you'll overflow an int when doing SUM or AVG
  const int max_key = 1000;
  const int max_value = 1000;

  // Create reference input columns
  std::tie(this->groupby_column, 
           this->aggregation_column) = this->create_reference_input(num_keys, 
                                                                    num_values_per_key,
                                                                    max_key,
                                                                    max_value);

  auto expected_values = this->compute_reference_solution(this->groupby_column, 
                                                          this->aggregation_column);

  // Create gdf_columns with same data as the reference input
  this->gdf_groupby_column = create_gdf_column(this->groupby_column);
  this->gdf_aggregation_column = create_gdf_column(this->aggregation_column);

  // Allocate buffers for output
  const size_t output_size = this->gdf_groupby_column->size;
  this->gdf_groupby_output = create_gdf_column(std::vector<typename TestFixture::key_type>(output_size));
  this->gdf_agg_output = create_gdf_column(std::vector<typename TestFixture::agg_output_type>(output_size));

  this->compute_gdf_result(this->gdf_groupby_column.get(), 
                           this->gdf_aggregation_column.get(), 
                           this->gdf_groupby_output.get(),
                           this->gdf_agg_output.get());
  
  this->verify_gdf_result(expected_values, 
                          this->gdf_groupby_output.get(),
                          this->gdf_agg_output.get());

}

TYPED_TEST(GDFGroupByTest, WarpKeysSame)
{
  const int num_keys = 1<<12;
  const int num_values_per_key = 32;
  // Note: If the maximum possible value for the aggregation column is large, it is very likely
  // you'll overflow an int when doing SUM or AVG
  const int max_key = 1000;
  const int max_value = 1000;

  // Create reference input columns
  std::tie(this->groupby_column, 
           this->aggregation_column) = this->create_reference_input(num_keys, 
                                                                    num_values_per_key,
                                                                    max_key,
                                                                    max_value);

  auto expected_values = this->compute_reference_solution(this->groupby_column, 
                                                          this->aggregation_column);

  // Create gdf_columns with same data as the reference input
  this->gdf_groupby_column = create_gdf_column(this->groupby_column);
  this->gdf_aggregation_column = create_gdf_column(this->aggregation_column);

  // Allocate buffers for output
  const size_t output_size = this->gdf_groupby_column->size;
  this->gdf_groupby_output = create_gdf_column(std::vector<typename TestFixture::key_type>(output_size));
  this->gdf_agg_output = create_gdf_column(std::vector<typename TestFixture::agg_output_type>(output_size));

  this->compute_gdf_result(this->gdf_groupby_column.get(), 
                           this->gdf_aggregation_column.get(), 
                           this->gdf_groupby_output.get(),
                           this->gdf_agg_output.get());
  
  this->verify_gdf_result(expected_values, 
                          this->gdf_groupby_output.get(),
                          this->gdf_agg_output.get());

}

TYPED_TEST(GDFGroupByTest, BlockKeysSame)
{
  const int num_keys = 1<<10;
  const int num_values_per_key = 256;
  // Note: If the maximum possible value for the aggregation column is large, it is very likely
  // you'll overflow an int when doing SUM or AVG
  const int max_key = 1000;
  const int max_value = 1000;

  // Create reference input columns
  std::tie(this->groupby_column, 
           this->aggregation_column) = this->create_reference_input(num_keys, 
                                                                    num_values_per_key,
                                                                    max_key,
                                                                    max_value);

  auto expected_values = this->compute_reference_solution(this->groupby_column, 
                                                          this->aggregation_column);

  // Create gdf_columns with same data as the reference input
  this->gdf_groupby_column = create_gdf_column(this->groupby_column);
  this->gdf_aggregation_column = create_gdf_column(this->aggregation_column);

  // Allocate buffers for output
  const size_t output_size = this->gdf_groupby_column->size;
  this->gdf_groupby_output = create_gdf_column(std::vector<typename TestFixture::key_type>(output_size));
  this->gdf_agg_output = create_gdf_column(std::vector<typename TestFixture::agg_output_type>(output_size));

  this->compute_gdf_result(this->gdf_groupby_column.get(), 
                           this->gdf_aggregation_column.get(), 
                           this->gdf_groupby_output.get(),
                           this->gdf_agg_output.get());
  
  this->verify_gdf_result(expected_values, 
                          this->gdf_groupby_output.get(),
                          this->gdf_agg_output.get());

}

