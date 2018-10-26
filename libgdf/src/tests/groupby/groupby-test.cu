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
#include <utility>
#include <type_traits>
#include <typeinfo>
#include <memory>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gdf_test_fixtures.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>

// See this header for all of the recursive handling of tuples of vectors
#include "test_parameters.cuh"
#include "groupby-test-helpers.cuh"

// A new instance of this class will be created for each *TEST(GroupTest, ...)
// Put all repeated setup and validation stuff here
template <class test_parameters>
struct GroupTest : public GdfTest {
  // The aggregation type is passed via a member of the template argument class
  const agg_op op = test_parameters::op;

  gdf_context ctxt = {0, test_parameters::group_type, 0};

  // multi_column_t is a tuple of vectors. The number of vectors in the tuple
  // determines the number of columns to be grouped, and the value_type of each
  // vector determiens the data type of the column
  using multi_column_t = typename test_parameters::multi_column_t;

  //output_t is the output type of the aggregation column
  using output_t = typename test_parameters::output_type;

  //map_t is used for reference solution
  using map_t = typename test_parameters::ref_map_type;

  //tuple_t is tuple of datatypes associated with each column to be grouped
  using tuple_t = typename test_parameters::tuple_t;

  //contains input generated for gdf calculation and reference solution
  multi_column_t input_key;

  //contains the input aggregation column
  std::vector<output_t> input_value;

  //contains grouped by column output of the gdf groupby call
  multi_column_t output_key;

  //contains the aggregated output column
  std::vector<output_t> output_value;

  // Type for a unique_ptr to a gdf_column with a custom deleter
  // Custom deleter is defined at construction
  using gdf_col_pointer = typename std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>;

  // Containers for unique_ptrs to gdf_columns that will be used in the gdf_group_by functions
  // unique_ptrs are used to automate freeing device memory
  std::vector<gdf_col_pointer> gdf_input_key_columns;
  gdf_col_pointer gdf_input_value_column;

  std::vector<gdf_col_pointer> gdf_output_key_columns;
  gdf_col_pointer gdf_output_value_column;

  // Containers for the raw pointers to the gdf_columns that will be used as input
  // to the gdf_group_by functions
  std::vector<gdf_column*> gdf_raw_input_key_columns;
  gdf_column* gdf_raw_input_val_column;
  std::vector<gdf_column*> gdf_raw_output_key_columns;
  gdf_column* gdf_raw_output_val_column;

  GroupTest()
  {
    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    static size_t number_of_instantiations{0};
    std::srand(number_of_instantiations++);
  }

  ~GroupTest()
  {
  }

  template <typename col_type>
  gdf_col_pointer create_gdf_column(std::vector<col_type> const & host_vector)
  {
    // Deduce the type and set the gdf_dtype accordingly
    gdf_dtype gdf_col_type = N_GDF_TYPES;
    if     (std::is_same<col_type,int8_t>::value) gdf_col_type = GDF_INT8;
    else if(std::is_same<col_type,uint8_t>::value) gdf_col_type = GDF_INT8;
    else if(std::is_same<col_type,int16_t>::value) gdf_col_type = GDF_INT16;
    else if(std::is_same<col_type,uint16_t>::value) gdf_col_type = GDF_INT16;
    else if(std::is_same<col_type,int32_t>::value) gdf_col_type = GDF_INT32;
    else if(std::is_same<col_type,uint32_t>::value) gdf_col_type = GDF_INT32;
    else if(std::is_same<col_type,int64_t>::value) gdf_col_type = GDF_INT64;
    else if(std::is_same<col_type,uint64_t>::value) gdf_col_type = GDF_INT64;
    else if(std::is_same<col_type,float>::value) gdf_col_type = GDF_FLOAT32;
    else if(std::is_same<col_type,double>::value) gdf_col_type = GDF_FLOAT64;

    // Create a new instance of a gdf_column with a custom deleter that will free
    // the associated device memory when it eventually goes out of scope
    auto deleter = [](gdf_column* col){col->size = 0; rmmFree(col->data, 0);};
    gdf_col_pointer the_column{new gdf_column, deleter};

    // Allocate device storage for gdf_column and copy contents from host_vector
    rmmAlloc(&(the_column->data), host_vector.size() * sizeof(col_type), 0);
    cudaMemcpy(the_column->data, host_vector.data(), host_vector.size() * sizeof(col_type), cudaMemcpyHostToDevice);

    // Fill the gdf_column members
    the_column->valid = nullptr;
    the_column->size = host_vector.size();
    the_column->dtype = gdf_col_type;
    gdf_dtype_extra_info extra_info;
    extra_info.time_unit = TIME_UNIT_NONE;
    the_column->dtype_info = extra_info;

    return the_column;
  }


  // Converts a tuple of host vectors into a vector of gdf_columns
  std::vector<gdf_col_pointer>
  initialize_gdf_columns(multi_column_t host_columns)
  {
    std::vector<gdf_col_pointer> gdf_columns;
    convert_tuple_to_gdf_columns(gdf_columns, host_columns);
    return gdf_columns;
  }

  /* --------------------------------------------------------------------------*/
  /**
   * @Synopsis  Initializes key columns and aggregation column for gdf group by call
   *
   * @Param key_count The number of unique keys
   * @Param value_per_key The number of times a random aggregation value is generated for a key
   * @Param max_key The maximum value of the key columns
   * @Param max_val The maximum value of aggregation column
   * @Param print Optionally print the keys and aggregation columns for debugging
   */
  /* ----------------------------------------------------------------------------*/
  void create_input(const size_t key_count, const size_t value_per_key,
                    const size_t max_key, const size_t max_val,
                    bool print = false) {
    size_t shuffle_seed = rand();
    initialize_keys(input_key, key_count, value_per_key, max_key, shuffle_seed);
    initialize_values(input_value, key_count, value_per_key, max_val, shuffle_seed);

    gdf_input_key_columns = initialize_gdf_columns(input_key);
    gdf_input_value_column = create_gdf_column(input_value);

    // Fill vector of raw pointers to gdf_columns
    for(auto const& c : gdf_input_key_columns){
      gdf_raw_input_key_columns.push_back(c.get());
    }
    gdf_raw_input_val_column = gdf_input_value_column.get();

    if(print)
    {
      std::cout << "Key column(s) created. Size: " << std::get<0>(input_key).size() << std::endl;
      print_tuple_vector(input_key);

      std::cout << "Value column(s) created. Size: " << input_value.size() << std::endl;
      print_vector(input_value);
    }
  }

    /* --------------------------------------------------------------------------*/
    /**
     * @Synopsis  Creates a unique_ptr that wraps a gdf_column structure intialized with a host vector
     *
     * @Param host_vector The host vector whose data is used to initialize the gdf_column
     *
     * @Returns A unique_ptr wrapping the new gdf_column
     */
    /* ----------------------------------------------------------------------------*/
  // Compile time recursion to convert each vector in a tuple of vectors into
  // a gdf_column and append it to a vector of gdf_columns
  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  convert_tuple_to_gdf_columns(std::vector<gdf_col_pointer> &gdf_columns,std::tuple<std::vector<Tp>...>& t)
  {
    //bottom of compile-time recursion
    //purposely empty...
  }

  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type
  convert_tuple_to_gdf_columns(std::vector<gdf_col_pointer> &gdf_columns,std::tuple<std::vector<Tp>...>& t)
  {
    // Creates a gdf_column for the current vector and pushes it onto
    // the vector of gdf_columns
    gdf_columns.push_back(create_gdf_column(std::get<I>(t)));

    //recurse to next vector in tuple
    convert_tuple_to_gdf_columns<I + 1, Tp...>(gdf_columns, t);
  }

  void create_gdf_output_buffers(const size_t key_count, const size_t value_per_key) {
      initialize_keys(output_key, key_count, value_per_key, 0, 0, false);
      initialize_values(output_value, key_count, value_per_key, 0, 0);

      gdf_output_key_columns = initialize_gdf_columns(output_key);
      gdf_output_value_column = create_gdf_column(output_value);
      for(auto const& c : gdf_output_key_columns){
        gdf_raw_output_key_columns.push_back(c.get());
      }
      gdf_raw_output_val_column = gdf_output_value_column.get();
  }

  map_t
  compute_reference_solution(void) {
      map_t key_val_map;
      if (test_parameters::op != agg_op::AVG) {
          AggOp<test_parameters::op> agg;
          for (size_t i = 0; i < input_value.size(); ++i) {
              auto l_key = extractKey(input_key, i);
              auto sch = key_val_map.find(l_key);
              if (sch != key_val_map.end()) {
                  key_val_map[l_key] = agg(sch->second, input_value[i]);
              } else {
                  key_val_map[l_key] = agg(input_value[i]);
              }
          }
      } else {
          std::map<tuple_t, size_t> counters;
          AggOp<agg_op::SUM> agg;
          for (size_t i = 0; i < input_value.size(); ++i) {
              auto l_key = extractKey(input_key, i);
              counters[l_key]++;
              auto sch = key_val_map.find(l_key);
              if (sch != key_val_map.end()) {
                  key_val_map[l_key] = agg(sch->second, input_value[i]);
              } else {
                  key_val_map[l_key] = agg(input_value[i]);
              }
          }
          for (auto& e : key_val_map) {
              e.second = e.second/counters[e.first];
          }
      }
      return key_val_map;
  }


  /* --------------------------------------------------------------------------*/
  /**
   * @Synopsis  Computes the gdf result of grouping the input_keys and input_value
   */
  /* ----------------------------------------------------------------------------*/
  void compute_gdf_result(void)
  {
    const int num_columns = std::tuple_size<multi_column_t>::value;

    gdf_error error{GDF_SUCCESS};

    gdf_column **group_by_input_key = gdf_raw_input_key_columns.data();
    gdf_column *group_by_input_value = gdf_raw_input_val_column;

    gdf_column **group_by_output_key = gdf_raw_output_key_columns.data();
    gdf_column *group_by_output_value = gdf_raw_output_val_column;

    switch(op)
    {
      case agg_op::MIN:
        {
          error = gdf_group_by_min(num_columns,
                                   group_by_input_key,
                                   group_by_input_value,
                                   nullptr,
                                   group_by_output_key,
                                   group_by_output_value,
                                   &ctxt);
          break;
        }
      case agg_op::MAX:
        {
          error = gdf_group_by_max(num_columns,
                                   group_by_input_key,
                                   group_by_input_value,
                                   nullptr,
                                   group_by_output_key,
                                   group_by_output_value,
                                   &ctxt);
          break;
        }
      case agg_op::SUM:
        {
          error = gdf_group_by_sum(num_columns,
                                   group_by_input_key,
                                   group_by_input_value,
                                   nullptr,
                                   group_by_output_key,
                                   group_by_output_value,
                                   &ctxt);
          break;
        }
      case agg_op::CNT:
        {
          error = gdf_group_by_count(num_columns,
                                   group_by_input_key,
                                   group_by_input_value,
                                   nullptr,
                                   group_by_output_key,
                                   group_by_output_value,
                                   &ctxt);
          break;
        }
      case agg_op::AVG:
        {
          error = gdf_group_by_avg(num_columns,
                                   group_by_input_key,
                                   group_by_input_value,
                                   nullptr,
                                   group_by_output_key,
                                   group_by_output_value,
                                   &ctxt);
          break;
        }
      default:
        error = GDF_INVALID_AGGREGATOR;
    }
    EXPECT_EQ(GDF_SUCCESS, error) << "The gdf group by function did not complete successfully";

    copy_output(
            group_by_output_key, output_key,
            group_by_output_value, output_value);
  }

  void compare_gdf_result(map_t& reference_map) {
      ASSERT_EQ(output_value.size(), reference_map.size()) <<
          "Size of gdf result does not match reference result\n";
      ASSERT_EQ(std::get<0>(output_key).size(), output_value.size()) <<
          "Mismatch between aggregation and group by column size.";
      for (size_t i = 0; i < output_value.size(); ++i) {
          auto sch = reference_map.find(extractKey(output_key, i));
          bool found = (sch != reference_map.end());
          EXPECT_EQ(found, true);
          if (!found) { continue; }
          if (std::is_integral<output_t>::value) {
              EXPECT_EQ(sch->second, output_value[i]);
          } else {
              EXPECT_NEAR(sch->second, output_value[i], sch->second/100.0);
          }
          //ensure no duplicates in gdf output
          reference_map.erase(sch);
      }
  }
};

TYPED_TEST_CASE(GroupTest, Implementations);

TYPED_TEST(GroupTest, GroupbyExampleTest)
{
    const size_t num_keys = 1;
    const size_t num_values_per_key = 8;
    const size_t max_key = num_keys*2;
    const size_t max_val = 1000;
    this->create_input(num_keys, num_values_per_key, max_key, max_val);
    auto reference_map = this->compute_reference_solution();
    this->create_gdf_output_buffers(num_keys, num_values_per_key);
    this->compute_gdf_result();
    this->compare_gdf_result(reference_map);
}

TYPED_TEST(GroupTest, AllKeysSame)
{
    const size_t num_keys = 1;
    const size_t num_values_per_key = 1<<14;
    const size_t max_key = num_keys*2;
    const size_t max_val = 1000;
    this->create_input(num_keys, num_values_per_key, max_key, max_val);
    auto reference_map = this->compute_reference_solution();
    this->create_gdf_output_buffers(num_keys, num_values_per_key);
    this->compute_gdf_result();
    this->compare_gdf_result(reference_map);
}

TYPED_TEST(GroupTest, AllKeysDifferent)
{
    const size_t num_keys = 1<<14;
    const size_t num_values_per_key = 1;
    const size_t max_key = num_keys*2;
    const size_t max_val = 1000;
    this->create_input(num_keys, num_values_per_key, max_key, max_val);
    auto reference_map = this->compute_reference_solution();
    this->create_gdf_output_buffers(num_keys, num_values_per_key);
    this->compute_gdf_result();
    this->compare_gdf_result(reference_map);
}

TYPED_TEST(GroupTest, WarpKeysSame)
{
    const size_t num_keys = 1<<10;
    const size_t num_values_per_key = 32;
    const size_t max_key = num_keys*2;
    const size_t max_val = 1000;
    this->create_input(num_keys, num_values_per_key, max_key, max_val);
    auto reference_map = this->compute_reference_solution();
    this->create_gdf_output_buffers(num_keys, num_values_per_key);
    this->compute_gdf_result();
    this->compare_gdf_result(reference_map);
}

TYPED_TEST(GroupTest, BlockKeysSame)
{
    const size_t num_keys = 1<<10;
    const size_t num_values_per_key = 256;
    const size_t max_key = num_keys*2;
    const size_t max_val = 1000;
    this->create_input(num_keys, num_values_per_key, max_key, max_val);
    auto reference_map = this->compute_reference_solution();
    this->create_gdf_output_buffers(num_keys, num_values_per_key);
    this->compute_gdf_result();
    this->compare_gdf_result(reference_map);
}

TYPED_TEST(GroupTest, EmptyInput)
{
    const size_t num_keys = 0;
    const size_t num_values_per_key = 0;
    const size_t max_key = 0;
    const size_t max_val = 0;
    this->create_input(num_keys, num_values_per_key, max_key, max_val);
    auto reference_map = this->compute_reference_solution();
    this->create_gdf_output_buffers(num_keys, num_values_per_key);
    this->compute_gdf_result();
    this->compare_gdf_result(reference_map);
}
