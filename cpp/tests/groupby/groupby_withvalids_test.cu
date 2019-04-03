/*
 * Copyright 2018 BlazingDB, Inc.
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

#include "gtest/gtest.h"

#include <cudf.h>
#include <cudf/functions.h>

#include <thrust/device_vector.h>

#include "tests/utilities/cudf_test_fixtures.h"
#include "tests/utilities/cudf_test_utils.cuh"

#include "tests/utilities/column_wrapper.cuh"
#include "tests/utilities/cudf_test_fixtures.h"
#include "utilities/type_dispatcher.hpp"
#include "utilities/wrapper_types.hpp"

// This is the main test feature
template <class T>
struct GroupByCountDTest : public GdfTest
{
  std::vector<T> key_column;
  std::vector<T> value_column;

  using column_wrapper_ptr = typename std::unique_ptr< cudf::test::column_wrapper<T> >;

  column_wrapper_ptr gdf_key_column;
  column_wrapper_ptr gdf_value_column;

  column_wrapper_ptr gdf_output_key_column;
  column_wrapper_ptr gdf_output_val_column;

  gdf_column* gdf_raw_key_column;
  gdf_column* gdf_raw_value_column;

  GroupByCountDTest()
  {
    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    static size_t number_of_instantiations{0};
    std::srand(number_of_instantiations++);
  }

  ~GroupByCountDTest()
  {
  }

  /* --------------------------------------------------------------------------*
   * @brief Initializes the input columns with the given values.
   *
   * @param key_column_list The original values
   * @param value_column_list The values that will be replaced
   * @param new_values_column_list The new values
   * @param print Optionally print the set of columns for debug
   * -------------------------------------------------------------------------*/
  void create_input(const std::initializer_list<T> &key_column_list,
                    const std::initializer_list<T> &value_column_list,
                    bool print = false)
  {
    key_column    = key_column_list;
    value_column = value_column_list;
  
    // auto even_bits_null = [](auto row) { return row > 5; };

    gdf_key_column    = std::make_unique<cudf::test::column_wrapper<T>>(key_column);
    gdf_value_column  = std::make_unique<cudf::test::column_wrapper<T>>(value_column);
    
    gdf_output_key_column = std::make_unique<cudf::test::column_wrapper<T>>(key_column);
    gdf_output_val_column = std::make_unique<cudf::test::column_wrapper<T>>(value_column);

    gdf_raw_key_column = gdf_key_column->get();
    gdf_raw_value_column = gdf_value_column->get();
 

    if(print)
    {
      std::cout << "key_column column(s) created. Size: " << key_column.size() << std::endl;
      print_vector(key_column);

      std::cout << "value_column column(s) created. Size: " << value_column.size() << std::endl;
      print_vector(value_column);
      
      std::cout << "\n";
    }
  }

  /* --------------------------------------------------------------------------*/
  /**
   * @brief Computes a reference solution
   *
   * @param print Option to print the solution for debug
   *
   * @returns A vector of 'T' with the old values replaced  
   */
  /* ----------------------------------------------------------------------------*/
  std::vector<T> compute_reference_solution(bool print = false)
  {
    std::vector<T> reference_result(key_column); 
    return reference_result;
  }

  /* --------------------------------------------------------------------------*/
  /**
   * @brief Replaces the values in a column given a map of old values to be replaced
   * and new values with the libgdf functions
   *
   * @param print Option to print the result computed by the libgdf function
   * 
   * @returns A vector of 'T' with the old values replaced  
   */
  /* ----------------------------------------------------------------------------*/
  std::vector<T> compute_gdf_result(bool print = false, gdf_error expected_result = GDF_SUCCESS)
  {
    gdf_error result_error{GDF_SUCCESS};

    const int num_columns = 1;

    gdf_column **group_by_input_key = &gdf_raw_key_column;
    gdf_column *group_by_input_value = gdf_raw_value_column;
    auto tmp = gdf_output_key_column->get();
    gdf_column **group_by_output_key = &(tmp);
    gdf_column *group_by_output_value = gdf_output_val_column->get();

    gdf_context ctxt = {0, GDF_SORT, 0};
		ctxt.flag_distinct = false;
		ctxt.flag_method = GDF_HASH;
		ctxt.flag_sort_result = 1;
  
    gdf_agg_op op{GDF_AVG};
    gdf_error status  = gdf_group_by_sort(group_by_input_key,
                                    num_columns,
                                    &group_by_input_value,
                                    1,
                                    &op,
                                    group_by_output_key,
                                    &group_by_output_value,
                                    &ctxt);

    EXPECT_EQ(expected_result, result_error) << "The gdf order by function did not complete successfully";

    // If the expected result was not GDF_SUCCESS, then this test was testing for a
    // specific error condition, in which case we return imediately and do not do
    // any further work on the output
    if(GDF_SUCCESS != expected_result){
      return std::vector<T>();
    }

    size_t output_size = gdf_output_key_column->get()->size;

    
    std::vector<T> host_key_values;
    std::vector<gdf_valid_type> host_key_bitmask;

    std::tie(host_key_values, host_key_bitmask) = gdf_output_key_column->to_host();
    std::vector<T> host_val_values;
    std::vector<gdf_valid_type> host_val_bitmask;
    std::tie(host_val_values, host_val_bitmask) = gdf_output_val_column->to_host();

    if(print){
      std::cout << "keys:\n";
      gdf_output_key_column->print();
      // print_vector(host_key_values);
      
      std::cout << "values:\n";
      gdf_output_val_column->print();
      // print_vector(host_val_values);
    }

    return host_key_values;
  }
};


using TestingTypes = ::testing::Types<int32_t>;

TYPED_TEST_CASE(GroupByCountDTest, TestingTypes);

// This test is used for debugging purposes and is disabled by default.
// The input sizes are small and has a large amount of debug printing enabled.
TYPED_TEST(GroupByCountDTest, Sample)
{
  this->create_input({0, 1, 2, 0, 1, 2, 15, 16, 15, 16}, 
                      {1, 1, 1, 1, 1, 1, 2, 2, 6, 6},  true);

  auto reference_result = this->compute_reference_solution(true);
  auto gdf_result = this->compute_gdf_result(true);
   
}