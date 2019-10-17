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

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cudf/cudf.h>
#include <cudf/groupby.hpp>
#include <cudf/legacy/table.hpp>

#include <utilities/cudf_utils.h>

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>

// See this header for all of the recursive handling of tuples of vectors
#include "test_parameters_wo.cuh"
#include "groupby_test_helpers.cuh"

// See this header for all valid handling
#include <bitmask/legacy/legacy_bitmask.hpp>

#include <utilities/cudf_utils.h>
#include <utilities/bit_util.cuh>
#include <rmm/thrust_rmm_allocator.h>

namespace without_agg {

// A new instance of this class will be created for each *TEST(GroupByWoAggTest, ...)
// Put all repeated setup and validation stuff here
template <class test_parameters>
struct GroupByWoAggTest : public GdfTest {
  // // The aggregation type is passed via a member of the template argument class
  // const agg_op op = test_parameters::op;

  const GroupByOutType group_output_type {test_parameters::group_output_type};

  gdf_context context;

  // multi_column_t is a tuple of vectors. The number of vectors in the tuple
  // determines the number of columns to be grouped, and the value_type of each
  // vector determiens the data type of the column
  using multi_column_t = typename test_parameters::multi_column_t;

  //output_t is the output type of the aggregation column
  using output_t = gdf_size_type;

  //map_t is used for reference solution
  using map_t = typename test_parameters::ref_map_type;

  //tuple_t is tuple of datatypes associated with each column to be grouped
  using tuple_t = typename test_parameters::tuple_t;

  //contains input generated for gdf calculation and reference solution
  multi_column_t input_key;

  //contains the input aggregation column
  std::vector<output_t> input_value;

  //contains grouped by column output of the gdf groupby call
  multi_column_t cpu_data_cols_out;

  //contains the aggregated output column
  std::vector<gdf_size_type> cpu_out_indices;

  // Containers for unique_ptrs to gdf_columns that will be used in the gdf_group_by functions
  // unique_ptrs are used to automate freeing device memory
  std::vector<gdf_col_pointer> gdf_input_key_columns;
  gdf_col_pointer gdf_input_value_column;

  // Containers for the raw pointers to the gdf_columns that will be used as input
  // to the gdf_group_by functions
  std::vector<gdf_column*> gdf_raw_input_key_columns;
  gdf_column* gdf_raw_input_val_column;

  GroupByWoAggTest()
  {
    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    static size_t number_of_instantiations{0};
    std::srand(number_of_instantiations++);

    if (this->group_output_type == GroupByOutType::SQL) {
      context.flag_groupby_include_nulls = true;
      context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;
    } else {
      context.flag_groupby_include_nulls = false;
      context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;
    }
  }

  ~GroupByWoAggTest()
  {
  }

  /**
   * @Synopsis  Initializes key columns and aggregation column for gdf group by call
   *
   * @Param key_count The number of unique keys
   * @Param value_per_key The number of times a random aggregation value is generated for a key
   * @Param max_key The maximum value of the key columns
   * @Param max_val The maximum value of aggregation column
   * @Param print Optionally print the keys and aggregation columns for debugging
   */
  void create_input(const size_t key_count, const size_t value_per_key,
                    const size_t max_key, const size_t max_val,
                    bool print = false) {
    size_t shuffle_seed = rand();
    initialize_keys(input_key, key_count, value_per_key, max_key, shuffle_seed);
    initialize_values(input_value, key_count, value_per_key, max_val, shuffle_seed);

    gdf_input_key_columns = initialize_gdf_columns(input_key);
    gdf_input_value_column = init_gdf_column(input_value, 0, [](size_t row, size_t col) { return true; });

    // Fill vector of raw pointers to gdf_columns
    for(auto const& c : gdf_input_key_columns){
      gdf_raw_input_key_columns.push_back(c.get());
    }
    gdf_raw_input_val_column = gdf_input_value_column.get();

    if(print)
    {
      std::cout << "Key column(s) created. Size: " << std::get<0>(input_key).size() << std::endl;
      print_tuple(input_key);

      std::cout << "Value column(s) created. Size: " << input_value.size() << std::endl;
      print_vector(input_value);

      std::cout << "\n==============================================" << std::endl;

    }
  }

  map_t
  compute_reference_solution() {
      map_t key_val_map;
      for (size_t i = 0; i < input_value.size(); ++i) {
          auto l_key = extractKey(input_key, i);
          auto sch = key_val_map.find(l_key);
          if (sch == key_val_map.end()) {
            key_val_map.emplace(l_key, input_value[i]);
          }
      }
      return key_val_map;
  }


  /* --------------------------------------------------------------------------*/
  /**
   * @Synopsis  Computes the gdf result of grouping the input_keys and input_value
   */
  /* ----------------------------------------------------------------------------*/
  void compute_gdf_result()
  {
    const int num_columns = std::tuple_size<multi_column_t>::value;

    gdf_column **group_by_input_key = this->gdf_raw_input_key_columns.data();

    std::vector<int> groupby_col_indices;
    for (size_t i = 0; i < this->gdf_raw_input_key_columns.size(); i++)
      groupby_col_indices.push_back(i);

    cudf::table input_table(group_by_input_key, num_columns);
    
    cudf::table output_table;
    gdf_column indices_col;
    EXPECT_NO_THROW(std::tie(output_table,
                            indices_col) = gdf_group_by_without_aggregations(input_table,
                                                                            num_columns,
                                                                            groupby_col_indices.data(),
                                                                            &context));

    copy_output_with_array(output_table.begin(), cpu_data_cols_out, static_cast<gdf_size_type *>(indices_col.data), indices_col.size, this->cpu_out_indices);

    // Free results
    gdf_column_free(&indices_col);
    std::for_each(output_table.begin(), output_table.end(), [](gdf_column* col){
      RMM_FREE(col->data, 0);
      RMM_FREE(col->valid, 0);
      delete col;
    });
  }

  void compare_gdf_result(map_t& reference_map) {

    ASSERT_EQ(cpu_out_indices.size(), reference_map.size()) << "Size of gdf result does not match reference result\n";

      auto ref_size = reference_map.size();
      for (size_t i = 0; i < ref_size; ++i) {
          auto sch = reference_map.find(extractKey(cpu_data_cols_out, cpu_out_indices[i]));
          bool found = (sch != reference_map.end());
     
          EXPECT_EQ(found, true);
          if (!found) { continue; }

          reference_map.erase(sch);
      }
  }

  void print_reference_solution(map_t & dicc) {
    std::stringstream ss;
    ss << "reference solution:\n";
    for(auto &iter : dicc) {
        print_tuple_value(ss, iter.first);
        ss << " => " <<  iter.second << std::endl;
    }
    std::cout << ss.str() << std::endl;
  }
};


TYPED_TEST_CASE(GroupByWoAggTest, Implementations);

TYPED_TEST(GroupByWoAggTest, GroupbyExampleTest)
{
    const size_t num_keys = 3;
    const size_t num_values_per_key = 8;
    const size_t max_key = num_keys*2;
    const size_t max_val = 10;
    this->create_input(num_keys, num_values_per_key, max_key, max_val, false);
    auto reference_map = this->compute_reference_solution();
    // this->print_reference_solution(reference_map);

    this->compute_gdf_result();
    this->compare_gdf_result(reference_map);
}

// Create a new derived class from JoinTest so we can do a new Typed Test set of tests
template <class test_parameters>
struct GroupValidTest : public GroupByWoAggTest<test_parameters>
{

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

  //contains input valid generated for gdf calculation and reference solution
  std::vector<host_valid_pointer> input_key_valids;

  //contains the input valid aggregation column
  host_valid_pointer input_value_valid;

  //contains grouped by column valid output of the gdf groupby call
  std::vector<host_valid_pointer> cpu_data_cols_out_valid;

  size_t shuffle_seed;

  GroupValidTest()
  {
    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    static size_t number_of_instantiations{0};
    shuffle_seed = number_of_instantiations++;
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
  void create_input_with_nulls(const size_t key_count, const size_t value_per_key,
                    const size_t max_key, const size_t max_val,
                    bool print = false) {
    initialize_keys(this->input_key, key_count, value_per_key, max_key, shuffle_seed);
    initialize_values(this->input_value, key_count, value_per_key, max_val, shuffle_seed);

    // Init valids and so ON!!!
    auto column_length = this->input_value.size();
    auto n_tuples = std::tuple_size<multi_column_t>::value;
    for (size_t i = 0; i < n_tuples; i++)
        input_key_valids.push_back(create_and_init_valid(column_length, column_length * 0.375));
    input_value_valid = create_and_init_valid(column_length, column_length * 0.375);

    this->gdf_input_key_columns = initialize_gdf_columns(this->input_key, 
                                                        [&](size_t row, size_t col) { return gdf_is_valid(input_key_valids[col].get(), row); });
    this->gdf_input_value_column = init_gdf_column(this->input_value, 0, 
                                                  [&](size_t row, size_t col) { return gdf_is_valid(input_value_valid.get(), row); });

    // Fill vector of raw pointers to gdf_columns
    for(auto const& c : this->gdf_input_key_columns){
      this->gdf_raw_input_key_columns.push_back(c.get());
    }
    this->gdf_raw_input_val_column = this->gdf_input_value_column.get();

    if(print)
    {
      std::cout << "Key column(s) created. Size: " << std::get<0>(this->input_key).size() << std::endl;
      print_tuples_and_valids(this->input_key, input_key_valids);

      std::cout << "Value column(s) created. Size: " << this->input_value.size() << std::endl;
      print_vector_and_valid(this->input_value, input_value_valid.get());

      std::cout << "\n==============================================" << std::endl;

    }
  }

  void create_gdf_output_buffers_with_nulls(const size_t key_count, const size_t value_per_key, const size_t max_key, const size_t max_val, bool print = false) {
      // Init Valids
      auto column_length = this->input_value.size();
      auto n_tuples = std::tuple_size<multi_column_t>::value;
      for (size_t i = 0; i < n_tuples; i++)
        cpu_data_cols_out_valid.push_back(create_and_init_valid(column_length, 0));
  }

  std::basic_string<bool> get_input_key_valids(std::vector<host_valid_pointer>& valids, int i, bool &all_valid_key) {
    std::basic_string<bool> valid_key;
    std::for_each(valids.begin(),
                  valids.end(),
                  [&i, &valid_key, &all_valid_key](host_valid_pointer& valid){
                      bool b1 = gdf_is_valid(valid.get(), i);
                      all_valid_key = all_valid_key && b1;
                      valid_key.push_back(b1);
                  });
    return valid_key;
  }

  map_t
  compute_reference_solution_with_nulls(void) {
      map_t key_val_map;

      for (size_t i = 0; i < this->input_value.size(); i++) {
          if (this->group_output_type == GroupByOutType::PANDAS) {
              bool all_valid_key = true;
              auto valid_key = get_input_key_valids(this->input_key_valids, i, all_valid_key);
              if (all_valid_key) {
                auto l_key = extractKeyWithNulls(this->input_key, valid_key, i);
                auto sch = key_val_map.find(l_key);
                if (sch == key_val_map.end()) {
                  key_val_map.emplace(l_key, this->input_value[i]);
                }
              }
          }
          else {
              bool all_valid_key = true;
              auto valid_key = get_input_key_valids(this->input_key_valids, i, all_valid_key);
              auto l_key = extractKeyWithNulls(this->input_key, valid_key, i);
              auto sch = key_val_map.find(l_key);
              if (sch == key_val_map.end()) {
                key_val_map.emplace(l_key, this->input_value[i]);
              }
          }
      }
      return key_val_map;
  }

  /* --------------------------------------------------------------------------*/
  /**
   * @Synopsis  Computes the gdf result of grouping the input_keys and input_value
   */
  /* ----------------------------------------------------------------------------*/
  void compute_gdf_result_with_nulls()
  {
    const int num_columns = std::tuple_size<multi_column_t>::value;

    gdf_column **group_by_input_key = this->gdf_raw_input_key_columns.data();

    std::vector<int> groupby_col_indices;
    for (size_t i = 0; i < this->gdf_raw_input_key_columns.size(); i++){
      groupby_col_indices.push_back(i);
    }

    cudf::table input_table(group_by_input_key, num_columns);

    cudf::table output_table;
    gdf_column indices_col;
    EXPECT_NO_THROW(std::tie(output_table,
                            indices_col) = gdf_group_by_without_aggregations(input_table,
                                                                            num_columns,
                                                                            groupby_col_indices.data(),
                                                                            &this->context));
    copy_output_with_array_with_nulls(
            output_table.begin(), this->cpu_data_cols_out, this->cpu_data_cols_out_valid, static_cast<gdf_size_type *>(indices_col.data), indices_col.size, this->cpu_out_indices);

    // Free results
    gdf_column_free(&indices_col);
    std::for_each(output_table.begin(), output_table.end(), [](gdf_column* col){
      RMM_FREE(col->data, 0);
      RMM_FREE(col->valid, 0);
      delete col;
    });    
  }

  void compare_gdf_result_with_nulls(map_t& reference_map) {

      size_t ref_size =  this->cpu_out_indices.size();
      for (size_t i = 0; i < ref_size; ++i) {
          bool all_valid_key = true;
          auto valid_key = this->get_input_key_valids(this->cpu_data_cols_out_valid, this->cpu_out_indices[i], all_valid_key);

          auto l_key = extractKeyWithNulls(this->cpu_data_cols_out, valid_key, this->cpu_out_indices[i]);
          
          auto sch = reference_map.find(l_key);
          bool found = (sch != reference_map.end());
          
          EXPECT_EQ(found, true);
          if (!found) { continue; }

          reference_map.erase(sch);
      }
  }
};

TYPED_TEST_CASE(GroupValidTest, ValidTestImplementations);

TYPED_TEST(GroupValidTest, GroupbyValidExampleTest)
{
    const size_t num_keys = 3;
    const size_t num_values_per_key = 8;
    const size_t max_key = num_keys*2;
    const size_t max_val = 10;
    this->create_input_with_nulls(num_keys, num_values_per_key, max_key, max_val, false);
    auto reference_map = this->compute_reference_solution_with_nulls();
    // this->print_reference_solution(reference_map);

    this->create_gdf_output_buffers_with_nulls(num_keys, num_values_per_key, max_key, max_val, true);
    this->compute_gdf_result_with_nulls();
    this->compare_gdf_result_with_nulls(reference_map);
}


TYPED_TEST(GroupValidTest, AllKeysDifferent)
{
    const size_t num_keys = 1 << 5;
    const size_t num_values_per_key = 1;
    const size_t max_key = num_keys*2;
    const size_t max_val = 1000;

    this->create_input_with_nulls(num_keys, num_values_per_key, max_key, max_val, false);
    auto reference_map = this->compute_reference_solution_with_nulls();
    // this->print_reference_solution(reference_map);

    this->create_gdf_output_buffers_with_nulls(num_keys, num_values_per_key, max_key, max_val, false);
    this->compute_gdf_result_with_nulls();
    this->compare_gdf_result_with_nulls(reference_map);
}

} //namespace: without_agg
