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

#include <cudf.h>
#include <cudf/functions.h>

#include "utilities/cudf_utils.h"

#include <tests/utilities/cudf_test_fixtures.h>

// See this header for all of the recursive handling of tuples of vectors
#include "test_parameters_wo.cuh"
#include "../groupby_test_helpers.cuh"

// See this header for all valid handling 
#include "../valid_vectors.h"

#include <utilities/cudf_utils.h>
#include <utilities/bit_util.cuh>
#include "rmm/thrust_rmm_allocator.h"

namespace without_agg {

// A new instance of this class will be created for each *TEST(GroupByWoAggTest, ...)
// Put all repeated setup and validation stuff here
template <class test_parameters>
struct GroupByWoAggTest : public GdfTest {
  // // The aggregation type is passed via a member of the template argument class
  // const agg_op op = test_parameters::op;

  const GroupByOutType group_output_type {test_parameters::group_output_type};

  gdf_context ctxt;
 
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

  // Type for a unique_ptr to a gdf_column with a custom deleter
  // Custom deleter is defined at construction
  using gdf_col_pointer = typename std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>;

  // Containers for unique_ptrs to gdf_columns that will be used in the gdf_group_by functions
  // unique_ptrs are used to automate freeing device memory
  std::vector<gdf_col_pointer> gdf_input_key_columns;
  gdf_col_pointer gdf_input_value_column;

  std::vector<gdf_col_pointer> gdf_cpu_data_cols_out_columns;
  // gdf_col_pointer gdf_output_indices_column;
  rmm::device_vector<gdf_size_type> gdf_output_indices_column;

  // Containers for the raw pointers to the gdf_columns that will be used as input
  // to the gdf_group_by functions
  std::vector<gdf_column*> gdf_raw_input_key_columns;
  gdf_column* gdf_raw_input_val_column;
  std::vector<gdf_column*> gdf_raw_gdf_data_cols_out_columns;
  // gdf_column* gdf_raw_out_indices;
  gdf_size_type* gdf_raw_out_indices;
  gdf_size_type gdf_raw_out_indices_size;

  GroupByWoAggTest()
  {
    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    static size_t number_of_instantiations{0};
    std::srand(number_of_instantiations++);

    if (this->group_output_type == GroupByOutType::SQL) {
      ctxt.flag_groupby_include_nulls = true;
      ctxt.flag_nulls_sort_behavior = GDF_NULL_AS_LARGEST;
    } else {
      ctxt.flag_groupby_include_nulls = false;
      ctxt.flag_nulls_sort_behavior = GDF_NULL_AS_LARGEST;
    }
  }

  ~GroupByWoAggTest()
  {
  }

  template <typename col_type>
  gdf_col_pointer create_gdf_column(std::vector<col_type> const & host_vector,
          const gdf_size_type n_count = 0, const gdf_valid_type* host_valid = nullptr)
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
    auto deleter = [](gdf_column* col){col->size = 0; RMM_FREE(col->data, 0); RMM_FREE(col->valid, 0); };
    gdf_col_pointer the_column{new gdf_column, deleter};

    // Allocate device storage for gdf_column and copy contents from host_vector
    EXPECT_EQ(RMM_ALLOC(&(the_column->data), host_vector.size() * sizeof(col_type), 0), RMM_SUCCESS);
    EXPECT_EQ(cudaMemcpy(the_column->data, host_vector.data(), host_vector.size() * sizeof(col_type), cudaMemcpyHostToDevice), cudaSuccess);
    if(host_valid != nullptr) {
      auto valid_size = gdf_get_num_chars_bitmask(host_vector.size());
      EXPECT_EQ(RMM_ALLOC((void**)&(the_column->valid), PaddedLength(valid_size), 0), RMM_SUCCESS);
      EXPECT_EQ(cudaMemcpy(the_column->valid, host_valid, valid_size, cudaMemcpyHostToDevice), cudaSuccess);
    }
    else {
      int valid_size = gdf_get_num_chars_bitmask(host_vector.size());
      EXPECT_EQ(RMM_ALLOC((void**)&(the_column->valid), PaddedLength(valid_size), 0), RMM_SUCCESS);
      EXPECT_EQ(cudaMemset(the_column->valid, 0xff, valid_size), cudaSuccess);
    }

     // Fill the gdf_column members
    // the_column->null_count = n_count;

    int valid_count;
    auto gdf_status = gdf_count_nonzero_mask(the_column->valid, host_vector.size(), &valid_count);
    the_column->null_count = host_vector.size() - valid_count;
    assert (gdf_status == GDF_SUCCESS);
    
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
                    bool print = false, const gdf_size_type n_count = 0) {
    size_t shuffle_seed = rand();
    initialize_keys(input_key, key_count, value_per_key, max_key, shuffle_seed);
    initialize_values(input_value, key_count, value_per_key, max_val, shuffle_seed);

    gdf_input_key_columns = initialize_gdf_columns(input_key);
    gdf_input_value_column = create_gdf_column(input_value, n_count);

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
      print_vector(input_value); //todo@!

      std::cout << "\n==============================================" << std::endl;

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

  void create_gdf_output_buffers(const size_t key_count, const size_t value_per_key, const size_t max_key, const size_t max_val, bool print = false) {
      // initialize_keys(cpu_data_cols_out, key_count, value_per_key, max_key, 0, false);
      // initialize_values(cpu_out_indices, key_count, value_per_key, max_val, 0);

      size_t shuffle_seed = rand();
      this->cpu_data_cols_out = this->input_key;
      this->cpu_out_indices = this->input_value;

      gdf_cpu_data_cols_out_columns = initialize_gdf_columns(cpu_data_cols_out);
      gdf_output_indices_column.resize(cpu_out_indices.size());
      cudaMemcpy(this->gdf_output_indices_column.data().get(), cpu_out_indices.data(), cpu_out_indices.size() * sizeof(gdf_size_type), cudaMemcpyHostToDevice);
      for(auto const& c : gdf_cpu_data_cols_out_columns){
        gdf_raw_gdf_data_cols_out_columns.push_back(c.get());
      }
      if(print)
      {
        std::cout << "gdf_cpu_data_cols_out_columns. Size: " << gdf_cpu_data_cols_out_columns.size() << "|" << gdf_cpu_data_cols_out_columns[0]->size << std::endl;
       
        std::cout << "gdf_output_indices_column. Size: " << gdf_output_indices_column.size() << std::endl;
       
        std::cout << "\n==============================================" << std::endl;

      }
      this->gdf_raw_out_indices = this->gdf_output_indices_column.data().get();
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
  void compute_gdf_result(const gdf_error expected_error = GDF_SUCCESS)
  {
    const int num_columns = std::tuple_size<multi_column_t>::value;

    // ctxt.flag_groupby_include_nulls = false;
    // if (num_columns > 1) {//@todo, it is working? 
    //   ctxt.flag_nulls_sort_behavior = 2;
    // }

    gdf_error error{GDF_SUCCESS};

    gdf_column **group_by_input_key = this->gdf_raw_input_key_columns.data();
    gdf_column **gdf_data_cols_out = gdf_raw_gdf_data_cols_out_columns.data();
    
    std::vector<int> groupby_col_indices;
    for (size_t i = 0; i < this->gdf_raw_input_key_columns.size(); i++) 
      groupby_col_indices.push_back(i);

    error = gdf_group_by_without_aggregations(num_columns, 
                                        group_by_input_key,
                                        num_columns,
                                        groupby_col_indices.data(),
                                        gdf_data_cols_out,
                                        this->gdf_raw_out_indices,
                                        &this->gdf_raw_out_indices_size,
                                        &ctxt
                                        );




    EXPECT_EQ(expected_error, error) << "The gdf group by function did not complete successfully";

    if (GDF_SUCCESS == expected_error) {
        
        copy_output_with_array(
                gdf_data_cols_out, cpu_data_cols_out, this->gdf_raw_out_indices, this->gdf_raw_out_indices_size, this->cpu_out_indices);

        //find the widest possible column
        
        // int widest_column = 0;
        // for(int i = 0; i < input.get_width();i++){
        //   int cur_width;
        //   get_column_byte_width(input.get_column(i).get_gdf_column(), &cur_width);
        //   if(cur_width > widest_column){
        //     widest_column = cur_width;
        //   }
        // }

        std::cout << "gdf_data_cols_out - " <<  num_columns << std::endl;
        for (size_t i = 0; i < num_columns; ++i) {
          tuple_each(cpu_data_cols_out, [this](auto& v) {
            for(size_t j = 0; j < this->cpu_out_indices.size(); j++) {
              std::cout << "\tcout_out:" <<  v.at(this->cpu_out_indices[j]) << std::endl;
            }   
          });
        }
        std::cout << "gdf_out_indices - " <<  cpu_out_indices.size() << std::endl;
        for (size_t i = 0; i < cpu_out_indices.size(); ++i) {
          std::cout << "\t index: " << cpu_out_indices[i] << std::endl;
        }
    }
  }

  void compare_gdf_result(map_t& reference_map) {
    ASSERT_EQ(cpu_out_indices.size(), reference_map.size()) << "Size of gdf result does not match reference result\n";

      for(auto &iter : reference_map) {
        tuple_each(iter.first, [&](auto& val) {
            std::cout << val <<  " => " <<  iter.second << std::endl;
        });
      }
      auto ref_size = reference_map.size();
      for (size_t i = 0; i < ref_size; ++i) {
          auto sch = reference_map.find(extractKey(cpu_data_cols_out, cpu_out_indices[i]));
          bool found = (sch != reference_map.end());
          if (found) {
            tuple_each(sch->first, [&](auto &val) {
              std::cout <<  "sch:  " << val  <<  " => " << sch->second << std::endl;
            });
          }

          EXPECT_EQ(found, true);
          if (!found) { continue; }

          // std:: cout << "cpu_out_indices: " <<  cpu_out_indices[i] << " | " << input_value.at(  cpu_out_indices[i] ) << std::endl;
          // if (std::is_integral<output_t>::value) {
          //    EXPECT_EQ(sch->second, input_value.at(  cpu_out_indices[i] ) ); //warning: not really, we don't sorted input_value like in cpu_data_cols_out.... see: v[this->cpu_out_indices[j]]
          // } else {
          //    // EXPECT_NEAR(sch->first, cpu_out_indices[i], sch->second/100.0);
          // }
          //ensure no duplicates in gdf output
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
    this->create_input(num_keys, num_values_per_key, max_key, max_val, true);
    auto reference_map = this->compute_reference_solution();
    this->print_reference_solution(reference_map);

    this->create_gdf_output_buffers(num_keys, num_values_per_key, max_key, max_val, true);
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

  using gdf_col_pointer = typename std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>;

  //contains input valid generated for gdf calculation and reference solution
  std::vector<host_valid_pointer> input_key_valids;

  //contains the input valid aggregation column
  host_valid_pointer input_value_valid;

  //contains grouped by column valid output of the gdf groupby call
  std::vector<host_valid_pointer> cpu_data_cols_out_valid;
  
  //contains the aggregated output column valid
  host_valid_pointer cpu_out_indices_valid;

 
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
  convert_tuple_to_gdf_columns_with_nulls(std::vector<gdf_col_pointer> &gdf_columns,std::tuple<std::vector<Tp>...>& t, std::vector<host_valid_pointer>& valids)
  {
    //bottom of compile-time recursion
    //purposely empty...
  }

  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type
  convert_tuple_to_gdf_columns_with_nulls(std::vector<gdf_col_pointer> &gdf_columns,std::tuple<std::vector<Tp>...>& t, std::vector<host_valid_pointer>& valids)
  {
    // Creates a gdf_column for the current vector and pushes it onto
    // the vector of gdf_columns
    if (valids.size() == 0) {
      gdf_columns.push_back(this->create_gdf_column(std::get<I>(t)));
    }
    else{
      auto valid = valids[I].data();
      auto column_size = std::get<I>(t).size();
      // auto null_count = gdf::util::null_count(valid, column_size);
      gdf_columns.push_back(this->create_gdf_column(std::get<I>(t), 0, valid));
    }

    //recurse to next vector in tuple
    convert_tuple_to_gdf_columns_with_nulls<I + 1, Tp...>(gdf_columns, t, valids);
  }

  // Converts a tuple of host vectors into a vector of gdf_columns
  std::vector<gdf_col_pointer>
  initialize_gdf_columns_with_nulls(multi_column_t host_columns, std::vector<host_valid_pointer>& cpu_data_cols_out_valid)
  {
    std::vector<gdf_col_pointer> gdf_columns;
    convert_tuple_to_gdf_columns_with_nulls(gdf_columns, host_columns, cpu_data_cols_out_valid);
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
  void create_input_with_nulls(const size_t key_count, const size_t value_per_key,
                    const size_t max_key, const size_t max_val,
                    bool print = false) {
    // size_t shuffle_seed = rand();
    // initialize_keys(this->input_key, key_count, value_per_key, max_key, shuffle_seed);
    // initialize_values(this->input_value, key_count, value_per_key, max_val, shuffle_seed);

    // this->gdf_input_key_columns = this->initialize_gdf_columns(this->input_key);
    
    // auto null_count = gdf::util::null_count(input_value_valid.get(), column_length);
    // this->gdf_input_value_column = this->create_gdf_column(this->input_value, null_count);

    // // Fill vector of raw pointers to gdf_columns
    // for(auto const& c : this->gdf_input_key_columns){
    //   this->gdf_raw_input_key_columns.push_back(c.get());
    // }
    // this->gdf_raw_input_val_column = this->gdf_input_value_column.get();

    // if(print)
    // {
    //   std::cout << "Key column(s) created. Size: " << std::get<0>(this->input_key).size() << std::endl;
    //   print_tuple_vector(this->input_key, input_key_valids);

    //   std::cout << "Value column(s) created. Size: " << this->input_value.size() << std::endl;
    //   print_vector(this->input_value, input_value_valid.get());
    //   std::cout << std::endl;
    // }

    size_t shuffle_seed = rand();
    initialize_keys(this->input_key, key_count, value_per_key, max_key, shuffle_seed);
    initialize_values(this->input_value, key_count, value_per_key, max_val, shuffle_seed);
    const gdf_size_type n_count = 0;

    // Init valids and so ON!!!
    auto column_length = this->input_value.size();
    auto n_tuples = std::tuple_size<multi_column_t>::value;
    for (size_t i = 0; i < n_tuples; i++)
        input_key_valids.push_back(create_and_init_valid(column_length)); 
    input_value_valid = create_and_init_valid(column_length);

    this->gdf_input_key_columns = this->initialize_gdf_columns_with_nulls(this->input_key, input_key_valids);
    this->gdf_input_value_column = this->create_gdf_column(this->input_value, n_count, input_value_valid.data());

    // Fill vector of raw pointers to gdf_columns
    for(auto const& c : this->gdf_input_key_columns){
      this->gdf_raw_input_key_columns.push_back(c.get());
    }
    this->gdf_raw_input_val_column = this->gdf_input_value_column.get();

    if(print)
    {
      std::cout << "Key column(s) created. Size: " << std::get<0>(this->input_key).size() << std::endl;
      print_tuple_vector(this->input_key, input_key_valids);

      std::cout << "Value column(s) created. Size: " << this->input_value.size() << std::endl;
      print_vector(this->input_value, input_value_valid.data()); //todo@!

      std::cout << "\n==============================================" << std::endl;

    }
  }
 
  void create_gdf_output_buffers_with_nulls(const size_t key_count, const size_t value_per_key, const size_t max_key, const size_t max_val, bool print = false) {
      size_t shuffle_seed = rand();
      this->cpu_data_cols_out = this->input_key;
      this->cpu_out_indices = this->input_value;

      // initialize_keys(this->cpu_data_cols_out, key_count, value_per_key, max_key, shuffle_seed);
      // initialize_values(this->cpu_out_indices, key_count, value_per_key, max_val, shuffle_seed);

      // Init Valids
      auto column_length = this->cpu_out_indices.size();
      auto n_tuples = std::tuple_size<multi_column_t>::value;
      for (size_t i = 0; i < n_tuples; i++)
        cpu_data_cols_out_valid.push_back(create_and_init_valid(column_length)); 
      cpu_out_indices_valid = create_and_init_valid(column_length);
    
      this->gdf_cpu_data_cols_out_columns = this->initialize_gdf_columns_with_nulls(this->cpu_data_cols_out, cpu_data_cols_out_valid);
      // auto null_count = gdf::util::null_count(cpu_out_indices_valid.data(), column_length);
       
      auto str = gdf::util::gdf_valid_to_str(cpu_out_indices_valid.data(), column_length);
      
      this->gdf_output_indices_column.resize(this->cpu_out_indices.size());
      cudaMemcpy(this->gdf_output_indices_column.data().get(), this->cpu_out_indices.data(), this->cpu_out_indices.size() * sizeof(gdf_size_type), cudaMemcpyHostToDevice);
      // this->gdf_output_indices_column = this->create_gdf_column(this->cpu_out_indices, 0, cpu_out_indices_valid.data());
      for(auto const& c : this->gdf_cpu_data_cols_out_columns){
        this->gdf_raw_gdf_data_cols_out_columns.push_back(c.get());
      }
      this->gdf_raw_out_indices = this->gdf_output_indices_column.data().get();
  }

  std::basic_string<bool> get_input_key_valids(std::vector<host_valid_pointer>& valids, int i, bool &all_valid_key) {
    std::basic_string<bool> valid_key;
    std::for_each(valids.begin(),
                  valids.end(),
                  [&i, &valid_key, &all_valid_key](host_valid_pointer& valid){
                      bool b1 = gdf_is_valid(valid.data(), i);
                      all_valid_key = all_valid_key && b1; 
                      valid_key.push_back(b1);
                  });
    return valid_key;
  } 

  map_t
  compute_reference_solution_with_nulls(void) {
      map_t key_val_map;
      
      for (size_t i = 0; i < this->input_value.size(); i++) {
          if (this->group_output_type == GroupByOutType::PANDAS) { // pandas style, todo what happens? 
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
          else { // sql style : todo solution with nulls
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
  void compute_gdf_result_with_nulls(const gdf_error expected_error = GDF_SUCCESS)
  {
    const int num_columns = std::tuple_size<multi_column_t>::value;

    // ctxt.flag_groupby_include_nulls = false;
    // if (num_columns > 1) {//@todo, it is working? 
    //   ctxt.flag_nulls_sort_behavior = 2;
    // }

    gdf_error error{GDF_SUCCESS};

    gdf_column **group_by_input_key = this->gdf_raw_input_key_columns.data();
    gdf_column **gdf_data_cols_out = this->gdf_raw_gdf_data_cols_out_columns.data();
    
    std::vector<int> groupby_col_indices;
    for (size_t i = 0; i < this->gdf_raw_input_key_columns.size(); i++) 
      groupby_col_indices.push_back(i);

    error = gdf_group_by_without_aggregations(num_columns, 
                                        group_by_input_key,
                                        num_columns,
                                        groupby_col_indices.data(),
                                        gdf_data_cols_out,
                                        this->gdf_raw_out_indices,
                                        &this->gdf_raw_out_indices_size,
                                        &this->ctxt
                                        );




    EXPECT_EQ(expected_error, error) << "The gdf group by function did not complete successfully";

    if (GDF_SUCCESS == expected_error) {
        copy_output_with_array_with_nulls(
                gdf_data_cols_out, this->cpu_data_cols_out, this->cpu_data_cols_out_valid, this->gdf_raw_out_indices, this->gdf_raw_out_indices_size, this->cpu_out_indices);
 
        std::cout << "gdf_data_cols_out - " <<  num_columns << std::endl;
        size_t index = 0;
        tuple_each(this->cpu_data_cols_out, [this, &index](auto& v) {
          auto valid = this->cpu_data_cols_out_valid[index].data();
          for(size_t j = 0; j < this->cpu_out_indices.size(); j++) {
            bool b1 = gdf_is_valid(valid, this->cpu_out_indices[j]); // too important
            if (b1) {
              std::cout << "\tcout_out:" <<  v.at(this->cpu_out_indices[j]) << std::endl;
            } else {
              std::cout << "\tcout_out:" <<   '@' << std::endl;
            }
          }
          index++;
        });

        std::cout << "gdf_out_indices - " <<  this->cpu_out_indices.size() << std::endl;
        for (size_t i = 0; i < this->cpu_out_indices.size(); ++i) {
          std::cout << "\t index: " << this->cpu_out_indices[i] << std::endl;
        }
    }
  }

  void compare_gdf_result_with_nulls(map_t& reference_map) {
    
      for(auto &iter : reference_map) {
      
        tuple_each(iter.first, [&](auto& val) {
            std::cout << val <<  " => " <<  iter.second << std::endl;
        });
      } 
      size_t ref_size = reference_map.size();
      for (size_t i = 0; i < ref_size; ++i) {
          bool all_valid_key = true;
          auto valid_key = this->get_input_key_valids(this->cpu_data_cols_out_valid, this->cpu_out_indices[i], all_valid_key);
          
          auto l_key = extractKeyWithNulls(this->cpu_data_cols_out, valid_key, this->cpu_out_indices[i]);
          std::cout <<  "lkey: \n  "; 
          print_tuple(l_key); //todo@!

          auto sch = reference_map.find(l_key);
          bool found = (sch != reference_map.end());
          if (found) {
            tuple_each(sch->first, [&](auto &val) {
              std::cout <<  "sch:  " << val  <<  " => " << sch->second << std::endl;
            });
          }

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
    this->create_input_with_nulls(num_keys, num_values_per_key, max_key, max_val, true);
    auto reference_map = this->compute_reference_solution_with_nulls();
    this->print_reference_solution(reference_map);

    this->create_gdf_output_buffers_with_nulls(num_keys, num_values_per_key, max_key, max_val, true);
    this->compute_gdf_result_with_nulls();
    this->compare_gdf_result_with_nulls(reference_map);
  }


// TYPED_TEST(GroupValidTest, AllKeysDifferent)
// {   
//     const size_t num_keys = 1<<14;
//     const size_t num_values_per_key = 1;
//     const size_t max_key = num_keys*2;
//     const size_t max_val = 1000;

//     this->create_input_with_nulls(num_keys, num_values_per_key, max_key, max_val, true);
//     auto reference_map = this->compute_reference_solution_with_nulls();
//     this->print_reference_solution(reference_map);
//     this->create_gdf_output_buffers_with_nulls(num_keys, num_values_per_key);
    
//     this->compute_gdf_result();

//     this->compare_gdf_result(reference_map);
// }


} //namespace: without_agg



