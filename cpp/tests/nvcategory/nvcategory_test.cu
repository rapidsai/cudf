/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Felipe Aramburu <felipe@blazingdb.com>
 *     Copyright 2018 Rommel Quintanilla <rommel@blazingdb.com>
 *     Copyright 2019 William Scott Malpica <william@blazingdb.com>
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


#include <gtest/gtest.h>

#include <cudf/binaryop.hpp>
#include <cudf/cudf.h>
#include <utilities/cudf_utils.h>

#include <cudf/functions.h>
#include <cudf/types.h>
#include <iostream>
#include <random>

#include <nvstrings/NVCategory.h>
#include <nvstrings/NVStrings.h>

#include <rmm/rmm.h>
#include <cstring>
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/nvcategory_utils.cuh>
#include <bitmask/legacy/bit_mask.cuh>

// See this header for all of the handling of valids' vectors 
#include <tests/utilities/valid_vectors.h>

#include <cudf/utilities/legacy/nvcategory_util.hpp>


gdf_column * create_column_ints(int32_t* host_data, gdf_size_type num_rows){
	gdf_column * column = new gdf_column{};
	int32_t * data;
	EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(int32_t) , 0), RMM_SUCCESS);
	CUDA_TRY( cudaMemcpy(data, host_data, sizeof(int32_t) * num_rows, cudaMemcpyHostToDevice) );

	bit_mask::bit_mask_t * valid;
	bit_mask::create_bit_mask(&valid, num_rows,1);

	gdf_error err = gdf_column_view(column,
			(void *) data,
			(gdf_valid_type *)valid,
			num_rows,
			GDF_INT32);
	return column;
}

gdf_column * create_column_constant(gdf_size_type num_rows, int value){
	gdf_column * column = new gdf_column{};
	int * data;
	bit_mask::bit_mask_t * valid;
	bit_mask::create_bit_mask(&valid, num_rows,1);
	EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(int) , 0), RMM_SUCCESS);
	cudaMemset(data,value,sizeof(int) * num_rows);
	gdf_error err = gdf_column_view(column,
			(void *) data,
			(gdf_valid_type *) valid,
			num_rows,
			GDF_INT32);
	return column;
}

int32_t* generate_int_data(gdf_size_type num_rows, size_t max_value, bool print=false){
	int32_t* host_data = new int32_t[num_rows];

	for(gdf_size_type row_index = 0; row_index < num_rows; row_index++){
		host_data[row_index] = std::rand() % max_value;

		if(print)
			std::cout<<host_data[row_index]<<"\t";
	}
	if(print)
		std::cout<<std::endl;

	return host_data;
}

struct NVCategoryTest : public GdfTest
{
	gdf_column * create_boolean_column(gdf_size_type num_rows){
		gdf_column * column = new gdf_column{};
		int * data;
		bit_mask::bit_mask_t * valid;
		bit_mask::create_bit_mask(&valid, num_rows,1);
		EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(int8_t) , 0), RMM_SUCCESS);
		gdf_error err = gdf_column_view(column,
				(void *) data,
				(gdf_valid_type *) valid,
				num_rows,
				GDF_INT8);
		return column;

	}

	gdf_column * create_indices_column(gdf_size_type num_rows){
		gdf_column * column = new gdf_column{};
		int * data;
		bit_mask::bit_mask_t * valid;
		bit_mask::create_bit_mask(&valid, num_rows,1);
		EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(int) , 0), RMM_SUCCESS);
		gdf_error err = gdf_column_view(column,
				(void *) data,
				(gdf_valid_type *) valid,
				num_rows,
				GDF_INT32);
		return column;

	}
};

//todo refactor tests
TEST_F(NVCategoryTest, TEST_NVCATEGORY_SORTING)
{
	bool print = false;
	const int rows_size = 64;
	const int length = 2;

	const char ** string_data = cudf::test::generate_string_data(rows_size, length, print);
	
	gdf_column * column = cudf::test::create_nv_category_column_strings(string_data, rows_size);
	gdf_column * output_column = create_indices_column(rows_size);

	gdf_column ** input_columns = new gdf_column *[1];
	input_columns[0] = column;

	if(print){
		print_gdf_column(input_columns[0]);
	}

	int8_t *asc_desc;
	EXPECT_EQ(RMM_ALLOC(&asc_desc, 1, 0), RMM_SUCCESS);
	int8_t minus_one = -1; //desc
  cudaMemset(asc_desc, minus_one, 1);
  
  gdf_context context;
  context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;

	//doesnt output nvcategory type columns so works as is
	gdf_error err = gdf_order_by(input_columns, asc_desc, 1, output_column, &context);
	EXPECT_EQ(GDF_SUCCESS, err);

	if(print){
		print_gdf_column(output_column);
	}

	int* host_data = new int[rows_size];
	CUDA_TRY( cudaMemcpy(
		host_data,
		output_column->data,
		sizeof(int) * output_column->size,
		cudaMemcpyDeviceToHost) );

	std::vector<std::string> strings_vector(string_data, string_data + rows_size);

	for(size_t i = 0; i < rows_size - 1; i++){
		EXPECT_TRUE(strings_vector[host_data[i]] >= strings_vector[host_data[i+1]]);
	}
}

// Selects the kind of join operation that is performed
enum struct agg_op
{
  MIN,//0
  MAX,//1
  SUM,//2
  CNT,//3
  AVG //4
};

template <agg_op op>
struct AggOp {
    template <typename T>
    T operator()(const T a, const T b) {
        return static_cast<T>(0);
    }
    template <typename T>
    T operator()(const T a) {
        return static_cast<T>(0);
    }
};

template<>
struct AggOp<agg_op::MIN> {
    template <typename T>
    T operator()(const T a, const T b) {
        return (a < b)? a : b;
    }
    template <typename T>
    T operator()(const T a) {
        return a;
    }
};

template<>
struct AggOp<agg_op::MAX> {
    template <typename T>
    T operator()(const T a, const T b) {
        return (a > b)? a : b;
    }
    template <typename T>
    T operator()(const T a) {
        return a;
    }
};

template<>
struct AggOp<agg_op::SUM> {
    template <typename T>
    T operator()(const T a, const T b) {
        return a + b;
    }
    template <typename T>
    T operator()(const T a) {
        return a;
    }
};

template<>
struct AggOp<agg_op::CNT> {
    size_t count{0};
    template <typename T>
    T operator()(const T a, const T b) {
        count = a+1;
        return count;
    }
    template <typename T>
    T operator()(const T a) {
        count = 1;
        return count;
    }
};

TEST_F(NVCategoryTest, TEST_NVCATEGORY_COMPARISON)
{
	bool print = false;
	const int rows_size = 64;
	const size_t length = 1;

	const char ** left_string_data = cudf::test::generate_string_data(rows_size, length, print);
  const char ** right_string_data = cudf::test::generate_string_data(rows_size, length, print);

  std::vector<std::string> left_host_column (left_string_data, left_string_data + rows_size);
  std::vector<std::string> right_host_column (right_string_data, right_string_data + rows_size);

  gdf_column * left_column = cudf::test::create_nv_category_column_strings(left_string_data, rows_size);
	gdf_column * right_column = cudf::test::create_nv_category_column_strings(right_string_data, rows_size);
	
	gdf_column * output_column = create_boolean_column(rows_size);

	NVStrings * temp_string = static_cast<NVCategory *>(right_column->dtype_info.category)->to_strings();
	NVCategory * new_category = static_cast<NVCategory *>(left_column->dtype_info.category)->add_strings(
			*temp_string);

	unsigned int * indices;
	EXPECT_EQ(RMM_ALLOC(&indices, sizeof(unsigned int) * new_category->size(), 0), RMM_SUCCESS);
	//now reset data
	new_category->get_values( (int*)indices, true);

	CUDA_TRY( cudaMemcpy(left_column->data,indices,sizeof(unsigned int) * left_column->size,cudaMemcpyDeviceToDevice) );
	CUDA_TRY( cudaMemcpy(right_column->data,indices + left_column->size,sizeof(unsigned int) * right_column->size,cudaMemcpyDeviceToDevice) );

	if(print){
		print_gdf_column(left_column);
		print_gdf_column(right_column);
	}

	left_column->dtype_info.category = new_category;
	right_column->dtype_info.category = new_category;

	CUDF_EXPECT_NO_THROW(cudf::binary_operation(output_column, left_column, right_column, gdf_binary_operator::GDF_EQUAL));

	int8_t * data = new int8_t[rows_size];
	CUDA_TRY( cudaMemcpy(data, output_column->data, sizeof(int8_t) * rows_size, cudaMemcpyDeviceToHost) );

  for(size_t i = 0; i < rows_size; ++i){
    EXPECT_EQ((bool)data[i], left_host_column[i] == right_host_column[i]);
	}

	delete data;
}

struct NVCategoryConcatTest : public GdfTest
{
	std::vector<gdf_column *> concat_columns;
	gdf_column * concat_out;
	const int length = 2;

	std::vector<std::string> compute_gdf_result(bool print = false){
		size_t keys_size = 0;
		for(size_t i=0;i<concat_columns.size();i++)
			keys_size+=concat_columns[i]->size;

		concat_out = cudf::test::create_nv_category_column(keys_size, true);

		gdf_error err = gdf_column_concat(concat_out, concat_columns.data(), concat_columns.size());
		EXPECT_EQ(GDF_SUCCESS, err);
	
		if(print){
			print_gdf_column(concat_out);
		}

		NVStrings * temp_strings = static_cast<NVCategory *>(concat_out->dtype_info.category)->gather_strings( 
			(nv_category_index_type *) concat_out->data, keys_size, DEVICE_ALLOCATED );
	
		char** host_strings = new char*[keys_size];
		for(size_t i=0;i<keys_size;i++){
			host_strings[i]=new char[length+1];
		}
	
		temp_strings->to_host(host_strings, 0, keys_size);
	
		for(size_t i=0;i<keys_size;i++){
			host_strings[i][length]=0;
		}
	
		std::vector<std::string> strings_vector(host_strings, host_strings + keys_size);

		NVStrings::destroy(temp_strings);

		for(size_t i = 0; i < keys_size; i++){
			delete host_strings[i];
		}
		delete host_strings;

		return strings_vector;
	}

};

TEST_F(NVCategoryConcatTest, concat_test){

	bool print = false;
	const int rows_size = 64;

	const char *** string_data = new const char**[2];
	string_data[0] = cudf::test::generate_string_data(rows_size, length, print);
	string_data[1] = cudf::test::generate_string_data(rows_size, length, print);

	concat_columns.resize(2);
	concat_columns[0] = cudf::test::create_nv_category_column_strings(string_data[0], rows_size);
	concat_columns[1] = cudf::test::create_nv_category_column_strings(string_data[1], rows_size);

	std::vector<std::string> reference_result;
	reference_result.insert(reference_result.end(), string_data[0], string_data[0] + rows_size);
	reference_result.insert(reference_result.end(), string_data[1], string_data[1] + rows_size);
	
	if(print){
		print_gdf_column(concat_columns[0]);
		print_gdf_column(concat_columns[1]);
	}

	std::vector<std::string> gdf_result = this->compute_gdf_result();

	ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

	// Compare the GDF and reference solutions
	for(size_t i = 0; i < reference_result.size(); ++i){
	  EXPECT_EQ(reference_result[i], gdf_result[i]);
	}
}

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
	  os << p.first << "\t" << p.second;
	  std::cout << "\n";
	  return os;
	}
  }

struct NVCategoryJoinTest : public GdfTest
{
  // Containers for the raw pointers to the gdf_columns that will be used as
  // input to the gdf_join functions
  std::vector<gdf_column*> gdf_raw_left_columns;
  std::vector<gdf_column*> gdf_raw_right_columns;
  std::vector<gdf_column*> gdf_raw_result_columns;

  std::vector<std::string> left_string_column;
  std::vector<std::string> right_string_column;

  gdf_context ctxt{0, GDF_HASH, 0};

  /* --------------------------------------------------------------------------*/
  /**
   * @Synopsis  Computes a reference solution for joining the left and right sets of columns
   *
   * @Param print Option to print the solution for debug
   * @Param sort Option to sort the solution. This is necessary for comparison against the gdf solution
   *
   * @Returns A vector of 'result_type' where result_type is a structure with a left_index, right_index
   * where left_columns[left_index] == right_columns[right_index]
   */
  /* ----------------------------------------------------------------------------*/
  std::vector<result_type> compute_reference_solution(join_op op, bool print = false, bool sort = true)
  {
    using key_type = std::string;
    using value_type = size_t;

    // Multimap used to compute the reference solution
    std::multimap<key_type, value_type> the_map;

    // Build hash table that maps the first right columns' values to their row index in the column
	std::vector<key_type> const & build_column = right_string_column;

    for(size_t right_index = 0; right_index < build_column.size(); ++right_index){
      the_map.insert(std::make_pair(build_column[right_index], right_index));
    }

    std::vector<result_type> reference_result;
	
    // Probe hash table with first left column
    std::vector<key_type> const & probe_column = left_string_column;

    for(size_t left_index = 0; left_index < probe_column.size(); ++left_index)
    {
      bool match{false};

        // Find all keys that match probe_key
        const auto probe_key = probe_column[left_index];
        auto range = the_map.equal_range(probe_key);

        // Every element in the returned range identifies a row in the first right column that
        // matches the probe_key. Need to check if all other columns also match
        for(auto i = range.first; i != range.second; ++i)
        {
          const auto right_index = i->second;

		  if(left_string_column[left_index] == right_string_column[right_index]){
            reference_result.emplace_back(left_index, right_index);
            match = true;
          }
        }

      // For left joins, insert a NULL if no match is found
      if((false == match) && ((op == join_op::LEFT) || (op == join_op::FULL))){
        constexpr int JoinNullValue{-1};
        reference_result.emplace_back(left_index, JoinNullValue);
      }
    }

    if (op == join_op::FULL)
    {
        the_map.clear();
        // Build hash table that maps the first left columns' values to their row index in the column
        for(size_t left_index = 0; left_index < probe_column.size(); ++left_index){
          the_map.insert(std::make_pair(probe_column[left_index], left_index));
        }
        // Probe the hash table with first right column
        // Add rows where a match for the right column does not exist
        for(size_t right_index = 0; right_index < build_column.size(); ++right_index)
        {
          const auto probe_key = build_column[right_index];
          auto search = the_map.find(probe_key);
          if ((search == the_map.end()))
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
      std::cout << "\nReference result size: " << reference_result.size() << std::endl;
      std::cout << "left index, right index" << std::endl;
      std::copy(reference_result.begin(), reference_result.end(), std::ostream_iterator<result_type>(std::cout, ""));
      std::cout << "\n";
    }

    return reference_result;
  }

  /* --------------------------------------------------------------------------*/
  /**
   * @Synopsis  Computes the result of joining the left and right sets of columns with the libgdf functions
   *
   * @Param op The join operator
   * @Param left_join_idx The vector of column indexes to join from left dataframe
   * @Param right_join_idx The vector of column indexes to join from right dataframe
   * @Param print Option to print the result computed by the libgdf function
   * @Param sort Option to sort the result. This is required to compare the result against the reference solution
   */
  /* ----------------------------------------------------------------------------*/
  std::vector<result_type> compute_gdf_result(join_op op, std::vector<int> left_join_idx, std::vector<int> right_join_idx, bool print = false, bool sort = true, gdf_error expected_result = GDF_SUCCESS)
  {
    EXPECT_EQ(gdf_raw_left_columns.size(), gdf_raw_right_columns.size()) << "Mismatch columns size";
    EXPECT_EQ(left_join_idx.size(), right_join_idx.size()) << "Mismatch join indexes size";

    gdf_column left_result{};
    gdf_column right_result{};
    left_result.size = 0;
    right_result.size = 0;

    size_t num_columns = gdf_raw_left_columns.size();
    size_t result_num_cols = gdf_raw_left_columns.size() + gdf_raw_right_columns.size() - left_join_idx.size();

    gdf_error result_error{GDF_SUCCESS};

    gdf_column ** left_gdf_columns = gdf_raw_left_columns.data();
    gdf_column ** right_gdf_columns = gdf_raw_right_columns.data();
    gdf_column ** result_columns = gdf_raw_result_columns.data();

    switch(op)
    {
      case join_op::LEFT:
        {
          result_error = gdf_left_join(
                                       left_gdf_columns, num_columns, left_join_idx.data(),
                                       right_gdf_columns, num_columns, right_join_idx.data(),
                                       left_join_idx.size(),
                                       result_num_cols, result_columns,
                                       &left_result, &right_result,
                                       &ctxt);
          break;
        }
      case join_op::INNER:
        {
          result_error =  gdf_inner_join(
                                         left_gdf_columns, num_columns, left_join_idx.data(),
                                         right_gdf_columns, num_columns, right_join_idx.data(),
                                         left_join_idx.size(),
                                         result_num_cols, result_columns,
                                         &left_result, &right_result,
                                         &ctxt);
          break;
        }
      case join_op::FULL:
        {
          result_error =  gdf_full_join(
                                         left_gdf_columns, num_columns, left_join_idx.data(),
                                         right_gdf_columns, num_columns, right_join_idx.data(),
                                         left_join_idx.size(),
                                         result_num_cols, result_columns,
                                         &left_result, &right_result,
                                         &ctxt);
          break;
        }
      default:
        std::cout << "Invalid join method" << std::endl;
        EXPECT_TRUE(false);
    }
   
    EXPECT_EQ(expected_result, result_error) << "The gdf join function did not complete successfully";

    // If the expected result was not GDF_SUCCESS, then this test was testing for a
    // specific error condition, in which case we return imediately and do not do
    // any further work on the output
    if(GDF_SUCCESS != expected_result){
      return std::vector<result_type>();
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
      std::cout << "\nGDF result size: " << host_pair_result.size() << std::endl;
      std::cout << "left index\tright index" << std::endl;
      std::copy(host_pair_result.begin(), host_pair_result.end(), std::ostream_iterator<result_type>(std::cout, ""));
      std::cout << "\n";
    }
    return host_pair_result;
  }

  void check_output(join_op op, std::vector<result_type>& reference_result, size_t length, bool print=false, bool sort=true){
    gdf_column* result_column = gdf_raw_result_columns[0];

    if(print){
      std::cout<<"Raw string result:\n";
      print_gdf_column(result_column);
    }
    
    size_t result_size = result_column->size;
    if(result_size>0){
      NVStrings * temp_strings = static_cast<NVCategory *>(result_column->dtype_info.category)->gather_strings( 
        (nv_category_index_type *) result_column->data, result_size , DEVICE_ALLOCATED );

      char** host_strings = new char*[result_size];
      for(size_t i=0;i<result_size;i++){
        host_strings[i]=new char[length+1];
      }

      temp_strings->to_host(host_strings, 0, result_size);

      for(size_t i=0;i<result_size;i++){
        host_strings[i][length]=0;
      }

      std::vector<std::string> result_output = std::vector<std::string>(host_strings, host_strings + result_size);
      std::vector<std::string> reference_output;

      for(size_t i=0; i<result_size; i++){
        if(reference_result[i].first != -1)
          reference_output.push_back(left_string_column[reference_result[i].first]);
        else
          reference_output.push_back(right_string_column[reference_result[i].second]);
      }

      EXPECT_EQ(reference_output.size(), result_size);

      if(sort){
        std::sort(result_output.begin(), result_output.end());
        std::sort(reference_output.begin(), reference_output.end());
      }

      if(print){
        for(auto str : result_output){
          std::cout<<str<<"\t";
        }
        std::cout<<std::endl;
      }

      NVStrings::destroy(temp_strings);

      for(size_t i = 0; i < result_size; i++){
        delete host_strings[i];
      }
      delete host_strings;

      for(size_t i=0; i<result_size; i++){
        EXPECT_EQ(reference_output[i], result_output[i]);
      }
    }
  }
};

TEST_F(NVCategoryJoinTest, join_test){

	bool print = false;
	size_t rows_size = 64;
	join_op op = join_op::INNER;

	size_t length = 1;
	const char ** left_string_data = cudf::test::generate_string_data(rows_size, length, print);
	const char ** right_string_data = cudf::test::generate_string_data(rows_size, length, print);

	left_string_column = std::vector<std::string> (left_string_data, left_string_data + rows_size);
	right_string_column = std::vector<std::string> (right_string_data, right_string_data + rows_size);

	gdf_column * left_column = cudf::test::create_nv_category_column_strings(left_string_data, rows_size);
  gdf_column * right_column = cudf::test::create_nv_category_column_strings(right_string_data, rows_size);
  gdf_column * result_column = cudf::test::create_nv_category_column_strings(right_string_data, rows_size);
	
	if(print){
		std::cout<<"Raw string indexes:\n";
    print_gdf_column(left_column);
		print_gdf_column(right_column);
	}

	gdf_raw_left_columns.push_back(left_column);
  gdf_raw_right_columns.push_back(right_column);
  gdf_raw_result_columns.push_back(result_column);

	std::vector<result_type> reference_result = this->compute_reference_solution(op, print);

  std::vector<int> left_join_idx={0};
  std::vector<int> right_join_idx={0};

	std::vector<result_type> gdf_result = this->compute_gdf_result(op, left_join_idx, right_join_idx, print);

	ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

	// Compare the GDF and reference solutions
	for(size_t i = 0; i < reference_result.size(); ++i){
	  EXPECT_EQ(reference_result[i], gdf_result[i]);
  }

  this->check_output(op, reference_result, length, print);
}

TEST_F(NVCategoryJoinTest, join_test_nulls){

  bool print = false;
  size_t rows_size = 16;
  join_op op = join_op::INNER;

  size_t length = 1;
  const char ** left_string_data = cudf::test::generate_string_data(rows_size, length, print);
  const char ** right_string_data = cudf::test::generate_string_data(rows_size, length, print);

  left_string_column = std::vector<std::string> (left_string_data, left_string_data + rows_size);
  right_string_column = std::vector<std::string> (right_string_data, right_string_data + rows_size);

  gdf_column * left_column = cudf::test::create_nv_category_column_strings(left_string_data, rows_size);
  gdf_column * right_column = cudf::test::create_nv_category_column_strings(right_string_data, rows_size);
  left_column->valid = nullptr;
  right_column->valid = nullptr;
  if(print){
    std::cout<<"Raw string indexes:\n";
    print_gdf_column(left_column);
    print_gdf_column(right_column);
  }

  gdf_raw_left_columns.push_back(left_column);
  gdf_raw_right_columns.push_back(right_column);

  std::vector<result_type> reference_result = this->compute_reference_solution(op, print);

  std::vector<int> left_join_idx={0};
  std::vector<int> right_join_idx={0};

  std::vector<result_type> gdf_result = this->compute_gdf_result(op, left_join_idx, right_join_idx, print);

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}


TEST_F(NVCategoryJoinTest, join_test_bug){

  bool print = false;
  join_op op = join_op::LEFT;

  const size_t left_size = 3;
  const char *column_left_b[] = {"one  ", "two  ", "NO MATCH"};
  int column_left_a[] = { 5, 14, 8 };

  const size_t right_size = 2;
  const char *column_right_b[] = {"two  ", "one  "};
  int column_left_c[] = { 0, 1 };

  left_string_column = std::vector<std::string> (column_left_b, column_left_b + left_size);
  right_string_column = std::vector<std::string> (column_right_b, column_right_b + right_size);

  gdf_column * left_column = cudf::test::create_nv_category_column_strings(column_left_b, left_size);
  left_column->valid = nullptr;
  gdf_column * left_non_join_column = create_column_ints(column_left_a, left_size);
  left_non_join_column ->valid = nullptr;
  gdf_column * right_column = cudf::test::create_nv_category_column_strings(column_right_b, right_size);
  right_column->valid = nullptr;
  gdf_column * right_non_join_column = create_column_ints(column_left_c, right_size);
  right_non_join_column->valid = nullptr;

  left_column->valid = nullptr;
  right_column->valid = nullptr;
  if(print){
    std::cout<<"Raw string indexes:\n";
    print_gdf_column(left_column);
    print_gdf_column(right_column);
  }

  gdf_raw_left_columns.push_back(left_non_join_column);
  gdf_raw_left_columns.push_back(left_column);
  
  gdf_raw_right_columns.push_back(right_non_join_column);
  gdf_raw_right_columns.push_back(right_column);

  gdf_column * result_column_nonjoin_left = create_column_ints(column_left_a, left_size);
  gdf_column * result_column_nonjoin_right = create_column_ints(column_left_a, left_size);
  gdf_column * result_column_joined = cudf::test::create_nv_category_column_strings(column_left_b, left_size);
  
  gdf_raw_result_columns.push_back(result_column_nonjoin_left);
  gdf_raw_result_columns.push_back(result_column_joined);
  gdf_raw_result_columns.push_back(result_column_nonjoin_right);

  std::vector<result_type> reference_result = this->compute_reference_solution(op, print);

  std::vector<int> left_join_idx={1};
  std::vector<int> right_join_idx={1};

  std::vector<result_type> gdf_result = this->compute_gdf_result(op, left_join_idx, right_join_idx, print);

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }

  if(print){
    std::cout<<"Output columns:\n";
    for(size_t i=0; i<gdf_raw_result_columns.size(); i++){
      print_gdf_column(gdf_raw_result_columns[i]);
      std::cout<<"\n-----\n";
    }
  }

}
