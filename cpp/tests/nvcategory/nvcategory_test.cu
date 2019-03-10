
#include "gtest/gtest.h"

#include <cudf.h>
#include <utilities/cudf_utils.h>

#include <cudf/functions.h>
#include <cudf/types.h>
#include <iostream>
#include <random>

#include <nvstrings/NVCategory.h>

#include "rmm/rmm.h"
#include <cstring>
#include "tests/utilities/cudf_test_utils.cuh"
#include "tests/utilities/cudf_test_fixtures.h"
#include "bitmask/bit_mask.h"

// See this header for all of the handling of valids' vectors 
#include "tests/utilities/valid_vectors.h"

namespace {
std::string const default_chars = 
	"abcdefghijklmnaoqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
}

std::string random_string(size_t len = 15, std::string const &allowed_chars = default_chars) {
	std::mt19937_64 gen { std::random_device()() };
	std::uniform_int_distribution<size_t> dist { 0, allowed_chars.length()-1 };

	std::string ret;
	std::generate_n(std::back_inserter(ret), len, [&] { return allowed_chars[dist(gen)]; });
	return ret;
}

gdf_column * create_column_ints(int32_t* host_data, gdf_size_type num_rows){
	gdf_column * column = new gdf_column;
	int32_t * data;
	EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(int32_t) , 0), RMM_SUCCESS);
	cudaMemcpy(data, host_data, sizeof(int32_t) * num_rows, cudaMemcpyHostToDevice);

	bit_mask::bit_mask_t * valid;
	bit_mask::create_bit_mask(&valid, num_rows,1);

	gdf_error err = gdf_column_view(column,
			(void *) data,
			(gdf_valid_type *)valid,
			num_rows,
			GDF_INT32);
	return column;
}

gdf_column * create_nv_category_column_strings(const char ** string_host_data, gdf_size_type num_rows){
	NVCategory* category = NVCategory::create_from_array(string_host_data, num_rows);

	gdf_column * column = new gdf_column;
	int * data;
	EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(gdf_nvstring_category) , 0), RMM_SUCCESS);

	category->get_values( (int *)data, true );
	bit_mask::bit_mask_t * valid;
	bit_mask::create_bit_mask(&valid, num_rows,1);

	gdf_error err = gdf_column_view(column,
			(void *) data,
			(gdf_valid_type *)valid,
			num_rows,
			GDF_STRING_CATEGORY);
	column->dtype_info.category = category;
	return column;
}

const char ** generate_string_data(gdf_size_type num_rows, size_t length, bool print=false){
	const char ** string_host_data = new const char *[num_rows];

	for(gdf_size_type row_index = 0; row_index < num_rows; row_index++){
		string_host_data[row_index] = new char[length+1];

		std::string rand_string = random_string(length);
		rand_string.push_back(0);
		if(print)
			std::cout<<rand_string<<"\t";
		std::memcpy((void *) string_host_data[row_index],rand_string.c_str(),rand_string.size());
	}
	if(print)
		std::cout<<std::endl;

	return string_host_data;
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
		gdf_column * column = new gdf_column;
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


	gdf_column * create_column_constant(gdf_size_type num_rows, int value){
		gdf_column * column = new gdf_column;
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


	gdf_column * create_indices_column(gdf_size_type num_rows){
		gdf_column * column = new gdf_column;
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

	gdf_column * create_nv_category_column(gdf_size_type num_rows, bool repeat_strings){

		const char ** string_host_data = new const char *[num_rows];


		for(gdf_size_type row_index = 0; row_index < num_rows; row_index++){
			string_host_data[row_index] = new char[(num_rows + 25) / 26]; //allows string to grow depending on numbe of rows
			std::string temp_string = "";
			int num_chars = repeat_strings ? 1 : (row_index / 26) + 1;
			char repeat_char = (26 - (row_index % 26)) + 65; //chars are Z,Y ...C,B,A,ZZ,YY,.....BBB,AAA.....
			for(int char_index = 0; char_index < num_chars; char_index++){
				temp_string.push_back(repeat_char);
			}
			temp_string.push_back(0);
			std::memcpy((void *) string_host_data[row_index],temp_string.c_str(),temp_string.size());

		}

		NVCategory* category = NVCategory::create_from_array(string_host_data, num_rows);

		gdf_column * column = new gdf_column;
		int * data;
		EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(gdf_nvstring_category) , 0), RMM_SUCCESS);


		category->get_values( (int *)data, true );
		bit_mask::bit_mask_t * valid;
		bit_mask::create_bit_mask(&valid, num_rows,1);

		gdf_error err = gdf_column_view(column,
				(void *) data,
				(gdf_valid_type *)valid,
				num_rows,
				GDF_STRING_CATEGORY);
		column->dtype_info.category = category;
		return column;
	}
};

//todo refactor tests
TEST_F(NVCategoryTest, TEST_NVCATEGORY_SORTING)
{
	bool print = false;

	gdf_column * column = create_nv_category_column(100,false);
	gdf_column * output_column = create_indices_column(100);
	gdf_column ** input_columns = new gdf_column *[1];
	input_columns[0] = column;

	int8_t *asc_desc;
	EXPECT_EQ(RMM_ALLOC(&asc_desc, 1 , 0), RMM_SUCCESS);
	int8_t minus_one = -1;
	cudaMemset(asc_desc,minus_one,1);

	//doesnt output nvcategory type columns so works as is
	gdf_error err = gdf_order_by(input_columns,asc_desc,1,output_column,false);
	EXPECT_EQ(GDF_SUCCESS, err);

	//    gather_strings( unsigned int* pos, unsigned int elems, bool devmem=true )->;

	if(print){
		print_valid_data(output_column->valid,100);

		/*print_typed_column((int32_t *) output_column->data,
				output_column->valid,
				100);*/
	}
}

TEST_F(NVCategoryTest, TEST_NVCATEGORY_GROUPBY)
{
	bool print = false;

	//left will be Z,Y,X..... C,B,A,ZZ,YY....
	gdf_column * category_column = create_nv_category_column(100,true);
	gdf_column * category_column_2 = create_nv_category_column(100,false);
	gdf_column * category_column_out = create_nv_category_column(100,false);

	gdf_column * constant_value_column_1 = create_column_constant(100,1);
	gdf_column * constant_value_column_5 = create_column_constant(100,5);

	gdf_column * category_column_groups_out = create_nv_category_column(100,true);
	gdf_column * out_col_agg = create_column_constant(100,1);

	gdf_context ctxt;
	ctxt.flag_distinct = false;
	ctxt.flag_method = GDF_HASH;
	ctxt.flag_sort_result = 1;
	gdf_column ** cols = new gdf_column*[1];
	gdf_column ** output_groups = new gdf_column*[1];
	cols[0] = category_column;
	output_groups[0] = category_column_groups_out;

	if(print){
		/*print_typed_column((int32_t *) constant_value_column_1->data,
				constant_value_column_1->valid,
				constant_value_column_1->size);*/
	}

	gdf_error err = gdf_group_by_sum(1,                    // # columns
			cols,            //input cols with 0 null_count otherwise GDF_VALIDITY_UNSUPPORTED is returned
			constant_value_column_1,          //column to aggregate on with 0 null_count otherwise GDF_VALIDITY_UNSUPPORTED is returned
			nullptr,  //if not null return indices of re-ordered rows
			output_groups,  //if not null return the grouped-by columns
			//(multi-gather based on indices, which are needed anyway)
			out_col_agg,      //aggregation result
			&ctxt);

	if(print){
		/*print_typed_column((int32_t *) output_groups[0]->data,
				output_groups[0]->valid,
				output_groups[0]->size);


		print_typed_column((int32_t *) out_col_agg->data,
				out_col_agg->valid,
				out_col_agg->size);*/
	}

	err = gdf_group_by_max(1,                    // # columns
			cols,            //input cols with 0 null_count otherwise GDF_VALIDITY_UNSUPPORTED is returned
			category_column_2,          //column to aggregate on with 0 null_count otherwise GDF_VALIDITY_UNSUPPORTED is returned
			nullptr,  //if not null return indices of re-ordered rows
			output_groups,  //if not null return the grouped-by columns
			//(multi-gather based on indices, which are needed anyway)
			category_column_out,      //aggregation result
			&ctxt);

	EXPECT_EQ(GDF_SUCCESS, err);

	if(print){
		/*print_typed_column((int32_t *) cols[0]->data,
				cols[0]->valid,
				cols[0]->size);


		print_typed_column((int32_t *) category_column_2->data,
				category_column_2->valid,
				category_column_2->size);

		print_typed_column((int32_t *) output_groups[0]->data,
				output_groups[0]->valid,
				output_groups[0]->size);


		print_typed_column((int32_t *) category_column_out->data,
				category_column_out->valid,
				category_column_out->size);*/
	}

	char ** data = new char *[200];
	for(int i = 0; i < 200; i++){
	  data[i] = new char[10];
	}

	static_cast<NVCategory *>(category_column_out->dtype_info.category)->to_strings()->to_host(data, 0, category_column_out->size);

	if(print){
		std::cout<<"maxes\n";
		for(int i = 0; i < category_column_out->size; i++){
			std::cout<<data[i]<<"\t";
		}
		std::cout<<std::endl;
  }

	static_cast<NVCategory *>(output_groups[0]->dtype_info.category)->to_strings()->to_host(data, 0, category_column_out->size);

	if(print){
		std::cout<<"groups\n";
		for(int i = 0; i < category_column_out->size; i++){
			std::cout<<data[i]<<"\t";
		}
		std::cout<<std::endl;
  }

	if(print){
		std::cout<<"lets concat the groups"<<std::endl;
	}

	gdf_column * concat[2];
	concat[0] = output_groups[0];
	concat[1] = output_groups[0];

	gdf_column * concat_out = create_nv_category_column(output_groups[0]->size*2,true);

	if(print){
		std::cout<<"calling concat"<<std::endl;
		std::cout<<"calliing concat category is null = "<<(concat[0]->dtype_info.category == nullptr)<<std::endl;
		std::cout<<"calliing concat category is null = "<<(output_groups[0]->dtype_info.category == nullptr)<<std::endl;
	}

	err = gdf_column_concat(concat_out,concat,2);
	EXPECT_EQ(GDF_SUCCESS, err);

	if(print){
	  std::cout<<"called concat category is null = "<<(category_column_groups_out->dtype_info.category == nullptr)<<std::endl;
  }

	static_cast<NVCategory *>(concat_out->dtype_info.category)->to_strings()->to_host(data, 0, category_column_out->size);

	if(print){
		std::cout<<"groups\n";
		for(int i = 0; i < category_column_out->size; i++){
			std::cout<<data[i]<<"\t";
		}
		std::cout<<std::endl;
  }

	for(int i = 0; i < 200; i++){
	  delete data[i];
	}
	delete data;
}

TEST_F(NVCategoryTest, TEST_NVCATEGORY_COMPARISON)
{
	bool print = false;

	//left will be Z,Y,X..... C,B,A,Z,Y....
	gdf_column * column_left = create_nv_category_column(100,true);
	//right will be Z,Y,X....C,B,A,ZZ,YY ....
	gdf_column * column_right = create_nv_category_column(100,false);
	gdf_column * output_column = create_boolean_column(100);

	NVStrings * temp_string = static_cast<NVCategory *>(column_right->dtype_info.category)->to_strings();
	NVCategory * new_category = static_cast<NVCategory *>(column_left->dtype_info.category)->add_strings(
			*temp_string);

	unsigned int * indices;
	EXPECT_EQ(RMM_ALLOC(&indices, sizeof(unsigned int) * new_category->size()  , 0), RMM_SUCCESS);
	//now reset data
	new_category->get_values((int*)indices,true);

	cudaMemcpy(column_left->data,indices,sizeof(unsigned int) * column_left->size,cudaMemcpyDeviceToDevice);
	cudaMemcpy(column_right->data,indices + column_left->size,sizeof(unsigned int) * column_right->size,cudaMemcpyDeviceToDevice);

	if(print){
		/*print_typed_column((int32_t *) column_left->data,
				column_left->valid,
				100);

		print_typed_column((int32_t *) column_right->data,
				column_right->valid,
				100);*/
	}

	//TODO: damn so this is just a regular silly pointer, i cant just assume i can free it...
	//without some locking mechanism theres no real way to clean this up easily, i could copy....
	column_left->dtype_info.category = new_category;
	column_right->dtype_info.category = new_category;

	if(print){
	//so a few options here, managing a single memory buffer is too annoying
	/*print_typed_column((int8_t *) output_column->data,
			output_column->valid,
			100);*/
	}

	gdf_error err = gdf_comparison(column_left, column_right, output_column,gdf_comparison_operator::GDF_EQUALS);
	EXPECT_EQ(GDF_SUCCESS, err);
	//    gather_strings( unsigned int* pos, unsigned int elems, bool devmem=true )->;

	if(print){
		print_valid_data(output_column->valid,100);

		/*print_typed_column((int8_t *) output_column->data,
				output_column->valid,
				100);*/
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

  std::vector<std::string> left_column;
  std::vector<std::string> right_column;

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
	std::vector<key_type> const & build_column = right_column;

    for(size_t right_index = 0; right_index < build_column.size(); ++right_index){
      the_map.insert(std::make_pair(build_column[right_index], right_index));
    }

    std::vector<result_type> reference_result;
	
    // Probe hash table with first left column
    std::vector<key_type> const & probe_column = left_column;

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

		  if(left_column[left_index] == right_column[right_index]){
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
   * @Param gdf_result A vector of result_type that holds the result of the libgdf join function
   * @Param print Option to print the result computed by the libgdf function
   * @Param sort Option to sort the result. This is required to compare the result against the reference solution
   */
  /* ----------------------------------------------------------------------------*/
  std::vector<result_type> compute_gdf_result(join_op op, bool print = false, bool sort = true, gdf_error expected_result = GDF_SUCCESS)
  {
	EXPECT_EQ(gdf_raw_left_columns.size(), gdf_raw_right_columns.size()) << "Mismatch columns size";

    gdf_column left_result;
    gdf_column right_result;
    left_result.size = 0;
    right_result.size = 0;

	size_t num_columns = gdf_raw_left_columns.size();

    gdf_error result_error{GDF_SUCCESS};

    gdf_column ** left_gdf_columns = gdf_raw_left_columns.data();
    gdf_column ** right_gdf_columns = gdf_raw_right_columns.data();
    std::vector<int> range;
    for (size_t i = 0; i < num_columns; ++i) {range.push_back(i);}
    switch(op)
    {
      case join_op::LEFT:
        {
          result_error = gdf_left_join(
                                       left_gdf_columns, num_columns, range.data(),
                                       right_gdf_columns, num_columns, range.data(),
                                       num_columns,
                                       0, nullptr,
                                       &left_result, &right_result,
                                       &ctxt);
          break;
        }
      case join_op::INNER:
        {
          result_error =  gdf_inner_join(
                                         left_gdf_columns, num_columns, range.data(),
                                         right_gdf_columns, num_columns, range.data(),
                                         num_columns,
                                         0, nullptr,
                                         &left_result, &right_result,
                                         &ctxt);
          break;
        }
      case join_op::FULL:
        {
          result_error =  gdf_full_join(
                                         left_gdf_columns, num_columns, range.data(),
                                         right_gdf_columns, num_columns, range.data(),
                                         num_columns,
                                         0, nullptr,
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
};

TEST_F(NVCategoryJoinTest, join_test){

	bool print = false;
	size_t rows_size = 16;
//	size_t max_int_value = 50;
	join_op op = join_op::INNER;

	size_t length = 1;
	const char ** left_string_data = generate_string_data(rows_size, length, print);
	const char ** right_string_data = generate_string_data(rows_size, length, print);

	left_column = std::vector<std::string> (left_string_data, left_string_data + rows_size);
	right_column = std::vector<std::string> (right_string_data, right_string_data + rows_size);

	gdf_column * left_column = create_nv_category_column_strings(left_string_data, rows_size);
	gdf_column * right_column = create_nv_category_column_strings(right_string_data, rows_size);
	
	if(print){
		std::cout<<"Raw string indexes:\n";
		print_gdf_column(left_column);
		print_gdf_column(right_column);
	}

	gdf_raw_left_columns.push_back(left_column);
  gdf_raw_left_columns.push_back(left_column);
	gdf_raw_right_columns.push_back(right_column);
  gdf_raw_right_columns.push_back(right_column);

	std::vector<result_type> reference_result = this->compute_reference_solution(op, print);

	std::vector<result_type> gdf_result = this->compute_gdf_result(op, print);

	ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

	// Compare the GDF and reference solutions
	for(size_t i = 0; i < reference_result.size(); ++i){
	  EXPECT_EQ(reference_result[i], gdf_result[i]);
	}
}

TEST_F(NVCategoryJoinTest, join_test_nulls){

  bool print = false;
  size_t rows_size = 16;
//  size_t max_int_value = 50;
  join_op op = join_op::INNER;

  size_t length = 1;
  const char ** left_string_data = generate_string_data(rows_size, length, print);
  const char ** right_string_data = generate_string_data(rows_size, length, print);

  left_column = std::vector<std::string> (left_string_data, left_string_data + rows_size);
  right_column = std::vector<std::string> (right_string_data, right_string_data + rows_size);

  gdf_column * left_column = create_nv_category_column_strings(left_string_data, rows_size);
  gdf_column * right_column = create_nv_category_column_strings(right_string_data, rows_size);
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

  std::vector<result_type> gdf_result = this->compute_gdf_result(op, print);

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  // Compare the GDF and reference solutions
  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}
