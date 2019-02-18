
#include "gtest/gtest.h"

#include <cudf.h>
#include <utilities/cudf_utils.h>

#include <cudf/functions.h>
#include <cudf/types.h>
#include <iostream>

#include <NVCategory.h>
#include <NVStrings.h>
#include "rmm/rmm.h"
#include <cstring>
#include "tests/utilities/cudf_test_utils.cuh"
#include "tests/utilities/cudf_test_fixtures.h"
#include "bitmask/bit_mask.h"
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

TEST_F(NVCategoryTest, TEST_NVCATEGORY_SORTING)
{
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



	//    gather_strings( unsigned int* pos, unsigned int elems, bool devmem=true )->;

	print_valid_data(output_column->valid,100);

	print_typed_column((int32_t *) output_column->data,
			output_column->valid,
			100);

}

TEST_F(NVCategoryTest, TEST_NVCATEGORY_GROUPBY)
{
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

	print_typed_column((int32_t *) constant_value_column_1->data,
			constant_value_column_1->valid,
			constant_value_column_1->size);

	gdf_error err = gdf_group_by_sum(1,                    // # columns
			cols,            //input cols with 0 null_count otherwise GDF_VALIDITY_UNSUPPORTED is returned
			constant_value_column_1,          //column to aggregate on with 0 null_count otherwise GDF_VALIDITY_UNSUPPORTED is returned
			nullptr,  //if not null return indices of re-ordered rows
			output_groups,  //if not null return the grouped-by columns
			//(multi-gather based on indices, which are needed anyway)
			out_col_agg,      //aggregation result
			&ctxt);

	print_typed_column((int32_t *) output_groups[0]->data,
			output_groups[0]->valid,
			output_groups[0]->size);


	print_typed_column((int32_t *) out_col_agg->data,
			out_col_agg->valid,
			out_col_agg->size);




	err = gdf_group_by_max(1,                    // # columns
			cols,            //input cols with 0 null_count otherwise GDF_VALIDITY_UNSUPPORTED is returned
			category_column_2,          //column to aggregate on with 0 null_count otherwise GDF_VALIDITY_UNSUPPORTED is returned
			nullptr,  //if not null return indices of re-ordered rows
			output_groups,  //if not null return the grouped-by columns
			//(multi-gather based on indices, which are needed anyway)
			category_column_out,      //aggregation result
			&ctxt);

	if(err != GDF_SUCCESS){
		std::cout<<"WE ARE NOT JAMMIN!"<<std::endl;
	}else{
		std::cout<<"WE ARE JAMMIN!"<<std::endl;
	}

	print_typed_column((int32_t *) cols[0]->data,
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
			category_column_out->size);

	char ** data = new char *[200];

	category_column_out->dtype_info.category->to_strings()->to_host(data, 0, category_column_out->size);
	std::cout<<"maxes\n";
	for(int i = 0; i < category_column_out->size; i++){
		std::cout<<data[i]<<"\t";
	}
	std::cout<<std::endl;

	output_groups[0]->dtype_info.category->to_strings()->to_host(data, 0, category_column_out->size);
	std::cout<<"groups\n";
	for(int i = 0; i < category_column_out->size; i++){
		std::cout<<data[i]<<"\t";
	}
	std::cout<<std::endl;


	std::cout<<"lets concat the groups"<<std::endl;

	gdf_column * concat[2];
	concat[0] = output_groups[0];
	concat[1] = output_groups[0];

	gdf_column * concat_out = create_nv_category_column(200,true);


	std::cout<<"calling concat"<<std::endl;

	std::cout<<"calliing concat category is null = "<<(concat[0]->dtype_info.category == nullptr)<<std::endl;
	std::cout<<"calliing concat category is null = "<<(output_groups[0]->dtype_info.category == nullptr)<<std::endl;


	err = gdf_column_concat(concat_out,concat,2);

	std::cout<<"called concat category is null = "<<(category_column_groups_out->dtype_info.category == nullptr)<<std::endl;

	if(err != GDF_SUCCESS){
		std::cout<<"WE ARE NOT JAMMIN! "<<err<<std::endl;
	}else{
		std::cout<<"CONCAT"<<std::endl;
	}

	concat_out->dtype_info.category->to_strings()->to_host(data, 0, category_column_out->size);
	std::cout<<"groups\n";
	for(int i = 0; i < category_column_out->size; i++){
		std::cout<<data[i]<<"\t";
	}
	std::cout<<std::endl;


}

TEST_F(NVCategoryTest, TEST_NVCATEGORY_COMPARISON)
{
	//left will be Z,Y,X..... C,B,A,Z,Y....
	gdf_column * column_left = create_nv_category_column(100,true);
	//right will be Z,Y,X....C,B,A,ZZ,YY ....
	gdf_column * column_right = create_nv_category_column(100,false);
	gdf_column * output_column = create_boolean_column(100);

	NVStrings * temp_string = column_right->dtype_info.category->to_strings();
	NVCategory * new_category = column_left->dtype_info.category->add_strings(
			*temp_string);

	unsigned int * indices;
	EXPECT_EQ(RMM_ALLOC(&indices, sizeof(unsigned int) * new_category->size()  , 0), RMM_SUCCESS);
	//now reset data
	new_category->get_values((int*)indices,true);

	cudaMemcpy(column_left->data,indices,sizeof(unsigned int) * column_left->size,cudaMemcpyDeviceToDevice);
	cudaMemcpy(column_right->data,indices + column_left->size,sizeof(unsigned int) * column_right->size,cudaMemcpyDeviceToDevice);

	print_typed_column((int32_t *) column_left->data,
			column_left->valid,
			100);

	print_typed_column((int32_t *) column_right->data,
			column_right->valid,
			100);


	//TODO: damn so this is just a regular silly pointer, i cant just assume i can free it...
	//without some locking mechanism theres no real way to clean this up easily, i could copy....
	column_left->dtype_info.category = new_category;
	column_right->dtype_info.category = new_category;

	//so a few options here, managing a single memory buffer is too annoying
	print_typed_column((int8_t *) output_column->data,
			output_column->valid,
			100);

	gdf_error err = gpu_comparison(column_left, column_right, output_column,gdf_comparison_operator::GDF_EQUALS);

	std::cout<<err<<std::endl;
	//    gather_strings( unsigned int* pos, unsigned int elems, bool devmem=true )->;

	print_valid_data(output_column->valid,100);

	print_typed_column((int8_t *) output_column->data,
			output_column->valid,
			100);

}
