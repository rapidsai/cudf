
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



	gdf_column * create_indices_column(gdf_size_type num_rows){
		gdf_column * column = new gdf_column;
		int * data;
	    bit_mask::bit_mask_t * valid;
	    bit_mask::create_bit_mask(&valid, 100,1);
		EXPECT_EQ(RMM_ALLOC(&data, num_rows * sizeof(int) , 0), RMM_SUCCESS);
		gdf_error err = gdf_column_view(column,
		                          (void *) data,
		                          (gdf_valid_type *) valid,
				                  num_rows,
		                          GDF_INT32);
		return column;

	}

	gdf_column * create_nv_category_column(gdf_size_type num_rows){

		const char ** string_host_data = new const char *[num_rows];


		for(gdf_size_type row_index = 0; row_index < num_rows; row_index++){
			string_host_data[row_index] = new char[200]; //allows up to 26 * 200 values
			std::string temp_string = "";
			int num_chars = (row_index / 26) + 1;
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


		category->get_values( (unsigned int *)data, true );
	    bit_mask::bit_mask_t * valid;
	    bit_mask::create_bit_mask(&valid, 100,1);

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
	gdf_column * column = create_nv_category_column(100);
	gdf_column * output_column = create_indices_column(100);
	gdf_column ** input_columns = new gdf_column *[1];
	input_columns[0] = column;

	int8_t *asc_desc;
    EXPECT_EQ(RMM_ALLOC(&asc_desc, 1 , 0), RMM_SUCCESS);
    int8_t minus_one = -1;
	cudaMemset(asc_desc,minus_one,1);
	std::cout<<column->size<<std::endl;
    gdf_error err = gdf_order_by(input_columns,asc_desc,1,output_column,false);
    print_valid_data(output_column->valid,100);
    bit_mask::bit_mask_t * valid;
    bit_mask::create_bit_mask(&valid, 100,1);
    output_column->valid = (gdf_valid_type *) valid;
	print_typed_column((int32_t *) output_column->data,
	                        nullptr,
	                        100);

}
