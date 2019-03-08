
#include "nvcategory_util.cuh"


#include <nvstrings/NVCategory.h>
#include <nvstrings/NVStrings.h>
#include "rmm/rmm.h"

#include "utilities/error_utils.hpp"
#include "utilities/nvtx/nvtx_utils.h"


gdf_error nvcategory_gather(gdf_column * column, NVCategory * nv_category){


	GDF_REQUIRE(nv_category != nullptr,GDF_INVALID_API_CALL );
	GDF_REQUIRE(column->dtype == GDF_STRING_CATEGORY,GDF_UNSUPPORTED_DTYPE);

	NVStrings * strings = nv_category->gather_strings(static_cast<nv_category_index_type *>(column->data),
			column->size,
			DEVICE_ALLOCATED);

	NVCategory * new_category = NVCategory::create_from_strings(*strings);
	new_category->get_values(static_cast<nv_category_index_type *>(column->data),
			DEVICE_ALLOCATED);

	//This is questionable behavior and should be reviewed by peers
	//Smart pointers would be lovely here
	if(column->dtype_info.category != nullptr){
		NVCategory::destroy(column->dtype_info.category);
	}
	column->dtype_info.category = new_category;

	NVStrings::destroy(strings);
	return GDF_SUCCESS;
}

gdf_error validate_categories(gdf_column * input_columns[], int num_columns, gdf_size_type & total_count){
	total_count = 0;
	for (int i = 0; i < num_columns; ++i) {
		gdf_column* current_column = input_columns[i];
		GDF_REQUIRE(current_column != nullptr,GDF_DATASET_EMPTY);
		GDF_REQUIRE(current_column->data != nullptr,GDF_DATASET_EMPTY);
		GDF_REQUIRE(current_column->dtype == GDF_STRING_CATEGORY,GDF_UNSUPPORTED_DTYPE);

		total_count += input_columns[i]->size;
	}
	return GDF_SUCCESS;
}

#include <iostream>
gdf_error concat_categories(gdf_column * input_columns[],gdf_column * output_column, int num_columns){

	gdf_size_type total_count;
	gdf_error err = validate_categories(input_columns,num_columns,total_count);
	GDF_REQUIRE(err == GDF_SUCCESS,err);
	GDF_REQUIRE(total_count <= output_column->size,GDF_COLUMN_SIZE_MISMATCH);
	GDF_REQUIRE(output_column->dtype == GDF_STRING_CATEGORY,GDF_UNSUPPORTED_DTYPE);
	//TODO: we have no way to jsut copy a category this will fail if someone calls concat
	//on a single input
	GDF_REQUIRE(num_columns >= 2,GDF_DATASET_EMPTY);
	NVCategory * new_category = input_columns[0]->dtype_info.category;
	NVCategory * temp_category;

	for (int i = 1; i < num_columns; i++) {
		NVStrings * temp_strings = input_columns[i]->dtype_info.category->to_strings();
		temp_category = new_category->add_strings(*temp_strings); //this is the only way to add to a category and keep the dictionary sorted
		if(i > 1){
			//only destroy categoryy after first iteration
			NVCategory::destroy(new_category);

		}

		NVStrings::destroy(temp_strings);

		new_category = temp_category;
	}

	new_category->get_values(
			static_cast<nv_category_index_type *>(output_column->data),
			true);
	output_column->dtype_info.category = new_category;



	return GDF_SUCCESS;
}

gdf_error sync_column_categories(gdf_column * input_columns[],gdf_column * output_columns[], int num_columns){

	GDF_REQUIRE(num_columns > 0,GDF_DATASET_EMPTY);
	gdf_size_type total_count;

	gdf_error err = validate_categories(input_columns,num_columns,total_count);
	GDF_REQUIRE(GDF_SUCCESS == err, err);

	err = validate_categories(output_columns,num_columns,total_count);
	GDF_REQUIRE(GDF_SUCCESS == err, err);

	for(int column_index = 0; column_index < num_columns; column_index++){
		GDF_REQUIRE(input_columns[column_index]->size == output_columns[column_index]->size,GDF_COLUMN_SIZE_MISMATCH);
	}

	NVStrings** temp_strs = new NVStrings*[num_columns];

	for(int column_index = 0; column_index < num_columns; column_index++){
		temp_strs[column_index] = input_columns[column_index]->dtype_info.category->to_strings();
	}

	NVCategory** new_categories = new NVCategory*[num_columns];

	for(int column_index_x = 0; column_index_x < num_columns; column_index_x++)
		for(int column_index_y = 0; column_index_y < num_columns; column_index_y++){
			if(column_index_x != column_index_y){
				new_categories[column_index_x] = output_columns[column_index_x]->dtype_info.category->add_strings(*temp_strs[column_index_y]);
			}
		}

	std::vector<cudaError_t> cuda_err(num_columns);
	for(int column_index = 0; column_index < num_columns; column_index++){
		if(output_columns[column_index]->dtype_info.category != nullptr){
			NVCategory::destroy(output_columns[column_index]->dtype_info.category);
		}

		output_columns[column_index]->dtype_info.category = new_categories[column_index];

		size_t size_to_copy = sizeof(nv_category_index_type) * output_columns[column_index]->size;
		cuda_err[column_index] = cudaMemcpy(output_columns[column_index]->data,
			output_columns[column_index]->dtype_info.category->values_cptr(),
			size_to_copy,
			cudaMemcpyDeviceToDevice);
	}

	for(int column_index = 0; column_index < num_columns; column_index++){
		GDF_REQUIRE(cuda_err[column_index] == cudaSuccess, GDF_CUDA_ERROR);
	}

	return GDF_SUCCESS;
}

gdf_error free_nvcategory(gdf_column * column){
	NVCategory::destroy(column->dtype_info.category);
	column->dtype_info.category = nullptr;
	return GDF_SUCCESS;
}
