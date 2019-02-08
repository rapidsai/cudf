
#include "nvcategory_util.cuh"


#include <NVCategory.h>
#include <NVStrings.h>
#include "rmm/rmm.h"

gdf_error create_nvcategory_from_indices(gdf_column * column, NVCategory * nv_category){

	if(nv_category == nullptr ||
			column == nullptr){
		return GDF_INVALID_API_CALL;
	}

	if(column->dtype != GDF_STRING_CATEGORY){
		return GDF_UNSUPPORTED_DTYPE;
	}

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


gdf_error concat_categories(gdf_column * input_columns[],gdf_column * output_column, int num_columns){
	gdf_dtype column_type = input_columns[0]->dtype;

	for (int i = 0; i < num_columns; ++i) {
		gdf_column* current_column = input_columns[i];
		if (nullptr == current_column) {
			return GDF_DATASET_EMPTY;
		}
		if ((current_column->size > 0) && (nullptr == current_column->data))
		{
			return GDF_DATASET_EMPTY;
		}
		if (column_type != current_column->dtype) {
			return GDF_DTYPE_MISMATCH;
		}
	}

	//TODO: we have no way to jsut copy a category this will fail if someone calls concat
	//on a single input
	if(num_columns < 2){
		return GDF_DATASET_EMPTY;
	}
	NVCategory * new_category = input_columns[0]->dtype_info.category;
	NVCategory * temp_category;

	for (int i = 1; i < num_columns; ++i) {
		NVStrings * temp_strings = input_columns[0]->dtype_info.category->to_strings();
		temp_category = new_category->add_strings(*temp_strings);
		NVCategory::destroy(new_category);
		NVStrings::destroy(temp_strings);
		new_category = temp_category;
	}

	new_category->get_values(
			static_cast<nv_category_index_type *>(output_column->data),
			true);
	output_column->dtype_info.category = new_category;



	return GDF_SUCCESS;
}
