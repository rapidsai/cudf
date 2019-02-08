
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

	NVStrings * strings = nv_category->gather_strings(static_cast<nv_category_type *>(column->data),
			column->size,
			DEVICE_ALLOCATED);

	NVCategory * new_category = NVCategory::create_from_strings(*strings);
	new_category->get_values(static_cast<nv_category_type *>(column->data),
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
