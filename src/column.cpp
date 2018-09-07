#include <gdf/gdf.h>
#include <gdf/errorutils.h>
#include <cuda_runtime_api.h>

gdf_size_type gdf_column_sizeof() {
	return sizeof(gdf_column);
}

gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid,
		gdf_size_type size, gdf_dtype dtype) {
	column->data = data;
	column->valid = valid;
	column->size = size;
	column->dtype = dtype;
	column->null_count = 0;
	return GDF_SUCCESS;
}


gdf_error gdf_column_view_augmented(gdf_column *column, void *data, gdf_valid_type *valid,
		gdf_size_type size, gdf_dtype dtype, gdf_size_type null_count) {
	column->data = data;
	column->valid = valid;
	column->size = size;
	column->dtype = dtype;
	column->null_count = null_count;
	return GDF_SUCCESS;
}

gdf_error gdf_column_free(gdf_column *column) {

  if(nullptr != column->data)
  {
    CUDA_TRY( cudaFree(column->data)  );
  }
  if(nullptr != column->valid)
  {
    CUDA_TRY( cudaFree(column->valid) );
  }
  return GDF_SUCCESS;
}


gdf_error get_column_byte_width(gdf_column * col, int * width){
	
	switch(col->dtype) {

	case GDF_INT8 :
		*width = 1;
		break;
	case GDF_INT16 :
		*width = 2;
		break;
	case GDF_INT32 :
		*width = 4;
		break;
	case GDF_INT64 :
		*width = 8;
		break;
	case GDF_FLOAT32 :
		*width = 4;
		break;
	case GDF_FLOAT64 :
		*width = 8;
		break;
	case GDF_DATE32 :
		*width = 4;
		break;
	case GDF_DATE64 :
		*width = 8;
		break;
	default :
		*width = -1;
		return GDF_UNSUPPORTED_DTYPE;
	}

	return GDF_SUCCESS;
}
