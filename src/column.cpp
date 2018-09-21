#include <gdf/gdf.h>
#include <gdf/errorutils.h>
#include <cuda_runtime_api.h>

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Concatenates the gdf_columns into a single, contiguous column,
 * including the validity bitmasks
 * 
 * @Param[in] columns_to_concat[] The columns to concatenate
 * @Param[in] num_columns The number of columns to concatenate
 * @Param[out] output_column A column whose buffers are already allocated that will 
 * contain the concatenation of the input columns
 * 
 * @Returns GDF_SUCCESS upon successful completion
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_column_concat(gdf_column * columns_to_concat[], int num_columns, gdf_column * output_column)
{

  if(nullptr == columns_to_concat){
    return GDF_DATASET_EMPTY;
  }

  if((nullptr == columns_to_concat[0])
      || (nullptr == output_column)){
    return GDF_DATASET_EMPTY;
  }

  const gdf_dtype column_type = columns_to_concat[0]->dtype;

  if(column_type != output_column->dtype){
    return GDF_DTYPE_MISMATCH;
  }

  // Ensure all the columns are properly allocated
  // and have matching types
  for(int i = 0; i < num_columns; ++i){
    gdf_column * current_column = columns_to_concat[i];
    if(nullptr == current_column){
      return GDF_DATASET_EMPTY;
    }
    if(current_column->size > 0){
      if((nullptr == current_column->data)
          || (nullptr == current_column->valid))
      {
        return GDF_DATASET_EMPTY;
      }
    }
    if(column_type != current_column->dtype){
      return GDF_DTYPE_MISMATCH;
    }
  }


  // Will the size of the output column already be set? If so, we should probably
  // make sure that the sum of the sizes of the input columns matches the size
  // of the output column



  // NOTE: You need to take into account the fact that the validity buffers 
  // for each column need to be concated into a single, contiguous validity 
  // buffer, but you cannot just concat them as is. This is because the number
  // of rows in the column may be less than the number of bits in the column's 
  // validity bitmask. Therefore, you must copy only the bits [0, col->size)
  // from each column's validity mask. E.g., the concatted bitmask will look like:
  // { col0->valid_bits[0, col0->size), col1->valid_bits[0, col1->size) ... }

  return GDF_SUCCESS;
}

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
	case GDF_TIMESTAMP :
		*width = 8;
		break;
	default :
		*width = -1;
		return GDF_UNSUPPORTED_DTYPE;
	}

	return GDF_SUCCESS;
}
