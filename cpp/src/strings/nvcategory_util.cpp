#include <utility>
#include <utilities/legacy/column_utils.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <cudf/legacy/replace.hpp>
#include <cudf/types.hpp>
#include <cudf/legacy/table.hpp>
#include <nvstrings/NVCategory.h>
#include <nvstrings/NVStrings.h>
#include <rmm/rmm.h>
#include <cudf/utilities/error.hpp>
#include <utilities/legacy/error_utils.hpp>
#include <cudf/legacy/binaryop.hpp>
#include <cudf/legacy/copying.hpp>

namespace {
  NVCategory * combine_column_categories(const gdf_column* const input_columns[], int num_columns){
    std::vector<NVCategory *> cats;
    std::transform(input_columns, input_columns + num_columns, std::back_inserter(cats), 
      [&](const gdf_column* c) {
        return const_cast<NVCategory *>(static_cast<const NVCategory *>(c->dtype_info.category));
      });

    return NVCategory::create_from_categories(cats);
  }


  gdf_error free_nvcategory(gdf_column * column){
    NVCategory::destroy(static_cast<NVCategory *>(column->dtype_info.category));
    column->dtype_info.category = nullptr;
    return GDF_SUCCESS;
  }
}

gdf_error nvcategory_gather_table(cudf::table source_table, cudf::table destination_table){
  GDF_REQUIRE(source_table.num_columns() == destination_table.num_columns(), GDF_TABLES_SIZE_MISMATCH);
  for(int i = 0; i < source_table.num_columns();i++){
    gdf_column * original_column = source_table.get_column(i);
    if(original_column->dtype == GDF_STRING_CATEGORY){
      gdf_column * output_column = destination_table.get_column(i);
      GDF_REQUIRE(output_column->dtype == original_column->dtype, GDF_DTYPE_MISMATCH);
      nvcategory_gather(output_column,static_cast<NVCategory *>(original_column->dtype_info.category));
    }

  }
  return GDF_SUCCESS;
}

gdf_error nvcategory_gather(gdf_column* column, NVCategory* nv_category){


  GDF_REQUIRE(nv_category != nullptr,GDF_INVALID_API_CALL );
  GDF_REQUIRE(column->dtype == GDF_STRING_CATEGORY,GDF_UNSUPPORTED_DTYPE);

  if(column->size == 0){
    column->dtype_info.category = nullptr;
    return GDF_SUCCESS;
  }

  //column may have null values here, we will do the following
  //check if the category we are gathering from has null if it does not
  bool destroy_category = false;
  if(column->null_count > 0){

    nv_category_index_type null_index = nv_category->get_value(nullptr);
    if(null_index == -1){
      const char* empty = 0;
      NVStrings* strs = NVStrings::create_from_array(&empty,1);
      nv_category = nv_category->add_keys_and_remap(*strs);

      gdf_scalar rhs;

      rhs.data.si32 = 1;
      rhs.is_valid = true;
      rhs.dtype = GDF_STRING_CATEGORY;
      cudf::binary_operation(column, column,  &rhs, GDF_ADD);
      destroy_category = true;
      null_index = nv_category->get_value(nullptr);

      NVStrings::destroy(strs);

    }
    GDF_REQUIRE(null_index == 0, GDF_INVALID_API_CALL);

    gdf_scalar null_index_scalar;

    null_index_scalar.data.si32 = null_index;
    null_index_scalar.is_valid = true;
    null_index_scalar.dtype = GDF_STRING_CATEGORY;

    const auto byte_width = cudf::size_of(column->dtype);
    gdf_column column_nulls_replaced = cudf::replace_nulls(*column, null_index_scalar);
    CUDA_TRY(cudaMemcpyAsync(
                             column->data,
                             column_nulls_replaced.data,
                             column->size * byte_width,
                             cudaMemcpyDefault,
                             0));

    gdf_column_free(&column_nulls_replaced);
  }

  CUDF_EXPECTS(column->data != nullptr, "Trying to gather nullptr data in nvcategory_gather");
  CUDF_EXPECTS(nv_category != nullptr, "Trying to gather nullptr data in nvcategory_gather");
  NVCategory * new_category = nv_category->gather(static_cast<nv_category_index_type *>(column->data),
                                                          column->size,
                                                          DEVICE_ALLOCATED);
  if(destroy_category){
    NVCategory::destroy(nv_category);
  }
  CHECK_CUDA(0);
  new_category->get_values(static_cast<nv_category_index_type *>(column->data),
                           DEVICE_ALLOCATED);
  CHECK_CUDA(0);
  //Python handles freeing the original column->dtype_info.category so we don't need to
  column->dtype_info.category = new_category;


  return GDF_SUCCESS;
}

gdf_error validate_categories(const gdf_column* const input_columns[], int num_columns, cudf::size_type & total_count){
  total_count = 0;
  for (int i = 0; i < num_columns; ++i) {
    const gdf_column* current_column = input_columns[i];
    GDF_REQUIRE(current_column != nullptr,GDF_DATASET_EMPTY);
    GDF_REQUIRE((current_column->data != nullptr || current_column->size == 0),GDF_DATASET_EMPTY);
    GDF_REQUIRE(current_column->dtype == GDF_STRING_CATEGORY,GDF_UNSUPPORTED_DTYPE);

    total_count += input_columns[i]->size;
  }
  return GDF_SUCCESS;
}


gdf_error concat_categories(const gdf_column* const input_columns[], gdf_column* output_column, int num_columns){

  cudf::size_type total_count;
  gdf_error err = validate_categories(input_columns,num_columns,total_count);
  GDF_REQUIRE(err == GDF_SUCCESS,err);
  GDF_REQUIRE(total_count <= output_column->size,GDF_COLUMN_SIZE_MISMATCH);
  GDF_REQUIRE(output_column->dtype == GDF_STRING_CATEGORY,GDF_UNSUPPORTED_DTYPE);
  //TODO: we have no way to jsut copy a category this will fail if someone calls concat
  //on a single input
  GDF_REQUIRE(num_columns >= 1,GDF_DATASET_EMPTY);

  NVCategory * combined_category = combine_column_categories(input_columns,num_columns);
  combined_category->get_values(
        static_cast<nv_category_index_type *>(output_column->data),
        true);
  output_column->dtype_info.category = (void *) combined_category;



  return GDF_SUCCESS;
}


gdf_error sync_column_categories(const gdf_column* const input_columns[], gdf_column* output_columns[], int num_columns){

  GDF_REQUIRE(num_columns > 0,GDF_DATASET_EMPTY);
  cudf::size_type total_count;

  gdf_error err = validate_categories(input_columns,num_columns,total_count);
  GDF_REQUIRE(GDF_SUCCESS == err, err);

  err = validate_categories(output_columns,num_columns,total_count);
  GDF_REQUIRE(GDF_SUCCESS == err, err);

  for(int column_index = 0; column_index < num_columns; column_index++){
    GDF_REQUIRE(input_columns[column_index]->size == output_columns[column_index]->size,GDF_COLUMN_SIZE_MISMATCH);
  }

  NVCategory * combined_category = combine_column_categories(input_columns,num_columns);

  cudf::size_type current_column_start_position = 0;
  for(int column_index = 0; column_index < num_columns; column_index++){
    cudf::size_type column_size = output_columns[column_index]->size;
    cudf::size_type size_to_copy =  column_size * sizeof(nv_category_index_type);
    CUDA_TRY( cudaMemcpy(output_columns[column_index]->data,
                          combined_category->values_cptr() + current_column_start_position,
                         size_to_copy,
                          cudaMemcpyDeviceToDevice));

    //TODO: becuase of how gather works we are making a copy to preserve dictionaries as the same
    //this has an overhead of having to store more than is necessary. remove when gather preserving dictionary is available for nvcategory
    output_columns[column_index]->dtype_info.category = combined_category->copy();


    current_column_start_position += column_size;
  }

  NVCategory::destroy(combined_category);

  return GDF_SUCCESS;
}
