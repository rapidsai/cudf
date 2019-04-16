#include "json_reader.hpp"

#include <iostream>
#include <vector>

#include <cudf.h>

#include "io/utilities/parsing_utils.cuh"
#include "io/utilities/wrapper_utils.hpp"


/*
 * Convert dtype strings into gdf_dtype enum
 */
gdf_dtype convertStringToDtype(const std::string &dtype) {
  if (dtype.compare( "str") == 0) return GDF_STRING;
  if (dtype.compare( "date") == 0) return GDF_DATE64;
  if (dtype.compare( "date32") == 0) return GDF_DATE32;
  if (dtype.compare( "date64") == 0) return GDF_DATE64;
  if (dtype.compare( "timestamp") == 0) return GDF_TIMESTAMP;
  if (dtype.compare( "category") == 0) return GDF_CATEGORY;
  if (dtype.compare( "float") == 0) return GDF_FLOAT32;
  if (dtype.compare( "float32") == 0) return GDF_FLOAT32;
  if (dtype.compare( "float64") == 0) return GDF_FLOAT64;
  if (dtype.compare( "double") == 0) return GDF_FLOAT64;
  if (dtype.compare( "short") == 0) return GDF_INT16;
  if (dtype.compare( "int") == 0) return GDF_INT32;
  if (dtype.compare( "int32") == 0) return GDF_INT32;
  if (dtype.compare( "int64") == 0) return GDF_INT64;
  if (dtype.compare( "long") == 0) return GDF_INT64;
  return GDF_invalid;
}

void JsonReader::parse(){
  for (int col = 0; col < args_->num_cols; ++col) {
    columns_.emplace_back(10, convertStringToDtype(args_->dtype[col]), gdf_dtype_extra_info{TIME_UNIT_NONE}, "c");
  }
}

void JsonReader::storeColumns(json_read_arg *out_args){

  // Transfer ownership to raw pointer output arguments
  out_args->data = (gdf_column **)malloc(sizeof(gdf_column *) * args_->num_cols);
  for (int i = 0; i < args_->num_cols; ++i) {
    out_args->data[i] = columns_[i].release();
  }
  out_args->num_cols_out = args_->num_cols;
  out_args->num_rows_out = 100;
}

gdf_error read_json(json_read_arg *args) {
  JsonReader reader(args);

  reader.parse();

  reader.storeColumns(args);

  return GDF_SUCCESS;
}