/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/column/column_view.hpp>
#include <cudf/legacy/interop.hpp>
#include <utilities/error_utils.hpp>

namespace cudf {
namespace legacy {

data_type gdf_dtype_to_data_type(gdf_dtype dtype) {
    switch(dtype){
        case GDF_INT8:     return data_type{INT8};
        case GDF_INT16:    return data_type{INT16};
        case GDF_INT32:    return data_type{INT32};
        case GDF_INT64:    return data_type{INT64};
        case GDF_FLOAT32:  return data_type{FLOAT32};
        case GDF_FLOAT64:  return data_type{FLOAT64};
        case GDF_BOOL8:    return data_type{BOOL8};
        case GDF_DATE32:   return data_type{DATE32};
        case GDF_CATEGORY: return data_type{CATEGORY};
        default:
          CUDF_FAIL("Unsuported gdf_dtype for converion to data_type.");
    }
}

gdf_dtype data_type_to_gdf_dtype( data_type type) {
  switch (type.id()) {
    case INT8:     return GDF_INT8;
    case INT16:    return GDF_INT16;
    case INT32:    return GDF_INT32;
    case INT64:    return GDF_INT64;
    case FLOAT32:  return GDF_FLOAT32;
    case FLOAT64:  return GDF_FLOAT64;
    case BOOL8:    return GDF_BOOL8;
    case DATE32:   return GDF_DATE32;
    case CATEGORY: return GDF_CATEGORY;
    default:
      CUDF_FAIL("Unsupported `data_type` for conversion to `gdf_dtype.");
  }
}

column_view gdf_column_to_view(gdf_column const& col) {
  return column_view{gdf_dtype_to_data_type(col.dtype), col.size, col.data,
                     reinterpret_cast<bitmask_type const*>(col.valid),
                     col.null_count};
}

mutable_column_view gdf_column_to_mutable_view(gdf_column* col) {
  CUDF_EXPECTS(nullptr != col, "Invalid column.");
  return mutable_column_view(
      gdf_dtype_to_data_type(col->dtype), col->size, col->data,
      reinterpret_cast<bitmask_type*>(col->valid), col->null_count);
}

gdf_column view_to_gdf_column(mutable_column_view view) {
  CUDF_EXPECTS(view.offset() == 0, "Unsupported view offset.");
  return gdf_column{
      view.head(), reinterpret_cast<gdf_valid_type*>(view.null_mask()),
      view.size(), data_type_to_gdf_dtype(view.type()), view.null_count()};
}
}  // namespace legacy
}  // namespace cudf