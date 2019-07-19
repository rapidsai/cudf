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

#include <utilities/column_utils.cuh>

namespace cudf {

template <typename T>
struct predicate_is_nan{
  
  CUDA_HOST_DEVICE_CALLABLE
  bool operator()(gdf_index_type index) const {
      return isnan(static_cast<T*>(input.data)[index]);
  }
  
  gdf_column input;

  predicate_is_nan() = delete;
  
  predicate_is_nan(const gdf_column input_): input(input_) {}

};


} // namespace cudf

bit_mask_t* nans_to_nulls(gdf_column const* col){
  
  const bit_mask_t* source_mask = reinterpret_cast<bit_mask_t*>(col->valid);
  
  switch(col->dtype){
    case GDF_FLOAT32:
      return cudf::null_if(source_mask, cudf::predicate_is_nan<float>(*col), col->size);
    case GDF_FLOAT64:
      return cudf::null_if(source_mask, cudf::predicate_is_nan<double>(*col), col->size);
    default:
      CUDF_EXPECTS(false, "Unsupported data type for is_nan()");
      return nullptr;
  }

} // namespace cudf
