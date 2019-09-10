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

#include <bitmask/valid_if.cuh>
#include <cudf/utilities/legacy/type_dispatcher.hpp>

namespace cudf {

namespace detail {

template <typename T>
struct predicate_not_nan{

  CUDA_HOST_DEVICE_CALLABLE
  bool operator()(gdf_index_type index) const {
      return !isnan(static_cast<T*>(input.data)[index]);
  }

  gdf_column input;

  predicate_not_nan() = delete;

  predicate_not_nan(gdf_column const& input_): input(input_) {}

};

template <typename T>
struct predicate_not_value{

  CUDA_HOST_DEVICE_CALLABLE

  bool operator()(gdf_index_type index) const {
      return !(static_cast<T*>(input.data)[index] == value);
  }

  gdf_column input;

  T value;

  predicate_not_value() = delete;

  predicate_not_value(gdf_column const& input_, T const&  value_): input(input_), value(value_) {}

};

struct replace_value_with_null{
  template <typename col_type>
  std::pair<bit_mask_t*, gdf_size_type> operator()(gdf_column const& input, 
                                                   gdf_scalar const& value)
  {
    const bit_mask_t* source_mask = reinterpret_cast<bit_mask_t*>(input.valid);

    switch (value.dtype) {
      case GDF_FLOAT32:
      {
        float val{};
        memcpy(&val, &value.data, sizeof(float));
        if (isnan(val)){
          return cudf::valid_if(source_mask, cudf::detail::predicate_not_nan<float>(input), input.size);
        }
      }

      case GDF_FLOAT64:
      {
        double val{};
        memcpy(&val, &value.data, sizeof(double));
        if (isnan(val)){
          return cudf::valid_if(source_mask, cudf::detail::predicate_not_nan<double>(input), input.size);
        }
      }

      default:
      {
        auto val = reinterpret_cast<const col_type*>(&value.data);
        return cudf::valid_if(source_mask, cudf::detail::predicate_not_value<col_type>(input, *val), input.size);
      }
    }
  }
};

} // namespace detail

std::pair<bit_mask_t*, gdf_size_type> values_to_nulls(gdf_column const& input, gdf_scalar const& value){

  CUDF_EXPECTS(input.dtype == value.dtype, "DTYPE mismatch");
    
  if(input.size == 0){
    return std::pair<bit_mask_t*, gdf_size_type>(nullptr, 0);
  }

  return cudf::type_dispatcher(input.dtype, cudf::detail::replace_value_with_null{}, input, value);
}
//
} // namespace cudf
