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
struct predicate_not_nat{

  CUDA_HOST_DEVICE_CALLABLE

  bool operator()(gdf_index_type index) const {
      return !(static_cast<T*>(input.data)[index] == value);
  }

  gdf_column input;

  T value;

  predicate_not_nat() = delete;

  predicate_not_nat(gdf_column const& input_, T const&  value_): input(input_), value(value_) {}

};

struct set_mask{
  template <typename col_type>
  std::pair<bit_mask_t*, gdf_size_type> operator()(gdf_column const& input, 
                                                   gdf_scalar const& value)
  {
      const bit_mask_t* source_mask = reinterpret_cast<bit_mask_t*>(input.valid);
      auto *val = reinterpret_cast<const col_type*>(&value.data);

      return cudf::valid_if(source_mask, cudf::detail::predicate_not_nat<col_type>(input, *val), input.size); 
  }
};

} // namespace detail

std::pair<bit_mask_t*, gdf_size_type> nats_to_nulls(gdf_column const& input, gdf_scalar const& value){

  std::cout<<"RGSL : In the start : "<<value.dtype<<std::endl;
  std::cout<<"RGSL : Input : "<<input.dtype<<std::endl;
  CUDF_EXPECTS(input.dtype == value.dtype, "DTYPE mismatch");
  std::cout<<"RGSL : After check "<<std::endl;
    
  if(input.size == 0){
    return std::pair<bit_mask_t*, gdf_size_type>(nullptr, 0);
  }
  std::cout<<"RGSL : innput size "<<std::endl;

  std::cout <<"Just before the dispatcher"<<std::endl;
  return cudf::type_dispatcher(input.dtype, cudf::detail::set_mask{}, input, value);
}
//
} // namespace cudf
