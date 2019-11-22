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

#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/legacy/cuda_utils.hpp>
#include <utilities/legacy/column_utils.hpp>
#include <bitmask/legacy/bit_mask.cuh>
#include <cudf/legacy/filling.hpp>

using bit_mask::bit_mask_t;

namespace cudf {

namespace detail {

gdf_column null_op(gdf_column const& input, bool nulls_are_false = true, cudaStream_t stream = 0) {
    auto output = cudf::allocate_column(GDF_BOOL8, input.size, false, 
		  gdf_dtype_extra_info{}, stream);

    if (not cudf::is_nullable(input)) {
	gdf_scalar value {nulls_are_false, GDF_BOOL8, true}; 
	cudf::fill(&output, value, 0, output.size);
    } else {
        const bit_mask_t* __restrict__ typed_input_valid = reinterpret_cast<bit_mask_t*>(input.valid);

        thrust::transform(rmm::exec_policy(stream)->on(stream),
                          thrust::make_counting_iterator(static_cast<gdf_size_type>(0)),
                          thrust::make_counting_iterator(static_cast<gdf_size_type>(input.size)),
                          static_cast<bool*>(output.data),
                          [=]__device__(auto index){
                              return (nulls_are_false ==
                                      bit_mask::is_valid(typed_input_valid, index));
                          });
    }

    return output;
}
}// detail

gdf_column is_null(gdf_column const& input) {
    return detail::null_op(input, false, 0);
}

gdf_column is_not_null(gdf_column const& input) {
    return detail::null_op(input, true, 0);
}

}// cudf
