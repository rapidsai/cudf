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
#include <utilities/cuda_utils.hpp>
#include <utilities/column_utils.hpp>
#include <bitmask/legacy/bit_mask.cuh>
#include <cudf/legacy/filling.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device_memory_resource.hpp>

using bit_mask::bit_mask_t;

namespace cudf {
namespace experimental {
namespace detail {

std::unique_ptr<cudf::column> 
null_op(cudf::column_view  input, 
        bool nulls_are_false = true,
        cudaStream_t stream = 0, 
        rmm::mr::device_memory_resource* mr = 
        rmm::mr::get_default_resource()) {

    std::unique_ptr<cudf::column> output = 
        cudf::make_numeric_column(cudf::data_type(cudf::BOOL8),
                                  input.size(), 
                                  cudf::UNALLOCATED, // #CH review
                                  stream, 
                                  mr);

    if (!input.nullable()) {
        // #CH to become cudf::scalar
        gdf_scalar value{nulls_are_false, GDF_BOOL8, true};
        // #CH dependent on FEA 2936 (filling.hpp)
        // cudf::fill(&output, value, 0, output.size);
    } else {
        const bit_mask_t* __restrict__ typed_input_valid = input.null_mask();
        auto exec = rmm::exec_policy(stream)->on(stream);

        auto output_view = output->mutable_view();
        auto d_output    = output_view.data<bool>(); // #CH review

        thrust::transform(exec,
                          thrust::make_counting_iterator(static_cast<size_type>(0)),
                          thrust::make_counting_iterator(static_cast<size_type>(input.size())),
                          d_output,
                          [=]__device__(auto index){
                              return (nulls_are_false ==
                                      bit_mask::is_valid(typed_input_valid, index));
                          });
    }

    return output;
}
} // namespace detail

std::unique_ptr<cudf::column> 
is_null(cudf::column_view input) {
    return detail::null_op(input, false, 0);
}

std::unique_ptr<cudf::column> 
is_not_null(cudf::column_view input) {
    return detail::null_op(input, true, 0);
}

} // expiremental
} // cudf
