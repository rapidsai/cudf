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
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>

namespace cudf {

namespace experimental {

namespace detail {

std::unique_ptr<column> null_op(column_view const& input,
                                bool nulls_are_false = true,
                                rmm::mr::device_memory_resource* mr = 
                                    rmm::mr::get_default_resource(), 
                                cudaStream_t stream = 0) {
    auto output = make_numeric_column(data_type(BOOL8), input.size(), UNALLOCATED, stream, mr);
    auto output_mutable_view = output->mutable_view();
    auto input_device_view = column_device_view::create(input);
    auto input_device_column_view = *input_device_view;
    auto exec = rmm::exec_policy(stream)->on(stream);
    auto output_data = output_mutable_view.data<bool>();

    thrust::transform(exec,
                      thrust::make_counting_iterator(static_cast<size_type>(0)),
                      thrust::make_counting_iterator(static_cast<size_type>(input.size())),
                      output_data,
                      [input_device_column_view, nulls_are_false]__device__(auto index){
                          return (nulls_are_false == input_device_column_view.is_valid(index));
                      });

    return output;
}
}// detail

std::unique_ptr<column> is_null(cudf::column_view const& input) {
    return detail::null_op(input, false);
}

std::unique_ptr<column> is_not_null(cudf::column_view const& input) {
    return detail::null_op(input, true);
}

}// experimental
}// cudf
