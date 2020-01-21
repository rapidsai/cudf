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
#include <cudf/detail/unary.hpp>

namespace cudf {

namespace experimental {

std::unique_ptr<column> is_null(cudf::column_view const& input,
                                rmm::mr::device_memory_resource* mr) {
    auto input_device_view = column_device_view::create(input);
    auto device_view = *input_device_view;
    auto predicate = [device_view] __device__(auto index){
                         return (device_view.is_null(index));
                     };
    return detail::true_if(thrust::make_counting_iterator(0),
                           thrust::make_counting_iterator(input.size()),
                           input.size(), predicate, mr);
}

std::unique_ptr<column> is_valid(cudf::column_view const& input,
                                 rmm::mr::device_memory_resource* mr) {
    auto input_device_view = column_device_view::create(input);
    auto device_view = *input_device_view;
    auto predicate = [device_view] __device__(auto index){
                         return device_view.is_valid(index);
                     };
    return detail::true_if(thrust::make_counting_iterator(0),
                           thrust::make_counting_iterator(input.size()),
                           input.size(), predicate, mr);
}

}// experimental
}// cudf
