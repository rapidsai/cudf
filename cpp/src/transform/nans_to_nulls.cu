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

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/detail/transform.hpp>

namespace cudf {
namespace experimental {
namespace detail {

struct dispatch_nan_to_null {
    template <typename T>
    std::enable_if_t<std::is_floating_point<T>::value, std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type>>
    operator ()(column_view const& input,
                rmm::mr::device_memory_resource* mr,
                cudaStream_t stream) {

        auto input_device_view_ptr = column_device_view::create(input, stream);
        auto input_device_view = *input_device_view_ptr;

        if (input.nullable()) {
            auto pred = [input_device_view] __device__ (cudf::size_type idx) {
                return not (std::isnan(input_device_view.element<T>(idx)) || input_device_view.is_null_nocheck(idx));
            };

            auto mask = detail::valid_if(thrust::make_counting_iterator<cudf::size_type>(0),
                                    thrust::make_counting_iterator<cudf::size_type>(input.size()),
                                    pred, stream, mr);

            return std::make_pair(std::make_unique<rmm::device_buffer>(std::move(mask.first)), mask.second);
        } else {
            auto pred = [input_device_view] __device__ (cudf::size_type idx) {
                return not (std::isnan(input_device_view.element<T>(idx)));
            };

            auto mask = detail::valid_if(thrust::make_counting_iterator<cudf::size_type>(0),
                                    thrust::make_counting_iterator<cudf::size_type>(input.size()),
                                    pred, stream, mr);

            return std::make_pair(std::make_unique<rmm::device_buffer>(std::move(mask.first)), mask.second);
        }
    }

    template <typename T>
    std::enable_if_t<!std::is_floating_point<T>::value, std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type>>
    operator ()(column_view const& input,
                rmm::mr::device_memory_resource* mr,
                cudaStream_t stream) {
        CUDF_FAIL("Input column can't be a non-floating type");
    }
};

std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> 
nans_to_nulls(column_view const& input, 
              rmm::mr::device_memory_resource * mr,
              cudaStream_t stream) {
   
    if (input.size() == 0){
        return std::make_pair(std::make_unique<rmm::device_buffer>(), 0);
    }

    return cudf::experimental::type_dispatcher(input.type(), dispatch_nan_to_null{}, 
                                               input, mr, stream);
}   

}// namespace detail

std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> 
nans_to_nulls(column_view const& input, rmm::mr::device_memory_resource * mr) {

    return detail::nans_to_nulls(input, mr);
}

}// namespace experimental
}// namespace cudf
