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
#include <cudf/detail/iterator.cuh>

namespace cudf {
namespace experimental {
namespace detail {

std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type>
bools_to_mask(column_view const& input,
                  rmm::mr::device_memory_resource * mr,
                  cudaStream_t stream) {
    CUDF_EXPECTS(input.type().id() == BOOL8, "Input is not of type bool");

    if(input.size() == 0){
        return std::make_pair(std::make_unique<rmm::device_buffer>(), 0);
    }

    auto input_device_view_ptr = column_device_view::create(input, stream);
    auto input_device_view = *input_device_view_ptr;
    auto pred = [] __device__ (experimental::bool8 element) {
        return element == true_v;
    };
    if(input.nullable()) {
        // Nulls are considered false
        auto input_begin = make_null_replacement_iterator<experimental::bool8>(input_device_view, false_v);

        auto mask = detail::valid_if(input_begin,
                                     input_begin + input.size(),
                                     pred, stream, mr);

        return std::make_pair(std::make_unique<rmm::device_buffer>(std::move(mask.first)), mask.second);
    } else {
        auto mask = detail::valid_if(input_device_view.begin<experimental::bool8>(),
                                     input_device_view.end<experimental::bool8>(),
                                     pred, stream, mr);

        return std::make_pair(std::make_unique<rmm::device_buffer>(std::move(mask.first)), mask.second);
    }
}  

}// namespace detail


std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type>
bools_to_mask(column_view const& input, rmm::mr::device_memory_resource * mr) {
    return detail::bools_to_mask(input, mr);
}

}// namespace experimental
}// namespace cudf
