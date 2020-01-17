/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/detail/iterator.cuh>

namespace cudf {
namespace experimental {

namespace detail{

struct nan_dispatcher {
    template<typename T, typename Predicate>
    std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<column>>
    operator()(cudf::column_view const& input,
               Predicate predicate,
               rmm::mr::device_memory_resource* mr,
               cudaStream_t stream) {

        auto input_device_view = column_device_view::create(input);

        if (input.has_nulls()) {
            auto input_pair_iterator = make_pair_iterator<T, true>(*input_device_view);
            return true_if(input_pair_iterator,
                           input_pair_iterator + input.size(),
                           input.size(), predicate, mr);
        } else {
            auto input_pair_iterator = make_pair_iterator<T, false>(*input_device_view);
            return true_if(input_pair_iterator,
                           input_pair_iterator + input.size(),
                           input.size(), predicate, mr);
        }
    }

    template<typename T, typename Predicate>
    std::enable_if_t<!std::is_floating_point<T>::value, std::unique_ptr<column>>
    operator()(cudf::column_view const& input,
               Predicate predicate,
               rmm::mr::device_memory_resource* mr,
               cudaStream_t stream) {

        CUDF_FAIL("NAN is not supported in a Non-floating point type column");
    }
};

/**
 * @copydoc cudf::experimental::is_nan
 *
 * @param[in] stream Optional CUDA stream on which to execute kernels
 */
std::unique_ptr<column> is_nan(cudf::column_view const& input,
                               rmm::mr::device_memory_resource* mr=
                                   rmm::mr::get_default_resource(),
                               cudaStream_t stream = 0) {
    
    auto predicate = [] __device__ (auto element_validity_pair) {
        return element_validity_pair.second and std::isnan(element_validity_pair.first);
    };

    return cudf::experimental::type_dispatcher(input.type(), nan_dispatcher{},
                                               input, predicate, mr, stream); 
}

/**
 * @copydoc cudf::experimental::is_not_nan
 *
 * @param[in] stream Optional CUDA stream on which to execute kernels
 */
std::unique_ptr<column> is_not_nan(cudf::column_view const& input,
                                 rmm::mr::device_memory_resource* mr=
                                     rmm::mr::get_default_resource(),
                                 cudaStream_t stream = 0) {
    
    auto predicate = [] __device__ (auto element_validity_pair) {
        return !element_validity_pair.second or !std::isnan(element_validity_pair.first);
    };
    
    return cudf::experimental::type_dispatcher(input.type(), nan_dispatcher{},
                                               input, predicate, mr, stream); 
}

}// namespace detail

std::unique_ptr<column> is_nan(cudf::column_view const& input,
                               rmm::mr::device_memory_resource* mr) {

    return detail::is_nan(input, mr);
}

std::unique_ptr<column> is_not_nan(cudf::column_view const& input,
                                 rmm::mr::device_memory_resource* mr) {

    return detail::is_not_nan(input, mr);
}

}// namespace experimental
}// namespace cudf

