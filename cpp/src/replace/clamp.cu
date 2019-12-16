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
#include <cudf/scalar/scalar.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/replace.hpp>
#include <cudf/detail/iterator.cuh>

namespace cudf {
namespace experimental {
namespace detail {
namespace {
struct dispatch_clamp {
    template <typename T, typename Transformer>
    void apply_transform (column_device_view  input,
                          mutable_column_device_view output,
                          Transformer trans,
                          cudaStream_t stream)
    {
        if (input.nullable()){
            auto input_begin = cudf::experimental::detail::make_null_replacement_iterator<T>(input);
            thrust::transform(rmm::exec_policy(stream)->on(stream),
                              input_begin,
                              input_begin+input.size(),
                              detail::make_validity_iterator(input),
                              output.begin<T>(),
                              trans);
        } else {
            thrust::transform(rmm::exec_policy(stream)->on(stream),
                              input.begin<T>(),
                              input.end<T>(),
                              output.begin<T>(),
                              trans);
        }
    }

    template <typename T>
    std::enable_if_t<cudf::is_fixed_width<T>(), std::unique_ptr<cudf::column>>
    operator()(column_view const& input,
              scalar const& lo,
              scalar const& hi,
              rmm::mr::device_memory_resource* mr,
              cudaStream_t stream) {
        using ScalarType = cudf::experimental::scalar_type_t<T>;
        auto lo_scalar = static_cast<ScalarType const&>(lo);
        auto hi_scalar = static_cast<ScalarType const&>(hi);
        auto lo_value = lo_scalar.value(stream);
        auto hi_value = hi_scalar.value(stream);
        auto output = detail::allocate_like(input, input.size(), mask_allocation_policy::RETAIN, mr, stream);
        // mask will not change
        if (input.nullable()){
            output->set_null_mask(copy_bitmask(input), input.null_count());
        }

        auto output_device_view  = cudf::mutable_column_device_view::create(output->mutable_view(), stream);
        auto input_device_view  = cudf::column_device_view::create(input, stream);

        if (lo.is_valid(stream) and hi.is_valid(stream)) {
            auto trans = [lo_value, hi_value] __device__ (T input, bool is_valid = true){
                if (is_valid) {
                    if (input < lo_value) {
                        return lo_value;
                    } else if (input > hi_value) {
                        return hi_value;
                    }
                }

                return input;
            };

            apply_transform<T>(*input_device_view, *output_device_view, trans, stream);
        } else if (not lo.is_valid(stream)) {
            auto trans = [hi_value] __device__ (T input, bool is_valid = true){
                if (is_valid and input > hi_value) {
                    return hi_value;
                }

                return input;
            };

            apply_transform<T>(*input_device_view, *output_device_view, trans, stream);
        } else {

            auto trans = [lo_value] __device__ (T input, bool is_valid = true){
                if (is_valid and input < lo_value) {
                    return lo_value;
                }

                return input;
            };

            apply_transform<T>(*input_device_view, *output_device_view, trans, stream);
        }
        
        return output;
    }
    
    template <typename T>
    std::enable_if_t<not cudf::is_fixed_width<T>(), std::unique_ptr<cudf::column>>
    operator()(column_view const& input,
              scalar const& lo,
              scalar const& hi,
              rmm::mr::device_memory_resource* mr,
              cudaStream_t stream) {
        CUDF_FAIL("Clamp is not yet supporting non-fixed types");
    }

};
} //namespace

/**
 * @copydoc cudf::experimental::clamp
 *
 * @param[in] stream Optional stream on which to issue all memory allocations
 */
std::unique_ptr<column> clamp(column_view const& input,
                              scalar const& lo,
                              scalar const& hi,
                              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                              cudaStream_t stream = 0) {
    CUDF_EXPECTS(lo.type() == hi.type(), "mimatching types of scalars");
    CUDF_EXPECTS(lo.type() == input.type(), "mimatching types of scalar and input");

    if ((not lo.is_valid(stream) and not hi.is_valid(stream)) or 
        (input.is_empty())) {
        // There will be no change
        return std::make_unique<column>(input, stream, mr);
    }

    return cudf::experimental::type_dispatcher(input.type(), dispatch_clamp{},
                                               input, lo, hi,
                                               mr, stream);
}   

}// namespace detail

// clamp input at lo and hi
std::unique_ptr<column> clamp(column_view const& input,
                              scalar const& lo,
                              scalar const& hi,
                              rmm::mr::device_memory_resource* mr) {

    return detail::clamp(input, lo, hi, mr);
}

}// namespace experimental
}// namespace cudf
