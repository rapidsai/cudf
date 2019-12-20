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
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/replace.hpp>
#include <cudf/detail/iterator.cuh>

namespace cudf {
namespace experimental {
namespace detail {
namespace {

template <typename Transformer>
std::pair<std::unique_ptr<column>, std::unique_ptr<column>>
form_offsets_and_char_column (cudf::column_device_view input,
                              Transformer offsets_transformer,
                              rmm::mr::device_memory_resource* mr,
                              cudaStream_t stream) {

    std::unique_ptr<column> offsets_column{};
    auto strings_count = input.size();

    if (input.nullable()) {
        auto input_begin = cudf::experimental::detail::make_null_replacement_iterator<string_view>(input, string_view{});
        auto offsets_transformer_itr = thrust::make_transform_iterator(input_begin, offsets_transformer);
        offsets_column = std::move(cudf::strings::detail::make_offsets_child_column(offsets_transformer_itr,
                    offsets_transformer_itr + strings_count,
                    mr, stream));
    } else {
        auto offsets_transformer_itr = thrust::make_transform_iterator(input.begin<string_view>(), offsets_transformer);
        offsets_column = std::move(cudf::strings::detail::make_offsets_child_column(offsets_transformer_itr,
                    offsets_transformer_itr + strings_count,
                    mr, stream));
    }

    auto d_offsets = offsets_column->view().template data<int32_t>();
    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[strings_count];
    auto chars_column = cudf::strings::detail::create_chars_child_column( strings_count, input.null_count(), bytes, mr, stream);

    return std::make_pair(std::move(offsets_column), std::move(chars_column));
}

std::unique_ptr<cudf::column> clamp_string_column (strings_column_view const& input,
                                                   string_scalar const& lo,
                                                   string_scalar const& hi,
                                                   rmm::mr::device_memory_resource* mr,
                                                   cudaStream_t stream) {

    auto input_device_column = column_device_view::create(input.parent(),stream);
    auto d_input = *input_device_column;
    auto d_lo = lo.value();
    auto d_hi = hi.value();
    auto strings_count = input.size();
    auto exec = rmm::exec_policy(stream);

    if (lo.is_valid(stream) and (hi.is_valid(stream))) {
        // build offset column
        auto offsets_transformer = [d_lo, d_hi] __device__ (string_view element, bool is_valid=true) {
            size_type bytes = 0;

            if (is_valid) {
                if (element < d_lo){
                    bytes = d_lo.size_bytes();
                } else if (d_hi < element) {
                    bytes = d_hi.size_bytes();
                } else {
                    bytes = element.size_bytes();
                }
            }
            return bytes;
        };

        auto offset_and_char = form_offsets_and_char_column(d_input, offsets_transformer, mr, stream);
        auto offsets_column(std::move(offset_and_char.first));
        auto chars_column(std::move(offset_and_char.second));

        auto d_offsets = offsets_column->view().template data<int32_t>();
        auto d_chars = chars_column->mutable_view().template data<char>();
        // fill in chars
        auto copy_transformer = [d_input, d_lo, d_hi, d_offsets, d_chars] __device__(size_type idx){
            if (d_input.is_null(idx)){
                return;
            }
            auto input_element = d_input.element<string_view>(idx);

            if (input_element < d_lo){
                memcpy(d_chars + d_offsets[idx], d_lo.data(), d_lo.size_bytes() );
            } else if (d_hi < input_element) {
                memcpy(d_chars + d_offsets[idx], d_hi.data(), d_hi.size_bytes() );
            } else {
                memcpy(d_chars + d_offsets[idx], input_element.data(), input_element.size_bytes() );
            }
        };
        thrust::for_each_n(exec->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count, copy_transformer);

        return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                input.null_count(), std::move(copy_bitmask(input.parent())), stream, mr);
    } else if (hi.is_valid(stream)) {
        // build offset column
        auto offsets_transformer = [d_hi] __device__ (string_view element, bool is_valid=true) {
            size_type bytes = 0;

            if (is_valid) {

                if (d_hi < element) {
                    bytes = d_hi.size_bytes();
                } else {
                    bytes = element.size_bytes();
                }
            }
            return bytes;
        };

        auto offset_and_char = form_offsets_and_char_column(d_input, offsets_transformer, mr, stream);
        auto offsets_column(std::move(offset_and_char.first));
        auto chars_column(std::move(offset_and_char.second));

        auto d_offsets = offsets_column->view().template data<int32_t>();
        auto d_chars = chars_column->mutable_view().template data<char>();
        // fill in chars
        auto copy_transformer = [d_input, d_hi, d_offsets, d_chars] __device__(size_type idx){
            if (d_input.is_null(idx)){
                return;
            }
            auto input_element = d_input.element<string_view>(idx);

            if (d_hi < input_element) {
                memcpy(d_chars + d_offsets[idx], d_hi.data(), d_hi.size_bytes() );
            } else {
                memcpy(d_chars + d_offsets[idx], input_element.data(), input_element.size_bytes() );
            }
        };
        thrust::for_each_n(exec->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count, copy_transformer);

        return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                input.null_count(), std::move(copy_bitmask(input.parent())), stream, mr);
    } else {
        // build offset column
        auto offsets_transformer = [d_lo] __device__ (string_view element, bool is_valid=true) {
            size_type bytes = 0;

            if (is_valid) {

                if (element < d_lo){
                    bytes = d_lo.size_bytes();
                } else {
                    bytes = element.size_bytes();
                }
            }
            return bytes;
        };

        auto offset_and_char = form_offsets_and_char_column(d_input, offsets_transformer, mr, stream);
        auto offsets_column(std::move(offset_and_char.first));
        auto chars_column(std::move(offset_and_char.second));

        auto d_offsets = offsets_column->view().template data<int32_t>();
        auto d_chars = chars_column->mutable_view().template data<char>();
        // fill in chars
        auto copy_transformer = [d_input, d_lo, d_offsets, d_chars] __device__(size_type idx){
            if ( d_input.is_null(idx)){
                return;
            }
            auto input_element = d_input.element<string_view>(idx);

            if (input_element < d_lo){
                memcpy(d_chars + d_offsets[idx], d_lo.data(), d_lo.size_bytes() );
            } else {
                memcpy(d_chars + d_offsets[idx], input_element.data(), input_element.size_bytes() );
            }
        };
        thrust::for_each_n(exec->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count, copy_transformer);

        return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                input.null_count(), std::move(copy_bitmask(input.parent())), stream, mr);
    }
}

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
        auto output = detail::allocate_like(input, input.size(), mask_allocation_policy::NEVER, mr, stream);
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
    std::enable_if_t<std::is_same<T, string_view>::value, std::unique_ptr<cudf::column>>
    operator()(column_view const& input,
              scalar const& lo,
              scalar const& hi,
              rmm::mr::device_memory_resource* mr,
              cudaStream_t stream) {

        using ScalarType = cudf::experimental::scalar_type_t<string_view>;
        auto lo_scalar = static_cast<ScalarType const*>(&lo);
        auto hi_scalar = static_cast<ScalarType const*>(&hi);
        return clamp_string_column (input, *lo_scalar, *hi_scalar, mr, stream);
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
