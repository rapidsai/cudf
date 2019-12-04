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

#include <cudf/null_mask.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/combine.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/detail/valid_if.cuh>
#include "../utilities.hpp"
#include "../utilities.cuh"

namespace cudf
{
namespace strings
{
namespace detail
{

std::unique_ptr<column> fill( strings_column_view const& strings,
                              size_type begin, size_type end,
                              string_scalar const& value,
                              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                              cudaStream_t stream = 0 )
{
    auto strings_count = strings.size();
    if( strings_count == 0 )
        return detail::make_empty_strings_column(mr,stream);
    CUDF_EXPECTS( (begin >= 0) && (end <= strings_count), "Parameters [begin,end) are outside the range of the provided strings column");
    CUDF_EXPECTS( begin <= end, "Parameters [begin,end) have invalid range values");
    if( begin==end ) // return a copy
        return std::make_unique<column>( strings.parent() );

    auto execpol = rmm::exec_policy(stream);
    string_view d_value(nullptr,0);
    if( value.is_valid() )
        d_value = string_view(value.data(),value.size());

    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_strings = *strings_column;

    // create resulting null mask
    auto valid_mask = cudf::experimental::detail::valid_if(
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(strings_count),
        [d_strings, begin, end, d_value] __device__ (size_type idx) {
            return ((begin <= idx) && (idx < end)) ? !d_value.is_null() : !d_strings.is_null(idx);
        }, stream, mr );
    rmm::device_buffer null_mask = valid_mask.first;
    auto null_count = valid_mask.second;

    // build offsets column
    auto offsets_transformer = [d_strings, begin, end, d_value] __device__ (size_type idx) {
            if( ((begin <= idx) && (idx < end)) ? d_value.is_null() : d_strings.is_null(idx) )
                return 0;
            int32_t bytes = d_value.size_bytes();
            if( (idx < begin) || (idx >= end) )
                bytes = d_strings.element<string_view>(idx).size_bytes();
            return bytes;
        };
    auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<size_type>(0), offsets_transformer );
    auto offsets_column = detail::make_offsets_child_column(offsets_transformer_itr,
                                                            offsets_transformer_itr+strings_count,
                                                            mr, stream);
    auto offsets_view = offsets_column->view();
    auto d_offsets = offsets_view.data<int32_t>();

    // create the chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[strings_count];
    auto chars_column = strings::detail::create_chars_child_column( strings_count, null_count, bytes, mr, stream );
    // fill the chars column
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<char>();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count,
        [d_strings, begin, end, d_value, d_offsets, d_chars] __device__(size_type idx){
            if( ((begin <= idx) && (idx < end)) ? d_value.is_null() : d_strings.is_null(idx) )
                return;
            string_view d_str = d_value;
            if( (idx < begin) || (idx >= end) )
                d_str = d_strings.element<string_view>(idx);
            memcpy( d_chars + d_offsets[idx], d_str.data(), d_str.size_bytes() );
        });

    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               null_count, std::move(null_mask), stream, mr);
}

} // namespace detail
} // namespace strings
} // namespace cudf
