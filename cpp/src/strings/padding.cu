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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/padding.hpp>
#include "./utilities.hpp"
#include "./utilities.cuh"


namespace cudf
{
namespace strings
{
namespace detail
{
namespace
{
struct compute_pad_output_length_fn
{
    column_device_view d_strings;
    size_type width;
    size_type fill_char_size;

    __device__ size_type operator()(size_type idx)
    {
        if( d_strings.is_null(idx) )
            return 0;
        string_view d_str = d_strings.element<string_view>(idx);
        size_type bytes = d_str.size_bytes();
        size_type length = d_str.length();
        if( width > length ) // no truncating
            bytes += fill_char_size * (width - length); // add padding
        return bytes;
    }
};

}

//
std::unique_ptr<column> pad( strings_column_view const& strings,
                             size_type width, pad_side side = pad_side::right,
                             std::string const& fill_char = " ",
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                             cudaStream_t stream = 0 )
{
    size_type strings_count = strings.size();
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);
    CUDF_EXPECTS( !fill_char.empty(), "fill_char parameter must not be empty" );
    char_utf8 d_fill_char = 0;
    size_type fill_char_size = to_char_utf8(fill_char.c_str(), d_fill_char );

    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(strings.parent(), stream);
    auto d_strings = *strings_column;

    rmm::device_buffer null_mask = copy_bitmask( strings.parent(), stream, mr );
    auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<int32_t>(0),
        compute_pad_output_length_fn{d_strings,width,fill_char_size} );
    auto offsets_column = make_offsets_child_column(offsets_transformer_itr,
                                       offsets_transformer_itr+strings_count,
                                       mr, stream);
    auto offsets_view = offsets_column->view();
    auto d_offsets = offsets_view.data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[strings_count];
    auto chars_column = strings::detail::create_chars_child_column( strings_count, strings.null_count(), bytes, mr, stream );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<char>();

    if( side==pad_side::left )
    {
        thrust::for_each_n(rmm::exec_policy(stream)->on(stream), thrust::make_counting_iterator<cudf::size_type>(0), strings_count,
            [d_strings, width, d_fill_char, d_offsets, d_chars] __device__ (size_type idx) {
                if( d_strings.is_null(idx) )
                    return;
                string_view d_str = d_strings.element<string_view>(idx);
                auto length = d_str.length();
                char* ptr = d_chars + d_offsets[idx];
                while( length++ < width )
                    ptr += from_char_utf8(d_fill_char,ptr);
                copy_string( ptr, d_str );
            });
    }
    else if( side==pad_side::right )
    {
        thrust::for_each_n(rmm::exec_policy(stream)->on(stream), thrust::make_counting_iterator<cudf::size_type>(0), strings_count,
            [d_strings, width, d_fill_char, d_offsets, d_chars] __device__ (size_type idx) {
                if( d_strings.is_null(idx) )
                    return;
                string_view d_str = d_strings.element<string_view>(idx);
                auto length = d_str.length();
                char* ptr = d_chars + d_offsets[idx];
                ptr = copy_string(ptr, d_str);
                while( length++ < width )
                    ptr += from_char_utf8(d_fill_char,ptr);
            });
    }
    else if( side==pad_side::both )
    {
        thrust::for_each_n(rmm::exec_policy(stream)->on(stream), thrust::make_counting_iterator<cudf::size_type>(0), strings_count,
            [d_strings, width, d_fill_char, d_offsets, d_chars] __device__ (size_type idx) {
                if( d_strings.is_null(idx) )
                    return;
                string_view d_str = d_strings.element<string_view>(idx);
                char* ptr = d_chars + d_offsets[idx];
                int32_t pad = static_cast<int32_t>(width - d_str.length());
                auto left_pad = pad/2;
                auto right_pad = pad - left_pad;
                while( left_pad-- > 0 )
                    ptr += from_char_utf8(d_fill_char,ptr);
                ptr = copy_string(ptr, d_str);
                while( right_pad-- > 0 )
                    ptr += from_char_utf8(d_fill_char,ptr);
            });
    }
    //
    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               strings.null_count(), std::move(null_mask), stream, mr);
}

std::unique_ptr<column> zfill( strings_column_view const& strings,
                               size_type width,
                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                               cudaStream_t stream = 0 )
{
    size_type strings_count = strings.size();
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);
    char_utf8 d_fill_char = static_cast<char_utf8>('0');

    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(strings.parent(), stream);
    auto d_strings = *strings_column;

    rmm::device_buffer null_mask = copy_bitmask( strings.parent(), stream, mr );
    auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<int32_t>(0),
        compute_pad_output_length_fn{d_strings,width,1} );
    auto offsets_column = make_offsets_child_column(offsets_transformer_itr,
                                       offsets_transformer_itr+strings_count,
                                       mr, stream);
    auto offsets_view = offsets_column->view();
    auto d_offsets = offsets_view.data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[strings_count];
    auto chars_column = strings::detail::create_chars_child_column( strings_count, strings.null_count(), bytes, mr, stream );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<char>();

    thrust::for_each_n(rmm::exec_policy(stream)->on(stream), thrust::make_counting_iterator<cudf::size_type>(0), strings_count,
        [d_strings, width, d_fill_char, d_offsets, d_chars] __device__ (size_type idx) {
            if( d_strings.is_null(idx) )
                return;
            string_view d_str = d_strings.element<string_view>(idx);
            auto length = d_str.length();
            auto bytes = d_str.size_bytes();
            char* out_ptr = d_chars + d_offsets[idx];
            if( length >= width )
                copy_string(out_ptr, d_str);
            else
            {
                const char* in_ptr = d_str.data();
                char first_char = length==0 ? 0 : *in_ptr;
                if( first_char=='+' || first_char=='-' )
                {
                    *out_ptr++ = first_char;
                    ++in_ptr;
                    --bytes;
                }
                while( length++ < width )
                    *out_ptr++ = '0';
                memcpy( out_ptr, in_ptr, bytes );
            }
        });
    //
    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               strings.null_count(), std::move(null_mask), stream, mr);
}


} // namespace detail

// external APIs

std::unique_ptr<column> pad( strings_column_view const& strings,
                             size_type width, pad_side side,
                             std::string const& fill_char,
                             rmm::mr::device_memory_resource* mr )
{
    return detail::pad(strings,width,side,fill_char,mr);
}                             

std::unique_ptr<column> zfill( strings_column_view const& strings,
                               size_type width,
                               rmm::mr::device_memory_resource* mr )
{
    return detail::zfill(strings,width,mr);
}

} // namespace strings
} // namespace cudf
