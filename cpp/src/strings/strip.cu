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

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strip.hpp>
#include <cudf/utilities/error.hpp>
#include <strings/utilities.hpp>
#include <strings/utilities.cuh>

#include <thrust/transform.h>
#include <thrust/logical.h>

namespace cudf
{
namespace strings
{
namespace detail
{
namespace
{

/**
 * @brief Used as template parameter to divide size calculation from
 * the actual string operation within a function.
 *
 * Useful when most of the logic is identical for both passes.
 */
enum TwoPass
{
    SizeOnly = 0, ///< calculate the size only
    ExecuteOp     ///< run the string operation
};

/**
 * @brief Strip characters from the beginning and/or end of a string.
 *
 * This functor strips the beginning and/or end of each string
 * of any characters found in d_to_strip or whitespace if
 * d_to_strip is empty.
 *
 * @tparam Pass Allows computing only the size of the output
 *              or writing the output to device memory.
 */
template <TwoPass Pass=SizeOnly>
struct strip_fn
{
    column_device_view const d_strings;
    strip_type stype; // right, left, or both
    string_view d_to_strip;
    int32_t const* d_offsets{};
    char* d_chars{};

    __device__ bool is_strip_character( char_utf8 chr )
    {
        return d_to_strip.empty() ? (chr <= ' ') : // whitespace check
                    thrust::any_of( thrust::seq, d_to_strip.begin(), d_to_strip.end(),
                                    [chr] __device__ (char_utf8 c) {return c==chr;});
    }

    __device__ size_type operator()( size_type idx )
    {
        if( d_strings.is_null(idx) )
            return 0;
        string_view d_str = d_strings.element<string_view>(idx);
        size_type length = d_str.length();
        size_type left_offset = 0;
        auto itr = d_str.begin();
        if( stype==strip_type::LEFT || stype==strip_type::BOTH )
        {
            for( ; itr != d_str.end(); )
            {
                if( !is_strip_character(*itr++) )
                    break;
                left_offset = itr.byte_offset();
            }
        }
        size_type right_offset = d_str.size_bytes();
        if( stype==strip_type::RIGHT || stype==strip_type::BOTH )
        {
            itr = d_str.end();
            for( size_type n=0; n < length; ++n )
            {
                if( !is_strip_character(*(--itr)) )
                    break;
                right_offset = itr.byte_offset();
            }
        }
        size_type bytes = 0;
        if( right_offset > left_offset )
            bytes = right_offset - left_offset;
        if( Pass==ExecuteOp )
            memcpy( d_chars + d_offsets[idx], d_str.data()+left_offset, bytes );
        return bytes;
    }
};

}

std::unique_ptr<column> strip( strings_column_view const& strings,
                               strip_type stype = strip_type::BOTH,
                               string_scalar const& to_strip = string_scalar(""),
                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                               cudaStream_t stream = 0 )
{
    auto strings_count = strings.size();
    if( strings_count == 0 )
        return detail::make_empty_strings_column(mr,stream);

    CUDF_EXPECTS( to_strip.is_valid(), "Parameter to_strip must be valid" );
    string_view d_to_strip(to_strip.data(), to_strip.size());

    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_column = *strings_column;

    // copy null mask
    rmm::device_buffer null_mask = copy_bitmask(strings.parent(),stream,mr);

    // build offsets column -- calculate the size of each output string
    auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<size_type>(0),
        strip_fn<SizeOnly>{d_column, stype, d_to_strip} );
    auto offsets_column = make_offsets_child_column(offsets_transformer_itr,
                                       offsets_transformer_itr+strings_count,
                                       mr, stream);
    auto offsets_view = offsets_column->view();
    auto d_offsets = offsets_view.data<int32_t>();

    // build the chars column -- convert characters based on case_flag parameter
    size_type bytes = thrust::device_pointer_cast(d_offsets)[strings_count];
    auto chars_column = create_chars_child_column( strings_count, d_column.null_count(), bytes, mr, stream );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<char>();
    thrust::for_each_n(execpol->on(stream),
        thrust::make_counting_iterator<size_type>(0), strings_count,
        strip_fn<ExecuteOp>{d_column, stype, d_to_strip, d_offsets, d_chars} );
    //
    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               d_column.null_count(), std::move(null_mask), stream, mr);
}

} // namespace detail

// external APIs

std::unique_ptr<column> strip( strings_column_view const& strings,
                               strip_type stype,
                               string_scalar const& to_strip,
                               rmm::mr::device_memory_resource* mr )
{
    return detail::strip(strings, stype, to_strip, mr);
}

} // namespace strings
} // namespace cudf
