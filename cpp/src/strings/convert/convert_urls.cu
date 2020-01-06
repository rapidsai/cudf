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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/convert/convert_urls.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <strings/utilities.hpp>
#include <strings/utilities.cuh>


namespace cudf
{
namespace strings
{
namespace detail
{
namespace
{

//
// This is the functor for the url_encode() method below.
// Specific requirements are documented in custrings issue #321.
// In summary it converts mostly non-ascii characters and control characters into UTF-8 hex characters
// prefixed with '%'. For example, the space character must be converted to characters '%20' where the
// '20' indicates the hex value for space in UTF-8. Likewise, multi-byte characters are converted to
// multiple hex charactes. For example, the é character is converted to characters '%C3%A9' where 'C3A9'
// is the UTF-8 bytes xc3a9 for this character.
//
struct url_encoder_fn
{
    column_device_view const d_strings;
    int32_t const* d_offsets{};
    char* d_chars{};

    // utility to create 2-byte hex characters from single binary byte
    __device__ void byte_to_hex( uint8_t byte, char* hex )
    {
        hex[0] = '0';
        if( byte >= 16 )
        {
            uint8_t hibyte = byte/16;
            hex[0] = hibyte < 10 ? '0'+hibyte : 'A'+(hibyte-10);
            byte = byte - (hibyte * 16);
        }
        hex[1] = byte < 10 ? '0'+byte : 'A'+(byte-10);
    }

    __device__ bool should_not_url_encode( char ch )
    {
        return ( (ch>='0' && ch<='9') || // these are the characters
                 (ch>='A' && ch<='Z') || // that are not to be url encoded
                 (ch>='a' && ch<='z') || // reference: docs.python.org/3/library/urllib.parse.html#urllib.parse.quote
                 (ch=='.') || (ch=='_') || (ch=='~') || (ch=='-') );
    }

    // main part of the functor the performs the url-encoding
    __device__ size_type operator()( size_type idx )
    {
        if( d_strings.is_null(idx) )
            return 0;
        string_view d_str = d_strings.element<string_view>(idx);
        //
        char* out_ptr = d_chars ? d_chars + d_offsets[idx] : nullptr;
        size_type nbytes = 0;
        char hex[2]; // two-byte hex max
        for( auto itr = d_str.begin(); itr!=d_str.end(); ++itr )
        {
            auto ch = *itr;
            if( ch < 128 )
            {
                if( should_not_url_encode( static_cast<char>(ch) ) )
                {
                    nbytes++;
                    if( out_ptr )
                        out_ptr = copy_and_increment( out_ptr, d_str.data() + itr.byte_offset(), 1);
                }
                else // url-encode everything else
                {
                    nbytes += 3;
                    if( out_ptr )
                    {
                        out_ptr = copy_and_increment(out_ptr,"%",1);  // add the '%' prefix
                        byte_to_hex( static_cast<uint8_t>(ch), hex);   // convert to 2 hex chars
                        out_ptr = copy_and_increment(out_ptr,hex,2);  // add them to the output
                    }
                }
            }
            else // these are to be utf-8 url-encoded
            {
                uint8_t char_bytes[4]; // holds utf-8 bytes for one character
                size_type char_width = from_char_utf8(ch, reinterpret_cast<char*>(char_bytes));
                nbytes += char_width * 3; // '%' plus 2 hex chars per byte (example: é is %C3%A9)
                // process each byte in this current character
                for( size_type chidx=0; out_ptr && (chidx < char_width); ++chidx )
                {
                    out_ptr = copy_and_increment(out_ptr,"%",1);  // add '%' prefix
                    byte_to_hex( char_bytes[chidx], hex);         // convert to 2 hex chars
                    out_ptr = copy_and_increment(out_ptr,hex,2);  // add them to the output
                }
            }
        }
        return nbytes;
    }
};

} // namespace

//
std::unique_ptr<column> url_encode( strings_column_view const& strings,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                    cudaStream_t stream = 0)
{
    size_type strings_count = strings.size();
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);

    auto strings_column = column_device_view::create(strings.parent(), stream);
    auto d_strings = *strings_column;

    // copy null mask
    rmm::device_buffer null_mask = copy_bitmask(strings.parent(),stream,mr);
    // build offsets column
    auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<size_type>(0),
                                                                    url_encoder_fn{d_strings} );
    auto offsets_column = make_offsets_child_column(offsets_transformer_itr,
                                                    offsets_transformer_itr+strings_count,
                                                    mr, stream);
    auto d_offsets = offsets_column->view().data<int32_t>();
    // build chars column
    auto chars_column = create_chars_child_column( strings_count, strings.null_count(),
                                                   thrust::device_pointer_cast(d_offsets)[strings_count],
                                                   mr, stream );
    auto d_chars = chars_column->mutable_view().data<char>();
    thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator<size_type>(0), strings_count,
                       url_encoder_fn{d_strings,d_offsets,d_chars});
    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               strings.null_count(), std::move(null_mask), stream, mr);
}

} // namespace detail

// external API
std::unique_ptr<column> url_encode( strings_column_view const& strings,
                                    rmm::mr::device_memory_resource* mr)
{
    return detail::url_encode(strings,mr);
}

namespace detail
{
namespace
{

//
// This is the functor for the url_decode() method below.
// Specific requirements are documented in custrings issue #321.
// In summary it converts all character sequences starting with '%' into bytes
// interpretting the following 2 characters as hex values to create the output byte.
// For example, the sequence '%20' is converted into byte (0x20) which is a single
// space character. Another example converts '%C3%A9' into 2 sequential bytes
// (0xc3 and 0xa9 respectively). Overall, 3 characters are converted into one byte
// whenever a '%' character is encountered in the string.
//
struct url_decoder_fn
{
    column_device_view const d_strings;
    int32_t const* d_offsets{};
    char* d_chars{};

    // utility to convert a hex char into a single byte
    __device__ uint8_t hex_char_to_byte( char ch )
    {
        if( ch >= '0' && ch <= '9' )
            return (ch-'0');
        if( ch >= 'A' && ch <= 'F' )
            return (ch-'A'+10); // in hex A=10,B=11,...,F=15
        if( ch >='a' && ch <= 'f' )
            return (ch-'a'+10); // same for lower case
        return 0;
    }

    // main functor method executed on each string
    __device__ size_type operator()(size_type idx)
    {
        if( d_strings.is_null(idx) )
            return 0;
        string_view d_str = d_strings.element<string_view>(idx);
        char* out_ptr = d_chars ? out_ptr = d_chars + d_offsets[idx] : nullptr;
        size_type nbytes = 0;
        const char* in_ptr = d_str.data();
        const char* end = in_ptr + d_str.size_bytes();
        while( in_ptr < end ) // walk through each byte
        {
            char ch = *in_ptr++;
            if( (ch == '%') && ((in_ptr+1) < end) )
            {   // found '%', convert hex to byte
                ch =  static_cast<char>(16 * hex_char_to_byte(*in_ptr++));
                ch += static_cast<char>(hex_char_to_byte(*in_ptr++));
            }
            ++nbytes; // keeping track of bytes and chars
            if( out_ptr )
                out_ptr = copy_and_increment(out_ptr, &ch, 1);
        }
        return nbytes;
    }
};

}

//
std::unique_ptr<column> url_decode( strings_column_view const& strings,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                    cudaStream_t stream = 0)
{
    size_type strings_count = strings.size();
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);

    auto strings_column = column_device_view::create(strings.parent(), stream);
    auto d_strings = *strings_column;

    // copy null mask
    rmm::device_buffer null_mask = copy_bitmask(strings.parent(),stream,mr);
    // build offsets column
    auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<size_type>(0),
                                                                    url_decoder_fn{d_strings} );
    auto offsets_column = make_offsets_child_column(offsets_transformer_itr,
                                                    offsets_transformer_itr+strings_count,
                                                    mr, stream);
    auto d_offsets = offsets_column->view().data<int32_t>();

    // build chars column
    auto chars_column = create_chars_child_column( strings_count, strings.null_count(),
                                                   thrust::device_pointer_cast(d_offsets)[strings_count],
                                                   mr, stream );
    auto d_chars = chars_column->mutable_view().data<char>();
    thrust::for_each_n(rmm::exec_policy(stream)->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count,
        url_decoder_fn{d_strings,d_offsets,d_chars});
    //
    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               strings.null_count(), std::move(null_mask), stream, mr);
}

} // namespace detail

// external API

std::unique_ptr<column> url_decode( strings_column_view const& strings,
                                    rmm::mr::device_memory_resource* mr)
{
    return detail::url_decode(strings,mr);
}

} // namespace strings
} // namespace cudf
