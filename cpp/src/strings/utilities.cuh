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
#pragma once

#include <bitmask/legacy/valid_if.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/detail/utilities.cuh>

#include <rmm/device_buffer.hpp>

#include <cstring>

namespace cudf
{
namespace strings
{
namespace detail
{

/**
 * @brief Copies input string data into a buffer and increments the pointer by the number of bytes copied.
 *
 * @param buffer Device buffer to copy to.
 * @param input Data to copy from.
 * @param bytes Number of bytes to copy.
 * @return Pointer to the end of the output buffer after the copy.
 */
__device__ inline char* copy_and_increment( char* buffer, const char* input, size_type bytes )
{
    memcpy( buffer, input, bytes );
    return buffer + bytes;
}

/**
 * @brief Copies input string data into a buffer and increments the pointer by the number of bytes copied.
 *
 * @param buffer Device buffer to copy to.
 * @param d_string String to copy.
 * @return Pointer to the end of the output buffer after the copy.
 */
__device__ inline char* copy_string( char* buffer, const string_view& d_string )
{
    return copy_and_increment( buffer, d_string.data(), d_string.size_bytes() );
}


/**
 * @brief Utility to create a null mask for a strings column using a custom function.
 *
 * @tparam BoolFn Function should return true/false given index for a strings column.
 * @param strings_count Number of strings for the column.
 * @param bfn The custom function used for identifying null string entries.
 * @param mr Memory resource to use.
 * @param stream Stream to use for any kernel calls.
 * @return Pair including null mask and null count
 */
template <typename BoolFn>
std::pair<rmm::device_buffer,cudf::size_type> make_null_mask( cudf::size_type strings_count,
    BoolFn bfn,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0)
{
    auto valid_mask = valid_if( static_cast<const bit_mask_t*>(nullptr),
                                bfn, strings_count, stream );
    auto null_count = valid_mask.second;
    rmm::device_buffer null_mask;
    if( null_count > 0 )
        null_mask = rmm::device_buffer(valid_mask.first,
                                       gdf_valid_allocation_size(strings_count),
                                       stream,mr); // does deep copy
    RMM_TRY( RMM_FREE(valid_mask.first,stream) ); // TODO valid_if to return device_buffer in future
    return std::make_pair(std::move(null_mask), null_count);
}

/**
 * @brief Converts a single UTF-8 character into a code-point value that
 * can be used for lookup in the character flags or the character case tables.
 *
 * @param utf8_char Single UTF-8 character to convert.
 * @return Code-point for the UTF-8 character.
 */
__device__ inline uint32_t utf8_to_codepoint(cudf::char_utf8 utf8_char)
{
    uint32_t unchr = 0;
    if( utf8_char < 0x00000080 ) // single-byte pass thru
        unchr = utf8_char;
    else if( utf8_char < 0x0000E000 ) // two bytes
    {
        unchr =  (utf8_char & 0x1F00) >> 2; // shift and
        unchr |= (utf8_char & 0x003F);      // unmask
    }
    else if( utf8_char < 0x00F00000 ) // three bytes
    {
        unchr =  (utf8_char & 0x0F0000) >> 4;  // get upper 4 bits
        unchr |= (utf8_char & 0x003F00) >> 2;  // shift and
        unchr |= (utf8_char & 0x00003F);       // unmask
    }
    else if( utf8_char <= (unsigned)0xF8000000 ) // four bytes
    {
        unchr =  (utf8_char & 0x03000000) >> 6; // upper 3 bits
        unchr |= (utf8_char & 0x003F0000) >> 4; // next 6 bits
        unchr |= (utf8_char & 0x00003F00) >> 2; // next 6 bits
        unchr |= (utf8_char & 0x0000003F);      // unmask
    }
    return unchr;
}

/**
 * @brief Converts a character code-point value into a UTF-8 character.
 *
 * @param unchr Character code-point to convert.
 * @return Single UTF-8 character.
 */
__host__ __device__ inline cudf::char_utf8 codepoint_to_utf8( uint32_t unchr )
{
    cudf::char_utf8 utf8 = 0;
    if( unchr < 0x00000080 ) // single byte utf8
        utf8 = unchr;
    else if( unchr < 0x00000800 )  // double byte utf8
    {
        utf8 =  (unchr << 2) & 0x1F00;  // shift bits for
        utf8 |= (unchr & 0x3F);         // utf8 encoding
        utf8 |= 0x0000C080;
    }
    else if( unchr < 0x00010000 )  // triple byte utf8
    {
        utf8 =  (unchr << 4) & 0x0F0000;  // upper 4 bits
        utf8 |= (unchr << 2) & 0x003F00;  // next 6 bits
        utf8 |= (unchr & 0x3F);           // last 6 bits
        utf8 |= 0x00E08080;
    }
    else if( unchr < 0x00110000 )  // quadruple byte utf8
    {
        utf8 =  (unchr << 6) & 0x07000000;  // upper 3 bits
        utf8 |= (unchr << 4) & 0x003F0000;  // next 6 bits
        utf8 |= (unchr << 2) & 0x00003F00;  // next 6 bits
        utf8 |= (unchr & 0x3F);             // last 6 bits
        utf8 |= (unsigned)0xF0808080;
    }
    return utf8;
}


} // namespace detail
} // namespace strings
} // namespace cudf
