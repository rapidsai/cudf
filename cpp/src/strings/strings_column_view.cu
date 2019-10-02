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
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <utilities/error_utils.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/for_each.h>
#include <thrust/transform_scan.h>

namespace cudf {

//
strings_column_view::strings_column_view( column_view strings_column )
    : _parent(strings_column)
{
    CUDF_EXPECTS( _parent.type().id()==STRING, "strings_column_view only supports strings");
    CUDF_EXPECTS( _parent.num_children()>0, "strings column must have children"); // revisit this (all nulls column?)
}

size_type strings_column_view::size() const
{
    return _parent.size();
}

column_view strings_column_view::parent() const
{
    return _parent;
}

column_view strings_column_view::offsets() const
{
    return _parent.child(0);
}

column_view strings_column_view::chars() const
{
    return _parent.child(1);
}

const bitmask_type* strings_column_view::null_mask() const
{
    return _parent.null_mask();
}

size_type strings_column_view::null_count() const
{
    return _parent.null_count();
}

namespace strings
{

// print strings to stdout
void print( strings_column_view strings,
            size_type start, size_type end,
            size_type max_width, const char* delimiter )
{
    size_type count = strings.size();
    if( end < 0 || end > count )
        end = count;
    if( start < 0 )
        start = 0;
    if( start >= end )
        throw std::invalid_argument("invalid parameter value");
    count = end - start;

    // stick with the default stream for this odd/rare stdout function
    auto execpol = rmm::exec_policy(0);
    auto strings_column = column_device_view::create(strings.parent());
    auto d_column = *strings_column;
    auto d_offsets = strings.offsets().data<int32_t>();
    auto d_strings = strings.chars().data<char>();

    // create output strings offsets
    rmm::device_vector<size_t> output_offsets(count+1,0);
    size_t* d_output_offsets = output_offsets.data().get();
    thrust::transform_inclusive_scan( execpol->on(0),
        thrust::make_counting_iterator<size_type>(start),
        thrust::make_counting_iterator<size_type>(end),
        d_output_offsets+1,
        [d_column, max_width] __device__ (size_type idx) {
            if( d_column.nullable() && d_column.is_null(idx) )
                return 0;
            string_view d_str = d_column.element<string_view>(idx);
            size_type bytes = d_str.size_bytes();
            if( (max_width > 0) && (d_str.length() > max_width) )
                bytes = d_str.byte_offset(max_width);
            return bytes+1; // allow for null-terminator on non-null strings
        },
        thrust::plus<int32_t>());
    cudaMemset( d_output_offsets, 0, sizeof(*d_output_offsets));

    // build output buffer
    size_t buffer_size = output_offsets.back(); // last element has total size
    if( buffer_size == 0 )
    {
        printf("all %d strings are null\n", count);
        return;
    }
    rmm::device_vector<char> buffer(buffer_size,0); // allocate and pre-null-terminate
    char* d_buffer = buffer.data().get();
    // copy strings into output buffer
    thrust::for_each_n(execpol->on(0),
        thrust::make_counting_iterator<size_type>(0), count,
        [d_strings, start, d_offsets, d_output_offsets, d_buffer] __device__(size_type idx) {
            size_t output_offset = d_output_offsets[idx];
            size_t length = d_output_offsets[idx+1] - output_offset; // bytes
            if( length ) // this is only 0 for nulls
            {
                idx += start;
                size_type offset = d_offsets[idx];
                memcpy(d_buffer + output_offset, d_strings + offset, length-1 );
            }
        });

    // copy output buffer to host
    std::vector<size_t> h_offsets(count+1);
    cudaMemcpy( h_offsets.data(), d_output_offsets, (count+1)*sizeof(size_t), cudaMemcpyDeviceToHost);
    std::vector<char> h_buffer(buffer_size);
    cudaMemcpy( h_buffer.data(), d_buffer, buffer_size, cudaMemcpyDeviceToHost );

    // print out the strings to stdout
    for( size_type idx=0; idx < count; ++idx )
    {
        size_t offset = h_offsets[idx];
        size_t length = h_offsets[idx+1] - offset;
        printf("%d:",idx);
        if( length )
            printf("[%s]", h_buffer.data()+offset);
        else
            printf("<null>");
        printf("%s",delimiter);
    }
}

std::pair<rmm::device_vector<char>, rmm::device_vector<size_type>>
    create_offsets( strings_column_view strings,
                    cudaStream_t stream,
                    rmm::mr::device_memory_resource* mr )
{
    std::pair<rmm::device_vector<char>, rmm::device_vector<size_type>> results;

    size_type count = strings.size();
    auto d_offsets = strings.offsets().data<size_type>();
    results.second = rmm::device_vector<size_type>(count+1);
    cudaMemcpyAsync( results.second.data().get(), d_offsets, (count+1)*sizeof(size_type),
                     cudaMemcpyDeviceToHost, stream);

    size_type bytes = thrust::device_pointer_cast(d_offsets)[count];
    auto d_chars = strings.chars().data<char>();
    results.first = rmm::device_vector<char>(bytes);
    cudaMemcpyAsync( results.first.data().get(), d_chars, bytes,
                     cudaMemcpyDeviceToHost, stream);

    return results;
}

} // namespace strings
} // namespace cudf
