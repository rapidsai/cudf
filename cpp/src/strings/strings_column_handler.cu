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

#include <cudf/strings/strings_column_handler.hpp>
#include <cudf/strings/strings_column_factories.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/column/column_device_view.cuh>
#include <utilities/error_utils.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/for_each.h>

namespace cudf {

#define STR_OFFSETS_CHILD_INDEX 0
#define STR_CHARS_CHILD_INDEX 1

//
strings_column_handler::strings_column_handler( const column_view& strings_column )
    : _parent(strings_column)
{
    CUDF_EXPECTS( _parent.type().id()==STRING, "string_column_view only support strings");
    CUDF_EXPECTS( _parent.num_children()>0, "string column must have children");
}

size_type strings_column_handler::count() const
{
    return _parent.child(STR_OFFSETS_CHILD_INDEX).size();
}

const char* strings_column_handler::chars_data() const
{
    return _parent.child(STR_CHARS_CHILD_INDEX).data<char>();
}

const int32_t* strings_column_handler::offsets_data() const
{
    return _parent.child(STR_OFFSETS_CHILD_INDEX).data<int32_t>();
}

size_type strings_column_handler::chars_column_size() const
{
    return _parent.child(STR_OFFSETS_CHILD_INDEX).size();
}

const bitmask_type* strings_column_handler::null_mask() const
{
    return _parent.null_mask();
}

size_type strings_column_handler::null_count() const
{
    return _parent.null_count();
}

// print strings to stdout
void strings_column_handler::print( size_type start, size_type end,
                                    size_type max_width, const char* delimiter ) const
{
    size_type count = this->count();
    if( end < 0 || end > count )
        end = count;
    if( start < 0 )
        start = 0;
    if( start >= end )
        return;
    count = end - start;

    // stick with the default stream for this odd/rare stdout function
    auto execpol = rmm::exec_policy(0);
    auto strings_column = column_device_view(_parent);
    auto d_offsets = offsets_data();
    auto d_strings = chars_data();

    // get individual strings sizes
    rmm::device_vector<size_t> output_offsets(count,0);
    thrust::transform( execpol->on(0),
        thrust::make_counting_iterator<size_type>(start), thrust::make_counting_iterator<size_type>(end),
        output_offsets.begin(),
        [strings_column, d_strings, max_width, d_offsets] __device__ (size_type idx) {
            if( strings_column.nullable() && strings_column.is_null(idx) )
                return 0;
            size_type offset = idx ? d_offsets[idx-1] : 0; // this logic will be a template
            size_type bytes = d_offsets[idx] - offset;     // specialization on element()
            string_view dstr( d_strings + offset, bytes ); // method of column_device_view
            if( (max_width > 0) && (dstr.characters() > max_width) )
                bytes = dstr.byte_offset_for(max_width);
            return bytes+1; // allow for null-terminator on non-null strings
        });
    // convert to offsets
    thrust::inclusive_scan( execpol->on(0), output_offsets.begin(), output_offsets.end(), output_offsets.begin() );
    // build output buffer
    size_t buffer_size = output_offsets[count-1];
    if( buffer_size == 0 )
    {
        printf("all %d strings are null\n", count);
        return;
    }
    rmm::device_vector<char> buffer(buffer_size,0); // allocate and pre-null-terminate
    char* d_buffer = buffer.data().get();
    // copy strings into output buffer
    size_t* d_output_offsets = output_offsets.data().get();
    thrust::for_each_n(execpol->on(0),
        thrust::make_counting_iterator<size_type>(0), (end-start),
        [d_strings, start, d_offsets, d_output_offsets, d_buffer] __device__(size_type idx) {
            size_t output_offset = (idx ? d_output_offsets[idx-1] : 0);
            size_t length = d_output_offsets[idx] - output_offset; // bytes
            if( length ) // this is only 0 for nulls
            {
                idx += start;
                size_type offset = (idx ? d_offsets[idx-1]:0);
                memcpy(d_buffer + output_offset, d_strings + offset, length-1 );
            }
        });

    // copy output buffer to host
    std::vector<size_t> h_offsets(count);
    cudaMemcpyAsync( h_offsets.data(), d_output_offsets, count*sizeof(size_t), cudaMemcpyDeviceToHost);
    std::vector<char> h_buffer(buffer_size);
    cudaMemcpyAsync( h_buffer.data(), d_buffer, buffer_size, cudaMemcpyDeviceToHost );
    cudaStreamSynchronize(0);

    // print out the strings to stdout
    for( size_type idx=0; idx < count; ++idx )
    {
        size_t offset = (idx ? h_offsets[idx-1]:0);
        size_t length = h_offsets[idx] - offset;
        printf("%d:",idx);
        if( length )
            printf("[%s]", h_buffer.data()+offset);
        else
            printf("<null>");
        printf("%s",delimiter);
    }
}

// new strings column from subset of given strings column
std::unique_ptr<cudf::column> strings_column_handler::sublist( size_type start, size_type end, size_type step )
{
    return make_strings_column(nullptr, 0);
}

// return sorted version of the given strings column
std::unique_ptr<cudf::column> strings_column_handler::sort( sort_type stype, bool ascending, bool nullfirst )
{
    return make_strings_column(nullptr, 0);
}

// return sorted indexes only -- returns integer column
std::unique_ptr<cudf::column> strings_column_handler::order( sort_type stype, bool ascending, bool nullfirst )
{
    return make_strings_column(nullptr, 0);
}

}  // namespace cudf