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

#include <bitmask/valid_if.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <utilities/error_utils.hpp>
#include "./utilities.hpp"
#include "./utilities.cuh"

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform_scan.h>

namespace cudf
{
namespace strings
{

std::unique_ptr<cudf::column> concatenate( strings_column_view strings,
                                           strings_column_view others,
                                           const char* separator,
                                           const char* narep,
                                           cudaStream_t stream,
                                           rmm::mr::device_memory_resource* mr )
{
    CUDF_EXPECTS( strings.size()==others.size(), "columns must be the same size");

    auto execpol = rmm::exec_policy(stream);
    size_type count = strings.size();

    if( !separator )
        separator = "";
    auto separator_ptr = detail::string_from_host(separator, stream);
    auto d_separator = *separator_ptr;
    auto narep_ptr = detail::string_from_host(narep, stream);
    string_view d_narep(nullptr,0);
    if( narep_ptr )
        d_narep = *narep_ptr;

    // create strings arrays
    auto strings_column_ptr = column_device_view::create(strings.parent(),stream);
    auto d_strings = *strings_column_ptr;
    auto others_column_ptr = column_device_view::create(others.parent(),stream);
    auto d_others = *others_column_ptr;

    // create resulting null mask
    auto valid_mask = valid_if( static_cast<const bit_mask_t*>(nullptr),
        [d_strings, d_others, d_narep] __device__ (size_type idx) {
            return !(((d_strings.nullable() && d_strings.is_null(idx)) ||
                     (d_others.nullable() && d_others.is_null(idx))) &&
                      d_narep.is_null());
        },
        count, stream );
    auto null_count = valid_mask.second;
    auto null_size = gdf_valid_allocation_size(count);
    rmm::device_buffer null_mask(valid_mask.first,null_size,stream,mr); // does deep copy
    RMM_TRY( RMM_FREE(valid_mask.first,stream) ); // TODO valid_if to return device_buffer in future

    // build offsets column
    auto offsets_column = make_numeric_column( data_type{INT32}, count+1, mask_state::UNALLOCATED,
                                               stream, mr );
    auto offsets_view = offsets_column->mutable_view();
    auto d_results_offsets = offsets_view.data<int32_t>();
    // compute offsets
    thrust::transform_inclusive_scan( execpol->on(stream),
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(count),
        d_results_offsets+1,
        [d_strings, d_others, d_separator, d_narep] __device__ (size_type idx) {
            string_view d_str1;
            if( d_strings.nullable() && d_strings.is_null(idx) )
                d_str1 = string_view(nullptr,0);
            else
                d_str1 = d_strings.element<string_view>(idx);
            string_view d_str2;
            if( d_others.nullable() && d_others.is_null(idx) )
                d_str2 = string_view(nullptr,0);
            else
                d_str2 = d_others.element<string_view>(idx);
            if( (d_str1.is_null() || d_str2.is_null()) && d_narep.is_null() )
                return 0; // null output case
            size_type bytes = 0;
            // left-side
            if( !d_str1.is_null() )
                bytes = d_str1.size_bytes();
            else if( !d_narep.is_null() )
                bytes = d_narep.size_bytes();
            // separator
            bytes += d_separator.size_bytes();
            if( !d_str2.is_null() )
                bytes += d_str2.size_bytes();
            else if( !d_narep.is_null() )
                bytes += d_narep.size_bytes();
            return bytes;
        },
        thrust::plus<int32_t>() );
    cudaMemsetAsync( d_results_offsets, 0, sizeof(*d_results_offsets), stream);

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_results_offsets)[count];
    if( (bytes==0) && (null_count < count) )
        bytes = 1; // all entries are empty strings
    auto chars_column = make_numeric_column( data_type{INT8}, bytes, mask_state::UNALLOCATED,
                                             stream, mr );
    auto chars_view = chars_column->mutable_view();
    auto d_results_chars = chars_view.data<char>();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), count,
        [d_strings, d_others, d_separator, d_narep, d_results_offsets, d_results_chars] __device__(size_type idx){
            string_view d_str1;
            if( d_strings.nullable() && d_strings.is_null(idx) )
                d_str1 = string_view(nullptr,0);
            else
                d_str1 = d_strings.element<string_view>(idx);
            string_view d_str2;
            if( d_others.nullable() && d_others.is_null(idx) )
                d_str2 = string_view(nullptr,0);
            else
                d_str2 = d_others.element<string_view>(idx);
            if( (d_str1.is_null() || d_str2.is_null()) && d_narep.is_null() )
                return; // null -- nothing to do
            // concat the two strings with appropriate separator and narep
            size_type offset = d_results_offsets[idx];
            char* d_buffer = d_results_chars + offset;
            if( !d_str1.is_null() )
                d_buffer = detail::copy_string(d_buffer, d_str1);
            else if( !d_narep.is_null() )
                d_buffer = detail::copy_string(d_buffer, d_narep);
            if( !d_separator.is_null() )
                d_buffer = detail::copy_string(d_buffer, d_separator);
            if( !d_str2.is_null() )
                d_buffer = detail::copy_string(d_buffer, d_str2);
            else if( !d_narep.is_null() )
                d_buffer = detail::copy_string(d_buffer, d_narep);
        });

    // build children vector
    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(offsets_column));
    children.emplace_back(std::move(chars_column));

    return std::make_unique<column>(
        data_type{STRING}, count, rmm::device_buffer{0,stream,mr},
        null_mask, null_count,
        std::move(children));
}

} // namespace strings
} // namespace cudf
