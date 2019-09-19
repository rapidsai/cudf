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

#include <cstring>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <utilities/error_utils.hpp>
#include "./utilities.h"

#include <rmm/rmm.h>
#include <thrust/transform_scan.h>
#include <thrust/transform_reduce.h>

namespace cudf 
{
namespace strings
{
namespace detail
{

// Used to build a temporary string_view object from a single host string.
std::unique_ptr<cudf::string_view, std::function<void(cudf::string_view*)>>
    string_from_host( const char* str, cudaStream_t stream )
{
    if( !str )
        return nullptr;
    size_type length = (size_type)std::strlen(str);

    char* d_str;
    RMM_TRY(RMM_ALLOC( &d_str, length, stream ));
    CUDA_TRY(cudaMemcpyAsync( d_str, str, length,
                              cudaMemcpyHostToDevice, stream ));
    CUDA_TRY(cudaStreamSynchronize(stream));

    auto deleter = [](cudf::string_view* sv) { RMM_FREE(const_cast<char*>(sv->data()),0); };
    return std::unique_ptr<cudf::string_view,
        decltype(deleter)>{ new cudf::string_view(d_str,length), deleter};
}

// build an array of string_view objects from a strings column
rmm::device_vector<cudf::string_view> create_string_array_from_column(
    cudf::strings_column_view strings,
    cudaStream_t stream )
{
    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_column = *strings_column;

    auto count = strings.size();
    rmm::device_vector<cudf::string_view> strings_array(count);
    cudf::string_view* d_strings = strings_array.data().get();
    thrust::for_each_n( execpol->on(stream), 
        thrust::make_counting_iterator<size_type>(0), count,
        [d_column, d_strings] __device__ (size_type idx) {
            if( d_column.nullable() && d_column.is_null(idx) )
                d_strings[idx] = cudf::string_view(nullptr,0);
            else
                d_strings[idx] = d_column.element<cudf::string_view>(idx);
        });
    return strings_array;
}

// build a strings offsets column from an array of string_views
std::unique_ptr<cudf::column> offsets_from_string_array(
    const rmm::device_vector<cudf::string_view>& strings,
    cudaStream_t stream, rmm::mr::device_memory_resource* mr )
{
    size_type count = strings.size();
    auto d_strings = strings.data().get();
    auto execpol = rmm::exec_policy(stream);
    auto offsets_column = make_numeric_column( data_type{INT32}, count,
                                               mask_state::UNALLOCATED,
                                               stream, mr );
    auto offsets_view = offsets_column->mutable_view();
    auto d_offsets = offsets_view.data<int32_t>();
    // create new offsets array
    thrust::transform_inclusive_scan( execpol->on(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(count),
        d_offsets,
        [d_strings] __device__ (size_type idx) {
            return d_strings[idx].size();
        },
        thrust::plus<int32_t>());

    return offsets_column;
}

// build a strings chars column from an array of string_views
std::unique_ptr<cudf::column> chars_from_string_array(
    const rmm::device_vector<cudf::string_view>& strings,
    const int32_t* d_offsets,
    cudaStream_t stream, rmm::mr::device_memory_resource* mr )
{
    size_type count = strings.size();
    auto d_strings = strings.data().get();
    auto execpol = rmm::exec_policy(stream);
    auto size_fn = [d_strings] __device__ (size_type idx) { return d_strings[idx].size(); };
    size_type bytes = thrust::device_pointer_cast(d_offsets)[count-1];
    // create column
    auto chars_column = make_numeric_column( data_type{INT8}, bytes,
                                             mask_state::UNALLOCATED,
                                             stream, mr );
    // get it's view
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<int8_t>();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), count,
        [d_strings, d_offsets, d_chars] __device__(size_type idx){
            string_view d_str = d_strings[idx];
            if( d_str.is_null() )
                return;
            size_type offset = (idx ? d_offsets[idx-1] : 0);
            memcpy(d_chars + offset, d_str.data(), d_str.size() );
        });

    return chars_column;
}

} // namespace detail
} // namespace strings
} // namespace cudf