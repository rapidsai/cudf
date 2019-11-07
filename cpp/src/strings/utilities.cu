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
#include <cudf/utilities/error.hpp>
#include "./utilities.hpp"
#include "./utilities.cuh"
#include "char_types/char_flags.h"

#include <mutex>
#include <rmm/rmm.h>
#include <rmm/rmm_api.h>
#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform_scan.h>
#include <thrust/transform_reduce.h>

namespace cudf
{
namespace strings
{
namespace detail
{

// Used to build a temporary string_view object from a single host string.
std::unique_ptr<string_view, std::function<void(string_view*)>>
    string_from_host( const char* str, cudaStream_t stream )
{
    if( !str )
        return nullptr;
    auto length = std::strlen(str);

    char* d_str{};
    RMM_TRY(RMM_ALLOC( &d_str, length, stream ));
    CUDA_TRY(cudaMemcpyAsync( d_str, str, length,
                              cudaMemcpyHostToDevice, stream ));
    CUDA_TRY(cudaStreamSynchronize(stream));

    auto deleter = [](string_view* sv) { RMM_FREE(const_cast<char*>(sv->data()),0); };
    return std::unique_ptr<string_view,
        decltype(deleter)>{ new string_view(d_str,length), deleter};
}

// build a vector of string_view objects from a strings column
rmm::device_vector<string_view> create_string_vector_from_column(
    cudf::strings_column_view strings,
    cudaStream_t stream )
{
    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_column = *strings_column;

    auto count = strings.size();
    rmm::device_vector<string_view> strings_vector(count);
    string_view* d_strings = strings_vector.data().get();
    thrust::for_each_n( execpol->on(stream),
        thrust::make_counting_iterator<size_type>(0), count,
        [d_column, d_strings] __device__ (size_type idx) {
            if( d_column.is_null(idx) )
                d_strings[idx] = string_view(nullptr,0);
            else
                d_strings[idx] = d_column.element<string_view>(idx);
        });
    return strings_vector;
}

// build a strings offsets column from a vector of string_views
std::unique_ptr<cudf::column> offsets_from_string_vector(
    const rmm::device_vector<string_view>& strings,
    cudaStream_t stream, rmm::mr::device_memory_resource* mr )
{
    auto transformer = [] __device__(string_view v) { return v.size_bytes(); };
    auto begin = thrust::make_transform_iterator(strings.begin(), transformer);
    return make_offsets_child_column(begin, begin + strings.size(), mr, stream);
}

// build a strings chars column from an vector of string_views
std::unique_ptr<cudf::column> chars_from_string_vector(
    const rmm::device_vector<string_view>& strings,
    const int32_t* d_offsets, cudf::size_type null_count,
    cudaStream_t stream, rmm::mr::device_memory_resource* mr )
{
    size_type count = strings.size();
    auto d_strings = strings.data().get();
    auto execpol = rmm::exec_policy(stream);
    size_type bytes = thrust::device_pointer_cast(d_offsets)[count];
    if( (bytes==0) && (null_count < count) )
        bytes = 1; // all entries are empty strings

    // create column
    auto chars_column = make_numeric_column( data_type{INT8}, bytes,
                                             mask_state::UNALLOCATED,
                                             stream, mr );
    // get it's view
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<int8_t>();
    thrust::for_each_n(execpol->on(stream),
        thrust::make_counting_iterator<size_type>(0), count,
        [d_strings, d_offsets, d_chars] __device__(size_type idx){
            string_view d_str = d_strings[idx];
            if( !d_str.is_null() )
                memcpy(d_chars + d_offsets[idx], d_str.data(), d_str.size_bytes() );
        });

    return chars_column;
}

//
std::unique_ptr<column> create_chars_child_column( cudf::size_type strings_count,
    cudf::size_type null_count, cudf::size_type total_bytes,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
    CUDF_EXPECTS(null_count <= strings_count, "Invalid null count");
    // If we have all nulls, a null chars column is allowed.
    // If all non-null strings are empty strings, we need a non-null chars column.
    // In this case we set the bytes to 1 to create a minimal one-byte chars column.
    if( (total_bytes==0) && (null_count < strings_count) )
        total_bytes = 1; // all entries are empty strings (not nulls)
    // return chars column
    return make_numeric_column( data_type{INT8}, total_bytes, mask_state::UNALLOCATED, stream, mr );
}

//
std::unique_ptr<column> make_empty_strings_column( rmm::mr::device_memory_resource* mr, cudaStream_t stream )
{
    return std::make_unique<column>( data_type{STRING}, 0,
                                     rmm::device_buffer{0,stream,mr}, // data
                                     rmm::device_buffer{0,stream,mr}, 0 ); // nulls
}

namespace
{

// This device variable is created here to avoid using a singleton that may cause issues
// with RMM initialize/finalize. See PR #3159 for details on this approach.
__device__ character_flags_table_type character_codepoint_flags[sizeof(g_character_codepoint_flags)];
std::mutex g_flags_table_mutex;
character_flags_table_type* d_character_codepoint_flags = nullptr;

} // namespace

// Return the flags table device pointer
const character_flags_table_type* get_character_flags_table()
{
    std::lock_guard<std::mutex> guard(g_flags_table_mutex);
    if( !d_character_codepoint_flags )
    {
        CUDA_TRY(cudaMemcpyToSymbol(character_codepoint_flags, g_character_codepoint_flags, sizeof(g_character_codepoint_flags)));
        CUDA_TRY(cudaGetSymbolAddress((void**)&d_character_codepoint_flags,character_codepoint_flags));
    }
    return d_character_codepoint_flags;
}

} // namespace detail
} // namespace strings
} // namespace cudf
