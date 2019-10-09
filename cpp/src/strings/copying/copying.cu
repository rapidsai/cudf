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
#include <cudf/strings/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include "../utilities.hpp"
#include "../utilities.cuh"

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>
#include <thrust/scatter.h>
#include <thrust/transform_scan.h>

namespace cudf
{
namespace strings
{
namespace detail
{

// new strings column from subset of this strings instance
std::unique_ptr<cudf::column> slice( strings_column_view strings,
                                     size_type start, size_type end,
                                     size_type step, cudaStream_t stream,
                                     rmm::mr::device_memory_resource* mr  )
{
    size_type num_strings = strings.size();
    CUDF_EXPECTS( num_strings > 0, "Column has no strings.");
    if( step == 0 )
        step = 1;
    CUDF_EXPECTS( step > 0, "Parameter step must be positive integer.");
    if( end < 0 || end > num_strings )
        end = num_strings;
    CUDF_EXPECTS( ((start >= 0) && (start < end)), "Invalid start parameter value.");
    num_strings = cudf::util::round_up_safe<size_type>((end - start),step);
    //
    auto execpol = rmm::exec_policy(stream);
    // build indices
    thrust::device_vector<size_type> indices(num_strings);
    thrust::sequence( execpol->on(stream), indices.begin(), indices.end(), start, step );
    // create a column_view as a wrapper of these indices
    column_view indices_view( data_type{INT32}, num_strings, indices.data().get(), nullptr, 0 );
    // build a new strings column from the indices
    return gather(strings, indices_view, stream, mr);
}

// return new strings column with strings from this instance as specified by the indices
std::unique_ptr<cudf::column> gather( strings_column_view strings,
                                      column_view gather_map, cudaStream_t stream,
                                      rmm::mr::device_memory_resource* mr  )
{
    CUDF_EXPECTS( strings.size() > 0, "Column has no strings.");
    auto num_strings = gather_map.size();
    // TODO use index-normalizing iterator to allow any numeric type for gather_map
    CUDF_EXPECTS( gather_map.type().id()==cudf::INT32, "strings gather method only supports int32 indices right now");
    auto d_indices = gather_map.data<int32_t>();

    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_column = *strings_column;
    auto d_offsets = strings.offsets().data<int32_t>();

    // build offsets column
    auto offsets_transformer = [d_column, d_offsets] __device__ (size_type idx) {
            if( d_column.nullable() && d_column.is_null(idx) )
                return 0;
            return d_offsets[idx+1] - d_offsets[idx];
        };
    auto offsets_transformer_itr = thrust::make_transform_iterator( d_indices, offsets_transformer );
    auto offsets_column = detail::make_offsets(offsets_transformer_itr,
                                               offsets_transformer_itr+num_strings,
                                               mr, stream);
    auto offsets_view = offsets_column->mutable_view();
    auto d_new_offsets = offsets_view.data<int32_t>();

    // build null mask
    auto valid_mask = valid_if( static_cast<const bit_mask_t*>(nullptr),
        [d_column, d_indices] __device__ (size_type idx) {
            return !d_column.nullable() || !d_column.is_null(d_indices[idx]);
        },
        num_strings, stream );
    auto null_count = valid_mask.second;
    auto null_size = gdf_valid_allocation_size(num_strings);
    rmm::device_buffer null_mask(valid_mask.first,null_size,stream,mr); // does deep copy
    RMM_TRY( RMM_FREE(valid_mask.first,stream) ); // TODO valid_if to return device_buffer in future

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_new_offsets)[num_strings]; // this may not be stream friendly
    if( (bytes==0) && (null_count < num_strings) )
        bytes = 1; // all entries are empty strings
    auto chars_column = make_numeric_column( data_type{INT8}, bytes, mask_state::UNALLOCATED,
                                             stream, mr );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<int8_t>();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), num_strings,
        [d_column, d_indices, d_new_offsets, d_chars] __device__(size_type idx){
            size_type index = d_indices[idx];
            if( d_column.nullable() && d_column.is_null(index) )
                return;
            string_view d_str = d_column.element<string_view>(index);
            memcpy(d_chars + d_new_offsets[idx], d_str.data(), d_str.size_bytes() );
        });

    return make_strings_column(num_strings, std::move(offsets_column), std::move(chars_column),
                               null_count, std::move(null_mask), stream, mr);
}


//
// s1 = ['a','b,'c','d']
// s2 = ['e','f']
// pos = [1,3]  -- must be the same length as s2
// s3 = s1.scatter(s2,pos)
// ['a','e','c','f']
//
std::unique_ptr<cudf::column> scatter( strings_column_view strings,
                                       strings_column_view values,
                                       cudf::column_view scatter_map,
                                       cudaStream_t stream,
                                       rmm::mr::device_memory_resource* mr )
{
    size_type num_strings = strings.size();
    CUDF_EXPECTS( num_strings > 0, "Column has no strings.");
    size_type elements = values.size();
    CUDF_EXPECTS( elements==scatter_map.size(), "number of values must match map size" );
    // TODO use index-normalizing iterator to allow any numeric type for gather_map
    CUDF_EXPECTS( scatter_map.type().id()==cudf::INT32, "strings scatter method only supports int32 indices right now");
    auto d_indices = scatter_map.data<int32_t>();
    auto execpol = rmm::exec_policy(stream);

    // create strings vector
    rmm::device_vector<string_view> strings_vector =
        detail::create_string_vector_from_column(strings,stream);
    string_view* d_strings = strings_vector.data().get();
    rmm::device_vector<string_view> values_vector =
        detail::create_string_vector_from_column(values,stream);
    string_view* d_values = values_vector.data().get();
    // do the scatter
    thrust::scatter( execpol->on(stream),
                     d_values, d_values+elements,
                     d_indices, d_strings );

    // build null mask
    auto valid_mask = valid_if( static_cast<const bit_mask_t*>(nullptr),
        [d_strings] __device__ (size_type idx) { return !d_strings[idx].is_null(); },
        num_strings, stream );
    auto null_count = valid_mask.second;
    auto null_size = gdf_valid_allocation_size(num_strings);
    rmm::device_buffer null_mask(valid_mask.first,null_size,stream,mr); // does deep copy
    RMM_TRY( RMM_FREE(valid_mask.first,stream) ); // TODO valid_if to return device_buffer in future

    // build offsets column
    auto offsets_column = detail::offsets_from_string_vector(strings_vector,stream,mr);
    auto offsets_view = offsets_column->view();
    auto d_offsets = offsets_view.data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[num_strings]; // this may not be stream friendly
    if( (bytes==0) && (null_count < num_strings) )
        bytes = 1; // all entries are empty strings
    auto chars_column = detail::chars_from_string_vector(strings_vector,d_offsets,null_count,stream,mr);

    return make_strings_column(num_strings, std::move(offsets_column), std::move(chars_column),
                               null_count, std::move(null_mask), stream, mr);
}

//
// s1 = ['a','b,'c','d']
// pos = [1,3]
// s3 = s1.scatter('e',pos,2)
// ['a','e','c','e']
//
std::unique_ptr<cudf::column> scatter( strings_column_view strings,
                                       const char* string,
                                       cudf::column_view scatter_map,
                                       cudaStream_t stream,
                                       rmm::mr::device_memory_resource* mr )
{
    size_type num_strings = strings.size();
    CUDF_EXPECTS( num_strings > 0, "Column has no strings.");
    size_type elements = scatter_map.size();
    auto execpol = rmm::exec_policy(0);
    // TODO use index-normalizing iterator to allow any numeric type for gather_map
    CUDF_EXPECTS( scatter_map.type().id()==cudf::INT32, "strings scatter method only supports int32 indices right now");
    auto d_indices = scatter_map.data<int32_t>();
    // copy string to device
    auto replace = detail::string_from_host(string, stream);
    auto d_replace = *replace;
    // create strings vector
    rmm::device_vector<string_view> strings_vector =
        detail::create_string_vector_from_column(strings, stream);
    auto d_strings = strings_vector.data().get();
    // replace specific elements
    thrust::for_each_n(execpol->on(0),
        thrust::make_counting_iterator<unsigned int>(0), elements,
        [d_indices, d_replace, d_strings] __device__ (unsigned int idx) {
            d_strings[d_indices[idx]] = d_replace;
        });

    // create strings column
    // build null mask
    auto valid_mask = valid_if( static_cast<const bit_mask_t*>(nullptr),
        [d_strings] __device__ (size_type idx) { return !d_strings[idx].is_null(); },
        num_strings, stream );
    auto null_count = valid_mask.second;
    auto null_size = gdf_valid_allocation_size(num_strings);
    rmm::device_buffer null_mask(valid_mask.first,null_size,stream,mr);
    RMM_TRY( RMM_FREE(valid_mask.first,stream) ); // TODO valid_if to return device_buffer in future

    // build offsets column
    auto offsets_column = detail::offsets_from_string_vector(strings_vector,stream,mr);
    auto offsets_view = offsets_column->view();
    auto d_offsets = offsets_view.data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[num_strings];
    if( (bytes==0) && (null_count < num_strings) )
        bytes = 1; // all entries are empty strings
    auto chars_column = detail::chars_from_string_vector(strings_vector,d_offsets,null_count,stream,mr);

    return make_strings_column(num_strings, std::move(offsets_column), std::move(chars_column),
                               null_count, std::move(null_mask), stream, mr);
}

} // namespace detail
} // namespace strings
} // namespace cudf
