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
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>
#include "./utilities.hpp"
#include "./utilities.cuh"

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

namespace cudf
{
namespace strings
{

// new strings column from subset of this strings instance
std::unique_ptr<cudf::column> sublist( strings_column_view strings,
                                       size_type start, size_type end,
                                       size_type step, cudaStream_t stream,
                                       rmm::mr::device_memory_resource* mr  )
{
    if( step <= 0 )
        step = 1;
    size_type count = strings.size();
    if( end < 0 || end > count )
        end = count;
    if( start < 0 || start > end )
        throw std::invalid_argument("invalid start parameter");
    count = (end - start)/step;
    //
    auto execpol = rmm::exec_policy(stream);
    // build indices
    thrust::device_vector<size_type> indices(count);
    thrust::sequence( execpol->on(stream), indices.begin(), indices.end(), start, step );
    // create a column_view as a wrapper of these indices
    column_view indices_view( data_type{INT32}, count, indices.data().get(), nullptr, 0 );
    // build a new strings column from the indices
    return gather(strings, indices_view, stream, mr);
}

// return new strings column with strings from this instance as specified by the indices
std::unique_ptr<cudf::column> gather( strings_column_view strings,
                                      column_view gather_map, cudaStream_t stream,
                                      rmm::mr::device_memory_resource* mr  )
{
    size_type count = gather_map.size();
    auto d_indices = gather_map.data<int32_t>();

    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_column = *strings_column;
    auto d_offsets = strings.offsets().data<int32_t>();

    // build offsets column
    auto offsets_column = make_numeric_column( data_type{INT32}, count+1, mask_state::UNALLOCATED,
                                               stream, mr );
    auto offsets_view = offsets_column->mutable_view();
    auto d_new_offsets = offsets_view.data<int32_t>();
    // fill new offsets array
    // using inclusive-scan to compute last entry which is the total size
    thrust::transform_inclusive_scan( execpol->on(stream),
        d_indices, d_indices + count,
        d_new_offsets+1, // fills in entries [1,count]
        [d_column, d_offsets] __device__ (size_type idx) {
            if( d_column.nullable() && d_column.is_null(idx) )
                return 0;
            return d_offsets[idx+1] - d_offsets[idx];
        },
        thrust::plus<int32_t>());
    // need to set the first entry to 0
    cudaMemsetAsync( d_new_offsets, 0, sizeof(*d_new_offsets), stream);

    // build null mask
    auto valid_mask = valid_if( static_cast<const bit_mask_t*>(nullptr),
        [d_column, d_indices] __device__ (size_type idx) {
            return !d_column.nullable() || !d_column.is_null(d_indices[idx]);
        },
        count, stream );
    auto null_count = valid_mask.second;
    auto null_size = gdf_valid_allocation_size(count);
    rmm::device_buffer null_mask(valid_mask.first,null_size,stream,mr); // does deep copy
    RMM_TRY( RMM_FREE(valid_mask.first,stream) ); // TODO valid_if to return device_buffer in future

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_new_offsets)[count]; // this may not be stream friendly
    if( (bytes==0) && (null_count < count) )
        bytes = 1; // all entries are empty strings
    auto chars_column = make_numeric_column( data_type{INT8}, bytes, mask_state::UNALLOCATED,
                                             stream, mr );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<int8_t>();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), count,
        [d_column, d_indices, d_new_offsets, d_chars] __device__(size_type idx){
            size_type index = d_indices[idx];
            if( d_column.nullable() && d_column.is_null(index) )
                return;
            string_view d_str = d_column.element<string_view>(index);
            memcpy(d_chars + d_new_offsets[idx], d_str.data(), d_str.size_bytes() );
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

// return sorted version of the given strings column
std::unique_ptr<cudf::column> sort( strings_column_view strings,
                                    sort_type stype,
                                    cudf::order order,
                                    cudf::null_order null_order,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource* mr  )
{
    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(strings.parent(), stream);
    auto d_column = *strings_column;

    // sort the indices of the strings
    size_type count = strings.size();
    thrust::device_vector<size_type> indices(count);
    thrust::sequence( execpol->on(stream), indices.begin(), indices.end() );
    thrust::sort( execpol->on(stream), indices.begin(), indices.end(),
        [d_column, stype, order, null_order] __device__ (size_type lhs, size_type rhs) {
            bool lhs_null{d_column.nullable() && d_column.is_null(lhs)};
            bool rhs_null{d_column.nullable() && d_column.is_null(rhs)};
            if( lhs_null || rhs_null )
                return (null_order==cudf::null_order::BEFORE ? !rhs_null : !lhs_null);
            string_view lhs_str = d_column.element<string_view>(lhs);
            string_view rhs_str = d_column.element<string_view>(rhs);
            int cmp = lhs_str.compare(rhs_str);
            return (order==cudf::order::ASCENDING ? (cmp<0) : (cmp>0));
        });

    // create a column_view as a wrapper of these indices
    column_view indices_view( data_type{INT32}, count, indices.data().get(), nullptr, 0 );
    // now build a new strings column from the indices
    return gather( strings, indices_view, stream, mr );
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
    size_type elements = values.size();
    CUDF_EXPECTS( elements==scatter_map.size(), "number of strings must match map size" );
    size_type count = strings.size();
    auto d_indices = scatter_map.data<int32_t>();
    auto execpol = rmm::exec_policy(stream);

    // create strings arrays
    rmm::device_vector<string_view> strings_array =
        detail::create_string_array_from_column(strings,stream);
    string_view* d_strings = strings_array.data().get();
    rmm::device_vector<string_view> values_array =
        detail::create_string_array_from_column(values,stream);
    string_view* d_values = values_array.data().get();
    // do the scatter
    thrust::scatter( execpol->on(stream),
                     d_values, d_values+elements,
                     d_indices, d_strings );

    // build null mask
    auto valid_mask = valid_if( static_cast<const bit_mask_t*>(nullptr),
        [d_strings] __device__ (size_type idx) { return !d_strings[idx].is_null(); },
        count, stream );
    auto null_count = valid_mask.second;
    auto null_size = gdf_valid_allocation_size(count);
    rmm::device_buffer null_mask(valid_mask.first,null_size,stream,mr); // does deep copy
    RMM_TRY( RMM_FREE(valid_mask.first,stream) ); // TODO valid_if to return device_buffer in future

    // build offsets column
    auto offsets_column = detail::offsets_from_string_array(strings_array,stream,mr);
    auto offsets_view = offsets_column->view();
    auto d_offsets = offsets_view.data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[count]; // this may not be stream friendly
    if( (bytes==0) && (null_count < count) )
        bytes = 1; // all entries are empty strings
    auto chars_column = detail::chars_from_string_array(strings_array,d_offsets,null_count,stream,mr);

    // build children vector
    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(offsets_column));
    children.emplace_back(std::move(chars_column));

    // return new strings column
    return std::make_unique<column>(
        data_type{STRING}, count, rmm::device_buffer{0,stream,mr},
        null_mask, null_count,
        std::move(children));
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
    size_type count = strings.size();
    size_type elements = scatter_map.size();
    auto execpol = rmm::exec_policy(0);
    auto d_indices = scatter_map.data<int32_t>();
    // copy string to device
    auto replace = detail::string_from_host(string, stream);
    auto d_replace = *replace;
    // create strings array
    rmm::device_vector<string_view> strings_vector =
        detail::create_string_array_from_column(strings, stream);
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
        count, stream );
    auto null_count = valid_mask.second;
    auto null_size = gdf_valid_allocation_size(count);
    rmm::device_buffer null_mask(valid_mask.first,null_size,stream,mr);
    RMM_TRY( RMM_FREE(valid_mask.first,stream) ); // TODO valid_if to return device_buffer in future

    // build offsets column
    auto offsets_column = detail::offsets_from_string_array(strings_vector,stream,mr);
    auto offsets_view = offsets_column->view();
    auto d_offsets = offsets_view.data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[count];
    if( (bytes==0) && (null_count < count) )
        bytes = 1; // all entries are empty strings
    auto chars_column = detail::chars_from_string_array(strings_vector,d_offsets,null_count,stream,mr);

    // build children vector
    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(offsets_column));
    children.emplace_back(std::move(chars_column));

    // return new strings column
    return std::make_unique<column>(
        data_type{STRING}, count, rmm::device_buffer{0,stream,mr},
        null_mask, null_count,
        std::move(children));
}

} // namespace strings
} // namespace cudf
