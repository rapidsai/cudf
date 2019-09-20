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

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

namespace cudf 
{
namespace strings
{

std::unique_ptr<cudf::column> characters_counts( strings_column_view strings,
                                                 cudaStream_t stream,
                                                 rmm::mr::device_memory_resource* mr )
{
    auto count = strings.size();
    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_column = *strings_column;
    // create output column
    auto result = std::make_unique<cudf::column>( data_type{INT32}, count,
        rmm::device_buffer(count * sizeof(int32_t), stream, mr),
        rmm::device_buffer(d_column.null_mask(), gdf_valid_allocation_size(count),
                           stream, mr),
        d_column.null_count());
    auto results_view = result->mutable_view();
    auto d_lengths = results_view.data<int32_t>();
    // set lengths
    thrust::transform( execpol->on(stream), 
        thrust::make_counting_iterator<int32_t>(0),
        thrust::make_counting_iterator<int32_t>(count),
        d_lengths,
        [d_column] __device__ (int32_t idx) {
            if( d_column.nullable() && d_column.is_null(idx) )
                return 0;
            return d_column.element<string_view>(idx).characters();
        });
    return result;
}

std::unique_ptr<cudf::column> bytes_counts( strings_column_view strings,
                                            cudaStream_t stream,
                                            rmm::mr::device_memory_resource* mr )
{
    auto count = strings.size();
    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_column = *strings_column;
    // create output column
    auto result = std::make_unique<cudf::column>( data_type{INT32}, count,
        rmm::device_buffer(count * sizeof(int32_t), stream, mr),
        rmm::device_buffer(d_column.null_mask(), gdf_valid_allocation_size(count),
                           stream, mr),
        d_column.null_count());
    auto results_view = result->mutable_view();
    auto d_lengths = results_view.data<int32_t>();
    // set sizes
    thrust::transform( execpol->on(stream), 
        thrust::make_counting_iterator<int32_t>(0),
        thrust::make_counting_iterator<int32_t>(count),
        d_lengths,
        [d_column] __device__ (int32_t idx) {
            if( d_column.nullable() && d_column.is_null(idx) )
                return 0;
            return d_column.element<string_view>(idx).size();
        });
    return result;
}

//
//
std::unique_ptr<cudf::column> code_points( strings_column_view strings,
                                           cudaStream_t stream,
                                           rmm::mr::device_memory_resource* mr )
{
    auto count = strings.size();
    auto execpol = rmm::exec_policy(0);
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_column = *strings_column;

    // offsets point to each individual integer range
    rmm::device_vector<size_type> offsets(count);
    size_type* d_offsets = offsets.data().get();
    thrust::transform_inclusive_scan(execpol->on(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(count),
        d_offsets,
        [d_column] __device__(size_type idx){
            if( d_column.nullable() && d_column.is_null(idx) )
                return 0;
            return d_column.element<string_view>(idx).characters();
        },
        thrust::plus<unsigned int>());
    
    // need the total size to build the column
    // the size is the last element from an inclusive-scan
    size_type size = offsets.back();
    // create output column
    auto result = make_numeric_column( data_type{INT32}, size,
                                       mask_state::UNALLOCATED,
                                       stream, mr );
    auto results_view = result->mutable_view();
    auto d_results = results_view.data<int32_t>();
    // now set the ranges from each strings' character values
    thrust::for_each_n(execpol->on(stream),
        thrust::make_counting_iterator<unsigned int>(0), count,
        [d_column, d_offsets, d_results] __device__(unsigned int idx){
            if( d_column.nullable() && d_column.is_null(idx) )
                return;
            auto d_str = d_column.element<string_view>(idx);
            auto result = d_results + (idx ? d_offsets[idx-1] :0);
            thrust::copy( thrust::seq, d_str.begin(), d_str.end(), result);
            //for( auto itr = d_str.begin(); itr != d_str.end(); ++itr )
            //    *result++ = (unsigned int)*itr;
        });
    //
    return result;
}

} // namespace strings
} // namespace cudf
