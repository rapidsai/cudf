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
#include <cudf/column/column.hpp>
#include <cudf/functions.h>
#include <cudf/null_mask.hpp>
#include <utilities/error_utils.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <thrust/for_each.h>


namespace cudf {

// Create a strings-type column from array of pointer/size pairs
std::unique_ptr<column> make_strings_column(
    const rmm::device_vector<thrust::pair<const char*,size_type>>& strings,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr)
{
    size_type num_strings = strings.size();
    // maybe a separate factory for creating null strings-column
    CUDF_EXPECTS(num_strings > 0, "must specify at least one pair");

    auto execpol = rmm::exec_policy(stream);
    auto d_strings = strings.data().get();

    // check total size is not too large for cudf column
    size_t bytes = thrust::transform_reduce( execpol->on(stream),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_strings),
        [d_strings] __device__ (size_t idx) {
            auto item = d_strings[idx];
            return (item.first!=nullptr) ? item.second : 0;
        },
        0, thrust::plus<size_t>());
    CUDF_EXPECTS( bytes < std::numeric_limits<size_type>::max(), "total size of strings is too large for cudf column" );

    // build offsets column -- last entry is the total size
    auto offsets_column = make_numeric_column( data_type{INT32}, num_strings+1, mask_state::UNALLOCATED, stream, mr );
    auto offsets_view = offsets_column->mutable_view();
    auto d_offsets = offsets_view.data<int32_t>();
    // Using inclusive-scan to compute last entry which is the total size.
    // Exclusive-scan is possible but will not compute that last entry.
    // Rather than manually computing the final offset using values in device memory,
    // we use inclusive-scan on a shifted output (d_offsets+1) and then set the first
    // zero offset manually.
    thrust::transform_inclusive_scan( execpol->on(stream),
        thrust::make_counting_iterator<size_type>(0), thrust::make_counting_iterator<size_type>(num_strings),
        d_offsets+1, // fills in offsets entries [1,num_strings]
        [d_strings] __device__ (size_type idx) {
            thrust::pair<const char*,size_type> item = d_strings[idx];
            return ( item.first!=nullptr ? static_cast<int32_t>(item.second) : 0 );
        },
        thrust::plus<int32_t>() );
    // set the first offset to 0
    CUDA_TRY(cudaMemsetAsync( d_offsets, 0, sizeof(*d_offsets), stream));

    // create null mask
    auto valid_mask = valid_if( static_cast<const bit_mask_t*>(nullptr),
        [d_strings] __device__ (size_type idx) { return d_strings[idx].first!=nullptr; },
        num_strings, stream );
    auto null_count = valid_mask.second;
    rmm::device_buffer null_mask(valid_mask.first, gdf_valid_allocation_size(num_strings),
                                 stream, mr);
    RMM_TRY( RMM_FREE(valid_mask.first,stream) ); // TODO valid_if to return device_buffer in future
    // if we have all nulls, a null chars column is allowed
    // if all non-null strings are empty strings, we need a non-null chars column
    // - in this case we set the bytes to 1 to create a minimal one-byte chars column
    if( (bytes==0) && (null_count < num_strings) )
        bytes = 1; // all entries are empty strings

    // build chars column
    auto chars_column = make_numeric_column( data_type{INT8}, bytes, mask_state::UNALLOCATED, stream, mr );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<char>();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), num_strings,
          [d_strings, d_offsets, d_chars] __device__(size_type idx){
              // place individual strings
              auto item = d_strings[idx];
              if( item.first!=nullptr )
                  memcpy(d_chars + d_offsets[idx], item.first, item.second );
          });

    // build children vector
    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(offsets_column));
    children.emplace_back(std::move(chars_column));

    // no data-ptr with num_strings elements plus children
    return std::make_unique<column>(
        data_type{STRING}, num_strings, rmm::device_buffer{0,stream,mr},
        null_mask, null_count,
        std::move(children));
}

// Create a strings-type column from array of chars and array of offsets.
std::unique_ptr<column> make_strings_column(
    const rmm::device_vector<char>& strings,
    const rmm::device_vector<size_type>& offsets,
    const rmm::device_vector<bitmask_type>& valid_mask,
    size_type null_count,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr )
{
    size_type num_strings = offsets.size()-1;
    CUDF_EXPECTS( num_strings > 0, "strings count must be greater than 0");
    CUDF_EXPECTS( null_count < num_strings, "null strings column not yet supported");
    if( null_count > 0 ) {
        CUDF_EXPECTS( !valid_mask.empty(), "Cannot have null elements without a null mask." );
    }

    auto execpol = rmm::exec_policy(stream);
    size_type bytes = offsets.back() - offsets[0];
    CUDF_EXPECTS( bytes >=0, "invalid offsets vector");

    // build offsets column -- this is the number of strings + 1
    auto offsets_column = make_numeric_column( data_type{INT32}, num_strings+1, mask_state::UNALLOCATED, stream, mr );
    auto offsets_view = offsets_column->mutable_view();
    CUDA_TRY(cudaMemcpyAsync( offsets_view.data<int32_t>(), offsets.data().get(),
                              (num_strings+1)*sizeof(int32_t),
                              cudaMemcpyDeviceToDevice, stream ));

    // build null bitmask
    rmm::device_buffer null_mask;
    if( null_count )
        null_mask = rmm::device_buffer(valid_mask.data().get(),
                                       gdf_valid_allocation_size(num_strings),
                                       stream, mr);

    // build chars column
    auto chars_column = make_numeric_column( data_type{INT8}, bytes, mask_state::UNALLOCATED, stream, mr );
    auto chars_view = chars_column->mutable_view();
    CUDA_TRY(cudaMemcpyAsync( chars_view.data<char>(), strings.data().get(), bytes,
                              cudaMemcpyDeviceToDevice, stream ));

    // build children vector
    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(offsets_column));
    children.emplace_back(std::move(chars_column));

    //
    return std::make_unique<column>(
        data_type{STRING}, num_strings, rmm::device_buffer{0,stream,mr},
        null_mask, null_count,
        std::move(children));
}

}  // namespace cudf
