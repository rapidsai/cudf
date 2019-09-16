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
#include <cudf/column/column.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_factories.hpp>
#include <utilities/error_utils.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <thrust/for_each.h>

namespace cudf {

// Create a strings-type column.
// A strings-column has children columns to manage the variable-length
// encoded character array.
// Use the strings_column_handler class to perform strings operations
// on this type of column.
std::unique_ptr<column> make_strings_column(
    std::pair<const char*,size_t>* strings, size_type count, cudaStream_t stream,
    rmm::mr::device_memory_resource* mr)
{
    // maybe a separate factory for creating null strings-column
    CUDF_EXPECTS(count > 0, "must have at least one pair");

    auto execpol = rmm::exec_policy(stream);
    auto strs = thrust::device_pointer_cast(reinterpret_cast<thrust::pair<const char*,size_t>*>(strings));
    auto d_strs = strs.get();

    // build offsets column
    auto offsets_column = make_numeric_column( data_type{INT32}, count, mask_state::UNALLOCATED, stream, mr );
    auto offsets_view = offsets_column->mutable_view();
    thrust::transform_inclusive_scan( execpol->on(stream),
        thrust::make_counting_iterator<size_type>(0), thrust::make_counting_iterator<size_type>(count),
        offsets_view.data<int32_t>(),
        [d_strs] __device__ (size_type idx) {
            thrust::pair<const char*,size_t> item = d_strs[idx];
            return ( item.first ? (int32_t)item.second : 0 );
        },
        thrust::plus<int32_t>() );

    // get number of bytes (last offset value)
    size_type bytes = thrust::device_pointer_cast(offsets_view.data<int32_t>())[count-1];

    // count nulls
    size_type null_count = thrust::transform_reduce( execpol->on(stream),
        thrust::make_counting_iterator<size_type>(0), thrust::make_counting_iterator<size_type>(count),
        [d_strs] __device__ (size_type idx) { return (size_type)(d_strs[idx].first==nullptr); },
        0, thrust::plus<size_type>() );

    // build null_mask
    mask_state state = mask_state::UNINITIALIZED;
    if( null_count==0 )
        state = mask_state::UNALLOCATED;
    else if( null_count==count )
        state = mask_state::ALL_NULL;
    auto null_mask = create_null_mask(count, state, stream, mr);
    if( (null_count > 0) && (null_count < count) )
    {
        uint8_t* d_null_mask = static_cast<uint8_t*>(null_mask.data());
        CUDA_TRY(cudaMemsetAsync(d_null_mask, 0, null_mask.size(), stream));
        thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), (count/8),
            [d_strs, count, d_null_mask] __device__(size_type byte_idx) {
                unsigned char byte = 0; // set one byte per thread -- init to all nulls
                for( size_type i=0; i < 8; ++i )
                {
                    size_type idx = i + (byte_idx*8);  // compute d_strs index
                    byte = byte >> 1;                  // shift until we are done
                    if( idx < count )                  // check boundary
                    {
                        if( d_strs[idx].first )
                            byte |= 128;               // string is not null, set high bit
                    }
                }
                d_null_mask[byte_idx] = byte;
            });
    }

    // build chars column
    auto chars_column = make_numeric_column( data_type{INT8}, bytes, mask_state::UNALLOCATED, stream, mr );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<int8_t>();
    auto d_offsets = offsets_view.data<int32_t>();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), count,
          [d_strs, d_offsets, d_chars] __device__(size_type idx){
              // place individual strings
              auto item = d_strs[idx];
              if( item.first )
              {
                  size_type offset = (idx ? d_offsets[idx-1] : 0);
                  memcpy(d_chars + offset, item.first, item.second );
              }
          });

    // build children vector
    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(offsets_column));
    children.emplace_back(std::move(chars_column));

    return std::make_unique<column>(
        data_type{STRING}, 0, rmm::device_buffer{0,stream,mr},
        null_mask, null_count,
        std::move(children));
}

}  // namespace cudf
