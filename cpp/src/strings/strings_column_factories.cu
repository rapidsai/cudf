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
    const rmm::device_vector<thrust::pair<const char*,size_t>>& strings,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr)
{
    size_type count = (size_type)strings.size();
    // maybe a separate factory for creating null strings-column
    CUDF_EXPECTS(count > 0, "must specify at least one pair");

    auto execpol = rmm::exec_policy(stream);
    auto d_strings = strings.data().get();

    // check total size is not too large for cudf column
    size_t bytes = thrust::transform_reduce( execpol->on(stream),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(count),
        [d_strings] __device__ (size_t idx) {
            auto item = d_strings[idx];
            return item.first ? item.second : (size_t)0;
        },
        (size_t)0,
        thrust::plus<size_t>());
    CUDF_EXPECTS( bytes < std::numeric_limits<size_type>::max(), "total size of strings is too large for cudf column" );

    // build offsets column
    auto offsets_column = make_numeric_column( data_type{INT32}, count, mask_state::UNALLOCATED, stream, mr );
    auto offsets_view = offsets_column->mutable_view();
    thrust::transform_inclusive_scan( execpol->on(stream),
        thrust::make_counting_iterator<size_type>(0), thrust::make_counting_iterator<size_type>(count),
        offsets_view.data<int32_t>(),
        [d_strings] __device__ (size_type idx) {
            thrust::pair<const char*,size_t> item = d_strings[idx];
            return ( item.first ? (int32_t)item.second : 0 );
        },
        thrust::plus<int32_t>() );

    // create null mask
    auto valid_mask = valid_if( static_cast<const bit_mask_t*>(nullptr),
        [d_strings] __device__ (size_type idx) { return d_strings[idx].first!=nullptr; },
        count, stream );
    auto null_count = valid_mask.second;
    rmm::device_buffer null_mask(valid_mask.first,gdf_valid_allocation_size(count)); // does deep copy
    RMM_TRY( RMM_FREE(valid_mask.first,stream) ); // TODO valid_if to return device_buffer in future

    // build chars column
    auto chars_column = make_numeric_column( data_type{INT8}, bytes, mask_state::UNALLOCATED, stream, mr );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<int8_t>();
    auto d_offsets = offsets_view.data<int32_t>();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), count,
          [d_strings, d_offsets, d_chars] __device__(size_type idx){
              // place individual strings
              auto item = d_strings[idx];
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

    // see column_view.cpp(45) to see why size must be 0 here
    return std::make_unique<column>(
        data_type{STRING}, 0, rmm::device_buffer{0,stream,mr},
        null_mask, null_count,
        std::move(children));
}

}  // namespace cudf
