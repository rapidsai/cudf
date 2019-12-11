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

#include <bitmask/legacy/valid_if.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/detail/concatenate.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include "../utilities.hpp"

#include <thrust/for_each.h>
#include <thrust/transform_reduce.h>

namespace cudf
{
namespace strings
{
namespace detail
{

std::unique_ptr<column> concatenate( std::vector<strings_column_view> const& strings_columns,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream )
{
    // calculate the size of the output column
    size_t strings_count = thrust::transform_reduce( strings_columns.begin(), strings_columns.end(),
        [] (auto scv) { return scv.size(); }, static_cast<size_t>(0), thrust::plus<size_t>());
    CUDF_EXPECTS( strings_count < std::numeric_limits<size_type>::max(), "total number of strings is too large for cudf column" );
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);
    size_t total_bytes = thrust::transform_reduce( strings_columns.begin(), strings_columns.end(),
        [] (auto scv) { return scv.chars_size(); }, static_cast<size_t>(0), thrust::plus<size_t>());
    CUDF_EXPECTS( total_bytes < std::numeric_limits<size_type>::max(), "total size of strings is too large for cudf column" );

    // create chars column
    auto chars_column = make_numeric_column( data_type{INT8}, total_bytes, mask_state::UNALLOCATED, stream, mr);
    auto chars_view = chars_column->mutable_view();
    auto d_new_chars = chars_view.data<char>();

    // create offsets column
    auto offsets_column = make_numeric_column( data_type{INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
    auto offsets_view = offsets_column->mutable_view();
    auto d_new_offsets = offsets_view.data<int32_t>();

    // copy over the data for all the columns
    ++d_new_offsets;
    size_type offset_adjust = 0;
    size_type null_count = 0;
    for( auto column = strings_columns.begin(); column != strings_columns.end(); ++column )
    {
        size_type size = column->size();
        if( size==0 )
            continue;
        column_view offsets_child = column->offsets();
        column_view chars_child = column->chars();

        // copy the offsets column
        auto d_offsets = offsets_child.data<int32_t>();
        CUDA_TRY(cudaMemcpyAsync( d_new_offsets, d_offsets+1+column->offset(), size*sizeof(int32_t), cudaMemcpyDeviceToDevice, stream ));

        // adjust offsets
        thrust::for_each_n( rmm::exec_policy(stream)->on(stream), thrust::make_counting_iterator<size_type>(0), size,
            [d_new_offsets, offset_adjust] __device__ (size_type idx) { d_new_offsets[idx] += offset_adjust; });

        // copy the chars column
        auto d_chars = chars_child.data<char*>();
        size_type bytes_offset = thrust::device_pointer_cast(d_offsets)[column->offset()];
        size_type bytes = chars_child.size() - bytes_offset;
        CUDA_TRY(cudaMemcpyAsync( d_new_chars, d_chars+bytes_offset, bytes, cudaMemcpyDeviceToDevice, stream ));

        // get ready for the next column
        offset_adjust += bytes;
        d_new_chars += bytes;
        d_new_offsets += size;
        null_count += column->null_count();
    }
    CUDA_TRY(cudaMemsetAsync( offsets_view.data<int32_t>(), 0, sizeof(int32_t), stream));

    // create blank null mask -- caller will be setting this
    rmm::device_buffer null_mask;
    if( null_count > 0 )
        null_mask = create_null_mask( strings_count, UNINITIALIZED, stream,mr );

    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               0, std::move(null_mask), stream, mr);
}

} // namespace detail
} // namespace strings
} // namespace cudf
