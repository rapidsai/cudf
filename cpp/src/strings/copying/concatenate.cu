/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/strings/detail/utilities.hpp>

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
    CUDF_EXPECTS( strings_count < std::numeric_limits<size_type>::max(), 
        "total number of strings is too large for cudf column" );
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);

    // build vector of column_device_views
    std::vector<std::unique_ptr<column_device_view,std::function<void(column_device_view*)> > > 
        device_cols(strings_columns.size());
    thrust::host_vector<column_device_view> h_device_views;
    for( auto&& scv : strings_columns )
    {
        device_cols.emplace_back(column_device_view::create(scv.parent(),stream));
        h_device_views.push_back(*(device_cols.back()));
    }
    rmm::device_vector<column_device_view> device_views(h_device_views);
    auto execpol = rmm::exec_policy(stream);
    auto d_views = device_views.data().get();
    // compute size of the output chars column
    size_t total_bytes = thrust::transform_reduce( execpol->on(stream),
        d_views, d_views + device_views.size(),
        [] __device__ (auto d_view) {
            if( d_view.size()==0 )
                return static_cast<size_t>(0);
            auto d_offsets = d_view.child(strings_column_view::offsets_column_index).template data<int32_t>();
            size_t size = d_offsets[d_view.size()+d_view.offset()] - d_offsets[d_view.offset()];
            return size;
        }, static_cast<size_t>(0), thrust::plus<size_t>());
    CUDF_EXPECTS( total_bytes < std::numeric_limits<size_type>::max(), "total size of strings is too large for cudf column" );

    // create chars column
    auto chars_column = make_numeric_column( data_type{INT8}, total_bytes, mask_state::UNALLOCATED, stream, mr);
    auto d_new_chars = chars_column->mutable_view().data<char>();

    // create offsets column
    auto offsets_column = make_numeric_column( data_type{INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
    auto offsets_view = offsets_column->mutable_view();
    auto d_new_offsets = offsets_view.data<int32_t>();

    // copy over the data for all the columns
    ++d_new_offsets; // skip the first element which will be set to 0 after the for-loop
    int32_t offset_adjust = 0; // each section of offsets must be adjusted
    size_type null_count = 0;  // add up the null counts
    for( auto column = strings_columns.begin(); column != strings_columns.end(); ++column )
    {
        size_type column_size = column->size();
        if( column_size==0 ) // nothing to do
            continue; // empty column may not have children
        size_type column_offset = column->offset();
        column_view offsets_child = column->offsets();
        column_view chars_child = column->chars();

        // copy the offsets column
        auto d_offsets = offsets_child.data<int32_t>() + column_offset;
        int32_t bytes_offset = thrust::device_pointer_cast(d_offsets)[0];
        
        thrust::transform( rmm::exec_policy(stream)->on(stream), d_offsets + 1, d_offsets + column_size + 1, d_new_offsets,
            [offset_adjust, bytes_offset] __device__ (int32_t old_offset) {
                return old_offset - bytes_offset + offset_adjust;
            } );

        // copy the chars column data
        auto d_chars = chars_child.data<char>() + bytes_offset;
        size_type bytes = thrust::device_pointer_cast(d_offsets)[column_size] - bytes_offset;
        CUDA_TRY(cudaMemcpyAsync( d_new_chars, d_chars, bytes, cudaMemcpyDeviceToDevice, stream ));
        // get ready for the next column
        offset_adjust += bytes;
        d_new_chars += bytes;
        d_new_offsets += column_size;
        null_count += column->null_count();
    }
    CUDA_TRY(cudaMemsetAsync( offsets_view.data<int32_t>(), 0, sizeof(int32_t), stream));

    // create blank null mask -- caller will be setting this
    rmm::device_buffer null_mask;
    if( null_count > 0 )
        null_mask = create_null_mask( strings_count, mask_state::UNINITIALIZED, stream,mr );
    offsets_column->set_null_count(0);  // reset the null counts
    chars_column->set_null_count(0);    // for children columns
    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               null_count, std::move(null_mask), stream, mr);
}

} // namespace detail
} // namespace strings
} // namespace cudf
