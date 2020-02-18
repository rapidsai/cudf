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

#include <cudf/detail/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/detail/concatenate.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <strings/utilities.hpp>

#include <thrust/for_each.h>
#include <thrust/transform_scan.h>

namespace cudf
{
namespace strings
{
namespace detail
{
namespace
{

// these hold offsets for offsets column and offsets for chars column
using offset_pair = thrust::pair<size_t,size_t>;

/**
 * @brief Concatenates multiple strings columns into a single column.
 *
 * The child columns are copied here for each strings column.
 * This will honor the strings column `offset` and `size` members appropriately.
 * The `d_columns_offsets` must be provided and include the output offsets
 * for each of the child columns. The `.first` member is the offset for the
 * output offsets column and the `.second` member is the offset for the
 * output chars column.
 *
 * The values in the child columns are copied directly while the output
 * offsets values must be updated according to where their new strings are placed.
 */
struct concatenate_fn
{
    column_device_view const* d_strings_columns; // strings columns to copy
    offset_pair const* d_columns_offsets;        // computed offsets for each input child columns
    int32_t* d_new_offsets;   // offsets for the output offsets column
    char* d_new_chars;        // chars for the output chars column

    /**
     * @brief Copy the strings child columns to the output child columns.
     *
     * @param idx Index of the strings column to process.
     */
    __device__ void operator()(size_type idx)
    {
        auto d_view = d_strings_columns[idx];
        if( d_view.size()==0 )
            return;
        auto column_offsets = d_columns_offsets[idx];
        // copy and adjust the offsets to the output
        auto d_input_offsets = d_view.child(strings_column_view::offsets_column_index).data<int32_t>() + d_view.offset();
        auto d_output_offsets = d_new_offsets + column_offsets.first;
        auto byte_offset = column_offsets.second; // add this to input offsets values
        auto first_offset = d_input_offsets[0];   // normalize input offsets values
        thrust::transform( thrust::seq, d_input_offsets + 1, d_input_offsets + d_view.size()+1, d_output_offsets,
            [first_offset, byte_offset] __device__ (int32_t old_offset) {
                return old_offset - first_offset + byte_offset;
            } );
        // copy the chars to the output
        auto byte_size = d_input_offsets[d_view.size()] - first_offset; // number of bytes to copy
        auto d_input_chars = d_view.child(strings_column_view::chars_column_index).data<char>() + first_offset;
        auto d_output_chars = d_new_chars + byte_offset; // point to the output memory slot
        thrust::copy( thrust::seq, d_input_chars, d_input_chars + byte_size, d_output_chars );
    }
};

}

std::unique_ptr<column> concatenate( std::vector<strings_column_view> const& strings_columns,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream )
{
    // build vector of column_device_views
    std::vector<std::unique_ptr<column_device_view,std::function<void(column_device_view*)> > >
        device_cols(strings_columns.size());
    size_type null_count = 0;  // add up the null counts too
    thrust::host_vector<column_device_view> h_device_views;
    for( auto&& scv : strings_columns )
    {
        device_cols.emplace_back(column_device_view::create(scv.parent(),stream));
        h_device_views.push_back(*(device_cols.back()));
        null_count += scv.null_count();
    }
    rmm::device_vector<column_device_view> device_views(h_device_views);
    auto d_views = device_views.data().get();
    auto execpol = rmm::exec_policy(stream);
    rmm::device_vector<offset_pair> columns_offsets(strings_columns.size()+1);
    // compute the offsets for the output columns
    thrust::transform_inclusive_scan( execpol->on(stream),
        d_views, d_views + device_views.size(), columns_offsets.data().get()+1,
        [] __device__ (auto d_view) {
            size_t bytes = 0;
            if( d_view.size()>0 )
            {
                auto d_offsets =
                    d_view.child(strings_column_view::offsets_column_index).template data<int32_t>()
                    + d_view.offset();
                bytes = d_offsets[d_view.size()] - d_offsets[0];
            }
            return offset_pair{d_view.size(),bytes};
        },
        [] __device__ (offset_pair const& lhs, offset_pair const& rhs) {
            return offset_pair{ lhs.first + rhs.first, lhs.second + rhs.second};
        });
    offset_pair last = columns_offsets.back();
    CUDF_EXPECTS( last.first < std::numeric_limits<size_type>::max(),
        "total number of strings is too large for a cudf column" );
    size_type strings_count = static_cast<size_type>(last.first);
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);

    CUDF_EXPECTS( last.second < std::numeric_limits<size_type>::max(),
        "total size of strings is too large for a cudf column" );
    size_type total_bytes = static_cast<size_type>(last.second);

    // set the first set of offsets to 0
    CUDA_TRY(cudaMemsetAsync( columns_offsets.data().get(), 0, sizeof(offset_pair)));

    // create chars column
    auto chars_column = create_chars_child_column( strings_count, null_count, total_bytes, mr, stream );
    auto d_new_chars = chars_column->mutable_view().data<char>();
    // create offsets column
    auto offsets_column = make_numeric_column( data_type{INT32}, strings_count + 1,
                                               mask_state::UNALLOCATED, stream, mr);
    auto d_new_offsets = offsets_column->mutable_view().data<int32_t>();

    // copy over the data for all the columns
    thrust::for_each_n( execpol->on(stream), thrust::make_counting_iterator<size_type>(0), strings_columns.size(),
        concatenate_fn{d_views, columns_offsets.data().get(), d_new_offsets+1, d_new_chars} );
    CUDA_TRY(cudaMemsetAsync(d_new_offsets,0,sizeof(int32_t)));

    // create blank null mask -- caller will be setting this
    rmm::device_buffer null_mask;
    if( null_count > 0 )
        null_mask = create_null_mask( strings_count, UNINITIALIZED, stream,mr );
    offsets_column->set_null_count(0);  // reset the null counts
    chars_column->set_null_count(0);    // for children columns
    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               null_count, std::move(null_mask), stream, mr);
}

} // namespace detail
} // namespace strings
} // namespace cudf
