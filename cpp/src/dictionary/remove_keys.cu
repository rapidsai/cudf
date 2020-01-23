/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/search.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <thrust/sequence.h>

namespace cudf
{
namespace dictionary
{
namespace detail
{
std::unique_ptr<column> remove_keys( dictionary_column_view const& dictionary_column,
                                     column_view const& keys_to_remove,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                     cudaStream_t stream = 0)
{
    auto keys_view = dictionary_column.dictionary_keys();
    auto device_view = column_device_view::create(dictionary_column.parent());
    auto indices_view = device_view->child(0);
    auto count = indices_view.size();
    auto execpol = rmm::exec_policy(stream);
    // locate keys to remove by searching the keys column
    auto matches = experimental::contains( keys_view, keys_to_remove, mr ); // TODO: need detail version of this API from PR 3823
    auto d_matches = matches->view().data<experimental::bool8>();
    // create keys indices column to identify original key positions after removing they keys
    rmm::device_vector<int32_t> keys_indices(keys_view.size()); // needed for remapping indices
    thrust::sequence( execpol->on(stream), keys_indices.begin(), keys_indices.end() );
    column_view keys_indices_view( data_type{INT32}, keys_view.size(), keys_indices.data().get() );
    // copy the non-removed keys ( d_matches: true=remove, false=keep )
    auto table_keys = experimental::detail::copy_if( table_view{{keys_view, keys_indices_view}},
        [d_matches]__device__(size_type idx) { return !d_matches[idx]; }, mr, stream )->release();
    rmm::device_vector<int32_t> map_indices(count,-1); // init -1 to identify new nulls
    // build indices mapper; example scatter([0,1,2][0,2,4][-1,-1,-1,-1,-1]) => [0,-1,1,-1,2]
    thrust::scatter( execpol->on(stream), thrust::make_counting_iterator<int32_t>(0),
                     thrust::make_counting_iterator<int32_t>(keys_indices.size()),
                     keys_indices_view.begin<int32_t>(), map_indices.begin() );
    // create new indices column
    auto indices_column = make_numeric_column( data_type{INT32}, count, cudf::mask_state::UNALLOCATED, stream, mr);
    auto d_new_indices = indices_column->mutable_view().data<int32_t>();
    auto d_column = *device_view; // this has the null mask
    auto d_map_indices = map_indices.data().get(); // mapping old indices to new values
    auto d_old_indices = indices_view.data<int32_t>();
    // map old indices to new indices -- this will probably become a utility method
    thrust::transform( execpol->on(stream), thrust::make_counting_iterator<size_type>(0),
                       thrust::make_counting_iterator<size_type>(count), d_new_indices,
                       [d_column, d_old_indices, d_map_indices] __device__ (size_type idx) {
                           if( d_column.is_null(idx) )
                                return -1; // this value can be anything
                           return d_map_indices[d_old_indices[idx]]; // get the new index
                       });
    // compute new nulls -- this one too
    auto new_nulls = experimental::detail::valid_if( thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(count),
                    [d_column, d_old_indices, d_map_indices] __device__ (size_type idx) {
                        if( d_column.is_null(idx) )
                            return false; // old nulls are unchanged
                        // new nulls identified by negative map values
                        return d_map_indices[d_old_indices[idx]] >=0;
                    }, stream, mr);

    std::shared_ptr<const column> keys_column(std::move(table_keys[0]));
    // create column with keys_column and indices_column
    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(indices_column));
    return std::make_unique<column>(
        data_type{DICTIONARY32}, count,
        rmm::device_buffer{0,stream,mr}, // no data in the parent
        new_nulls.first, new_nulls.second,
        std::move(children),
        std::move(keys_column));
}

std::unique_ptr<column> remove_unused_keys( dictionary_column_view const& dictionary_column,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                            cudaStream_t stream = 0)
{
    // locate the keys to remove
    auto keys = dictionary_column.dictionary_keys();
    auto indices = dictionary_column.indices();
    auto execpol = rmm::exec_policy(stream);
    // build keys index to verify against indices values
    rmm::device_vector<int32_t> keys_indices(keys.size());
    thrust::sequence( execpol->on(stream), keys_indices.begin(), keys_indices.end());
    // wrap the indices for comparison with column_views
    column_view keys_indices_view( data_type{INT32}, keys.size(), keys_indices.data().get() );
    column_view indices_view( data_type{INT32}, indices.size(), indices.data<int32_t>(),
        dictionary_column.null_mask(), dictionary_column.null_count() );
    // search the indices values with key indices to look for any holes
    auto matches = experimental::contains( keys_indices_view, indices_view, mr ); // TODO: need detail version of this API from PR 3823
    auto d_matches = matches->view().data<experimental::bool8>();
    // copy any keys that are not found
    auto table_keys = experimental::detail::copy_if( table_view{{keys}},
        [d_matches]__device__(size_type idx) { return !d_matches[idx]; }, mr, stream )->release();
    // call remove_keys to remove those keys
    return remove_keys( dictionary_column, table_keys[0]->view(), mr, stream);
}

} // namespace detail


std::unique_ptr<column> remove_keys( dictionary_column_view const& dictionary_column,
                                     column_view const& keys_to_remove,
                                     rmm::mr::device_memory_resource* mr)
{
    return detail::remove_keys(dictionary_column, keys_to_remove, mr);
}

std::unique_ptr<column> remove_unused_keys( dictionary_column_view const& dictionary_column,
                                            rmm::mr::device_memory_resource* mr)
{
    return detail::remove_unused_keys(dictionary_column,mr);
}

} // namespace dictionary
} // namespace cudf
