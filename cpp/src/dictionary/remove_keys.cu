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

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/detail/search.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <thrust/sequence.h>
#include <thrust/scatter.h>

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
    CUDF_EXPECTS( !keys_to_remove.has_nulls(), "keys_to_remove must not have nulls" );
    auto keys_view = dictionary_column.keys();
    CUDF_EXPECTS( keys_view.type()==keys_to_remove.type(), "keys types must match");
    auto indices_view = dictionary_column.indices();
    auto count = indices_view.size();
    auto execpol = rmm::exec_policy(stream);
    // locate keys to remove by searching the keys column
    auto matches = experimental::detail::contains( keys_view, keys_to_remove, mr, stream);
    auto d_matches = matches->view().data<experimental::bool8>();
    // create keys indices column to identify original key positions after removing they keys
    rmm::device_vector<int32_t> keys_indices(keys_view.size()); // needed for remapping indices
    thrust::sequence( execpol->on(stream), keys_indices.begin(), keys_indices.end() );
    column_view keys_indices_view( data_type{INT32}, keys_view.size(), keys_indices.data().get() );
    // copy the non-removed keys ( d_matches: true=remove, false=keep )
    auto table_keys = experimental::detail::copy_if( table_view{{keys_view, keys_indices_view}},
        [d_matches]__device__(size_type idx) { return !d_matches[idx]; }, mr, stream )->release();
    std::unique_ptr<column> keys_column(std::move(table_keys.front()));
    keys_indices_view = table_keys[1]->view();
    rmm::device_vector<int32_t> map_indices(keys_view.size(),-1); // init -1 to identify new nulls
    // build indices mapper; example scatter([0,1,2][0,2,4][-1,-1,-1,-1,-1]) => [0,-1,1,-1,2]
    thrust::scatter( execpol->on(stream), thrust::make_counting_iterator<int32_t>(0),
                     thrust::make_counting_iterator<int32_t>(keys_indices_view.size()),
                     keys_indices_view.begin<int32_t>(), map_indices.begin() );
    // create new indices column
    // gather([4,0,3,1,2,2,2,4,0],[0,-1,1,-1,2]) => [2,0,-1,-1,1,1,1,2,0]
    column_view map_indices_view( data_type{INT32}, keys_view.size(), map_indices.data().get() );
    auto table_indices = experimental::detail::gather( table_view{{map_indices_view}},
                    indices_view, false, false, false, mr, stream )->release();
    std::unique_ptr<column> indices_column(std::move(table_indices.front()));

    // compute new nulls -- merge the current nulls with the newly created ones (value<0)
    auto d_null_mask = dictionary_column.null_mask();
    auto d_indices = indices_column->view().data<int32_t>();
    auto new_nulls = experimental::detail::valid_if( thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(count),
                    [d_null_mask, d_indices] __device__ (size_type idx) {
                        if( d_null_mask && !bit_is_set(d_null_mask,idx) )
                            return false;
                        return (d_indices[idx] >= 0); // new nulls have negative values
                    }, stream, mr);

    // create column with keys_column and indices_column
    return make_dictionary_column( std::move(keys_column), std::move(indices_column), 
                                   std::move(new_nulls.first), new_nulls.second );
}

std::unique_ptr<column> remove_unused_keys( dictionary_column_view const& dictionary_column,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                            cudaStream_t stream = 0)
{
    // locate the keys to remove
    auto keys = dictionary_column.keys();
    auto indices = dictionary_column.indices();
    auto execpol = rmm::exec_policy(stream);
    // build keys index to verify against indices values
    rmm::device_vector<int32_t> keys_positions(keys.size());
    thrust::sequence( execpol->on(stream), keys_positions.begin(), keys_positions.end());
    // wrap the indices for comparison with column_views
    column_view keys_positions_view( data_type{INT32}, keys.size(), keys_positions.data().get() );
    column_view indices_view( data_type{INT32}, indices.size(), indices.data<int32_t>(),
        dictionary_column.null_mask(), dictionary_column.null_count(), dictionary_column.offset() );
    // search the indices values with key indices to look for any holes
    auto matches = experimental::detail::contains( keys_positions_view, indices_view, mr, stream);
    auto d_matches = matches->view().data<experimental::bool8>();
    // copy any keys that are not found
    auto table_keys = experimental::detail::copy_if( table_view{{keys}},
        [d_matches]__device__(size_type idx) { return !d_matches[idx]; }, mr, stream )->release();
    // call remove_keys to remove those keys
    return remove_keys( dictionary_column, table_keys.front()->view(), mr, stream);
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
