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
#include <thrust/transform.h>

namespace cudf
{
namespace dictionary
{
namespace detail
{
namespace
{

std::unique_ptr<column> remove_keys_fn( dictionary_column_view const& dictionary_column,
                                        experimental::bool8 const* d_matches,
                                        rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                        cudaStream_t stream = 0)
{
    auto keys_view = dictionary_column.keys();
    auto indices_view = dictionary_column.indices();
    auto execpol = rmm::exec_policy(stream);

    // create keys positions column to identify original key positions after removing they keys
    rmm::device_vector<int32_t> keys_positions(keys_view.size()); // needed for remapping indices
    thrust::sequence( execpol->on(stream), keys_positions.begin(), keys_positions.end() );
    column_view keys_positions_view( data_type{INT32}, keys_view.size(), keys_positions.data().get() );

    // copy the non-removed keys ( d_matches: true=remove, false=keep )
    std::unique_ptr<column> keys_column;
    rmm::device_vector<int32_t> map_indices(keys_view.size(),-1); // init -1 to identify new nulls
    {
        auto table_keys = experimental::detail::copy_if( table_view{{keys_view, keys_positions_view}},
            [d_matches]__device__(size_type idx) { return !d_matches[idx]; }, mr, stream )->release();
        keys_column = std::move(table_keys.front());
        keys_positions_view = table_keys[1]->view();
        // build indices mapper
        // Example scatter([0,1,2][0,2,4][-1,-1,-1,-1,-1]) => [0,-1,1,-1,2]
        thrust::scatter( execpol->on(stream), thrust::make_counting_iterator<int32_t>(0),
                         thrust::make_counting_iterator<int32_t>(keys_positions_view.size()),
                         keys_positions_view.begin<int32_t>(), map_indices.begin() );
    } // frees up the temporary table_keys objects

    // create new indices column
    // Example: gather([4,0,3,1,2,2,2,4,0],[0,-1,1,-1,2]) => [2,0,-1,-1,1,1,1,2,0]
    column_view map_indices_view( data_type{INT32}, keys_view.size(), map_indices.data().get() );
    auto table_indices = experimental::detail::gather( table_view{{map_indices_view}},
                    indices_view, false, false, false, mr, stream )->release();
    std::unique_ptr<column> indices_column(std::move(table_indices.front()));

    // compute new nulls -- merge the existing nulls with the newly created ones (value<0)
    auto d_null_mask = dictionary_column.null_mask();
    auto d_indices = indices_column->view().data<int32_t>();
    auto new_nulls = experimental::detail::valid_if( thrust::make_counting_iterator<size_type>(dictionary_column.offset()),
                    thrust::make_counting_iterator<size_type>(dictionary_column.size()),
                    [d_null_mask, d_indices] __device__ (size_type idx) {
                        if( d_null_mask && !bit_is_set(d_null_mask,idx) )
                            return false;
                        return (d_indices[idx] >= 0); // new nulls have negative values
                    }, stream, mr);
    rmm::device_buffer new_null_mask;
    if( new_nulls.second > 0 )
        new_null_mask = std::move(new_nulls.first);

    // create column with keys_column and indices_column
    return make_dictionary_column( std::move(keys_column), std::move(indices_column),
                                   std::move(new_null_mask), new_nulls.second );
}

} // namespace

std::unique_ptr<column> remove_keys( dictionary_column_view const& dictionary_column,
                                     column_view const& keys_to_remove,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                     cudaStream_t stream = 0)
{
    CUDF_EXPECTS( !keys_to_remove.has_nulls(), "keys_to_remove must not have nulls" );
    auto keys_view = dictionary_column.keys();
    CUDF_EXPECTS( keys_view.type()==keys_to_remove.type(), "keys types must match");

    // locate keys to remove by searching the keys column
    auto matches = experimental::detail::contains( keys_view, keys_to_remove, mr, stream);
    // call common utility method to remove the keys not found
    return remove_keys_fn( dictionary_column, matches->view().data<experimental::bool8>(),
                           mr, stream );
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
    auto d_matches = matches->mutable_view().data<experimental::bool8>();
    // need to negate the values
    thrust::transform( execpol->on(stream), d_matches, d_matches + keys.size(), d_matches,
        [] __device__ ( auto bv ) { return !bv; });

    // call common utility method to remove the keys not found
    return remove_keys_fn( dictionary_column, d_matches, mr, stream );
}

} // namespace detail

// external APIs

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
