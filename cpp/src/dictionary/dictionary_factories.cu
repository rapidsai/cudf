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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cudf
{
namespace detail
{
namespace
{

// Sort functor for column row positions.
template <bool has_nulls = true>
struct sort_functor
{
    column_device_view d_column;
    __host__ __device__ bool operator()(size_type lhs_index, size_type rhs_index)
    {
        auto comparator = experimental::element_relational_comparator<has_nulls>{
                          d_column, d_column, null_order::AFTER }; // put nulls at the end
        auto result = experimental::type_dispatcher(d_column.type(), comparator,
                                                    lhs_index, rhs_index);
        return result == experimental::weak_ordering::LESS; // always sort ascending
    }
};

// Copy functor used for locating and marking unique values.
template <bool has_nulls = true>
struct copy_unique_functor
{
    column_device_view d_column;
    const int32_t* d_ordinals;
    int32_t* d_indices;

    __device__ bool operator()(int32_t idx)
    {
        if( idx==0 )
        {   // first element is always unique
            d_indices[0] = 0;
            return true;
        }
        // check if adjacent elements match
        auto lhs_index = d_ordinals[idx-1];
        auto rhs_index = d_ordinals[idx];
        auto result = !cudf::experimental::type_dispatcher( d_column.type(),
                             experimental::element_equality_comparator<has_nulls>{d_column, d_column, true},
                             lhs_index, rhs_index);
        d_indices[idx] = static_cast<int32_t>(result); // convert bool to integer [0,1]
        return result;
    }
};

} // namespace
} // detail

/**
 * @brief Create dictionary column from a column_view.
 *
 */
std::unique_ptr<column> make_dictionary_column( column_view const& input_column,
                                                rmm::mr::device_memory_resource* mr,
                                                cudaStream_t stream)
{
    auto count = input_column.size();
    auto execpol = rmm::exec_policy(stream);
    auto d_view = column_device_view::create(input_column,stream);
    auto d_column = *d_view;

    // Example using a strings column:  [e,a,d,b,c,c,c,e,a]
    //    row positions for reference:   0,1,2,3,4,5,6,7,8

    rmm::device_vector<size_type> ordinals(count);
    auto d_ordinals = ordinals.data().get();
    thrust::sequence(execpol->on(stream), ordinals.begin(), ordinals.end()); // [0,1,2,3,4,5,6,7,8]
    if( input_column.has_nulls() )
        thrust::sort(execpol->on(stream), ordinals.begin(), ordinals.end(), detail::sort_functor<true>{d_column} );
    else
        thrust::sort(execpol->on(stream), ordinals.begin(), ordinals.end(), detail::sort_functor<false>{d_column} );
    // output of sort:
    //  ordinals: [1,8,3,4,5,6,2,0,7]  => these represent sorted strings as: [a,a,b,c,c,c,d,e,e]
    // create empty indices_column
    auto indices_column = make_numeric_column( data_type{INT32}, count,
                                               mask_state::UNALLOCATED, // nulls managed by parent
                                               stream, mr);
    auto indices = indices_column->mutable_view();
    auto d_indices = indices.data<int32_t>();
    // build map and initialize indices
    rmm::device_vector<int32_t> map_indices(count);
    // The copy-if here does 2 things in one kernel; (trying to minimize element compares)
    // 1) compute indices of only the unique elements from the sorted result
    // 2) mark in indices with 1 where unique values are found and 0 otherwise
    auto map_nend = map_indices.end();
    if( input_column.has_nulls() )
        map_nend = thrust::copy_if( execpol->on(stream), thrust::make_counting_iterator<int32_t>(0),
                                    thrust::make_counting_iterator<int32_t>(count), map_indices.begin(),
                                    detail::copy_unique_functor<true>{d_column, d_ordinals, d_indices} );
    else
        map_nend = thrust::copy_if( execpol->on(stream), thrust::make_counting_iterator<int32_t>(0),
                                    thrust::make_counting_iterator<int32_t>(count), map_indices.begin(),
                                    detail::copy_unique_functor<false>{d_column, d_ordinals, d_indices} );

    // output of copy_if:
    //  map_indices: [0,2,3,6,7]     => start of unique values        0,1,2,3,4,5,6,7,8
    //  indices: [0,0,1,1,0,0,1,1,0] => identifies unique positions   a,a,b,c,c,c,d,e,e

    // gather the positions of the unique values
    size_type unique_count = static_cast<size_type>(std::distance(map_indices.begin(),map_nend)); // 5
    rmm::device_vector<size_type> keys_indices(unique_count);
    thrust::gather( execpol->on(stream), map_indices.begin(), map_nend, d_ordinals, keys_indices.begin() );
    // output of gathering [0,2,3,6,7] from [1,8,3,4,5,6,2,0,7] is
    //  keys_indices: [1,3,4,2,0]

    // in-place scan will produce the actual indices
    thrust::inclusive_scan(execpol->on(stream), d_indices, d_indices + count, d_indices);
    // output of scan indices [0,0,1,1,0,0,1,1,0] is now [0,0,1,2,2,2,3,4,4]
    // sort will put the indices in the correct order
    thrust::sort_by_key(execpol->on(stream), ordinals.begin(), ordinals.end(), d_indices);
    // output of sort: indices is now [4,0,3,1,2,2,2,4,0]

    // if there are nulls, just truncate the null entry -- it was sorted to the end;
    // the indices do not need to be updated since any value for a null entry is technically undefined
    if( input_column.has_nulls() )
        unique_count--;
    indices_column->set_null_count(0);
    // gather the keys using keys_indices: [1,3,4,2,0] => ['a','b','c','d','e']
    auto table_keys = experimental::detail::gather( table_view{std::vector<column_view>{input_column}},
                                                    keys_indices.begin(), keys_indices.begin()+unique_count,
                                                    false, false, false, mr, stream)->release();
    std::shared_ptr<const column> keys_column = std::move(table_keys[0]);

    // create column with keys_column and indices_column
    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(indices_column));
    return std::make_unique<column>(
        data_type{DICTIONARY32}, count, rmm::device_buffer{0,stream,mr},
        copy_bitmask( input_column, stream, mr), input_column.null_count(),
        std::move(children),
        std::move(keys_column));
}

std::unique_ptr<column> make_dictionary_column( column_view const& keys_column,
                                                column_view const& indices_column,
                                                rmm::mr::device_memory_resource* mr,
                                                cudaStream_t stream)
{
    auto indices_copy = std::make_unique<column>( indices_column, stream, mr);
    std::shared_ptr<const column> keys_copy = std::make_unique<column>(keys_column, stream, mr);

    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(indices_copy));
    return std::make_unique<column>(
        data_type{DICTIONARY32}, indices_column.size(),
        rmm::device_buffer{0,stream,mr},
        copy_bitmask(indices_column,stream,mr), indices_column.null_count(),
        std::move(children),
        std::move(keys_copy));
}

}  // namespace cudf
