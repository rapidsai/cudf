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
// TODO add has_nulls template parameter
struct sort_functor
{
    column_device_view d_column;
    __host__ __device__ bool operator()(size_type lhs_index, size_type rhs_index)
    {
        auto comparator = experimental::element_relational_comparator<true>{ // true=has_nulls
                          d_column, d_column, null_order::AFTER }; // put nulls at the end
        auto result = experimental::type_dispatcher(d_column.type(), comparator,
                                                    lhs_index, rhs_index);
        return result == experimental::weak_ordering::LESS; // always sort ascending
    }
};

// Copy functor used for locating and marking unique values.
// TODO add has_nulls template parameter
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
        auto result = cudf::experimental::type_dispatcher( d_column.type(),
                            experimental::element_equality_comparator<true>{d_column, d_column, true},
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
                                                rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                                cudaStream_t stream = 0)
{
    auto count = input_column.size();
    auto execpol = rmm::exec_policy(stream);
    auto column = column_device_view::create(input_column,stream);
    auto d_column = *column;

    // Example using a strings column:  [e,a,d,b,c,c,c,e,a]
    //    row positions for reference:   0,1,2,3,4,5,6,7,8

    rmm::device_vector<size_type> ordinals(count);
    auto d_ordinals = ordinals.data().get();
    thrust::sequence(execpol->on(stream), ordinals.begin(), ordinals.end()); // [0,1,2,3,4,5,6,7,8]
    thrust::sort(execpol->on(stream), ordinals.begin(), ordinals.end(), detail::sort_functor{d_column} );
    // output of sort:
    //  ordinals: [1,8,3,4,5,6,2,0,7]  => these represent sorted strings as: [a,a,b,c,c,c,d,e,e]
    // create empty indices_column
    auto indices_column = make_numeric_column( data_type{INT32}, count,
         copy_bitmask( input_column, stream, mr), input_column.null_count(), stream, mr);
    auto indices = indices_column->mutable_view();
    auto d_indices = indices.data<int32_t>();
    // build indices map and initialize indices
    rmm::device_vector<int32_t> map_indices(count);
    auto d_map_indices = map_indices.data().get();
    // The copy-if here does 2 things in one kernel; (trying to minimize element compares)
    // 1) compute indices of only the unique elements from the sorted result
    // 2) mark in indices with 1 where unique values are found and 0 otherwise
    int* d_map_nend = thrust::copy_if( execpol->on(stream), thrust::make_counting_iterator<int32_t>(0),
                                       thrust::make_counting_iterator<int32_t>(count), d_map_indices,
                                       detail::copy_unique_functor{d_column, d_ordinals, d_indices} );
    // output of copy_if:
    //  map_indices: [0,2,3,6,7]       => start of unique values        0,1,2,3,4,5,6,7,8
    //  indices: [0,0,1,1,0,0,1,1,0,0] => identifies unique positions   a,a,b,c,c,c,d,e,e
    // in-place scan will produce the actual indices
    thrust::inclusive_scan(execpol->on(stream), d_indices, d_indices + count, d_indices);
    // output of scan indices [0,0,1,1,0,0,1,1,0,0] is now [0,0,1,2,2,2,3,4,4]
    // and sort will put the indices in the correct order
    thrust::sort_by_key(execpol->on(stream), ordinals.begin(), ordinals.end(), d_indices);
    // output of sort; indices is now [4,0,3,1,2,2,2,4,0]
    indices.set_null_count(input_column.null_count());
    // done with indices_column
    // gather the positions of the unique values
    size_type unique_count = static_cast<size_type>(std::distance(d_map_indices,d_map_nend)); // 5
    rmm::device_vector<size_type> keys_indices(unique_count);
    thrust::gather( execpol->on(stream), d_map_indices, d_map_nend, ordinals.begin(), keys_indices.begin() );
    // output of gather [0,2,3,6,7] from [1,8,3,4,5,6,2,0,7]
    //  keys_indices: [1,3,4,2,0]
    // if there are nulls, just truncate the null entry -- it was sorted to the end;
    // the indices do not need to be updated since any value for a null entry is technically undefined
    if( input_column.has_nulls() )
        unique_count--;
    // gather the keys using keys_indices: [1,3,4,2,0] => ['a','b','c','d','e']
    auto d_keys_indices = keys_indices.data().get();
    auto table_keys = experimental::detail::gather( table_view{std::vector<column_view>{input_column}},
                                                    d_keys_indices, d_keys_indices+unique_count,
                                                    false, false, false, mr, stream).release();
    auto keys_column = std::move(table_keys[0]);

    // create column with keys_column and indices_column
    return nullptr;
}

}  // namespace cudf
