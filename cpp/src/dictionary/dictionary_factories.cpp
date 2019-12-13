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

// use existing cudf sort or row-ordering classes
struct sort_functor
{
    column_device_view _column;
    __host__ __device__ bool operator()(size_type lhs_index, size_type rhs_index)
    {
        auto comparator = experimental::element_relational_comparator<true>{
                          _column, _column, null_order::BEFORE };
        auto result = experimental::type_dispatcher(_column.type(), comparator,
                                                    lhs_index, rhs_index);
        return result == experimental::weak_ordering::LESS;
    }
};

struct copy_unique_functor
{
    column_device_view _column;
    const int32_t* d_ordinals;
    int32_t* d_indices;

    __device__ bool operator()(int32_t idx)
    {
        if( idx==0 )
        {
            d_indices[0] = 0;
            return true;
        }
        auto lhs_index = d_ordinals[idx-1];
        auto rhs_index = d_ordinals[idx];
        auto result = cudf::experimental::type_dispatcher( _column.type(),
                                experimental::element_equality_comparator<true>{_column, _column, true},
                                lhs_index, rhs_index);
        d_indices[idx] = static_cast<int32_t>(result);
        return result;
    }
};

} // namespace
} // detail

std::unique_ptr<column> make_dictionary_column( column_view const& input_column,
                                                rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                                cudaStream_t stream = 0)
{
    auto count = input_column.size();
    auto execpol = rmm::exec_policy(stream);

    auto column = column_device_view::create(input_column,stream);
    auto d_column = *column;

    rmm::device_vector<size_type> ordinals(count);
    auto d_ordinals = ordinals.data().get();
    thrust::sequence(execpol->on(stream), ordinals.begin(), ordinals.end()); // 0,1,2,3,4,5,6,7,8
    thrust::sort(execpol->on(stream), ordinals.begin(), ordinals.end(), detail::sort_functor{d_column} );

    auto indices_column = make_numeric_column( data_type{INT32}, count,
         copy_bitmask( input_column, stream, mr), input_column.null_count(), stream, mr);
    auto indices = indices_column->mutable_view();
    auto d_indices = indices.data<int32_t>();
    rmm::device_vector<int32_t> map_indices(count);
    auto d_map_indices = map_indices.data().get();
    int* d_map_nend = thrust::copy_if( execpol->on(stream), thrust::make_counting_iterator<int32_t>(0),
                                       thrust::make_counting_iterator<int32_t>(count), d_map_indices,
                                       detail::copy_unique_functor{d_column, d_ordinals, d_indices} );
    size_type unique_count = static_cast<size_type>(std::distance(d_map_indices,d_map_nend));
    rmm::device_vector<size_type> keys_indices(unique_count);
    //
    thrust::gather( execpol->on(stream), d_map_indices, d_map_nend, ordinals.begin(), keys_indices.begin() );
    // scan will produce the resulting indices
    thrust::inclusive_scan(execpol->on(stream), d_indices, d_indices+count, d_indices);
    // sort will put them in the correct order
    thrust::sort_by_key(execpol->on(stream), ordinals.begin(), ordinals.end(), d_indices);
    // gather the keys
    // remove nulls -- adjust indices values
}

}  // namespace cudf
