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

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/dictionary/encode.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/binary_search.h>

namespace cudf
{
namespace dictionary
{
namespace detail
{
namespace
{

template<typename Element>
struct dictionary_resolver
{
    column_device_view d_dictionary;

    __device__ Element operator()(size_type idx)
    {
        if( d_dictionary.is_null(idx) )
            return Element{};
        
        column_device_view d_indices = d_dictionary.child(0);
        column_device_view* d_keys = d_dictionary.dictionary_keys();
        auto index = d_indices.element<int32_t>(idx + d_dictionary.offset());
        printf("     keys=%p,%d; index=%d\n", d_keys, (int)d_keys->size(), (int)index);
        return d_keys->element<Element>(index);
    }
};

template<>
__device__ string_view dictionary_resolver<string_view>::operator()(size_type idx)
{
        column_device_view d_indices = d_dictionary.child(0);
        column_device_view* d_keys = d_dictionary.dictionary_keys();
        auto index = d_indices.element<int32_t>(idx + d_dictionary.offset());
        printf(":keys=%p,%d; index=%d\n", d_keys, (int)d_keys->size(), (int)index);
        string_view rtn = d_keys->element<string_view>(index);
        printf(":string_view=%p,%d\n", rtn.data(), (int)rtn.size_bytes() );
        return rtn;
}

template<typename Element>
struct less_operator
{
    column_device_view d_dictionary;
    column_device_view d_keys;

    __device__ Element resolve_dictionary_element(size_type idx)
    {
//        if( d_dictionary.is_null(idx) )
//            return Element{};
        column_device_view d_indices = d_dictionary.child(0);
        column_device_view* d_keys = d_dictionary.dictionary_keys();
        auto index = d_indices.element<int32_t>(idx + d_dictionary.offset());
        printf("     keys=%p,%d; index=%d\n", d_keys, (int)d_keys->size(), (int)index);
        return d_keys->element<Element>(index);
    }

    __device__ bool operator()(size_type lhs_index, size_type rhs_index) const
    {
        Element lhs = d_keys.element<Element>(lhs_index);
        Element rhs = resolve_dictionary_element(rhs_index);
        return lhs < rhs;
    }
};

template<>
__device__ bool less_operator<string_view>::operator()(size_type lhs_index, size_type rhs_index) const
{
    string_view lhs = d_keys.element<string_view>(lhs_index);
    printf(":lhs=(%p,%d)\n", lhs.data(), (int)lhs.size_bytes() );

//    string_view rhs = resolve_dictionary_element<string_view>(rhs_index);
    column_device_view d_indices = d_dictionary.child(0);
    column_device_view* d_keys = d_dictionary.dictionary_keys();
    auto index = d_indices.element<int32_t>(rhs_index + d_dictionary.offset());
    printf(":keys=%p,%d; index=%d\n", d_keys, (int)d_keys->size(), (int)index);
    string_view rhs = d_keys->element<string_view>(index);
    printf(":rhs=%p,%d\n", rhs.data(), (int)rhs.size_bytes() );
    
    return lhs < rhs;
}                                                               <- 
                                                                 |
struct dispatch_compute_indices                                  |
{                                                                |
    template<typename Element>                                   |
    std::unique_ptr<column> operator()( dictionary_column_view co|nst& input, column_view const& new_keys,
                                        rmm::mr::device_memory_re|source* mr, cudaStream_t stream )
    {                                                            |
        auto d_dictionary = column_device_view::create(input.pare|nt());
        auto transformer_itr = thrust::make_transform_iterator(  | thrust::make_counting_iterator<int32_t>(0),
                                                                 | dictionary_resolver<Element>{*d_dictionary} );
                                                                 | 
        auto result = make_numeric_column(data_type{INT32},      |input.size(),
                                         mask_state::UNALLOCATED,|stream, mr);
        auto d_result = result->mutable_view().data<int32_t>();  |
        auto execpol = rmm::exec_policy(stream);                 |
        printf("before lower_bound\n");                          |
        //                                                       |
        // THIS IS FAILING WITH cudaErrorIllegalAddress          |
        // need to investigate this with a standalone program -- |lower_bound may only work with numerics
        // if so, look at launch_search in search.cu to see how  |they use index iterators instead of element ones
        // (some of that is started above in the less_operator)--/
        thrust::lower_bound( execpol->on(stream), new_keys.begin<Element>(), new_keys.end<Element>(), 
                             transformer_itr, transformer_itr + input.size(),
                             d_result, less_operator<Element>{} );//thrust::less<Element>() );
        printf("after lower_bound\n");
        result->set_null_count(0);
        return result;
    }
};

} // namespace

/**
 * @brief Create a new dictionary by applying the given keys.
 *
 */
std::unique_ptr<column> set_keys( dictionary_column_view const& dictionary_column,
                                  column_view const& new_keys,
                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                  cudaStream_t stream = 0)
{
    CUDF_EXPECTS( !new_keys.has_nulls(), "keys parameter must not have nulls");
    auto keys = dictionary_column.dictionary_keys();
    CUDF_EXPECTS( keys.type()==new_keys.type(), "keys types must match");

    // copy the keys
    std::shared_ptr<const column> keys_column = std::make_shared<column>(new_keys);
    // compute the new indices
    auto indices_column = experimental::type_dispatcher( new_keys.type(), dispatch_compute_indices{}, 
        dictionary_column, new_keys, mr, stream );

    // compute the new nulls
    auto matches = experimental::detail::contains( keys, new_keys, mr, stream );
    auto d_matches = matches->view().data<experimental::bool8>();
    auto d_null_mask = dictionary_column.null_mask();
    auto new_nulls = experimental::detail::valid_if( thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(dictionary_column.size()),
                    [d_null_mask, d_matches] __device__ (size_type idx) {
                        return /*d_matches[idx] &&*/ (d_null_mask ? bit_is_set(d_null_mask,idx) : true);
                    }, stream, mr);

    // create column with keys_column and indices_column
    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(indices_column));
    return std::make_unique<column>(
        data_type{DICTIONARY32}, dictionary_column.size(),
        rmm::device_buffer{0,stream,mr}, // no data in the parent
        new_nulls.first, new_nulls.second,
        std::move(children),
        std::move(keys_column));
}
} // namespace detail

// external API

std::unique_ptr<column> set_keys( dictionary_column_view const& dictionary_column,
                                  column_view const& keys,
                                  rmm::mr::device_memory_resource* mr)
{
    return detail::set_keys(dictionary_column, keys, mr);
}

} // namespace dictionary
} // namespace cudf
