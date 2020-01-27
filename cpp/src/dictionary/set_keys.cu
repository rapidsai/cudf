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

#include <cudf/column/column.hpp>
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
        column_device_view d_keys = d_dictionary.dictionary_keys();

        auto index = d_indices.element<int32_t>(idx + d_dictionary.offset());
        return d_keys.element<Element>(index);
    }
};

struct dispatch_dictionary_resolver
{
    template<typename Element>
    std::unique_ptr<column> operator()( dictionary_column_view const& input, column_view const& new_keys,
                                        rmm::mr::device_memory_resource* mr, cudaStream_t stream )
    {
        column_device_view d_dictionary = column_device_view::create(input.parent());
        auto transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<int32_t>(0),
                                                                dictionary_resolver<Element>{*d_dictionary} );
        
        auto result = make_numeric_column(data_type{INT32}, input.size(),
                                          mask_state::UNALLOCATED, stream, mr);
        auto result_view = result->mutable_view();
        auto execpol = rmm::exec_policy(stream);
        thrust::lower_bound( execpol->on(stream), new_keys.begin<Element>(), new_keys.end<Element>(), 
                             transformer_itr, transformer_itr + input.keys_size(),
                             result_view.begin<int32_t>(), thrust::less<Element>() );

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

    std::shared_ptr<const column> keys_column = std::make_shared<column>();


    auto indices_column = experimental::type_dispatcher( new_keys.type(), dispatch_dictionary_resolver{}, 
        dictionary_column, new_keys, mr, stream );

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
