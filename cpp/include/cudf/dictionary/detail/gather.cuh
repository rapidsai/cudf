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

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
//#include <cudf/detail/copy.hpp>
//#include <cudf/detail/valid_if.cuh>
#include <cudf/detail/gather.cuh>

namespace cudf
{
namespace dictionary
{
namespace detail
{

/**
 * @brief Returns a new dictionary column using the specified indices to select
 * elements from the input column.
 *
 * Caller must update the validity mask in the output column.
 *
 * ```
 * d1 = {["a", "b", "c"],[2,0,0,1]}
 * map = [3, 0]
 * d2 = gather<true>( d1, map.begin(), map.end() )
 * d2 is {["a", "b", "c"],[1,2]}
 * ```
 *
 * @tparam NullifyOutOfBounds If true, indices outside the column's range are nullified.
 * @tparam MapIterator Iterator for retrieving integer indices of the column.
 *
 * @param dictionary Column instance for this operation.
 * @param begin Start of index iterator.
 * @param end End of index iterator.
 * @param remove_unused_keys If true, do not include any non-referenced keys from the output column.
 * @param mr Resource for allocating device memory.
 * @param stream CUDA stream to use kernels in this method.
 * @return New dictionary column containing the gathered elements.
 */
template<bool NullifyOutOfBounds, typename MapIterator>
std::unique_ptr<cudf::column> gather( dictionary_column_view const& dictionary,
                                      MapIterator begin, MapIterator end,
                                      bool remove_unused_keys,
                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                      cudaStream_t stream=0 )
{
    auto output_count = std::distance(begin, end);
    auto elements_count = dictionary.size();
    if( output_count == 0 || elements_count == 0 )
        return make_empty_column(data_type{DICTIONARY32});

    // create null mask -- caller must update this
    rmm::device_buffer null_mask;
    if( dictionary.parent().nullable() or NullifyOutOfBounds )
        null_mask = create_null_mask(output_count, mstate, stream, mr);

    // gather the new indices
    // problem here is handling nulls
    // -- the indices-column does not have any nulls -- they are in the parent
    // -- any new nulls need to be accounted for
    // -- finally, we need to somehow remove the nulls from the output here
    auto table = experimental::detail::gather( table_view{std::vector<column_view>{dictionary.indices()}},
                                               begin, end,
                                               false, NullifyOutOfBounds, false, mr, stream)->release();
    auto indices_column = table[0]; // can we remove the null_mask somehow?
    auto keys_copy = column( dictionary.dictionary_keys() );

    // TODO: handle remove_unused_keys

    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(indices_column));
    return std::make_unique<column>(
        data_type{DICTIONARY32}, indices_column.size(),
        rmm::device_buffer{0,stream,mr},
        null_mask, dictionary.null_count(),
        std::move(children),
        std::move(keys_copy));
}

/**
 * @brief Returns a new strings column using the specified indices to select
 * elements from the `strings` column.
 *
 * Caller must update the validity mask in the output column.

 * ```
 *
 * @tparam MapIterator Iterator for retrieving integer indices of the column.
 *
 * @param dictionary Column instance for this operation.
 * @param begin Start of index iterator.
 * @param end End of index iterator.
 * @param nullify_out_of_bounds If true, indices outside the column's range are nullified.
 * @param remove_unused_keys If true, do not include any non-referenced keys from the output column.
 * @param mr Resource for allocating device memory.
 * @param stream CUDA stream to use kernels in this method.
 * @return New strings column containing the gathered strings.
 */
template<typename MapIterator>
std::unique_ptr<cudf::column> gather( dictionary_column_view const& dictionary,
                                      MapIterator begin, MapIterator end,
                                      bool nullify_out_of_bounds,
                                      bool remove_unused_keys,
                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                      cudaStream_t stream=0 )
{
    if( nullify_out_of_bounds )
        return gather<true>( dictionary, begin, end, remove_unused_keys, mr, stream );
    return gather<false>( dictionary, begin, end, remove_unused_keys, mr, stream );
}


} // namespace detail
} // namespace strings
} // namespace cudf
