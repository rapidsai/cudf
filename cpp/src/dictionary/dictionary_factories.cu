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
#include <cudf/dictionary/dictionary_factories.hpp>

#include <rmm/thrust_rmm_allocator.h>

namespace cudf
{

std::unique_ptr<column> make_dictionary_column( column_view const& keys_column,
                                                column_view const& indices_column,
                                                rmm::mr::device_memory_resource* mr,
                                                cudaStream_t stream)
{
    CUDF_EXPECTS( !keys_column.has_nulls(), "keys column must not have nulls" );
    if( keys_column.size()==0 )
        return make_empty_column( data_type{DICTIONARY32} );

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

std::unique_ptr<column> make_dictionary_column( std::shared_ptr<column>&& keys_column,
                                                std::unique_ptr<column>&& indices_column,
                                                rmm::device_buffer&& null_mask,
                                                size_type null_count )
{
    CUDF_EXPECTS( !keys_column->has_nulls(), "keys column must not have nulls" );
    CUDF_EXPECTS( !indices_column->has_nulls(), "indices column must not have nulls" );

    size_type count = indices_column->size();
    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(indices_column));
    return std::make_unique<column>(
        data_type{DICTIONARY32}, count,
        rmm::device_buffer{0},
        null_mask, null_count,
        std::move(children),
        std::move(keys_column));
}

}  // namespace cudf
