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
    CUDF_EXPECTS( indices_column.type().id()==cudf::type_id::INT32, "indices column must be INT32" );

    auto keys_copy = std::make_unique<column>( keys_column, stream, mr );
    auto indices_copy = std::make_unique<column>( indices_column, stream, mr);
    rmm::device_buffer null_mask;
    auto null_count = indices_column.null_count();
    if( null_count )
    {
        indices_copy->set_null_mask(null_mask,0);
        null_mask = copy_bitmask(indices_column,stream,mr);
    }

    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(indices_copy));
    children.emplace_back(std::move(keys_copy));
    return std::make_unique<column>(
        data_type{DICTIONARY32}, indices_column.size(),
        rmm::device_buffer{0,stream,mr},
        null_mask, null_count,
        std::move(children));
}

std::unique_ptr<column> make_dictionary_column( std::unique_ptr<column> keys_column,
                                                std::unique_ptr<column> indices_column,
                                                rmm::device_buffer&& null_mask,
                                                size_type null_count )
{
    CUDF_EXPECTS( !keys_column->has_nulls(), "keys column must not have nulls" );
    CUDF_EXPECTS( !indices_column->has_nulls(), "indices column must not have nulls" );
    CUDF_EXPECTS( indices_column->type().id()==cudf::type_id::INT32, "indices must be type INT32" );

    auto count = indices_column->size();
    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(indices_column));
    children.emplace_back(std::move(keys_column));
    return std::make_unique<column>(
        data_type{DICTIONARY32}, count,
        rmm::device_buffer{0},
        null_mask, null_count,
        std::move(children));
}

}  // namespace cudf
