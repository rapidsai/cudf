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
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/dictionary/encode.hpp>

#include <rmm/thrust_rmm_allocator.h>

namespace cudf
{
namespace dictionary
{

/**
 * @brief Create a new dictionary column from a column_view.
 *
 */
std::unique_ptr<column> decode( dictionary_column_view const& source,
                                rmm::mr::device_memory_resource* mr,
                                cudaStream_t stream)
{
    if( source.size()==0 || source.keys_size()==0 )
        return make_empty_column( data_type{EMPTY} );
    auto keys = source.keys();
    auto indices = source.indices();
    if( indices.size()==0 )
        return make_empty_column( keys.type() );
        
    return nullptr;
}

} // namespace dictionary
} // namespace cudf
