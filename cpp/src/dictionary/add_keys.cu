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
#include <cudf/copying.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/search.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/dictionary/update_keys.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/sequence.h>

namespace cudf
{
namespace dictionary
{

/**
 * @brief Create a new dictionary column by adding the new keys elements
 * to the existing dictionary_column.
 * 
 * ```
 * Example:
 * d1 = {[a,b,c,d,f],{4,0,3,1,2,2,2,4,0}}
 * d2 = add_keys(d1,[d,b,e])
 * d2 is now {[a,b,c,d,e,f],[5,0,3,1,2,2,2,5,0]}
 * ```
 * 
 */
std::unique_ptr<column> add_keys( dictionary_column_view const& dictionary_column,
                                  column_view const& new_keys,
                                  rmm::mr::device_memory_resource* mr,
                                  cudaStream_t stream)
{
    CUDF_EXPECTS( !new_keys.has_nulls(), "Keys must not have nulls" );

    auto count = dictionary_column.size();
    // first, concatenate the keys together
    auto combined_keys = cudf::concatenate( std::vector<column_view>{dictionary_column.dictionary_keys(), new_keys}, mr, stream);
    // create sequence column to help with remapping indices values
    auto execpol = rmm::exec_policy(stream);
    rmm::device_vector<int32_t> map_indices(count);
    thrust::sequence( execpol->on(stream), map_indices.begin(), map_indices.end() );
    column_view keys_indices_view( data_type{INT32}, count, map_indices.data().get() );
    // drop_duplicates will sort and remove any duplicate keys we may been given
    // the keys_indices values will also be sorted according to the keys
    auto table_keys = experimental::detail::drop_duplicates( table_view{{keys_indices_view,*combined_keys}},
                            std::vector<size_type>{1},
                            experimental::duplicate_keep_option::KEEP_FIRST,
                            true, mr, stream )->release();
    // the new keys for the dictionary are now in table_keys[1]
    std::shared_ptr<const column> keys_column(std::move(table_keys[1].release()));

    // the table_keys[0] values contain the old indices placed in their new positions
    auto indices_column = cudf::experimental::lower_bound(table_view{{table_keys[0]->view()}},
                    table_view{{dictionary_column.indices()}}, // map old indices to new positions
                    std::vector<order>{order::ASCENDING},
                    std::vector<null_order>{null_order::AFTER}, // should be no nulls
                    mr );

    // create column with keys_column and indices_column
    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(indices_column));
    return std::make_unique<column>(
        data_type{DICTIONARY32}, count,
        rmm::device_buffer{0,stream,mr}, // no data in the parent
        copy_bitmask( dictionary_column.parent(), stream, mr), // nulls have
        dictionary_column.null_count(),                        // not changes
        std::move(children),
        std::move(keys_column));
}

} // namespace dictionary
} // namespace cudf
