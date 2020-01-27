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
std::unique_ptr<column> encode( column_view const& input_column,
                                data_type indices_type,
                                rmm::mr::device_memory_resource* mr,
                                cudaStream_t stream)
{
    // side effects of this function were are now dependent on:
    // - resulting column elements are sorted ascending
    // - nulls are sorted to the beginning
    auto table_keys = experimental::detail::drop_duplicates( table_view{{input_column}},
                        std::vector<size_type>{0},
                        experimental::duplicate_keep_option::KEEP_FIRST,
                        true, mr, stream )->release(); // true == nulls are equal
    std::unique_ptr<column> keys(std::move(table_keys[0]));

    if( input_column.has_nulls() )
    {
        // the single null entry should be at the beginning -- side effect from drop_duplicates
        // copy the column without the null entry
        keys = std::make_unique<column>(experimental::slice(*keys, std::vector<size_type>{1,keys->size()})[0],stream,mr);
        keys->set_null_mask( rmm::device_buffer{0,stream,mr}, 0 ); // remove the null-mask
    }
    std::shared_ptr<const column> keys_column(keys.release());

    // this returns a column with no null entries
    // - it actually appears to ignore the null entries and tries to place the value regardless
    auto indices_column = cudf::experimental::lower_bound(table_view{{*keys_column}},
                    table_view{{input_column}},
                    std::vector<order>{order::ASCENDING},
                    std::vector<null_order>{null_order::AFTER},
                    mr );

    // create column with keys_column and indices_column
    std::vector<std::unique_ptr<column>> children;
    children.emplace_back(std::move(indices_column));
    return std::make_unique<column>(
        data_type{DICTIONARY32}, input_column.size(),
        rmm::device_buffer{0,stream,mr}, // no data in the parent
        copy_bitmask( input_column, stream, mr), input_column.null_count(),
        std::move(children),
        std::move(keys_column));
}

} // namespace dictionary
} // namespace cudf
