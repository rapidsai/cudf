/* Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Eyal Rozenberg <eyalroz@blazingdb.com>
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

#include "column_sequence_utils.hpp"
#include <utilities/column_utils.hpp>

namespace cudf {

void validate_all(const gdf_column* const * columns, gdf_num_columns_type num_columns)
{
    CUDF_EXPECTS ( (columns != nullptr) , "Null pointer to gdf_columns");
    for(gdf_num_columns_type i = 0; i < num_columns; i++) {
        validate(columns[i]);
    }
}

bool has_uniform_column_sizes(const gdf_column* const * validated_column_sequence, gdf_num_columns_type num_columns)
{
    if (num_columns == 0) { return true; }
    auto uniform_size = validated_column_sequence[0]->size;
    auto has_appropriate_size =
        [&uniform_size](const gdf_column* cp) { return cp->size == uniform_size; };
    return std::all_of(
        validated_column_sequence + 1,
        validated_column_sequence + num_columns,
        has_appropriate_size);
}

// Note: The type matching is strict, i.e. extra_dtype_info is taken into account
bool have_matching_types(
    const gdf_column* const * validated_column_sequence_ptr_1,
    const gdf_column* const * validated_column_sequence_ptr_2,
    gdf_num_columns_type num_columns)
{
    // I'd use `std::all_of` but that would require a zip iterator.
    for (gdf_column_index_type i = 0; i < num_columns; i++) {
        const auto& lhs = *(validated_column_sequence_ptr_1[i]);
        const auto& rhs = *(validated_column_sequence_ptr_2[i]);
        if (not cudf::have_same_type(lhs, rhs)) { return false; }
    }
    return true;
}

} // namespace cudf
