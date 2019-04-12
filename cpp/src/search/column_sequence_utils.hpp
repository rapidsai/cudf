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

/**
 * @file Extension of column_utils.hpp for sequences of columns (
 * represented as arrays in main memory)
 */
#ifndef COLUMN_SEQUENCE_UTILS_HPP_
#define COLUMN_SEQUENCE_UTILS_HPP_

#include <utilities/column_utils.hpp>

// Note: If libcudf relied on having std::span / gsl::span, we could
// have just templatized this using either iterators or containers

namespace cudf {

void validate_all(const gdf_column* const * columns, gdf_num_columns_type num_columns);

bool has_uniform_column_sizes(const gdf_column* const * validated_column_sequence, gdf_num_columns_type num_columns);

bool have_matching_types(
    const gdf_column* const * validated_column_sequence_ptr_1,
    const gdf_column* const * validated_column_sequence_ptr_2,
    gdf_num_columns_type num_columns);

} // namespace cudf

#endif // COLUMN_SEQUENCE_UTILS_HPP_
