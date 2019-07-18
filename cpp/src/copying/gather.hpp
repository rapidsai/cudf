/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#ifndef GATHER_HPP
#define GATHER_HPP

#include <cudf/types.h>

namespace cudf {

// Forward declaration
struct table;

namespace detail {

/**---------------------------------------------------------------------------*
 * @brief Gathers the rows (including null values) of a set of source columns
 * into a set of destination columns.
 *
 * Gathers the rows of the source columns into the destination columns according
 * to a gather map such that row "i" in the destination columns will contain
 * row "gather_map[i]" from the source columns.
 *
 * The datatypes between coresponding columns in the source and destination
 * columns must be the same.
 *
 * The number of elements in the gather_map must equal the number of rows in the
 * destination columns.
 *
 * Optionally performs bounds checking on the values of the `gather_map` that
 * ignores values outside [0, num_source_rows). It is undefined behavior if a
 * value in `gather_map` is outside these bounds and bounds checking is not
 * enabled.
 *
 * If the same index appears more than once in gather_map, the result is
 * undefined.
 *
 * @param[in] source_table The input columns whose rows will be gathered
 * @param[in] gather_map An array of indices that maps the rows in the source
 * columns to rows in the destination columns.
 * @param[out] destination_table A preallocated set of columns with a number
 * of rows equal in size to the number of elements in the gather_map that will
 * contain the rearrangement of the source columns based on the mapping
 * determined by the gather_map. Can be the same as `source_table` (in-place
 * gather).
 * @param check_bounds Optionally perform bounds checking on the values of
 * `gather_map`
 * @param stream Optional CUDA stream on which to execute kernels
 * @return gdf_error
 *---------------------------------------------------------------------------**/
void gather(table const* source_table, gdf_index_type const gather_map[],
                 table* destination_table, bool check_bounds = false,
                 cudaStream_t stream = 0);
}  // namespace detail
}  // namespace cudf

#endif
