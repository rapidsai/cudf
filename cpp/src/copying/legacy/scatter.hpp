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

#ifndef SCATTER_HPP
#define SCATTER_HPP

#include <cudf/types.h>

namespace cudf {

// Forward declaration
struct table;

namespace detail {
/**
 * @brief Scatters a set of source columns into the rows of a set of
 * destination columns.
 *
 * Scatters the source columns into the destination columns according
 * to a scatter map such that row "i" of the source columns will replace
 * row "scatter_map[i]" of the destination columns.
 *
 * The datatypes between coresponding columns in the source and destination
 * columns must be the same.
 *
 * The number of elements in the scattermap must equal the number of rows in the
 * source columns.
 *
 * Optionally considers negative values in the scattermap. If enabled, a negative
 * value `i` in `scatter_map` is interpreted as `i + num_destination_rows`.
 *
 * Optionally performs bounds checking on the values of the `scatter_map` and
 * raises a runtime error if any of its values are outside the range
 * [0, num_source_rows).
 *
 * Optionally ignores values in the scattermap outside of the range
 * [0, num_source_rows).
 *
 * @param[in] source_table The input columns containing values that will replace
 * values in the destination columns.
 * @param[in] scatter_map A non-nullable column of integral indices that maps the
 * rows in the source columns to rows in the destination columns. Must be equal
 * in size to the number of elements in the source columns.
 * @param[out] destination_table The set of columns whose rows will be replaced
 * by elements in the corresponding columns of the `source_table`.
 * @param check_bounds Optionally perform bounds checking on the values of
 * `scatter_map` and throw an error if any of its values are out of bounds.
 * @param ignore_out_of_bounds Ignore values in `scatter_map` that are
 * out of bounds. Currently incompatible with `allow_negative_indices`,
 * i.e., setting both to `true` is undefined.
 * @param allow_negative_indices Interpret each negative index `i` in the
 * scattermap as the positive index `i+num_destination_rows`.
 *---------------------------------------------------------------------------**/
void scatter(table const* source_table, gdf_column const& scatter_map,
	     table* destination_table, bool check_bounds = false,
	     bool allow_negative_indices = false);

/**
 * @overload This function accepts `scatter_map` as an array instead of
 * a `gdf_column`.
 */
void scatter(table const* source_table, gdf_index_type const scatter_map[],
	     table* destination_table, bool check_bounds = false,
	     bool allow_negative_indices = false);

}  // namespace detail

}  // namespace cudf

#endif
