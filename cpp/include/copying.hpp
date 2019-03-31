/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#ifndef COPYING_HPP
#define COPYING_HPP

#include "cudf.h"
#include "types.hpp"

namespace cudf {
/**
 * @brief Scatters the rows (including null values) of a set of source columns
 * into a set of destination columns.
 * 
 * The two sets of columns must have equal numbers of columns.
 *
 * Scatters the rows of the source columns into the destination columns
 * according to a scatter map such that row "i" from the source columns will be
 * scattered to row "scatter_map[i]" in the destination columns.
 *
 * The datatypes between coresponding columns in the source and destination
 * columns must be the same.
 *
 * The number of elements in the scatter_map must equal the number of rows in
 * the source columns.
 *
 * If any index in scatter_map is outside the range of [0, num rows in
 * destination_columns), the result is undefined.
 *
 * If the same index appears more than once in scatter_map, the result is
 * undefined.
 *
 * @Param[in] source_table The columns whose rows will be scattered
 * @Param[in] scatter_map An array that maps rows in the input columns
 * to rows in the output columns.
 * @Param[out] destination_table A preallocated set of columns with a number
 * of rows equal in size to the maximum index contained in scatter_map
 *
 * @Returns GDF_SUCCESS upon successful completion
 */
void scatter(table const* source_table, gdf_index_type const scatter_map[],
                  table* destination_table);

/**
 * @brief Gathers the rows (including null values) of a set of source columns
 * into a set of destination columns.
 * 
 * The two sets of columns must have equal numbers of columns.
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
 * If any index in the gather_map is outside the range [0, num rows in
 * source_columns), the result is undefined.
 *
 * If the same index appears more than once in gather_map, the result is
 * undefined.
 *
 * @param[in] source_table The input columns whose rows will be gathered
 * @param[in] gather_map An array of indices that maps the rows in the source
 * columns to rows in the destination columns.
 * @param[out] destination_table A preallocated set of columns with a number
 * of rows equal in size to the number of elements in the gather_map that will
 * contain the rearrangement of the source columns based on the mapping. Can be
 * the same as `source_table` (in-place gather).
 *
 * @Returns GDF_SUCCESS upon successful completion
 */
void gather(table const* source_table, gdf_index_type const gather_map[],
                 table* destination_table);

/**
 * @brief Slices a column (including null values) into a set of columns
 * according to a set of indices.
 *
 * The "slice" function divides part of the input column into multiple intervals
 * of rows using the indices values and it stores the intervals into the output
 * columns. Regarding the interval of indices, a pair of values are taken from
 * the indices array in a consecutive manner. The pair of indices are left-closed
 * and right-open.
 *
 * The pairs of indices in the array are required to comply with the following
 * conditions:
 * a, b belongs to Range[0, input column size]
 * a <= b, where the position of a is less or equal to the position of b.
 * There is no relationship between the values of each pair of indices.
 *
 * Exceptional cases for the indices array are:
 * When the values in the pair are equal, the function return an empty column.
 * When the values in the pair are 'strictly decreasing', the outcome is
 * undefined.
 * When any of the values in the pair don't belong to the range[0, input column
 * size), the outcome is undefined.
 *
 * It is required that the output columns will be preallocated. The size of each
 * of the columns can be of different value. The number of columns must be equal
 * to the number of indices in the array divided by two. The indices array must
 * have an even size. The datatypes of the input column and the output columns
 * must be the same.
 *
 * Example:
 * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
 * indexes: {1, 3, 5, 9, 2, 4, 8, 8}
 * output:  {{12, 14}, {20, 22, 24, 26}, {14, 16}, {}}
 *
 * @param[in] input_column The input column whose rows will be sliced.
 * @param[in] indexes An array of indices that are used to take 'slices'
 * of the input column.
 * @param[out] output_columns A preallocated set of columns. Each column
 * has a different number of rows that are equal to the difference of two
 * consecutive indices in the indices array.
 */
void slice(gdf_column const*   input_column,
           gdf_column const*   indexes,
           cudf::column_array* output_columns);

/**
 * @brief Splits all the column (including null values) into a set of columns
 * according to a set of indices.
 *
 * The "split" function divides all the input column into multiple intervals
 * of rows using the indices values and it stores the intervals into the output
 * columns. Regarding the interval of indices, a pair of values are taken from
 * the indices array in a consecutive manner. The pair of indices are left-closed
 * and right-open.
 *
 * The indices array ('indexes') is require to be a monotonic non-decreasing set.
 * The indices in the array are required to comply with the following conditions:
 * a, b belongs to Range[0, input column size]
 * a <= b, where the position of a is less or equal to the position of b.
 *
 * The split function will take a pair of indices from the indices array
 * ('indexes') in a consecutive manner. For the first pair, the function will
 * take the value 0 and the first element of the indices array. For the last pair,
 * the function will take the last element of the indices array and the size of
 * the input column.
 *
 * Exceptional cases for the indices array are:
 * When the values in the pair are equal, the function return an empty column.
 * When the values in the pair are 'strictly decreasing', the outcome is
 * undefined.
 * When any of the values in the pair don't belong to the range[0, input column
 * size), the outcome is undefined.
 *
 * It is required that the output columns will be preallocated. The size of each
 * of the columns can be of different value. The number of columns must be equal
 * to the number of indices in the array plus one. The datatypes of the input
 * column and the output columns must be the same.
 *
 * Example:
 * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
 * indexes: {2, 5, 9}
 * output:  {{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}}
 *
 * @param[in] input_column The input column whose rows will be split.
 * @param[in] indexes An array of indices that are used to divide the input
 * column into multiple columns.
 * @param[out] output_columns A preallocated set of columns. Each column
 * has a different number of rows that are equal to the difference of two
 * consecutive indices in the indices array.
 */
void split(gdf_column const*   input_column,
           gdf_column const*   indexes,
           cudf::column_array* output_columns);

}  // namespace cudf

#endif  // COPYING_H
