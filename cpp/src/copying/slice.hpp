/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Christian Noboa Mardini <christian@blazingdb.com>
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

#ifndef COPYING_SLICE_HPP
#define COPYING_SLICE_HPP

namespace cudf {

// Forward declaration
struct column_array;

namespace detail {

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
 * It uses an array of 'cudaStream_t' in order to process in a parallel manner
 * the different output columns. In case of the size of the streams is less than
 * the size of the output columns, it reassigns again the streams with the remaining
 * output columns.
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
 * @param[in] streams An array of 'cudaStream_t' in order to process every output
 * column in a parallel way. It can be null.
 * @param[in] stream_size The size of the 'cudaStream_t' array.
 */
void slice(gdf_column const*   input_column,
           gdf_column const*   indexes,
           cudf::column_array* output_columns,
           cudaStream_t*       streams,
           gdf_size_type       streams_size);

}  // namespace detail
}  // namespace cudf

#endif
