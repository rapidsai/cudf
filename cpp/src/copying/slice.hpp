/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Christian Noboa Mardini <christian@blazingdb.com>
 *     Copyright 2019 William Scott Malpica <william@blazingdb.com>
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

#include <cudf/types.h>
#include <vector>

namespace cudf {

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
 * When the values in the pair are equal, the function returns an empty column.
 * When the values in the pair are 'strictly decreasing', the outcome is
 * undefined.
 * When any of the values in the pair don't belong to the range[0, input column
 * size), the outcome is undefined.
 *
 * The output columns will be allocated by the function.
 * 
 * It uses an vector of 'cudaStream_t' in order to process in a parallel manner
 * the different output columns. In case of the size of the streams is less than
 * the size of the output columns, it reassigns again the streams with the remaining
 * output columns.
 *
 * Example:
 * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
 * indices: {1, 3, 5, 9, 2, 4, 8, 8}
 * output:  {{12, 14}, {20, 22, 24, 26}, {14, 16}, {}}
 *
 * @param[in] input_column  The input column whose rows will be sliced.
 * @param[in] indices       An array of indices that are used to take 'slices'
 * of the input column.
 * @param[in] streams       An vector of 'cudaStream_t' in order to process every output
 * @Returns output_columns  A set of gdf_column*. Each column can have
 * a different number of rows that are equal to the difference of two
 * consecutive indices in the indices array.
 */
std::vector<gdf_column*> slice(gdf_column const &                input_column,
                               gdf_index_type const*             indices,
                               gdf_size_type                     num_indices,
                               std::vector<cudaStream_t> const & streams = std::vector<cudaStream_t>{});

}  // namespace detail
}  // namespace cudf

#endif
