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

#pragma once 

#include <cudf/cudf.h>
#include <cudf/types.hpp>

namespace cudf {
namespace experimental {

/**
 * @brief Slices a column_view (including null values) into a set of column_views
 * according to a set of indices.
 *
 * The `slice` function creates `column_view`s from `input` with multiple intervals
 * of rows using the `indices` values. Regarding the interval of indices, a pair of 
 * values are taken from the indices vector in a consecutive manner.
 * The pair of indices are left-closed and right-open.
 *
 * The pairs of indices in the vector are required to comply with the following
 * conditions:
 * a, b belongs to Range[0, input column size]
 * a <= b, where the position of a is less or equal to the position of b.
 *
 * Exceptional cases for the indices array are:
 * When the values in the pair are equal, the function returns an empty `column_view`.
 * When the values in the pair are 'strictly decreasing', the outcome is
 * undefined.
 * When any of the values in the pair don't belong to the range[0, input column
 * size), the outcome is undefined.
 * When the indices vector is empty, an empty vector of `column_view` unique_ptr is returned.
 *
 * @example:
 * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
 * indices: {1, 3, 5, 9, 2, 4, 8, 8}
 * output:  {{12, 14}, {20, 22, 24, 26}, {14, 16}, {}}
 *
 * @param input Immutable view of input column for slicing
 * @param indices A vector indices that are used to take 'slices' of the `input`.
 * @return vector<std::unique_ptr<column_view>> A vector of `column_view` unique_ptr each of which may have a different number of rows. 
 */

std::vector<std::unique_ptr<column_view>> slice(column_view const& input_table,
                                                std::vector<size_type> const& indices);


/**
 * @brief Splits a `column_view` (including null values) into a set of `column_view`s
 * according to a set of splits.
 *
 * The `splits` vector is required to be a monotonic non-decreasing set.
 * The indices in the vector are required to comply with the following conditions:
 * a, b belongs to Range[0, input column size]
 * a <= b, where the position of a is less or equal to the position of b.
 *
 * The split function will take a pair of indices from the `splits` vector
 * in a consecutive manner. For the first pair, the function will
 * take the value 0 and the first element of the `splits` vector. For the last pair,
 * the function will take the last element of the `splits` vector and the size of
 * the `input`.
 *
 * Exceptional cases for the indices array are:
 * When the values in the pair are equal, the function return an empty `column_view`.
 * When the values in the pair are 'strictly decreasing', the outcome is
 * undefined.
 * When any of the values in the pair don't belong to the range[0, input column
 * size), the outcome is undefined.
 * When the indices array is empty, an empty vector of `column_view` unique_ptr is returned.
 *
 * Example:
 * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
 * splits: {2, 5, 9}
 * output:  {{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}}
 *
 * @param input Immutable view of input column for spliting
 * @param indices A vector indices that are used to split the `input`
 * @return vector<std::unique_ptr<column_view>> A vector of `column_view` unique_ptr each of which may have a different number of rows. 
 */
std::vector<std::unique_ptr<column_view>> split(column_view const& input_table,
                                                std::vector<size_type> const& splits);

}  // namespace experimental
}  // namespace cudf
