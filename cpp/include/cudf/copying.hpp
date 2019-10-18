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

/** ---------------------------------------------------------------------------*
* @brief Indicates when to allocate a mask, based on an existing mask.
* ---------------------------------------------------------------------------**/
enum mask_allocation_policy {
    NEVER, ///< Do not allocate a null mask, regardless of input
    RETAIN, ///< Allocate a null mask if the input contains one
    ALWAYS ///< Allocate a null mask, regardless of input
};

/*
 * Initializes and returns an empty column of the same type as the `input`.
 *
 * @param input Immutable view of input column to emulate
 * @return std::unique_ptr<column> An empty column of same type as `input`
 */
std::unique_ptr<column> empty_like(column_view input);

/**
 * @brief Creates an uninitialized new column of the same size and type as the `input`.
 * Supports only fixed-width types.
 *
 * @param input Immutable view of input column to emulate
 * @param mask_alloc Optional, Policy for allocating null mask. Defaults to RETAIN.
 * @param mr Optional, The resource to use for all allocations
 * @return std::unique_ptr<column> A column with sufficient uninitialized capacity to hold the same number of elements as `input` of the same type as `input.type()`
 */
std::unique_ptr<column> allocate_like(column_view input,
                                      mask_allocation_policy mask_alloc = RETAIN,
                                      rmm::mr::device_memory_resource *mr =
                                          rmm::mr::get_default_resource());

/**
 * @brief Creates an uninitialized new column of the specified size and same type as the `input`.
 * Supports only fixed-width types.
 *
 * @param input Immutable view of input column to emulate
 * @param size The desired number of elements that the new column should have capacity for
 * @param mask_alloc Optional, Policy for allocating null mask. Defaults to RETAIN.
 * @param mr Optional, The resource to use for all allocations
 * @return std::unique_ptr<column> A column with sufficient uninitialized capacity to hold the specified number of elements as `input` of the same type as `input.type()`
 */
std::unique_ptr<column> allocate_like(column_view input, size_type size,
                                      mask_allocation_policy mask_alloc = RETAIN,
                                      rmm::mr::device_memory_resource *mr =
                                          rmm::mr::get_default_resource());

/**
 * @brief Creates a table of empty columns with the same types as the `input_table`
 *
 * Creates the `cudf::column` objects, but does not allocate any underlying device
 * memory for the column's data or bitmask.
 *
 * @param input_table Immutable view of input table to emulate
 * @return std::unique_ptr<table> A table of empty columns with the same types as the columns in `input_table`
 */
std::unique_ptr<table> empty_like(table_view input_table);

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
 * @throws `cudf::logic_error` if `indices` size is not even.
 *
 * @param input Immutable view of input column for slicing
 * @param indices A vector indices that are used to take 'slices' of the `input`.
 * @return vector<std::unique_ptr<column_view>> A vector of `column_view` unique_ptr each of which may have a different number of rows. 
 */

std::vector<std::unique_ptr<column_view>> slice(column_view const& input,
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
 * When the value in `splits` is not in the range [0, size], the outcome is
 * undefined..
 * When the values in the `splits` are 'strictly decreasing', the outcome is
 * undefined.
 * When the `splits` is empty, an empty vector of `column_view` unique_ptr is returned.
 *
 * Example:
 * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
 * splits: {2, 5, 9}
 * output:  {{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}}
 *
 * @throws `cudf::logic_error` if `splits` has end index > size of `input`.
 *
 * @param input Immutable view of input column for spliting
 * @param indices A vector indices that are used to split the `input`
 * @return vector<std::unique_ptr<column_view>> A vector of `column_view` unique_ptr each of which may have a different number of rows. 
 */
std::vector<std::unique_ptr<column_view>> split(column_view const& input,
                                                std::vector<size_type> const& splits);

}  // namespace experimental
}  // namespace cudf
