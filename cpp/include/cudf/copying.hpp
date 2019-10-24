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

#include "cudf.h"
#include "types.hpp"

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
 * @brief Copies a range of elements in-place from one column to another.
 *
 * Copies N elements of @p input starting at @p in_begin to the N
 * elements of @p output starting at @p out_begin, where
 * N = (@p out_end - @p out_begin).
 *
 * This function updates in-place assuming that no memory reallocation is
 * necessary for @p output. Use the out-of-place copy function returning
 * std::unique_ptr<column> for uses cases requiring memory reallocation.
 *
 * If the input and output are the same object and the ranges overlap, the
 * behavior is undefined.
 *
 * @throws `cudf::logic_error` if memory reallocation is required (e.g. for
 * variable width types).
 * @throws `cudf::logic_error` for invalid range (if @p out_begin < 0,
 * @p out_begin > @p out_end, @p out_begin >= @p output.size(),
 * @p out_end > @p output.size(), @p in_begin < 0, in_begin >= @p input.size(),
 * or @p in_begin + @p out_end - @p out_begin > @p input.size()).
 * @throws `cudf::logic_error` if @p output and @p input have different types.
 * @throws `cudf::logic_error` if @p input has null values and @p output is not
 * nullable.
 *
 * @param output The preallocated column to copy into
 * @param input The column to copy from
 * @param out_begin The starting index of the output range
 * @param out_end The index one past the end of the output range
 * @param in_begin The starting index of the input range
 * @return void
 */
void copy_range(mutable_column_view& output, column_view const& input,
                size_type out_begin, size_type out_end, size_type in_begin);

/**
 * @brief Copies a range of elements out-of-place from one column to another.
 *
 * This copy function updates out-of-place creating a new column object to
 * return. The returned column and @p input holds the same values for the
 * N (= @p out_end - @p out_begin) elements from @p out_beign and @p in_begin,
 * respectively. The returned column stores the same values to @p output outside
 * the copy range (i.e. [@p out_begin, @p out_end) of the returned column and
 * [@p in_begin, @p in_begin + N) of @p input are identical, and the returned
 * column and @p output holds the same values in [0, @p out_begin) and
 * [@p out_end, output.size())).
 *
 * If the input and output are the same object and the ranges overlap, the
 * behavior is undefined.
 *
 * @throws `cudf::logic_error` for invalid range (if @p out_begin < 0,
 * @p out_begin > @p out_end, @p out_begin >= @p output.size(),
 * @p out_end > @p output.size(), @p in_begin < 0, in_begin >= @p input.size(),
 * or @p in_begin + @p out_end - @p out_begin > @p input.size()).
 * @throws `cudf::logic_error` if @p output and @p input have different types.
 *
 * @param output The column to copy from outside the range.
 * @param input The column to copy from inside the range.
 * @param out_begin The starting index of the output range
 * @param out_end The index one past the end of the output range
 * @param in_begin The starting index of the input range
 * @param mr Memory resource to allocate the result output column.
 * @return std::unique_ptr<column> The result output column
 */
std::unique_ptr<column> copy_range(column_view const& output,
                                   column_view const& input,
                                   size_type out_begin, size_type out_end,
                                   size_type in_begin,
                                   rmm::mr::device_memory_resource* mr =
                                       rmm::mr::get_default_resource());

}  // namespace experimental
}  // namespace cudf
