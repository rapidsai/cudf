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
 * Overwrites the range of elements in @p output indicated by the indices
 * [@p out_begin, @p out_ned) with the elements from @p input indicated by the
 * indices [@p in_begin, @p in_begin + N) (where N =
 * (@p out_end - @p out_begin)). Use the out-of-place copy function returning
 * std::unique_ptr<column> for uses cases requiring memory reallocation.
 *
 * If @p input and @p output refer to the same elements and the ranges overlap,
 * the behavior is undefined.
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
 * @param out_begin The starting index of the output range (inclusive)
 * @param out_end The index of the last element in the output range (exclusive)
 * @param in_begin The starting index of the input range (inclusive)
 * @return void
 */
void copy_range(mutable_column_view& output, column_view const& input,
                size_type out_begin, size_type out_end, size_type in_begin);

/**
 * @brief Copies a range of elements out-of-place from one column to another.
 *
 * Creates a new column as-if an in-place copy was performed into @p output;
 * i.e. it is as if a copy of @p output was created first and then the elements
 * indicated by the indices [@p out_begin, @p out_end) were overwritten by the
 * elements from the indices [@p in_begin, @p in_begin + N) (where N =
 * (@p out_end - @p out_begin)).
 *
 * If @p input and @p output refer to the same elements and the ranges overlap,
 * the behavior is undefined.
 *
 * @throws `cudf::logic_error` for invalid range (if @p out_begin < 0,
 * @p out_begin > @p out_end, @p out_begin >= @p output.size(),
 * @p out_end > @p output.size(), @p in_begin < 0, in_begin >= @p input.size(),
 * or @p in_begin + @p out_end - @p out_begin > @p input.size()).
 * @throws `cudf::logic_error` if @p output and @p input have different types.
 *
 * @param output The column to copy from outside the range.
 * @param input The column to copy from inside the range.
 * @param out_begin The starting index of the output range (inclusive)
 * @param out_end The index of the last element in the output range (exclusive)
 * @param in_begin The starting index of the input range (inclusive)
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
