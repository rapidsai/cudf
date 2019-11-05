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
#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>

#include <memory>

namespace cudf {
namespace experimental {

/**
 * @brief Gathers the specified rows (including null values) of a set of columns.
 *
 * Gathers the rows of the source columns according to `gather_map` such that row "i"
 * in the resulting table's columns will contain row "gather_map[i]" from the source columns.
 * The number of rows in the result table will be equal to the number of elements in
 * `gather_map`.
 *
 * A negative value `i` in the `gather_map` is interpreted as `i+n`, where
 * `n` is the number of rows in the `source_table`.
 *
 * @throws `cudf::logic_error` if `check_bounds == true` and an index exists in
 * `gather_map` outside the range `[-n, n)`, where `n` is the number of rows in
 * the source table. If `check_bounds == false`, the behavior is undefined.
 *
 * @param[in] source_table The input columns whose rows will be gathered
 * @param[in] gather_map View into a non-nullable column of integral indices that maps the
 * rows in the source columns to rows in the destination columns.
 * @param[in] check_bounds Optionally perform bounds checking on the values
 * of `gather_map` and throw an error if any of its values are out of bounds.
 * @params[in] mr The resource to use for all allocations
 * @return std::unique_ptr<table> Result of the gather
 */
std::unique_ptr<table> gather(table_view const& source_table, column_view const& gather_map,
			      bool check_bounds = false,
			      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Creates a new table that results from scattering values from the rows
 * of a source table into the rows of a target table according to a scatter map.
 *
 * Scatters values from the source table into the target table out-of-place,
 * returning a "destination table". The scatter is performed according to a scatter map
 * such that row "i" of the source table will be equal to row "scatter_map[i]"
 * of the destination table. All other rows of the destination table will be equal to
 * corresponding rows of the target table.
 *
 * The datatypes between coresponding columns in the source and destination
 * must be the same.
 *
 * Optionally performs bounds checking on the values of the `scatter_map` and
 * raises a runtime error if any of its values are outside the range
 * [0, num_source_rows).
 *
 * @throws `cudf::logic_error` if `check_bounds == true` and an index exists in
 * `scatter_map` outside the range `[-n, n)`, where `n` is the number of rows in
 * the target table. If `check_bounds == false`, the behavior is undefined.
 *
 * @param[in] source_table The input columns containing values to be scattered
 * into the target columns
 * @param[in] scatter_map A non-nullable column of integral indices that maps the
 * rows in the source table to rows in the target table. Must be equal
 * in size to the number of elements in the source columns.
 * @param[out] target_table The set of columns into which values from the source_table
 * are to be scattered
 * @param check_bounds Optionally perform bounds checking on the values of
 * `scatter_map` and throw an error if any of its values are out of bounds.
 * @param ignore_out_of_bounds Ignore values in `scatter_map` that are
 * out of bounds.
 * @return std::unique_ptr<table> Result of scattering values from the source table
 * to the target table
 *---------------------------------------------------------------------------**/
std::unique_ptr<table> scatter(table_view const& source_table, column_view const& scatter_map,
                               table_view const& target_table, bool check_bounds = false);


/**
 * @brief Creates a new table that results from scattering a set of scalar source values
 * into the rows of a target table according to a scatter map.
 *
 * Scatters a set of scalar values into the target table out-of-place,
 * returning a "destination table". The scatter is performed according to a scatter map
 * such that each row of the destination table specified in the scatter map will be
 * equal to the source values. All other rows of the destination table will be equal to
 * corresponding rows of the target table.
 *
 * The datatypes between coresponding columns in the source and destination
 * must be the same.
 *
 * The number of elements in the scatter map must equal the number of rows in the
 * source table.
 *
 * @throws `cudf::logic_error` if `check_bounds == true` and an index exists in
 * `scatter_map` outside the range `[-n, n)`, where `n` is the number of rows in
 * the target table. If `check_bounds == false`, the behavior is undefined.
 *
 * @param[in] source_table The input columns containing values to be scattered
 * into the target columns
 * @param[in] scatter_map A non-nullable column of integral indices that maps the
 * rows in the source table to rows in the target table. Must be equal
 * in size to the number of elements in the source columns.
 * @param[out] target_table The set of columns into which values from the source_table
 * are to be scattered
 * @param check_bounds Optionally perform bounds checking on the values of
 * `scatter_map` and throw an error if any of its values are out of bounds.
 * @param ignore_out_of_bounds Ignore values in `scatter_map` that are
 * out of bounds.
 * @return std::unique_ptr<table> Result of scattering values from the source table
 * to the target table
 *---------------------------------------------------------------------------**/
std::unique_ptr<table> scatter(std::vector<std::unique_ptr<cudf::scalar>> const& source,
	      column_view const& scatter_map,
	      table_view const& target);

/** ---------------------------------------------------------------------------*
* @brief Indicates when to allocate a mask, based on an existing mask.
* ---------------------------------------------------------------------------**/
enum class  mask_allocation_policy {
    NEVER, ///< Do not allocate a null mask, regardless of input
    RETAIN, ///< Allocate a null mask if the input contains one
    ALWAYS ///< Allocate a null mask, regardless of input
};


/*
 * Initializes and returns an empty column of the same type as the `input`.
 *
 * @param[in] input Immutable view of input column to emulate
 * @return std::unique_ptr<column> An empty column of same type as `input`
 */
std::unique_ptr<column> empty_like(column_view input);

/**
 * @brief Creates an uninitialized new column of the same size and type as the `input`.
 * Supports only fixed-width types.
 *
 * @param[in] input Immutable view of input column to emulate
 * @param[in] mask_alloc Optional, Policy for allocating null mask. Defaults to RETAIN.
 * @param[in] mr Optional, The resource to use for all allocations
 * @return std::unique_ptr<column> A column with sufficient uninitialized capacity to hold the same number of elements as `input` of the same type as `input.type()`
 */
std::unique_ptr<column> allocate_like(column_view input,
                                      mask_allocation_policy mask_alloc = mask_allocation_policy::RETAIN,
                                      rmm::mr::device_memory_resource *mr =
                                          rmm::mr::get_default_resource());

/**
 * @brief Creates an uninitialized new column of the specified size and same type as the `input`.
 * Supports only fixed-width types.
 *
 * @param[in] input Immutable view of input column to emulate
 * @param[in] size The desired number of elements that the new column should have capacity for
 * @param[in] mask_alloc Optional, Policy for allocating null mask. Defaults to RETAIN.
 * @param[in] mr Optional, The resource to use for all allocations
 * @return std::unique_ptr<column> A column with sufficient uninitialized capacity to hold the specified number of elements as `input` of the same type as `input.type()`
 */
std::unique_ptr<column> allocate_like(column_view input, size_type size,
                                      mask_allocation_policy mask_alloc = mask_allocation_policy::RETAIN,
                                      rmm::mr::device_memory_resource *mr =
                                          rmm::mr::get_default_resource());

/**
 * @brief Creates a table of empty columns with the same types as the `input_table`
 *
 * Creates the `cudf::column` objects, but does not allocate any underlying device
 * memory for the column's data or bitmask.
 *
 * @param[in] input_table Immutable view of input table to emulate
 * @return std::unique_ptr<table> A table of empty columns with the same types as the columns in `input_table`
 */
std::unique_ptr<table> empty_like(table_view input_table);

/**
 * @brief Slices a `column_view` into a set of `column_view`s according to a set of indices.
 * The returned views of `input` are constructed from an even number indices where
 * the `i`th returned `column_view` views the elements in `input` indicated by the range
 * `[indices[2*i], indices[(2*i)+1])`.
 *
 * For all `i` it is expected `indices[i] <= input.size()`
 * For all `i%2==0`, it is expected that `indices[i] <= indices[i+1]`
 *
 * @note It is the caller's responsibility to ensure that the returned view
 * does not outlive the viewed device memory.
 *
 * @example:
 * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
 * indices: {1, 3, 5, 9, 2, 4, 8, 8}
 * output:  {{12, 14}, {20, 22, 24, 26}, {14, 16}, {}}
 *
 * @throws `cudf::logic_error` if `indices` size is not even.
 * @throws `cudf::logic_error` When the values in the pair are strictly decreasing.
 * @throws `cudf::logic_error` When any of the values in the pair don't belong to
 * the range [0, input.size()).
 *
 * @param input View of column to slice
 * @param indices A vector of indices used to take slices of `input`.
 * @return Vector of views of `input` indicated by the ranges in `indices`.
 */
std::vector<column_view> slice(column_view const& input,
                               std::vector<size_type> const& indices);

/**
 * @brief Splits a `column_view` into a set of `column_view`s according to a set of indices
 * derived from expected splits.
 *
 * The returned view's of `input` are constructed from vector of splits, which indicates
 * where the split should occur. The `i`th returned `column_view` is sliced as
 * `[0, splits[i])` if `i`=0, else `[splits[i], input.size())` if `i` is the last view and
 * `splits[i] != input.size()`, or `[splits[i-1], splits[i]]` otherwise.
 *
 * For all `i` it is expected `splits[i] <= splits[i+1] <= input.size()`
 *
 * @note It is the caller's responsibility to ensure that the returned view
 * does not outlive the viewed device memory.
 *
 * Example:
 * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
 * splits:  {2, 5, 9}
 * output:  {{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}}
 *
 * @throws `cudf::logic_error` if `splits` has end index > size of `input`.
 * @throws `cudf::logic_error` When the value in `splits` is not in the range [0, input.size()).
 * @throws `cudf::logic_error` When the values in the `splits` are 'strictly decreasing'.
 *
 * @param input View of column to split
 * @param splits A vector of indices where the view will be split
 * @return The set of requested views of `input` indicated by the `splits`.
 */
std::vector<column_view> split(column_view const& input,
                               std::vector<size_type> const& splits);

}  // namespace experimental
}  // namespace cudf
