/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <rmm/mr/default_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>

// for gdf_scalar, unnecessary once we switch to cudf::scalar
#include <cudf/types.h>

#include <memory>

namespace cudf {
namespace experimental {

/**---------------------------------------------------------------------------*
 * @brief Fills a range of elements in a column with a scalar value.
 * 
 * Fills N elements of @p destination starting at @p begin with @p value, where
 * N = (@p end - @p begin).
 *
 * The datatypes of @p destination and @p value must be the same. This function
 * assumes that no memory reallocation is necessary for @p destination. This
 * function updates in-place and throws an exception if memory reallocation is
 * necessary (e.g. for variable width types). Use the out-of-place fill function
 * returning std::unique_ptr<column> for use cases requiring memory
 * reallocation.
 *
 * @param destination The preallocated column to fill into
 * @param begin The starting index of the fill range
 * @param end The index one past the end of the fill range
 * @param value The scalar value to fill
 * @return void
 *---------------------------------------------------------------------------**/
void fill(mutable_column_view& destination, size_type begin, size_type end,
          gdf_scalar const& value);

/**---------------------------------------------------------------------------*
 * @brief Fills a range of elements in a column with a scalar value.
 * 
 * This fill function updates out-of-place creating a new column object to
 * return. The returned column holds @p value for N elements from @p begin,
 * where N = (@p end - @p begin). The returned column stores same values to
 * @p input outside the fill range.
 *
 * The datatypes of @p input and @p value must be the same.
 *
 * @param input The input column used to create a new column. The new column
 * is created by replacing the values of @p input in the specified range with
 * @p value.
 * @param begin The starting index of the fill range
 * @param end The index one past the end of the fill range
 * @param value The scalar value to fill
 * @param mr Memory resource to allocate the result output column
 * @return std::unique_ptr<column> The result output column
 *---------------------------------------------------------------------------**/
std::unique_ptr<column> fill(
    column_view const& input, size_type begin, size_type end,
    gdf_scalar const& value,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
 * @brief Repeat rows of a Table.
 * 
 * Creates a new table by repeating the rows of @p input_table. The number of 
 * repetitions of each element is defined by the value at the corresponding 
 * index of @p count
 * Example:
 * ```
 * in = [4,5,6]
 * count = [1,2,3]
 * return = [4,5,5,6,6,6]
 * ```
 * @p count should be non-nullable; should not contain negative values; and the
 * sum of count elements should not overflow the size_type's limit. It is
 * undefined behavior if @p count has negative values or the sum overflows and
 * @p check_count is set to false.
 *
 * @param input_table Input table
 * @param count Non-nullable column of a integral type
 * @param check_count Whether to check count (negative values and overflow)
 * @param mr Memory resource to allocate the result output table
 * @return std::unique_ptr<table> The result table containing the repetitions
 *---------------------------------------------------------------------------**/
std::unique_ptr<table> repeat(
    table_view const& input_table, column_view const& count,
    bool check_nonnegative = false,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
 * @brief Repeat rows of a Table.
 * 
 * Creates a new table by repeating @p count times the rows of @p input_table.
 * Example:
 * ```
 * in = [4,5,6]
 * count = 2
 * return = [4,4,5,5,6,6]
 * ```
 * @p count should be non-null and should hold a non-negative value.
 * 
 * @param input_table Input table
 * @param count Non-null scalar of a integral type
 * @param mr Memory resource to allocate the result output table
 * @return std::unique_ptr<table> The result table containing the repetitions
 *---------------------------------------------------------------------------**/
std::unique_ptr<table> repeat(
    table_view const& input_table, gdf_scalar const& count,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace experimental
}  // namespace cudf
