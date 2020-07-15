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

#include <memory>

namespace cudf {
/**
 * @addtogroup transformation_fill
 * @{
 */

/**
 * @brief Fills a range of elements in-place in a column with a scalar value.
 *
 * Fills N elements of @p destination starting at @p begin with @p value, where
 * N = (@p end - @p begin).
 *
 * Overwrites the range of elements in @p destination indicated by the indices
 * [@p begin, @p end) with @p value. Use the out-of-place fill function
 * returning std::unique_ptr<column> for use cases requiring memory
 * reallocation.
 *
 * @throws cudf::logic_error if memory reallocation is required (e.g. for
 * variable width types).
 * @throws cudf::logic_error for invalid range (if @p begin < 0,
 * @p begin > @p end, or @p end > @p destination.size()).
 * @throws cudf::logic_error if @p destination and @p value have different
 * types.
 * @throws cudf::logic_error if @p value is invalid but @p destination is not
 * nullable.
 *
 * @param destination The preallocated column to fill into
 * @param begin The starting index of the fill range (inclusive)
 * @param end The index of the last element in the fill range (exclusive)
 * @param value The scalar value to fill
 * @return void
 */
void fill_in_place(mutable_column_view& destination,
                   size_type begin,
                   size_type end,
                   scalar const& value);

/**
 * @brief Fills a range of elements in a column out-of-place with a scalar
 * value.
 *
 * Creates a new column as-if an in-place fill was performed into @p input;
 * i.e. it is as if a copy of @p input was created first and then the elements
 * indicated by the indices [@p begin, @p end) were overwritten by @p value.
 *
 * @throws cudf::logic_error for invalid range (if @p begin < 0,
 * @p begin > @p end, or @p end > @p destination.size()).
 * @throws cudf::logic_error if @p destination and @p value have different
 * types.
 *
 * @param input The input column used to create a new column. The new column
 * is created by replacing the values of @p input in the specified range with
 * @p value.
 * @param begin The starting index of the fill range (inclusive)
 * @param end The index of the last element in the fill range (exclusive)
 * @param value The scalar value to fill
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return The result output column
 */
std::unique_ptr<column> fill(column_view const& input,
                             size_type begin,
                             size_type end,
                             scalar const& value,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
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
 * @p count should not have null values; should not contain negative values;
 * and the sum of count elements should not overflow the size_type's limit.
 * It is undefined behavior if @p count has negative values or the sum overflows
 * and @p check_count is set to false.
 *
 * @throws cudf::logic_error if the data type of @p count is not size_type.
 * @throws cudf::logic_error if @p input_table and @p count have different
 * number of rows.
 * @throws cudf::logic_error if @p count has null values.
 * @throws cudf::logic_error if @p check_count is set to true and @p count
 * has negative values or the sum of @p count elements overflows.
 *
 * @param input_table Input table
 * @param count Non-nullable column of an integral type
 * @param check_count Whether to check count (negative values and overflow)
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return The result table containing the repetitions
 */
std::unique_ptr<table> repeat(
  table_view const& input_table,
  column_view const& count,
  bool check_count                    = false,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Repeat rows of a Table.
 *
 * Creates a new table by repeating @p count times the rows of @p input_table.
 * Example:
 * ```
 * in = [4,5,6]
 * count = 2
 * return = [4,4,5,5,6,6]
 * ```
 * @throws cudf::logic_error if the data type of @p count is not size_type.
 * @throws cudf::logic_error if @p count is invalid or @p count is negative.
 * @throws cudf::logic_error if @p input_table.num_rows() * @p count overflows
 * size_type.
 *
 * @param input_table Input table
 * @param count Number of repetitions
 * @param mr Device memory resource used to allocate the returned table's device memory.
 * @return The result table containing the repetitions
 */
std::unique_ptr<table> repeat(
  table_view const& input_table,
  size_type count,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Fills a column with a sequence of value specified by an initial value and a step.
 *
 * Creates a new column and fills with @p size values starting at @p init and
 * incrementing by @p step, generating the sequence
 * [ init, init+step, init+2*step, ... init + (size - 1)*step]
 *
 * ```
 * size = 3
 * init = 0
 * step = 2
 * return = [0, 2, 4]
 * ```
 * @throws cudf::logic_error if @p init and @p @step are not the same type.
 * @throws cudf::logic_error if scalar types are not numeric.
 * @throws cudf::logic_error if @p size is < 0.
 *
 * @param size Size of the output column
 * @param init First value in the sequence
 * @param step Increment value
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return std::unique_ptr<column> The result table containing the sequence
 **/
std::unique_ptr<column> sequence(
  size_type size,
  scalar const& init,
  scalar const& step,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Fills a column with a sequence of value specified by an initial value and a step of 1.
 *
 * Creates a new column and fills with @p size values starting at @p init and
 * incrementing by 1, generating the sequence
 * [ init, init+1, init+2, ... init + (size - 1)]
 *
 * ```
 * size = 3
 * init = 0
 * return = [0, 1, 2]
 * ```
 * @throws cudf::logic_error if @p init is not numeric.
 * @throws cudf::logic_error if @p size is < 0.
 *
 * @param size Size of the output column
 * @param init First value in the sequence
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return std::unique_ptr<column> The result table containing the sequence
 **/
std::unique_ptr<column> sequence(
  size_type size,
  scalar const& init,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/** @} */  // end of group
}  // namespace cudf
