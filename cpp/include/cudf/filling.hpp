/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>

namespace cudf {
/**
 * @addtogroup transformation_fill
 * @{
 * @file
 * @brief Column APIs for fill, repeat, and sequence
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
std::unique_ptr<column> fill(
  column_view const& input,
  size_type begin,
  size_type end,
  scalar const& value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

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
 * The behavior of this function is undefined if @p count has negative values
 * or the sum overflows.
 *
 * @throws cudf::logic_error if the data type of @p count is not size_type.
 * @throws cudf::logic_error if @p input_table and @p count have different
 * number of rows.
 * @throws cudf::logic_error if @p count has null values.
 *
 * @param input_table Input table
 * @param count Non-nullable column of an integral type
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return The result table containing the repetitions
 */
std::unique_ptr<table> repeat(
  table_view const& input_table,
  column_view const& count,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

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
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return The result table containing the repetitions
 */
std::unique_ptr<table> repeat(
  table_view const& input_table,
  size_type count,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

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
 * @throws cudf::logic_error if @p init and @p step are not the same type.
 * @throws cudf::logic_error if scalar types are not numeric.
 * @throws cudf::logic_error if @p size is < 0.
 *
 * @param size Size of the output column
 * @param init First value in the sequence
 * @param step Increment value
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return The result column containing the generated sequence
 */
std::unique_ptr<column> sequence(
  size_type size,
  scalar const& init,
  scalar const& step,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

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
 * @return The result column containing the generated sequence
 */
std::unique_ptr<column> sequence(
  size_type size,
  scalar const& init,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Generate a sequence of timestamps beginning at `init` and incrementing by `months` for
 * each successive element, i.e., `output[i] = init + i * months` for `i` in `[0, size)`.
 *
 * If a given date is invalid, the date is scaled back to the last available day of that month.
 *
 * Example:
 * ```
 * size = 3
 * init = 2020-01-31 08:00:00
 * months = 1
 * return = [2020-01-31 08:00:00, 2020-02-29 08:00:00, 2020-03-31 08:00:00]
 * ```
 *
 * @throw cudf::logic_error if input datatype is not a TIMESTAMP
 *
 * @param size Number of timestamps to generate
 * @param init The initial timestamp
 * @param months Months to increment
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @return Timestamps column with sequences of months
 */
std::unique_ptr<cudf::column> calendrical_month_sequence(
  size_type size,
  scalar const& init,
  size_type months,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
