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

#include <cudf/cudf.h>
#include <cudf/types.hpp>

namespace cudf {
namespace experimental {


/**
 * @brief Filters a table to remove null elements.
 *
 * Filters the rows of the `input` considering specified columns from
 * `keys` for validity / null values.
 *
 * Given an input table_view, row `i` from the input columns is copied to
 * the output if the same row `i` of @p keys has at leaast @p keep_threshold
 * non-null fields.
 *
 * This operation is stable: the input order is preserved in the output.
 * 
 * Any non-nullable column in the input is treated as all non-null.
 *
 * @example input   {col1: {1, 2,    3,    null},
 *                   col2: {4, 5,    null, null},
 *                   col3: {7, null, null, null}}
 *          keys = input
 *          keep_threshold = 2
 *
 *          output {col1: {1, 2}
 *                  col2: {4, 5}
 *                  col3: {7, null}}
 *
 * @note if @p input.num_rows() is zero, or @p keys is empty or has no nulls,
 * there is no error, and an empty `std::unique_ptr<table>` is returned
 * 
 * @throws cudf::logic_error if @p keys is non-empty and keys.num_rows() is less
 * than input.num_rows()
 *
 * @param[in] input The input `table_view` to filter.
 * @param[in] keys The `table_view` to filter `input`.
 * @param[in] keep_threshold The minimum number of non-null fields in a row
 *                           required to keep the row.
 * @return unique_ptr<table> Table containing all rows of the `input` with at least @p keep_threshold non-null fields in @p keys.
 */
std::unique_ptr<experimental::table> drop_nulls(table_view const& input,
                 table_view const& keys,
                 cudf::size_type keep_threshold,
                 rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

/**
 * @brief Filters a table to remove null elements.
 *
 * @example input   {col1: {1, 2,    3,    null},
 *                   col2: {4, 5,    null, null},
 *                   col3: {7, null, null, null}}
 *          keys = input
 *          keep_threshold = 2
 *
 *          output {col1: {1}
 *                  col2: {4}
 *                  col3: {7}}
 *
 * @overload drop_nulls
 *
 * Same as drop_nulls but defaults keep_threshold to the number of columns in
 * @p keys.
 *
 * @param[in] input The input `table_view` to filter.
 * @param[in] keys The `table_view` to filter `input`.
 * @return unique_ptr<table> Table containing all rows of the `input` without nulls in the columns of @p keys.
 */
std::unique_ptr<experimental::table> drop_nulls(table_view const &input,
                 table_view const &keys,
                 rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

} // namespace experimental
} // namespace cudf
