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
namespace detail {

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
 * @param[in] mr Optional, The resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to execute kernels
 * @return unique_ptr<table> Table containing all rows of the `input` with at least @p keep_threshold non-null fields in @p keys.
 */
std::unique_ptr<experimental::table> drop_nulls(table_view const& input,
                 table_view const& keys,
                 cudf::size_type keep_threshold,
                 rmm::mr::device_memory_resource *mr =
                     rmm::mr::get_default_resource(),
                 cudaStream_t stream = 0);

/**
 * @brief Filters `input` using `boolean_mask` of boolean values as a mask.
 *
 * Given an input `table_view` and a mask `column_view`, an element `i` from
 * each column_view of the `input` is copied to the corresponding output column
 * if the corresponding element `i` in the mask is non-null and `true`.
 * This operation is stable: the input order is preserved.
 *
 * @note if @p input.num_rows() is zero, there is no error, and an empty table
 * is returned.
 *
 * @throws cudf::logic_error if The `input` size  and `boolean_mask` size mismatches.
 * @throws cudf::logic_error if `boolean_mask` is not `BOOL8` type.
 *
 * @param[in] input The input tablei_view to filter
 * @param[in] boolean_mask A column_view of type BOOL8 used as a mask to filter
 * the `input`.
 * @param[in] mr Optional, The resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to execute kernels
 * @return unique_ptr<table> Table containing copy of all rows of @p input passing
 * the filter defined by @p boolean_mask.
 */
std::unique_ptr<experimental::table>
    apply_boolean_mask(table_view const& input,
                       column_view const& boolean_mask,
                       rmm::mr::device_memory_resource *mr =
                           rmm::mr::get_default_resource(),
                       cudaStream_t stream = 0);

std::unique_ptr<experimental::table>
    drop_duplicates(table_view const& input,
                            table_view const& keys,
                            duplicate_keep_option const& keep,
                            bool const& nulls_are_equal=true,
                            rmm::mr::device_memory_resource *mr =
                               rmm::mr::get_default_resource(),
                            cudaStream_t stream = 0);

cudf::size_type unique_count(column_view const& input,
                           bool const& ignore_nulls,
                           bool const& nan_as_null,
                           rmm::mr::device_memory_resource *mr =
                               rmm::mr::get_default_resource(),
                           cudaStream_t stream = 0);

} // namespace detail
} // namespace experimental
} // namespace cudf
