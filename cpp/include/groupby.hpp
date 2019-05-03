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

#ifndef GROUPBY_HPP
#define GROUPBY_HPP

#include <cudf.h>
#include <types.hpp>
#include <tuple>
#include <rmm/thrust_rmm_allocator.h>

/**
 * @brief Returns the first index of each unique row. Assumes the data is already sorted 
 *
 * @param[in]  input_table          The input columns whose rows are sorted.
 * @param[in]  context              The options for controlling treatment of nulls
 *             context->flag_null_sort_behavior
 *                    GDF_NULL_AS_LARGEST = Nulls are treated as largest, 
 *                    GDF_NULL_AS_SMALLEST = Nulls are treated as smallest, 
 *                    GDF_NULL_AS_LARGEST_FOR_MULTISORT = Special multicolumn-sort case: A row with null in any column is largest
 *
 * @returns A device vector containing the first index of every unique row.
 */
rmm::device_vector<gdf_index_type>
gdf_unique_indices(cudf::table const& input_table,
                  gdf_context const& context);

/**
 * @brief Sorts a set of columns based on specified "key" columns. Returns a column containing
 * the offset to the start of each set of unique keys.
 *
 * @param[in]  input_table           The input columns whose rows will be grouped.
 * @param[in]  num_key_cols             The number of key columns.
 * @param[in]  key_col_indices          The indices of the of the key columns by which data will be grouped.
 * @param[in]  context                  The context used to control how nulls are treated in group by
 *             context->flag_null_sort_behavior
 *                      GDF_NULL_AS_LARGEST = Nulls are treated as largest, 
 *                      GDF_NULL_AS_SMALLEST = Nulls are treated as smallest, 
 *                      GDF_NULL_AS_LARGEST_FOR_MULTISORT = Special multicolumn-sort case: A row with null in any column is largest
 *             context-> flag_groupby_include_nulls 
 *                      false = Nulls keys are ignored (Pandas style),
 *                      true = Nulls keys are treated as values. NULL keys will compare as equal NULL == NULL (SQL style)
 *
 * @returns A tuple containing:
 *          - A cudf::table containing a set of columns sorted by the key columns.
 *          - A device vector containing the first index of every unique row
 */
std::tuple<cudf::table, rmm::device_vector<gdf_index_type>> 
gdf_group_by_without_aggregations(cudf::table const& input_table,
                                  gdf_size_type num_key_cols,
                                  gdf_index_type const * key_col_indices,
                                  gdf_context* context);

#endif
