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

namespace cudf {
// joins

/** 
 * @brief  Performs an inner join on the specified columns of two
 * dataframes (left, right)
 * If join_context->flag_method is set to GDF_SORT then the null_count of the
 * columns must be set to 0 otherwise a GDF_VALIDITY_UNSUPPORTED error is
 * returned.
 * 
 * @param[in] left_cols The table of left dataframe
 * @param[in] left_join_cols The column indices of columns from the left dataframe
 * to join on
 * @param[in] right_cols The table of right dataframe
 * @param[in] right_join_cols The column indices of columns from the right dataframe
 * to join on
 * @param[out] gdf_column * left_indices If not nullptr, indices of rows from the left table that match rows in the right table
 * @param[out] gdf_column * right_indices If not nullptr, indices of rows from the right table that match rows in the left table
 * @param[in] join_context The context to use to control how the join is performed,e.g.,
 * sort vs hash based implementation
 * 
 * @returns If Success, pair of left and right table which are merged 
 * 
 */
std::pair<cudf::table, cudf::table> gdf_inner_join(
                         cudf::table & left_cols,
                         std::vector <int> left_join_cols,
                         cudf::table & right_cols,
                         std::vector <int> right_join_cols,
                         gdf_column * left_indices,
                         gdf_column * right_indices,
                         gdf_context *join_context,
                         std::vector <int> left_join_result_cols,
                         std::vector <int> right_join_result_cols);
/** 
 * @brief  Performs a left join (also known as left outer join) on the
 * specified columns of two dataframes (left, right)
 * If join_context->flag_method is set to GDF_SORT then the null_count of the
 * columns must be set to 0 otherwise a GDF_VALIDITY_UNSUPPORTED error is
 * returned.
 *
 * @param[in] left_cols The table of left dataframe
 * @param[in] left_join_cols The column indices of columns from the left dataframe
 * to join on
 * @param[in] right_cols The table of right dataframe
 * @param[in] right_join_cols The column indices of columns from the right dataframe
 * to join on
 * @param[out] gdf_column * left_indices If not nullptr, indices of rows from the left table that match rows in the right table
 * @param[out] gdf_column * right_indices If not nullptr, indices of rows from the right table that match rows in the left table
 * @param[in] join_context The context to use to control how the join is performed,e.g.,
 * sort vs hash based implementation
 * 
 * @returns If Success, pair of left and right table which are merged * 
 */
std::pair<cudf::table, cudf::table> gdf_left_join(
                         cudf::table & left_cols,
                         std::vector <int> left_join_cols,
                         cudf::table & right_cols,
                         std::vector <int> right_join_cols,
                         gdf_column * left_indices,
                         gdf_column * right_indices,
                         gdf_context *join_context,
                         std::vector <int> left_join_result_cols,
                         std::vector <int> right_join_result_cols);

/** 
 * @brief  Performs a full join (also known as full outer join) on the
 * specified columns of two dataframes (left, right)
 * If join_context->flag_method is set to GDF_SORT then the null_count of the
 * columns must be set to 0 otherwise a GDF_VALIDITY_UNSUPPORTED error is
 * returned.
 *
 * @param[in] left_cols The table of left dataframe
 * @param[in] left_join_cols The column indices of columns from the left dataframe
 * to join on
 * @param[in] right_cols The table of right dataframe
 * @param[in] right_join_cols The column indices of columns from the right dataframe
 * to join on
 * @param[out] gdf_column * left_indices If not nullptr, indices of rows from the left table that match rows in the right table
 * @param[out] gdf_column * right_indices If not nullptr, indices of rows from the right table that match rows in the left table
 * @param[in] join_context The context to use to control how the join is performed,e.g.,
 * sort vs hash based implementation
 * 
 * @returns If Success, pair of left and right table which are merged * 
 */
std::pair<cudf::table, cudf::table> gdf_full_join(
                         cudf::table & left_cols,
                         std::vector <int> left_join_cols,
                         cudf::table & right_cols,
                         std::vector <int> right_join_cols,
                         gdf_column * left_indices,
                         gdf_column * right_indices,
                         gdf_context *join_context,
                         std::vector <int> left_join_result_cols,
                         std::vector <int> right_join_result_cols);
} //namespace cudf
