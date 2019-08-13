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
 *
 * If join_context->flag_method is set to GDF_SORT then the null_count of the
 * columns must be set to 0 otherwise a GDF_VALIDITY_UNSUPPORTED error is
 * returned.
 * 
 * @param[in] left_cols The table of left dataframe
 * @param[in] right_cols The table of right dataframe
 * @param[in] join_cols The table containing indices of left and right join columns
 * respectively
 * @param[in] merging_join_cols The table containing indices of left and right
 * join columns which have same name respectively and which will merge into a single column
 * @param[out] out_indices The table containing joined indices of left and right table 
 * @param[in] join_context The context to use to control how the join is performed,e.g.,
 * sort vs hash based implementation
 * 
 * @returns If Success, pair of left and right table which are merged 
 */
std::pair<cudf::table, cudf::table> inner_join(
                         cudf::table const& left_cols,
                         cudf::table const& right_cols,
                         cudf::table const& join_cols,
                         cudf::table const& merging_join_cols,
                         cudf::table const& out_indices,
                         gdf_context *join_context);
/** 
 * @brief  Performs a left join (also known as left outer join) on the
 * specified columns of two dataframes (left, right)
 *
 * If join_context->flag_method is set to GDF_SORT then the null_count of the
 * columns must be set to 0 otherwise a GDF_VALIDITY_UNSUPPORTED error is
 * returned.
 *
 * @param[in] left_cols The table of left dataframe
 * @param[in] right_cols The table of right dataframe
 * @param[in] join_cols The table containing indices of left and right join columns
 * respectively
 * @param[in] merging_join_cols The table containing indices of left and right
 * join columns which have same name respectively and which will merge into a single column
 * @param[out] out_indices The table containing joined indices of left and right table 
 * @param[in] join_context The context to use to control how the join is performed,e.g.,
 * sort vs hash based implementation
 * 
 * @returns If Success, pair of left and right table which are merged
 */
std::pair<cudf::table, cudf::table> left_join(
                         cudf::table const& left_cols,
                         cudf::table const& right_cols,
                         cudf::table const& join_cols,
                         cudf::table const& merging_join_cols,
                         cudf::table const& out_indices,
                         gdf_context *join_context);

/** 
 * @brief  Performs a full join (also known as full outer join) on the
 * specified columns of two dataframes (left, right)
 *
 * If join_context->flag_method is set to GDF_SORT then the null_count of the
 * columns must be set to 0 otherwise a GDF_VALIDITY_UNSUPPORTED error is
 * returned.
 *
 * @param[in] left_cols The table of left dataframe
 * @param[in] right_cols The table of right dataframe
 * @param[in] join_cols The table containing indices of left and right join columns
 * respectively
 * @param[in] merging_join_cols The table containing indices of left and right
 * join columns which have same name respectively and which will merge into a single column
 * @param[out] out_indices The table containing joined indices of left and right table 
 * @param[in] join_context The context to use to control how the join is performed,e.g.,
 * sort vs hash based implementation
 * 
 * @returns If Success, pair of left and right table which are merged 
 */
std::pair<cudf::table, cudf::table> full_join(
                         cudf::table const& left_cols,
                         cudf::table const& right_cols,
                         cudf::table const& join_cols,
                         cudf::table const& merging_join_cols,
                         cudf::table const& out_indices,
                         gdf_context *join_context);
} //namespace cudf
