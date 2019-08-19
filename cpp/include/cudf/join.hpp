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
 * columns must be set to 0 otherwise @throws GDF Validity is unsupported by sort_join.
 * 
 * @example TableA a:{0,1,2}
 *          TableB b:{1,2,3}, a:{1,2,5} And TableB joins TableA 
 *          on column 'a' with column 'a'.
 * The result would be Table-res a:{1, 2}, b:{1, 2}
 * Result is intersection of column 'a' from A and column 'a' from B
 *
 * @param[in] left The left dataframe
 * @param[in] right The right dataframe
 * @param[in] left_on The column indices from left dataframe 
 * @param[in] right_on The column indices from right dataframe 
 * @param[in] joining_ind contains two columns, left and right
 * indices, which are derived from left_on and right_on.
 *
 * @example Considering the example provided above, column name 'a'
 * is same in both tables and being joined, this will result into
 * single column in the joined result. This table contains the
 * indices of such columns. For above example, this table would be
 * Table left:{0}, right:{1}.
 *
 * @param[out] out_index_table The table containing 
 * joined indices of left and right table 
 * @param[in] join_context The context to use to control how 
 * the join is performed,e.g., sort vs hash based implementation*
 * 
 * @returns If Success, pair of left and right table which are joined 
 */
std::pair<cudf::table, cudf::table> inner_join(
                         cudf::table const& left,
                         cudf::table const& right,
                         cudf::table const& left_on,
                         cudf::table const& right_on,
                         std::vector<std::pair<int, int>> const& joining_ind,
                         cudf::table *out_index_table,
                         gdf_context *join_context);
/** 
 * @brief  Performs a left join (also known as left outer join) on the
 * specified columns of two dataframes (left, right)
 *
 * If join_context->flag_method is set to GDF_SORT then the null_count of the
 * columns must be set to 0 otherwise @throws GDF Validity is unsupported by sort_join.
 * 
 * @example TableA a:{0,1,2}
 *          TableB b:{1,2,3}, a:{1,2,5} And TableB joins TableA 
 *          on column 'a' with column 'b'.
 * The result would be Table-res a:{1, 2, 0}, b:{1, 2, NULL}
 * Result is left of column 'a' only
 *
 * @param[in] left The left dataframe
 * @param[in] right The right dataframe
 * @param[in] left_on The column indices from left dataframe 
 * @param[in] right_on The column indices from right dataframe 
 * @param[in] joining_ind contains two columns, 
 * but these column have the indices which have same name 
 * in dataframes and will eventually merge into a single column.
 * @param[out] out_index_table The table containing 
 * joined indices of left and right table 
 * @param[in] join_context The context to use to control how 
 * the join is performed,e.g., sort vs hash based implementation
 * 
 * @returns If Success, pair of left and right table which are joined
 */
std::pair<cudf::table, cudf::table> left_join(
                         cudf::table const& left,
                         cudf::table const& right,
                         cudf::table const& left_on,
                         cudf::table const& right_on,
                         std::vector<std::pair<int, int>> const& joining_ind,
                         cudf::table *out_index_table,
                         gdf_context *join_context);

/** 
 * @brief  Performs a full join (also known as full outer join) on the
 * specified columns of two dataframes (left, right)
 *
 * If join_context->flag_method is set to GDF_SORT then the null_count of the
 * columns must be set to 0 otherwise @throws GDF Validity is unsupported by sort_join.
 * 
 * @example TableA a:{0,1,2}
 *          TableB b:{1,2,3}, c:{4,5,6} And TableB joins TableA 
 *          on column 'a' with column 'b'.
 * The result would be Table-res a:{0, 1, 2, NULL}, b:{NULL, 1, 2, 3}, c:{NULL, 4, 5, 6}
 * Result is union of column 'a' and column 'b'
 *
 * @param[in] left The left dataframe
 * @param[in] right The right dataframe
 * @param[in] left_on The column indices from left dataframe 
 * @param[in] right_on The column indices from right dataframe 
 * @param[in] joining_ind contains two columns, 
 * but these column have the indices which have same name 
 * in dataframes and will eventually merge into a single column.
 * @param[out] out_index_table The table containing 
 * joined indices of left and right table 
 * @param[in] join_context The context to use to control how 
 * the join is performed,e.g., sort vs hash based implementation
 * 
 * @returns If Success, pair of left and right table which are joined 
 */
std::pair<cudf::table, cudf::table> full_join(
                         cudf::table const& left,
                         cudf::table const& right,
                         cudf::table const& left_on,
                         cudf::table const& right_on,
                         std::vector<std::pair<int, int>> const& joining_ind,
                         cudf::table *out_index_table,
                         gdf_context *join_context);
} //namespace cudf
