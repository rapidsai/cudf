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
 * @example TableA a:{0,1,2}
 *          TableB b:{1,2,3}, a:{1,2,5} And TableB joins TableA 
 *          on column 'a' with column 'a'.
 * The result would be Table-res a:{1, 2}, b:{1, 2}
 * Result is intersection of column 'a' from A and column 'a' from B
 *
 * @throws cudf::logic_error 
 * if a sort-based join is requested and either `right_on` or `left_on` contains null values.
 *
 * @param[in] left The left dataframe
 * @param[in] right The right dataframe
 * @param[in] left_on left_on The columns from left to join on.
 * Column i from left_on will be compared against column i of right_on.
 * @param[in] right_on The columns from right to join on.
 * Column i from right_on will be compared with column i of left_on. 
 * @param[in] joining_ind is a vector of pairs of left and right
 * join indcies derived from left_on and right_on. This contains
 * the indices with the same name which evetually result into a 
 * single column.
 *
 * @example Considering the example provided above, column name 'a'
 * is same in both tables and being joined, this will result into
 * single column in the joined result. This vector will have {(0,1)}.
 *
 * @param[out] out_ind The table containing joined indices of left 
 * and right table 
 * @param[in] join_context The context to use to control how 
 * the join is performed,e.g., sort vs hash based implementation*
 * 
 * @returns If Success, joined table
 */
cudf::table inner_join(
                         cudf::table const& left,
                         cudf::table const& right,
                         cudf::table const& left_on,
                         cudf::table const& right_on,
                         std::vector<std::pair<int, int>> const& joining_ind,
                         cudf::table *out_ind,
                         gdf_context *join_context);
/** 
 * @brief  Performs a left join (also known as left outer join) on the
 * specified columns of two dataframes (left, right)
 *
 * @example TableA a:{0,1,2}
 *          TableB b:{1,2,3}, a:{1,2,5} And TableB joins TableA 
 *          on column 'a' with column 'b'.
 * The result would be Table-res a:{0, 1, 2}, b:{NULL, 1, 2}
 * Result is left of column 'a' only
 * @throws cudf::logic_error 
 * if a sort-based join is requested and either `right_on` or `left_on` contains null values.
 *
 * @param[in] left The left dataframe
 * @param[in] right The right dataframe
 * @param[in] left_on left_on The columns from left to join on.
 * Column i from left_on will be compared against column i of right_on.
 * @param[in] right_on The columns from right to join on.
 * Column i from right_on will be compared with column i of left_on. 
 * @param[in] joining_ind is a vector of pairs of left and right
 * join indcies derived from left_on and right_on. This contains
 * the indices with the same name which evetually result into a 
 * single column.
 *
 * @example Considering the example provided above, column name 'a'
 * is same in both tables and being joined, this will result into
 * single column in the joined result. This vector will have {(0,1)}.
 *
 * @param[out] out_ind The table containing joined indices of left 
 * and right table 
 * @param[in] join_context The context to use to control how 
 * the join is performed,e.g., sort vs hash based implementation
 * 
 * @returns If Success, joined table
 */
cudf::table left_join(
                         cudf::table const& left,
                         cudf::table const& right,
                         cudf::table const& left_on,
                         cudf::table const& right_on,
                         std::vector<std::pair<int, int>> const& joining_ind,
                         cudf::table *out_ind,
                         gdf_context *join_context);

/** 
 * @brief  Performs a full join (also known as full outer join) on the
 * specified columns of two dataframes (left, right)
 * 
 * @example TableA a:{0,1,2}
 *          TableB b:{1,2,3}, a:{1,2,5} And TableB joins TableA 
 *          on column 'a' with column 'a'.
 * The result would be Table-res a:{0, 1, 2, 5}, b:{NULL, 1, 2, 3}
 * Result is union of column 'a' and column 'b'
 *
 * @throws cudf::logic_error 
 * if a sort-based join is requested and either `right_on` or `left_on` contains null values.
 *
 * @param[in] left The left dataframe
 * @param[in] right The right dataframe
 * @param[in] left_on left_on The columns from left to join on.
 * Column i from left_on will be compared against column i of right_on.
 * @param[in] right_on The columns from right to join on.
 * Column i from right_on will be compared with column i of left_on. 
 * @param[in] joining_ind is a vector of pairs of left and right
 * join indcies derived from left_on and right_on. This contains
 * the indices with the same name which evetually result into a 
 * single column.
 *
 * @example Considering the example provided above, column name 'a'
 * is same in both tables and being joined, this will result into
 * single column in the joined result. This vector will have {(0,1)}.
 *
 * @param[out] out_ind The table containing joined indices of left 
 * and right table 
 * @param[in] join_context The context to use to control how 
 * the join is performed,e.g., sort vs hash based implementation
 * 
 * @returns If Success, joined table 
 */
cudf::table full_join(
                         cudf::table const& left,
                         cudf::table const& right,
                         cudf::table const& left_on,
                         cudf::table const& right_on,
                         std::vector<std::pair<int, int>> const& joining_ind,
                         cudf::table *out_ind,
                         gdf_context *join_context);
} //namespace cudf
