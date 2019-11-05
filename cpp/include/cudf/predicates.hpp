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

namespace experimental {

/** 
 * @brief Checks whether the rows of a `table` are sorted in a lexicographical order.
 * 
 * @param[in] in                table whose rows need to be compared for ordering
 * @param[in] column_order      The expected sort order for each column. Size
 *                              must be equal to `in.num_columns()` or empty. If
 *                              empty, it is expected all columns are in
 *                              ascending order.
 * @param[in] numm_precedence   The desired order of null compared to other
 *                              elements for each column. Size must be equal to
 *                              `input.num_columns()` or empty. If empty,
 *                              `null_order::BEFORE` is assumed for all columns.
 * 
 * @returns bool                true if sorted as expected, false if not.
 */
bool is_sorted(cudf::table_view const& table,
               std::vector<order> const& column_order,
               std::vector<null_order> const& null_precedence);

} // namespace experimental

} // namespace cudf
