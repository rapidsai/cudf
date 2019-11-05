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

/** 
 * @brief Checks whether the rows of a `table` are sorted in a lexicographical order.
 * 
 * @param[in] input              table whose rows need to be compared for ordering
 * @param[in] descending         vector that specifies the expected ordering of each input column
 *                               (0 is ascending order and 1 is descending)
 *                               If this an empty vector, then it will be assumed that each column is in ascending order.
 * @param[in] nulls_are_smallest true indicates nulls are to be considered
 *                               smaller than non-nulls ; false indicates opposite
 * 
 * @returns true - if sorted , false - if not.
 */
bool is_sorted(cudf::table const& table,
                       std::vector<int8_t> const& descending,
                       bool nulls_are_smallest);

} // namespace cudf
