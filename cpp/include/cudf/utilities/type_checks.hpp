/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>

namespace cudf {

/**
 * @brief Compares the type of two `column_view`s
 *
 * This function returns true if the type of `lhs` equals that of `rhs`.
 * - For fixed point types, the scale is compared.
 * - For dictionary types, the type of the keys are compared if both are
 *   non-empty columns.
 * - For lists types, the type of child columns are compared recursively.
 * - For struct types, the type of each field are compared in order.
 * - For all other types, the `id` of `data_type` is compared.
 *
 * @param lhs The first `column_view` to compare
 * @param rhs The second `column_view` to compare
 * @return true if column types match
 */
bool column_types_equal(column_view const& lhs, column_view const& rhs);

/**
 * @brief Compare the type IDs of two `column_view`s
 * This function returns true if the type of `lhs` equals that of `rhs`.
 * - For fixed point types, the scale is ignored.
 *
 * @param lhs The first `column_view` to compare
 * @param rhs The second `column_view` to compare
 * @return true if column types match
 */
bool column_types_equivalent(column_view const& lhs, column_view const& rhs);

}  // namespace cudf
