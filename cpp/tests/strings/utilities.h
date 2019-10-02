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

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <vector>

namespace cudf {
namespace test {

/**---------------------------------------------------------------------------*
 * @brief Utility for creating a strings column from a vector of host strings
 *
 * @param h_strings Pointer to null-terminated, UTF-8 encode chars arrays.
 * @return column instance of type STRING
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> create_strings_column( const std::vector<const char*>& h_strings );

/**---------------------------------------------------------------------------*
 * @brief Verifies the element-wise equality of two strings columns.
 *
 * Treats null elements as equivalent.
 * Based on `expect_columns_equal()` in tests/utilities/column_utilities.cu
 *
 * @param lhs The first column
 * @param rhs The second column
 *---------------------------------------------------------------------------**/
void expect_strings_columns_equal(cudf::strings_column_view lhs, cudf::strings_column_view rhs);

}  // namespace test
}  // namespace cudf
