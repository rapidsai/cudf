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

#include <cudf/types.hpp>
#include <cudf/column/column_view.hpp>
#include <tests/utilities/cudf_test_utils.cuh>

namespace cudf {
namespace test {
/**---------------------------------------------------------------------------*
 * @brief Verifies the element-wise equality of two columns.
 *
 * Treats null elements as equivalent.
 *
 * @param lhs The first column
 * @param rhs The second column
 *---------------------------------------------------------------------------**/
void expect_columns_equal(cudf::column_view lhs, cudf::column_view rhs);

/**---------------------------------------------------------------------------*
 * @brief Verifies the bitwise equality of two device memory buffers.
 *
 * @param lhs The first buffer
 * @param rhs The second buffer
 * @param size_bytes The number of bytes to check for equality
 *---------------------------------------------------------------------------**/
void expect_equal_buffers(void const* lhs, void const* rhs,
                          std::size_t size_bytes);

/**---------------------------------------------------------------------------*
 * @brief Print a column for debugging.
 *
 * @param col The column to print
 *---------------------------------------------------------------------------**/
template <typename Element>
void print_column(cudf::mutable_column_view col) {
  print_typed_column<Element>(
    static_cast<const Element*>(col.data<Element>()),
    reinterpret_cast<gdf_valid_type*>(col.null_mask()),
    col.size(),
    1);
}

}  // namespace test
}  // namespace cudf
