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
#include <cudf/table/table_view.hpp>

namespace cudf {
/**
 * @addtogroup reshape_transpose
 * @{
 * @file
 */

/**
 * @brief Transposes a table.
 *
 * Stores output in a contiguous column, exposing the transposed table as
 * a `table_view`.
 *
 * @throw cudf::logic_error if column types are non-homogenous
 * @throw cudf::logic_error if column types are non-fixed-width
 *
 * @param[in] input A table (M cols x N rows) to be transposed.
 * @return          The transposed input (N cols x M rows) as a `column` and
 *                  `table_view`, representing the owner and transposed table,
 *                  respectively.
 */
std::pair<std::unique_ptr<column>, table_view> transpose(
  table_view const& input,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
