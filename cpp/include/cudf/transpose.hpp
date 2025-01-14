/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT cudf {
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
 * @throw cudf::logic_error if column types are non-homogeneous
 * @throw cudf::logic_error if column types are non-fixed-width
 *
 * @param[in] input   A table (M cols x N rows) to be transposed
 * @param[in] stream  CUDA stream used for device memory operations and kernel launches
 * @param[in] mr      Device memory resource used to allocate the device memory of returned value
 * @return            The transposed input (N cols x M rows) as a `column` and
 *                    `table_view`, representing the owner and transposed table,
 *                    respectively.
 */
std::pair<std::unique_ptr<column>, table_view> transpose(
  table_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
