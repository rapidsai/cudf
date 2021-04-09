/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/strings/strings_column_view.hpp>

namespace cudf {
namespace strings {

/**
 * @addtogroup strings_json
 * @{
 * @file
 */

/**
 * @brief Apply a JSONPath string to all rows in an input strings column.
 *
 * Applies a JSONPath string to an incoming strings column where each row in the column
 * is a valid json string.  The output is returned by row as a strings column.
 *
 * https://tools.ietf.org/id/draft-goessner-dispatch-jsonpath-00.html
 * Implements only the operators: $ . [] *
 *
 * @param col The input strings column. Each row must contain a valid json string
 * @param json_path The JSONPath string to be applied to each row
 * @param mr Resource for allocating device memory.
 * @return New strings column containing the retrieved json object strings
 */
std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& col,
  cudf::string_scalar const& json_path,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
