/*
 * Copyright (c) 2019-2021, BAIDU CORPORATION.
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
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace cudf {
namespace strings {

/**
 * @addtogroup strings_json
 * @{
 * @file
 */

/**
 * @brief Convert input json strings column to lists.
 *
 * Parse input json strings to a list column, which is combined by struct columns of two column, key
 * and value. and when input json string element type is OBJECT, the list size is 1, while when element
 * type is ARRAY, the list size is euqal to number of json object in the json arrray.
 *
 * @param col The input strings column. Each row must contain a valid json string
 * @param mr Resource for allocating device memory.
 * @return A LIST column of STRUCT column of a pair of string columns, key and value.
 */
std::unique_ptr<cudf::column> json_to_array(
  cudf::strings_column_view const& col,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
