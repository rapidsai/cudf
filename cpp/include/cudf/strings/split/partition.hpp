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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>

namespace cudf
{
namespace strings
{

/**
 * @brief Returns a set of 3 columns by splitting each string using the
 * specified delimiter.
 *
 * The number of rows in the output columns will be the same as the
 * input column. The first column will contain the first tokens of
 * each string as a result of the split. The second column will contain
 * the delimiter. The third column will contain the remaining characters
 * of each string after the delimiter.
 *
 * Any null string entries return corresponding null output columns.
 *
 * ```
 * s = ["ab_cd","def_g_h"]
 * r = rpartition(s,"_")
 * r[0] is ["ab","def"]
 * r[1] is ["_","_"]
 * r[2] is ["cd","g_h"]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param delimiter UTF-8 encoded string indentifying where to split each string.
 *        Default of empty string indicates split on whitespace.
 * @param mr Resource for allocating device memory.
 * @return New table of strings columns.
 */
std::unique_ptr<experimental::table> partition( strings_column_view const& strings,
                                                string_scalar const& delimiter = string_scalar(""),
                                                rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a set of 3 columns by splitting each string using the
 * specified delimiter starting from the end of each string.
 *
 * The number of rows in the output columns will be the same as the
 * input column. The first column will contain the characters of
 * each string before the last delimiter found. The second column will contain
 * the delimiter. The third column will contain the remaining characters
 * of each string after the delimiter.
 *
 * Any null string entries return corresponding null output columns.
 *
 * ```
 * s = ["ab_cd","def_g_h"]
 * r = rpartition(s,"_")
 * r[0] is ["ab","def_g"]
 * r[1] is ["_","_"]
 * r[2] is ["cd","h"]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param delimiter UTF-8 encoded string indentifying where to split each string.
 *        Default of empty string indicates split on whitespace.
 * @param mr Resource for allocating device memory.
 * @return New strings columns.
 */
std::unique_ptr<experimental::table> rpartition( strings_column_view const& strings,
                                                 string_scalar const& delimiter = string_scalar(""),
                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace strings
} // namespace cudf
