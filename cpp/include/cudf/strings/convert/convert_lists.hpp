/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_convert
 * @{
 * @file
 */

/**
 * @brief Convert a list column of strings into a formatted strings column.
 *
 * The `separators` column should contain 3 strings elements in the following order:
 * - element separator (default is comma `,`)
 * - left-hand enclosure (default is `[`)
 * - right-hand enclosure (default is `]`)
 *
 * @code{.pseudo}
 * l1 = { [[a,b,c], [d,e]], [[f,g], [h]] }
 * s1 = format_list_column(l1)
 * s1 is now ["[[a,b,c],[d,e]]", "[[f,g],[h]]"]
 *
 * l2 = { [[a,b,c], [d,e]], [NULL], [[f,g], NULL, [h]] }
 * s2 = format_list_column(l1, '-', [':', '{', '}'])
 * s2 is now ["{{a:b:c}:{d:e}}", "{-}", "{{f:g}:-:{h}}"]
 * @endcode
 *
 * @throw cudf::logic_error if the input column is not a LIST type with a STRING child.
 *
 * @param input Lists column to format
 * @param na_rep Replacement string for null elements
 * @param separators Strings to use for enclosing list components and separating elements
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> format_list_column(
  lists_column_view const& input,
  string_scalar const& na_rep           = string_scalar(""),
  strings_column_view const& separators = strings_column_view(column_view{
    data_type{type_id::STRING}, 0, nullptr, nullptr, 0}),
  rmm::cuda_stream_view stream          = cudf::get_default_stream(),
  cudf::device_async_resource_ref mr    = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
