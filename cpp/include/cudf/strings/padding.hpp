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
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/side_type.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_modify
 * @{
 * @file
 */

/**
 * @brief Add padding to each string using a provided character.
 *
 * If the string is already `width` or more characters, no padding is performed.
 * Also, no strings are truncated.
 *
 * Null string entries result in corresponding null entries in the output column.
 *
 * @code{.pseudo}
 * Example:
 * s = ['aa','bbb','cccc','ddddd']
 * r = pad(s,4)
 * r is now ['aa  ','bbb ','cccc','ddddd']
 * @endcode
 *
 * @param input Strings instance for this operation
 * @param width The minimum number of characters for each string
 * @param side Where to place the padding characters;
 *        Default is pad right (left justify)
 * @param fill_char Single UTF-8 character to use for padding;
 *        Default is the space character
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with padded strings
 */
std::unique_ptr<column> pad(
  strings_column_view const& input,
  size_type width,
  side_type side                    = side_type::RIGHT,
  std::string_view fill_char        = " ",
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Add '0' as padding to the left of each string.
 *
 * This is equivalent to `pad(width,left,'0')` but preserves the sign character
 * if it appears in the first position.
 *
 * If the string is already width or more characters, no padding is performed.
 * No strings are truncated.
 *
 * Null rows in the input result in corresponding null rows in the output column.
 *
 * @code{.pseudo}
 * Example:
 * s = ['1234','-9876','+0.34','-342567', '2+2']
 * r = zfill(s,6)
 * r is now ['001234','-09876','+00.34','-342567', '0002+2']
 * @endcode
 *
 * @param input Strings instance for this operation
 * @param width The minimum number of characters for each string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column of strings
 */
std::unique_ptr<column> zfill(
  strings_column_view const& input,
  size_type width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
