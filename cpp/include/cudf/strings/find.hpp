/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <rmm/mr/device/per_device_resource.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_find
 * @{
 * @file
 */

/**
 * @brief Returns a column of character position values where the target
 * string is first found in each string of the provided column.
 *
 * If `target` is not found, -1 is returned for that row entry in the output column.
 *
 * The target string is searched within each string in the character
 * position range [start,stop). If the stop parameter is -1, then the
 * end of each string becomes the final position to include in the search.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @throw cudf::logic_error if start position is greater than stop position.
 *
 * @param strings Strings instance for this operation.
 * @param target UTF-8 encoded string to search for in each string.
 * @param start First character position to include in the search.
 * @param stop Last position (exclusive) to include in the search.
 *             Default of -1 will search to the end of the string.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New integer column with character position values.
 */
std::unique_ptr<column> find(
  strings_column_view const& strings,
  string_scalar const& target,
  size_type start                     = 0,
  size_type stop                      = -1,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a column of character position values where the target
 * string is first found searching from the end of each string.
 *
 * If `target` is not found, -1 is returned for that entry.
 *
 * The target string is searched within each string in the character
 * position range [start,stop). If the stop parameter is -1, then the
 * end of each string becomes the final position to include in the search.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @throw cudf::logic_error if start position is greater than stop position.
 *
 * @param strings Strings instance for this operation.
 * @param target UTF-8 encoded string to search for in each string.
 * @param start First position to include in the search.
 * @param stop Last position (exclusive) to include in the search.
 *             Default of -1 will search starting at the end of the string.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New integer column with character position values.
 */
std::unique_ptr<column> rfind(
  strings_column_view const& strings,
  string_scalar const& target,
  size_type start                     = 0,
  size_type stop                      = -1,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * the target string was found within that string in the provided column.
 *
 * If the `target` is not found for a string, false is returned for that entry in the output column.
 * If `target` is an empty string, true is returned for all non-null entries in the output column.
 *
 * Any null string entries return corresponding null entries in the output columns.
 *
 * @param strings Strings instance for this operation.
 * @param target UTF-8 encoded string to search for in each string.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New type_id::BOOL8 column.
 */
std::unique_ptr<column> contains(
  strings_column_view const& strings,
  string_scalar const& target,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * the corresponding target string was found within that string in the provided column.
 *
 * The 'output[i] = true` if string `targets[i]` is found inside `strings[i]` otherwise
 * `output[i] = false`.
 * If `target[i]` is an empty string, true is returned for `output[i]`.
 * If `target[i]` is null, false is returned for `output[i]`.
 *
 * Any null `strings[i]` row results in a null `output[i]` row.
 *
 * @throw cudf::logic_error if `strings.size() != targets.size()`.
 *
 * @param strings Strings instance for this operation.
 * @param targets Strings column of targets to check row-wise in `strings`.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New type_id::BOOL8 column.
 */
std::unique_ptr<column> contains(
  strings_column_view const& strings,
  strings_column_view const& targets,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * the target string was found at the beginning of that string in the provided column.
 *
 * If `target` is not found at the beginning of a string, false is set for
 * that row entry in the output column.
 * If `target` is an empty string, true is returned for all non-null entries in the output column.
 *
 * Any null string entries return corresponding null entries in the output columns.
 *
 * @param strings Strings instance for this operation.
 * @param target UTF-8 encoded string to search for in each string.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New type_id::BOOL8 column.
 */
std::unique_ptr<column> starts_with(
  strings_column_view const& strings,
  string_scalar const& target,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * corresponding string in target column was found at the beginning of that string in
 * the provided column.
 *
 * If `targets[i]` is not found at the beginning of a string in `strings[i]`, false is set for
 * that row entry in the output column.
 * If `targets[i]` is an empty string, true is returned for corresponding entry in the
 * output column.
 *
 * Any null string entries in `targets` return corresponding null entries in the output columns.
 *
 * @throw cudf::logic_error if `strings.size() != targets.size()`.
 *
 * @param strings Strings instance for this operation.
 * @param targets Strings instance for this operation.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New type_id::BOOL8 column.
 */
std::unique_ptr<column> starts_with(
  strings_column_view const& strings,
  strings_column_view const& targets,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * the target string was found at the end of that string in the provided column.
 *
 * If `target` is not found at the end of a string, false is set for
 * that row entry in the output column.
 * If `target` is an empty string, true is returned for all non-null entries in the output column.
 *
 * Any null string entries return corresponding null entries in the output columns.
 *
 * @param strings Strings instance for this operation.
 * @param target UTF-8 encoded string to search for in each string.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New type_id::BOOL8 column.
 */
std::unique_ptr<column> ends_with(
  strings_column_view const& strings,
  string_scalar const& target,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * corresponding string in target column was found at the end of that string in
 * the provided column.
 *
 * If `targets[i]` is not found at the end of a string in `strings[i]`, false is set for
 * that row entry in the output column.
 * If `targets[i]` is an empty string, true is returned for the corresponding entry in the
 * output column.
 *
 * Any null string entries in `targets` return corresponding null entries in the output columns.
 *
 * @throw cudf::logic_error if `strings.size() != targets.size()`.
 *
 * @param strings Strings instance for this operation.
 * @param targets Strings instance for this operation.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New type_id::BOOL8 column.
 */
std::unique_ptr<column> ends_with(
  strings_column_view const& strings,
  strings_column_view const& targets,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
