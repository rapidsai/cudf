/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <rmm/mr/device/device_memory_resource.hpp>

namespace cudf::lists {
/**
 * @addtogroup set_operations
 * @{
 * @file
 */

/**
 * @brief TBA
 *
 *
 * Example:
 * @code{.pseudo}
 * @endcode
 *
 * @throws cudf::logic_error
 *
 */
std::unique_ptr<column> set_overlap(
  lists_column_view const& lhs,
  lists_column_view const& rhs,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<column> set_intersect(
  lists_column_view const& lhs,
  lists_column_view const& rhs,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<column> set_union(
  lists_column_view const& lhs,
  lists_column_view const& rhs,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<column> set_difference(
  lists_column_view const& lhs,
  lists_column_view const& rhs,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf::lists
