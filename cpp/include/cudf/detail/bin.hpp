/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace cudf {

namespace detail {

/**
 * @addtogroup binning
 * @{
 * @file
 * @brief Internal APIs for binning values.
 */

/**
 * @brief Enum used to define whether or not bins include their boundary points.
 */
enum class inclusive { YES, NO };

/**
 * @copydoc cudf::bin(column_view const& input, column_view const& left_edges, inclusive
 * left_inclusive, column_view const& right_edges, inclusive right_inclusive, null_order
 * edge_null_precedence null_order::BEFORE, rmm::mr::device_memory_resource* mr)
 *
 * @param stream Stream view on which to allocate resources and queue execution.
 */
std::unique_ptr<column> bin(
  column_view const& input,
  column_view const& left_edges,
  inclusive left_inclusive,
  column_view const& right_edges,
  inclusive right_inclusive,
  null_order edge_null_precedence     = null_order::BEFORE,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default);

/** @} */  // end of group
}  // namespace detail
}  // namespace cudf
