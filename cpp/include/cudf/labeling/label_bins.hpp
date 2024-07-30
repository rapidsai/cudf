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
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup label_bins
 * @{
 * @file
 * @brief APIs for labeling values by bin.
 */

/**
 * @brief Enum used to define whether or not bins include their boundary points.
 */
enum class inclusive { YES, NO };

/**
 * @brief Labels elements based on membership in the specified bins.
 *
 * A bin `i` is defined by `left_edges[i], right_edges[i]`. Whether the edges are inclusive or
 * not is determined by `left_inclusive` and `right_inclusive`, respectively.
 *
 * A value `input[j]` belongs to bin `i` if `value[j]` is contained in the range `left_edges[i],
 * right_edges[i]` (with the specified inclusiveness) and `label[j] == i`. If  `input[j]` does not
 * belong to any bin, then `label[j]` is NULL.
 *
 * Notes:
 *   - If an empty set of edges is provided, all elements in `input` are labeled NULL.
 *   - NULL elements in `input` belong to no bin and their corresponding label is NULL.
 *   - NaN elements in `input` belong to no bin and their corresponding label is NULL.
 *   - Bins must be provided in monotonically increasing order, otherwise behavior is undefined.
 *   - If two or more bins overlap, behavior is undefined.
 *
 * @throws cudf::logic_error if `input.type() == left_edges.type() == right_edges.type()` is
 * violated.
 * @throws cudf::logic_error if `left_edges.size() != right_edges.size()`
 * @throws cudf::logic_error if `left_edges.has_nulls()` or `right_edges.has_nulls()`
 *
 * @param input The input elements to label according to the specified bins.
 * @param left_edges Values of the left edge of each bin.
 * @param left_inclusive Whether or not the left edge is inclusive.
 * @param right_edges Value of the right edge of each bin.
 * @param right_inclusive Whether or not the right edge is inclusive.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device.
 * @return The integer labels of the elements in `input` according to the specified bins.
 */
std::unique_ptr<column> label_bins(
  column_view const& input,
  column_view const& left_edges,
  inclusive left_inclusive,
  column_view const& right_edges,
  inclusive right_inclusive,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
