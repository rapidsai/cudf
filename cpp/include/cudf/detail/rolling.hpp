/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/rolling.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace detail {

namespace rolling {
/**
 * @brief Direction tag for a range-based rolling window.
 */
enum class direction : bool {
  PRECEDING,  ///< A preceding window.
  FOLLOWING,  ///< A following window.
};
/**
 * @brief Wrapper for preprocessed group information from sorted group keys.
 */
struct preprocessed_group_info {
  rmm::device_uvector<size_type> const& labels;   ///< Mapping from row index to group label
  rmm::device_uvector<size_type> const& offsets;  ///< Mapping from group label to row offsets
  rmm::device_uvector<size_type> const&
    nulls_per_group;  ///< Mapping from group label to null count in the group
};

}  // namespace rolling

/**
 * @brief Compute the number of nulls in each group.
 *
 * @param orderby Column with null mask.
 * @param offsets Offset array defining the (sorted) groups.
 * @param stream CUDA stream used for kernel launches
 * @return device_uvector containing the null count per group.
 */
[[nodiscard]] rmm::device_uvector<cudf::size_type> nulls_per_group(
  column_view const& orderby,
  rmm::device_uvector<size_type> const& offsets,
  rmm::cuda_stream_view stream);

/**
 * @copydoc std::unique_ptr<column> rolling_window(
 *            column_view const& input,
 *            column_view const& preceding_window,
 *            column_view const& following_window,
 *            size_type min_periods,
 *            rolling_aggregation const& agg,
 *            rmm::device_async_resource_ref mr)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& preceding_window,
                                       column_view const& following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @brief Make a column representing the window offsets for a range-based window
 *
 * @param orderby Column use to define window ranges. If @p grouping is empty,
 * must be sorted. If
 * @p grouping is non-empty, must be sorted within each group. As well as
 * being sorted, must be sorted consistently with the @p order and @p null_order
 * parameters.
 * @param grouping Optional preprocessed grouping information.
 * @param order The sort order of the @p orderby column.
 * @param null_order The sort order of nulls in the @p orderby column.
 * @param window Descriptor specifying the window type.
 * @param stream CUDA stream used for device memory operations and kernel
 * launches.
 * @param mr Device memory resource used for allocations.
 * @return Column representing the window offsets as requested, suitable for passing to
 * `rolling_window`.
 */
[[nodiscard]] std::unique_ptr<column> make_range_window(
  column_view const& orderby,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  rolling::direction direction,
  order order,
  null_order null_order,
  range_window_type window,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
