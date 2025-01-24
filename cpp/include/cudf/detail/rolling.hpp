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
 * @tparam Direction Is this a preceding window or a following one.
 *
 * @param group_keys Table defining grouping of the windows. May be empty. If
 * non-empty, group keys must be sorted.
 * @param orderby Column use to define window ranges. If @p group_keys is empty,
 * must be sorted. If
 * @p group_keys is non-empty, must be sorted within each group. As well as
 * being sorted, must be sorted consistently with the @p order and @p null_order
 * parameters.
 * @param order The sort order of the @p orderby column.
 * @param null_order The sort order of nulls in the @p orderby column.
 * @param row_delta Pointer to scalar providing the delta for the window range.
 * May be null, but only if the @p window_type is @p CURRENT_ROW or @p
 * UNBOUNDED. Note that @p row_delta is always added to the current row value.
 * @param window_type The type of window we are computing bounds for.
 * @param stream CUDA stream used for device memory operations and kernel
 * launches.
 * @param mr Device memory resource used for allocations.
 */
template <rolling::direction Direction>
[[nodiscard]] std::unique_ptr<column> make_range_window_bound(
  column_view const& orderby,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  order order,
  null_order null_order,
  range_window_type window,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

[[nodiscard]] std::unique_ptr<column> make_preceding_range_window_bound(
  column_view const& orderby,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  order order,
  null_order null_order,
  range_window_type window,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

[[nodiscard]] std::unique_ptr<column> make_following_range_window_bound(
  column_view const& orderby,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  order order,
  null_order null_order,
  range_window_type window,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
