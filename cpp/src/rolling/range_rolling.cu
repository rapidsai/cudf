/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/rolling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/rolling.hpp>
#include <cudf/rolling/range_window_bounds.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/functional>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <optional>

namespace CUDF_EXPORT cudf {
namespace detail {

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
  rmm::cuda_stream_view stream)
{
  auto d_orderby        = column_device_view::create(orderby, stream);
  auto const num_groups = offsets.size() - 1;
  std::size_t bytes{0};
  auto is_null_it = cudf::detail::make_counting_transform_iterator(
    cudf::size_type{0},
    cuda::proclaim_return_type<size_type>(
      [orderby = *d_orderby] __device__(size_type i) -> size_type {
        return static_cast<size_type>(orderby.is_null_nocheck(i));
      }));
  rmm::device_uvector<cudf::size_type> null_counts{num_groups, stream};
  cub::DeviceSegmentedReduce::Sum(nullptr,
                                  bytes,
                                  is_null_it,
                                  null_counts.begin(),
                                  num_groups,
                                  offsets.begin(),
                                  offsets.begin() + 1,
                                  stream.value());
  auto tmp = rmm::device_buffer(bytes, stream);
  cub::DeviceSegmentedReduce::Sum(tmp.data(),
                                  bytes,
                                  is_null_it,
                                  null_counts.begin(),
                                  num_groups,
                                  offsets.begin(),
                                  offsets.begin() + 1,
                                  stream.value());
  return null_counts;
}

/**
 * @brief Deduce the `null_order` of a column given sort order and group offsets
 *
 * @param orderby The orderby column to check.
 * @param order The sort order of the column.
 * @param offsets Group offsets.
 * @param stream CUDA stream used for kernel launches.
 *
 * @return The deduced `null_order`.
 */
[[nodiscard]] null_order deduce_null_order(column_view const& orderby,
                                           order order,
                                           rmm::device_uvector<size_type> const& offsets,
                                           rmm::device_uvector<size_type> const& per_group_nulls,
                                           rmm::cuda_stream_view stream)
{
  auto d_orderby = column_device_view::create(orderby, stream);
  if (order == order::ASCENDING) {
    // Sort order is ASCENDING
    // null_order was either BEFORE or AFTER
    // If at least one group has a null at the beginning and that
    // group has more entries than the null count of the group, must
    // be nulls at starts of groups (BEFORE),otherwise must be nulls at end (AFTER)
    auto it = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<bool>(
        [d_orderby       = *d_orderby,
         d_offsets       = offsets.data(),
         nulls_per_group = per_group_nulls.data()] __device__(size_type i) -> bool {
          return nulls_per_group[i] < (d_offsets[i + 1] - d_offsets[i]) &&
                 d_orderby.is_null_nocheck(d_offsets[i]);
        }));
    auto is_before = thrust::reduce(
      rmm::exec_policy_nosync(stream), it, it + offsets.size() - 1, false, thrust::logical_or<>{});
    return is_before ? null_order::BEFORE : null_order::AFTER;
  } else {
    // Sort order is DESCENDING
    // null_order was either BEFORE or AFTER
    // If at least one group has a null at the end and that group has
    // more entries than the null count of the group must be nulls at ends of groups (BEFORE).
    // Otherwise must be nulls at start (AFTER)
    auto it = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<bool>(
        [d_orderby       = *d_orderby,
         d_offsets       = offsets.data(),
         nulls_per_group = per_group_nulls.data()] __device__(size_type i) -> bool {
          return nulls_per_group[i] < (d_offsets[i + 1] - d_offsets[i]) &&
                 d_orderby.is_null_nocheck(d_offsets[i + 1] - 1);
        }));
    auto is_before = thrust::reduce(
      rmm::exec_policy_nosync(stream), it, it + offsets.size() - 1, false, thrust::logical_or<>{});
    return is_before ? null_order::BEFORE : null_order::AFTER;
  }
}

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_range_window_bounds(
  table_view const& group_keys,
  column_view const& orderby,
  order order,
  std::optional<null_order> null_order,
  range_window_type preceding,
  range_window_type following,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto make_preceding = [&](std::optional<detail::rolling::preprocessed_group_info> const& grouping,
                            cudf::null_order null_order) {
    return make_preceding_range_window(orderby, grouping, order, null_order, preceding, stream, mr);
  };
  auto make_following = [&](std::optional<detail::rolling::preprocessed_group_info> const& grouping,
                            cudf::null_order null_order) {
    return make_following_range_window(orderby, grouping, order, null_order, following, stream, mr);
  };

  if (group_keys.num_columns() > 0) {
    using sort_helper = cudf::groupby::detail::sort::sort_groupby_helper;
    sort_helper helper{group_keys, null_policy::INCLUDE, sorted::YES, {}};
    auto const& labels   = helper.group_labels(stream);
    auto const& offsets  = helper.group_offsets(stream);
    auto per_group_nulls = orderby.has_nulls() ? nulls_per_group(orderby, offsets, stream)
                                               : rmm::device_uvector<size_type>{0, stream};
    detail::rolling::preprocessed_group_info grouping{labels, offsets, per_group_nulls};
    auto deduced_null_order = [&]() {
      if (null_order.has_value()) { return null_order.value(); }
      if (!orderby.has_nulls()) {
        // Doesn't matter in this case
        return null_order::BEFORE;
      }
      return deduce_null_order(orderby, order, offsets, per_group_nulls, stream);
    }();
    return {make_preceding(grouping, deduced_null_order),
            make_following(grouping, deduced_null_order)};
  } else {
    auto deduced_null_order = [&]() {
      if (null_order.has_value()) { return null_order.value(); }
      if (!orderby.has_nulls()) {
        // Doesn't matter in this case.
        return null_order::BEFORE;
      }
      if ((order == order::ASCENDING && orderby.null_count(0, 1, stream) == 1) ||
          (order == order::DESCENDING &&
           orderby.null_count(orderby.size() - 1, orderby.size(), stream) == 1)) {
        return null_order::BEFORE;
      } else {
        return null_order::AFTER;
      }
    }();
    return {make_preceding(std::nullopt, deduced_null_order),
            make_following(std::nullopt, deduced_null_order)};
  }
}
}  // namespace detail

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_range_windows(
  table_view const& group_keys,
  column_view const& orderby,
  order order,
  range_window_type preceding,
  range_window_type following,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(
    group_keys.num_columns() == 0 || group_keys.num_rows() == orderby.size(),
    "If a grouping table is provided, it must have same number of rows as the orderby column.");
  // This interface is a stop-gap until we can migrate the old grouped rolling code.
  return detail::make_range_window_bounds(
    group_keys, orderby, order, std::nullopt, preceding, following, stream, mr);
}

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_range_windows(
  table_view const& group_keys,
  column_view const& orderby,
  order order,
  null_order null_order,
  range_window_type preceding,
  range_window_type following,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(
    group_keys.num_columns() == 0 || group_keys.num_rows() == orderby.size(),
    "If a grouping table is provided, it must have same number of rows as the orderby column.");
  return detail::make_range_window_bounds(
    group_keys, orderby, order, null_order, preceding, following, stream, mr);
}

}  // namespace CUDF_EXPORT cudf
