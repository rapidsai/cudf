/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/range_utils.cuh"
#include "detail/rolling.hpp"

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
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/reduce.h>

#include <optional>
#include <variant>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace detail {

rmm::device_uvector<cudf::size_type> nulls_per_group(column_view const& orderby,
                                                     rmm::device_uvector<size_type> const& offsets,
                                                     rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
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

std::unique_ptr<column> make_range_window(
  column_view const& orderby,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  rolling::direction direction,
  order order,
  null_order null_order,
  range_window_type window,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  bool const nulls_at_start = (order == order::ASCENDING && null_order == null_order::BEFORE) ||
                              (order == order::DESCENDING && null_order == null_order::AFTER);

  auto dispatch = [&](auto&& clamper, scalar const* row_delta) {
    return type_dispatcher(orderby.type(),
                           clamper,
                           orderby,
                           direction,
                           order,
                           grouping,
                           nulls_at_start,
                           row_delta,
                           stream,
                           mr);
  };
  return std::visit(
    [&](auto&& window) -> std::unique_ptr<column> {
      using WindowType = cuda::std::decay_t<decltype(window)>;
      return dispatch(rolling::range_window_clamper<WindowType>{}, window.delta());
    },
    window);
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
  if (group_keys.num_columns() > 0) {
    using sort_helper = cudf::groupby::detail::sort::sort_groupby_helper;
    sort_helper helper{group_keys, null_policy::INCLUDE, sorted::YES, {}};
    auto const& labels   = helper.group_labels(stream);
    auto const& offsets  = helper.group_offsets(stream);
    auto per_group_nulls = orderby.has_nulls() ? nulls_per_group(orderby, offsets, stream)
                                               : rmm::device_uvector<size_type>{0, stream};
    auto grouping = detail::rolling::preprocessed_group_info{labels, offsets, per_group_nulls};
    return {
      make_range_window(
        orderby, grouping, rolling::direction::PRECEDING, order, null_order, preceding, stream, mr),
      make_range_window(orderby,
                        grouping,
                        rolling::direction::FOLLOWING,
                        order,
                        null_order,
                        following,
                        stream,
                        mr)};
  } else {
    return {make_range_window(orderby,
                              std::nullopt,
                              rolling::direction::PRECEDING,
                              order,
                              null_order,
                              preceding,
                              stream,
                              mr),
            make_range_window(orderby,
                              std::nullopt,
                              rolling::direction::FOLLOWING,
                              order,
                              null_order,
                              following,
                              stream,
                              mr)};
  }
}

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_range_windows(
  table_view const& group_keys,
  table_view const& orderby,
  host_span<order const> orders,
  host_span<null_order const> null_orders,
  range_window_type preceding,
  range_window_type following,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(orderby.num_columns() > 0, "orderby must be non-empty");
  CUDF_EXPECTS(group_keys.num_columns() == 0 || group_keys.num_rows() == orderby.num_rows(),
               "Size mismatch between group_keys and orderby table.");
  CUDF_EXPECTS(orderby.num_columns() == static_cast<size_type>(orders.size()),
               "ORDER BY column count must match order vector");
  CUDF_EXPECTS(orderby.num_columns() == static_cast<size_type>(null_orders.size()),
               "ORDER BY column count must match null-order vector");

  if (orderby.num_columns() == 1) {
    return detail::make_range_windows(group_keys,
                                      orderby.column(0),
                                      orders.front(),
                                      null_orders.front(),
                                      preceding,
                                      following,
                                      stream,
                                      mr);
  }

  auto const is_peer_bound = [](range_window_type const& w) {
    return std::holds_alternative<unbounded>(w) || std::holds_alternative<current_row>(w);
  };
  CUDF_EXPECTS(is_peer_bound(preceding) && is_peer_bound(following),
               "Multi-column RANGE windows support only UNBOUNDED and CURRENT ROW bounds");

  using sort_helper = cudf::groupby::detail::sort::sort_groupby_helper;

  std::vector<column_view> peer_keys(group_keys.begin(), group_keys.end());
  peer_keys.insert(peer_keys.end(), orderby.begin(), orderby.end());
  sort_helper peer_helper{table_view{peer_keys}, null_policy::INCLUDE, sorted::YES, {}};
  auto const peers = rolling::grouped{peer_helper.group_labels(stream).data(),
                                      peer_helper.group_offsets(stream).data()};

  std::optional<sort_helper> group_helper;
  if (group_keys.num_columns() > 0 && (std::holds_alternative<unbounded>(preceding) ||
                                       std::holds_alternative<unbounded>(following))) {
    group_helper.emplace(group_keys, null_policy::INCLUDE, sorted::YES, std::vector<null_order>{});
  }

  auto const num_rows   = orderby.num_rows();
  auto preceding_result = make_numeric_column(
    data_type{type_to_id<size_type>()}, num_rows, mask_state::UNALLOCATED, stream, mr);
  auto following_result = make_numeric_column(
    data_type{type_to_id<size_type>()}, num_rows, mask_state::UNALLOCATED, stream, mr);

  auto write_offsets = [&](auto preceding_grouping, auto following_grouping) {
    auto preceding_view = preceding_result->mutable_view();
    auto following_view = following_result->mutable_view();

    auto const source_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<cuda::std::tuple<size_type, size_type>>(
        [preceding_functor =
           rolling::unbounded_distance_functor{preceding_grouping, rolling::direction::PRECEDING},
         following_functor = rolling::unbounded_distance_functor{
           following_grouping, rolling::direction::FOLLOWING}] __device__(size_type i) {
          return {preceding_functor(i), following_functor(i)};
        }));

    thrust::copy_n(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      source_iter,
      num_rows,
      cuda::zip_iterator(preceding_view.begin<size_type>(), following_view.begin<size_type>()));
  };

  using offset_grouping      = std::variant<rolling::grouped, rolling::ungrouped>;
  auto const select_grouping = [&](range_window_type const& window) -> offset_grouping {
    if (std::holds_alternative<current_row>(window)) {
      return peers;
    } else if (group_helper.has_value()) {
      return rolling::grouped{group_helper->group_labels(stream).data(),
                              group_helper->group_offsets(stream).data()};
    } else {
      return rolling::ungrouped{num_rows};
    }
  };

  auto preceding_grouping = select_grouping(preceding);
  auto following_grouping = select_grouping(following);
  std::visit(write_offsets, preceding_grouping, following_grouping);

  return std::pair{std::move(preceding_result), std::move(following_result)};
}
}  // namespace detail

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
  return detail::make_range_windows(
    group_keys, orderby, order, null_order, preceding, following, stream, mr);
}

}  // namespace CUDF_EXPORT cudf
