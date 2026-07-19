/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/rolling.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <optional>
#include <utility>

namespace cudf::detail {

[[nodiscard]] std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_range_windows(
  table_view const& group_keys,
  column_view const& orderby,
  order order,
  null_order null_order,
  range_window_type preceding,
  range_window_type following,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  unbounded,
  column_view const& orderby,
  rolling::direction direction,
  order order,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  bool nulls_at_start,
  scalar const* row_delta,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  current_row,
  column_view const& orderby,
  rolling::direction direction,
  order order,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  bool nulls_at_start,
  scalar const* row_delta,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  bounded_closed,
  column_view const& orderby,
  rolling::direction direction,
  order order,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  bool nulls_at_start,
  scalar const* row_delta,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  bounded_open,
  column_view const& orderby,
  rolling::direction direction,
  order order,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  bool nulls_at_start,
  scalar const* row_delta,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
