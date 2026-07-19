/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/range_rolling.hpp"
#include "range_rolling/dispatch.hpp"

#include <memory>

namespace cudf::detail {

std::unique_ptr<column> dispatch_range_window(
  bounded_closed window,
  column_view const& orderby,
  rolling::direction direction,
  order order,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  bool nulls_at_start,
  scalar const* row_delta,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return rolling::dispatch_range_window_by_family(
    window,
    rolling::range_window_dispatch_args{
      orderby, direction, order, grouping, nulls_at_start, row_delta, stream, mr});
}

}  // namespace cudf::detail
