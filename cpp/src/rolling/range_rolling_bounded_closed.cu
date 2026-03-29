/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/range_utils.cuh"
#include "detail/range_window_dispatch.hpp"

namespace cudf::detail::rolling {

std::unique_ptr<column> make_range_window_bounded_closed(
  column_view const& orderby,
  std::optional<preprocessed_group_info> const& grouping,
  direction direction,
  order order,
  bool nulls_at_start,
  scalar const* row_delta,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return type_dispatcher(orderby.type(),
                         range_window_clamper<bounded_closed>{},
                         orderby,
                         direction,
                         order,
                         grouping,
                         nulls_at_start,
                         row_delta,
                         stream,
                         mr);
}

}  // namespace cudf::detail::rolling
