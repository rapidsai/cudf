/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/rolling.hpp>
#include <cudf/rolling.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <optional>

namespace cudf::detail::rolling {

/**
 * @brief Compute range window bounds for bounded_closed window type.
 */
std::unique_ptr<column> make_range_window_bounded_closed(
  column_view const& orderby,
  std::optional<preprocessed_group_info> const& grouping,
  direction direction,
  order order,
  bool nulls_at_start,
  scalar const* row_delta,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Compute range window bounds for bounded_open window type.
 */
std::unique_ptr<column> make_range_window_bounded_open(
  column_view const& orderby,
  std::optional<preprocessed_group_info> const& grouping,
  direction direction,
  order order,
  bool nulls_at_start,
  scalar const* row_delta,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::detail::rolling
