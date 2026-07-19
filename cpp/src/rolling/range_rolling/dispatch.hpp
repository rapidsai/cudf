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
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <optional>
#include <type_traits>

namespace cudf::detail::rolling {

struct range_window_dispatch_args {
  column_view const& orderby;
  direction window_direction;
  cudf::order sort_order;
  std::optional<preprocessed_group_info> const& grouping;
  bool nulls_at_start;
  scalar const* row_delta;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;
};

struct signed_integral_orderby {};
struct unsigned_integral_orderby {};
struct floating_point_orderby {};
struct timestamp_orderby {};
struct fixed_point_orderby {};
struct string_orderby {};

}  // namespace cudf::detail::rolling

namespace cudf::detail {

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  bounded_closed,
  rolling::signed_integral_orderby,
  rolling::range_window_dispatch_args const& args);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  bounded_closed,
  rolling::unsigned_integral_orderby,
  rolling::range_window_dispatch_args const& args);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  bounded_closed, rolling::floating_point_orderby, rolling::range_window_dispatch_args const& args);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  bounded_closed, rolling::timestamp_orderby, rolling::range_window_dispatch_args const& args);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  bounded_closed, rolling::fixed_point_orderby, rolling::range_window_dispatch_args const& args);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  bounded_open, rolling::signed_integral_orderby, rolling::range_window_dispatch_args const& args);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  bounded_open,
  rolling::unsigned_integral_orderby,
  rolling::range_window_dispatch_args const& args);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  bounded_open, rolling::floating_point_orderby, rolling::range_window_dispatch_args const& args);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  bounded_open, rolling::timestamp_orderby, rolling::range_window_dispatch_args const& args);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  bounded_open, rolling::fixed_point_orderby, rolling::range_window_dispatch_args const& args);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  current_row, rolling::signed_integral_orderby, rolling::range_window_dispatch_args const& args);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  current_row, rolling::unsigned_integral_orderby, rolling::range_window_dispatch_args const& args);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  current_row, rolling::floating_point_orderby, rolling::range_window_dispatch_args const& args);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  current_row, rolling::timestamp_orderby, rolling::range_window_dispatch_args const& args);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  current_row, rolling::fixed_point_orderby, rolling::range_window_dispatch_args const& args);

[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  current_row, rolling::string_orderby, rolling::range_window_dispatch_args const& args);

namespace rolling {

template <typename WindowType>
[[nodiscard]] std::unique_ptr<column> dispatch_range_window_by_family(
  WindowType window, range_window_dispatch_args const& args)
{
  switch (args.orderby.type().id()) {
    case type_id::INT8:
    case type_id::INT16:
    case type_id::INT32:
    case type_id::INT64:
      return cudf::detail::dispatch_range_window(window, signed_integral_orderby{}, args);
    case type_id::UINT8:
    case type_id::UINT16:
    case type_id::UINT32:
    case type_id::UINT64:
      return cudf::detail::dispatch_range_window(window, unsigned_integral_orderby{}, args);
    case type_id::FLOAT32:
    case type_id::FLOAT64:
      return cudf::detail::dispatch_range_window(window, floating_point_orderby{}, args);
    case type_id::TIMESTAMP_DAYS:
    case type_id::TIMESTAMP_SECONDS:
    case type_id::TIMESTAMP_MILLISECONDS:
    case type_id::TIMESTAMP_MICROSECONDS:
    case type_id::TIMESTAMP_NANOSECONDS:
      return cudf::detail::dispatch_range_window(window, timestamp_orderby{}, args);
    case type_id::DECIMAL32:
    case type_id::DECIMAL64:
    case type_id::DECIMAL128:
      return cudf::detail::dispatch_range_window(window, fixed_point_orderby{}, args);
    case type_id::STRING:
      if constexpr (std::is_same_v<WindowType, current_row>) {
        return cudf::detail::dispatch_range_window(window, string_orderby{}, args);
      }
      break;
    default: break;
  }
  CUDF_FAIL("Unsupported rolling window type.", cudf::data_type_error);
}

}  // namespace rolling
}  // namespace cudf::detail
