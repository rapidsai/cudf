/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/optional>

namespace cudf::io::parquet::detail {

/**
 * @brief Computes the timestamp scale between a Parquet timestamp logical type's
 * native precision and the output column's clock rate.
 *
 * @param logical_type Logical type of the column
 * @param clock_rate Clock rate of the output timestamp column (e.g. 1000 for ms)
 * @return Scale factor; <0 means divide by `-scale`, >0 means multiply by `scale`, 0 means no-op
 */
[[nodiscard]] CUDF_HOST_DEVICE inline int32_t calc_timestamp_scale(
  cuda::std::optional<LogicalType> logical_type, int32_t clock_rate)
{
  if (not logical_type.has_value()) { return 0; }

  auto units = 0;
  if (logical_type->is_timestamp_millis()) {
    units = cudf::timestamp_ms::period::den;
  } else if (logical_type->is_timestamp_micros()) {
    units = cudf::timestamp_us::period::den;
  } else if (logical_type->is_timestamp_nanos()) {
    units = cudf::timestamp_ns::period::den;
  }

  if (units and units != clock_rate) {
    return (clock_rate < units) ? -(units / clock_rate) : (clock_rate / units);
  }

  return 0;
}

}  // namespace cudf::io::parquet::detail
