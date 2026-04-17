/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/parquet_schema.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>

namespace cudf {
namespace io {

struct get_period {
  template <typename T>
  int32_t operator()()
  {
    if constexpr (is_chrono<T>()) { return T::period::den; }
    CUDF_FAIL("Invalid, non chrono type");
  }
};

/**
 * @brief Function that translates cuDF time unit to clock frequency
 */
inline int32_t to_clockrate(type_id timestamp_type_id)
{
  return timestamp_type_id == type_id::EMPTY
           ? 0
           : type_dispatcher(data_type{timestamp_type_id}, get_period{});
}

/**
 * @brief Computes the timestamp scale factor between a Parquet timestamp logical type's
 * native precision and the output column's clock rate.
 *
 * Callable from host and device. Callers are expected to have already determined that the
 * column is a timestamp (e.g. `logical_type.type == LogicalType::TIMESTAMP`) and that
 * `output_clock_rate` is non-zero.
 *
 * @param logical_type Parquet logical type of the column (must be `TIMESTAMP`)
 * @param output_clock_rate Clock rate of the output timestamp column (e.g. 1000 for ms)
 * @return Scale factor: <0 means divide by `-scale`, >0 means multiply by `scale`, 0 means no-op
 */
[[nodiscard]] CUDF_HOST_DEVICE inline int32_t calc_timestamp_scale(
  parquet::LogicalType const& logical_type, int32_t output_clock_rate)
{
  // Extracted from
  // https://github.com/rapidsai/cudf/blob/c89c83c00c729a86c56570693b627f31408bc2c9/cpp/src/io/parquet/page_decode.cuh#L1219-L1236
  int32_t native_units = 0;
  if (logical_type.is_timestamp_millis()) {
    native_units = cudf::timestamp_ms::period::den;
  } else if (logical_type.is_timestamp_micros()) {
    native_units = cudf::timestamp_us::period::den;
  } else if (logical_type.is_timestamp_nanos()) {
    native_units = cudf::timestamp_ns::period::den;
  }
  if (native_units != 0 && native_units != output_clock_rate) {
    return (output_clock_rate < native_units) ? -(native_units / output_clock_rate)
                                              : (output_clock_rate / native_units);
  }
  return 0;
}

}  // namespace io
}  // namespace cudf
