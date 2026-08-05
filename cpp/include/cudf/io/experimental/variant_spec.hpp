/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cudf::io::parquet::experimental {

/**
 * @brief Low 2 bits of a VARIANT value's metadata byte: the basic type.
 */
enum class variant_basic_type : uint8_t {
  PRIMITIVE    = 0,
  SHORT_STRING = 1,
  OBJECT       = 2,
  ARRAY        = 3,
};

/**
 * @brief Physical type ID carried in the value_header of a primitive VARIANT value.
 */
enum class variant_primitive_type : uint8_t {
  NULLVAL              = 0,
  BOOLEAN_TRUE         = 1,
  BOOLEAN_FALSE        = 2,
  INT8                 = 3,
  INT16                = 4,
  INT32                = 5,
  INT64                = 6,
  FLOAT64              = 7,
  DECIMAL4             = 8,
  DECIMAL8             = 9,
  DECIMAL16            = 10,
  DATE                 = 11,
  TIMESTAMP_MICROS     = 12,
  TIMESTAMP_NTZ_MICROS = 13,
  FLOAT32              = 14,
  BINARY               = 15,
  LONG_STRING          = 16,
  TIME_NTZ_MICROS      = 17,
  TIMESTAMP_NANOS      = 18,
  TIMESTAMP_NTZ_NANOS  = 19,
  UUID                 = 20,
};

/**
 * @brief Per-row outcome of a VARIANT extraction or conversion operation.
 *
 * A SQL-null input row produces a null status (the status column entry is null).
 * Every other row receives one of these valid status values.
 */
enum class variant_operation_status : uint8_t {
  success,             ///< The requested output was produced.
  missing_path,        ///< Path resolution failed: key absent, index out of range,
                       ///< or a non-container/null value before the final step.
  variant_null,        ///< The resolved value is an encoded VARIANT null.
  type_mismatch,       ///< Source type is not accepted for the requested operation.
  malformed_variant,   ///< Bytes needed by the requested operation are invalid or truncated.
  overflow,            ///< Conversion is outside the target range or precision.
  invalid_conversion,  ///< Conversion failed for another value-dependent reason.
};

}  // namespace cudf::io::parquet::experimental
