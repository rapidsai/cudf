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
  PRIMITIVE         = 0,
  SHORT_STRING = 1,
  OBJECT               = 2,
  ARRAY                 = 3,
};

/**
 * @brief Physical type ID carried in the value_header of a primitive VARIANT value.
 */
enum class variant_primitive_type : uint8_t {
  null                 = 0,
  boolean_true         = 1,
  boolean_false        = 2,
  int8                 = 3,
  int16                = 4,
  int32                = 5,
  int64                = 6,
  float64              = 7,
  decimal4             = 8,
  decimal8             = 9,
  decimal16            = 10,
  date                 = 11,
  timestamp_micros     = 12,
  timestamp_ntz_micros = 13,
  float32              = 14,
  binary               = 15,
  long_string          = 16,
  time_ntz_micros      = 17,
  timestamp_nanos      = 18,
  timestamp_ntz_nanos  = 19,
  uuid                 = 20,
};

}  // namespace cudf::io::parquet::experimental
