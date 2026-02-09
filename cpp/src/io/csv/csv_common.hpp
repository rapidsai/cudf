/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/utilities/column_type_histogram.hpp"

#include <cstdint>

namespace cudf {
namespace io {
namespace csv {
namespace column_parse {
/**
 * @brief Per-column parsing flags used for dtype detection and data conversion
 */
enum : uint8_t {
  disabled       = 0,   ///< data is not read
  enabled        = 1,   ///< data is read and parsed as usual
  inferred       = 2,   ///< infer the dtype
  as_default     = 4,   ///< no special decoding
  as_hexadecimal = 8,   ///< decode with base-16
  as_datetime    = 16,  ///< decode as date and/or time
};
using flags = uint8_t;

}  // namespace column_parse

}  // namespace csv
}  // namespace io
}  // namespace cudf
