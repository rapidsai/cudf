/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
