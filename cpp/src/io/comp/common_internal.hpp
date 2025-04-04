/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "nvcomp_adapter.hpp"

#include <cudf/io/types.hpp>

#include <optional>

namespace cudf::io::detail {

/**
 * @brief GZIP header flags
 * See https://tools.ietf.org/html/rfc1952
 */
namespace GZIPHeaderFlag {
constexpr uint8_t ftext    = 0x01;  // ASCII text hint
constexpr uint8_t fhcrc    = 0x02;  // Header CRC present
constexpr uint8_t fextra   = 0x04;  // Extra fields present
constexpr uint8_t fname    = 0x08;  // Original file name present
constexpr uint8_t fcomment = 0x10;  // Comment present
};                                  // namespace GZIPHeaderFlag

// Threshold for buffer count where device compression is favored over host in AUTO mode
constexpr size_t host_compression_auto_threshold = 128;
// Threshold for buffer count where device decompression is favored over host in AUTO mode
constexpr size_t host_decompression_auto_threshold = 128;

[[nodiscard]] std::optional<nvcomp::compression_type> to_nvcomp_compression(
  compression_type compression);

}  // namespace cudf::io::detail
