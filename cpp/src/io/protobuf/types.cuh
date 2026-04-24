/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <cudf/io/protobuf.hpp>

namespace cudf::io::protobuf::detail {

// Protobuf varint encoding uses at most 10 bytes to represent a 64-bit value.
constexpr int MAX_VARINT_BYTES = 10;

// CUDA kernel launch configuration.
constexpr int THREADS_PER_BLOCK = 256;

// Error codes for kernel error reporting.
constexpr int ERR_BOUNDS                  = 1;
constexpr int ERR_VARINT                  = 2;
constexpr int ERR_FIELD_NUMBER            = 3;
constexpr int ERR_WIRE_TYPE               = 4;
constexpr int ERR_OVERFLOW                = 5;
constexpr int ERR_FIELD_SIZE              = 6;
constexpr int ERR_SKIP                    = 7;
constexpr int ERR_FIXED_LEN               = 8;
constexpr int ERR_REQUIRED                = 9;
constexpr int ERR_SCHEMA_TOO_LARGE        = 10;
constexpr int ERR_MISSING_ENUM_META       = 11;
constexpr int ERR_REPEATED_COUNT_MISMATCH = 12;

// Threshold for using a direct-mapped lookup table for field_number -> field_index.
// Field numbers above this threshold fall back to linear search.
constexpr int FIELD_LOOKUP_TABLE_MAX = 4096;

/**
 * Structure to record field location within a message.
 * offset < 0 means field was not found.
 */
struct field_location {
  int32_t offset;  // Offset of field data within the message (-1 if not found)
  int32_t length;  // Length of field data in bytes
};

/**
 * Field descriptor passed to the scanning kernel.
 */
struct field_descriptor {
  int field_number;        // Protobuf field number
  int expected_wire_type;  // Expected wire type for this field
  bool is_repeated;        // Repeated children are scanned via count/scan kernels
};

/**
 * Information about repeated field occurrences in a row.
 */
struct repeated_field_info {
  int32_t count;         // Number of occurrences in this row
  int32_t total_length;  // Total bytes for all occurrences (for varlen fields)
};

/**
 * Location of a single occurrence of a repeated field.
 */
struct repeated_occurrence {
  int32_t row_idx;  // Which row this occurrence belongs to
  int32_t offset;   // Offset within the message
  int32_t length;   // Length of the field data
};

/**
 * Per-field descriptor passed to the combined occurrence scan kernel.
 * Contains device pointers so the kernel can write to each field's output.
 */
struct repeated_field_scan_desc {
  int field_number;
  int wire_type;
  int32_t const* row_offsets;        // Pre-computed prefix-sum offsets [num_rows + 1]
  repeated_occurrence* occurrences;  // Output buffer [total_count]
};

/**
 * Device-side descriptor for nested schema fields.
 */
struct device_nested_field_descriptor {
  int field_number;
  int parent_idx;
  int depth;
  int wire_type;
  int output_type_id;
  int encoding;
  bool is_repeated;
  bool is_required;
  bool has_default_value;

  device_nested_field_descriptor() = default;

  // Wire type and encoding are stored as int (not typed enums) because CUDA device code
  // historically had limited constexpr enum support, and the kernel comparison sites use
  // int-typed wire_type_value()/encoding_value() helpers throughout.
  explicit device_nested_field_descriptor(nested_field_descriptor const& src)
    : field_number(src.field_number),
      parent_idx(src.parent_idx),
      depth(src.depth),
      wire_type(static_cast<int>(src.wire_type)),
      output_type_id(static_cast<int>(src.output_type)),
      encoding(static_cast<int>(src.encoding)),
      is_repeated(src.is_repeated),
      is_required(src.is_required),
      has_default_value(src.has_default_value)
  {
  }
};

}  // namespace cudf::io::protobuf::detail
