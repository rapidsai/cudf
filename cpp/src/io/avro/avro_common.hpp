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

#include "io/utilities/column_buffer.hpp"

#include <cstdint>
#include <cstdio>

namespace cudf {
namespace io {
namespace avro {
struct block_desc_s {
  block_desc_s() = default;  // required to compile on ctk-12.2 + aarch64

  explicit constexpr block_desc_s(
    size_t offset_, uint32_t size_, uint32_t row_offset_, uint32_t first_row_, uint32_t num_rows_)
    : offset(offset_),
      size(size_),
      row_offset(row_offset_),
      first_row(first_row_),
      num_rows(num_rows_)
  {
  }

  // Offset of this block, in bytes, from the start of the file.
  size_t offset;

  // Size of this block, in bytes.
  uint32_t size;

  // The absolute row offset that needs to be added to each row index in order
  // to derive the offset of the decoded data in the destination array.  E.g.
  // `const ptrdiff_t dst_row = ((row - first_row) + row_offset)`.  See
  // `avro_decode_row()` for details.
  uint32_t row_offset;

  // The index of the first row to be *saved* from this block.  That is, the
  // number of rows to skip in this block before starting to save values.  If
  // this is 0, then no rows will be skipped (all rows will be saved).  If a
  // user has requested `read_avro()` to skip rows, that will materialize as a
  // non-zero `first_row` value in the appropriate block containing the first
  // row to be saved.
  //
  // N.B. We explicitly use the word "saved" here, not "decoded".  Technically,
  //      all rows are decoded, one column at a time, as the process of decoding
  //      a column value is what informs us of the value's size in bytes (in its
  //      encoded form), and thus, where the next column starts.  However, we
  //      only *save* these decoded values based on the `first_row`.
  uint32_t first_row;

  // The number of rows to save from this block.  If a user has requested
  // `read_avro()` to limit the number of rows to return, this will materialize
  // as a `num_rows` value less than the total number of rows in the appropriate
  // block.  Otherwise, `num_rows` will be equal to the total number of rows in
  // the block, after skipping `first_row` rows (if applicable).
  //
  // N.B. Unlike `first_rows`, where all rows and columns are decoded prior to
  //      reaching the point we've been requested to start *saving* values --
  //      once the `num_rows` limit has been reached, no further decoding takes
  //      place.
  uint32_t num_rows;
};

enum type_kind_e {
  type_not_set = -1,
  // Primitive types
  type_null = 0,
  type_boolean,
  type_int,
  type_long,
  type_float,
  type_double,
  type_bytes,
  type_string,
  // Complex types
  type_enum,
  type_record,
  type_union,
  type_array,
  type_fixed,
  // Logical types
  type_decimal,
  type_uuid,
  type_date,
  type_time_millis,
  type_time_micros,
  type_timestamp_millis,
  type_timestamp_micros,
  type_local_timestamp_millis,
  type_local_timestamp_micros,
  type_duration,
};

enum logicaltype_kind_e {
  logicaltype_not_set = 0,
  // N.B. We intentionally mirror the logicaltype enum values with their
  //      equivalent type enum value, as this allows us to cast the type
  //      value directly to a logical type without an intermediate
  //      mapping step, and vice versa, e.g.:
  //
  //        auto kind = type_date;
  //        auto logical_kind = static_cast<logical_kind_e>(type_date);
  //        // logical_kind == logicaltype_kind_e::logicaltype_date
  //
  //      And:
  //
  //        auto logical_kind = logicaltype_date;
  //        auto kind = static_cast<type_kind_e>(logical_kind);
  //        // kind == type_kind_e::type_date
  //
  logicaltype_decimal = type_decimal,
  logicaltype_uuid,
  logicaltype_date,
  logicaltype_time_millis,
  logicaltype_time_micros,
  logicaltype_timestamp_millis,
  logicaltype_timestamp_micros,
  logicaltype_local_timestamp_millis,
  logicaltype_local_timestamp_micros,
  logicaltype_duration,
};

/**
 * @brief Determines if the supplied logical type is currently supported.
 *
 * @param[in] logical_kind Supplies the logicaltype_kind_e enum value.
 *
 * @return true if the logical type is supported, false otherwise.
 */
CUDF_HOST_DEVICE inline constexpr bool is_supported_logical_type(logicaltype_kind_e logical_kind)
{
  switch (logical_kind) {
    case logicaltype_date: return true;

    case logicaltype_not_set: [[fallthrough]];
    case logicaltype_decimal: [[fallthrough]];
    case logicaltype_uuid: [[fallthrough]];
    case logicaltype_time_millis: [[fallthrough]];
    case logicaltype_time_micros: [[fallthrough]];
    case logicaltype_timestamp_millis: [[fallthrough]];
    case logicaltype_timestamp_micros: [[fallthrough]];
    case logicaltype_local_timestamp_millis: [[fallthrough]];
    case logicaltype_local_timestamp_micros: [[fallthrough]];
    case logicaltype_duration: [[fallthrough]];
    default: return false;
  }
}

using cudf::io::detail::string_index_pair;

}  // namespace avro
}  // namespace io
}  // namespace cudf
