/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <io/utilities/column_buffer.hpp>

#include <cstdint>
#include <cstdio>

namespace cudf {
namespace io {
namespace avro {
struct block_desc_s {
  block_desc_s() {}
  explicit constexpr block_desc_s(size_t offset_,
                                  uint32_t size_,
                                  uint32_t first_row_,
                                  uint32_t num_rows_)
    : offset(offset_), size(size_), first_row(first_row_), num_rows(num_rows_)
  {
  }

  size_t offset;
  uint32_t size;
  uint32_t first_row;
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
inline constexpr bool is_supported_logical_type(logicaltype_kind_e logical_kind)
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
