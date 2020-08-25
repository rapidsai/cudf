/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <stdint.h>

namespace cudf {
namespace io {
namespace avro {
struct block_desc_s {
  block_desc_s() {}
  explicit constexpr block_desc_s(size_t offset_,
                                  uint32_t size_,
                                  uint32_t first_row_,
                                  uint32_t num_rows_)
    : offset(offset_), first_row(first_row_), num_rows(num_rows_), size(size_)
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
};

}  // namespace avro
}  // namespace io
}  // namespace cudf
