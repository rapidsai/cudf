/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cstdint>

#pragma once

namespace cudf::io::parquet::detail {

enum class testfile : int8_t {
  NONE  = 0,  // Do not prune any pages
  FILE1 = 1,  // list<str>, int, float, str (plain encoding)
  FILE2 = 2,  // list<str>, list<list<str>>, list<list<list<int64>>> (plain encoding)
  FILE3 = 3,  // structs (plain encoding)
  FILE4 = 4,  // FLBA flat (byte stream split encoding)
  FILE5 = 5,  // int, str, bool, float, list<bool>, list<list<str>>, list<str> (plain encoding)
  FILE6 =
    6,  // str (DELTA_BYTE_ARRAY), str (DELTA_LENGTH_BYTE_ARRAY), list<str> (DELTA_BYTE_ARRAY),
        // list<str> (DELTA_LENGTH_BYTE_ARRAY), int (DELTA_BINARY_PACKED), float (BYTE_STREAM_SPLIT)

  // kernel_masks to check still => GENERAL, BYTE_STREAM_SPLIT
  FILE7  = 7,   // unused
  FILE8  = 8,   // unused
  FILE9  = 9,   // unused
  FILE10 = 10,  // unused
};

static constexpr testfile _file = testfile::FILE6;

}  // namespace cudf::io::parquet::detail
