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
  NONE   = 0,
  FILE1  = 1,  // list<str>, int, float, str (plain encoding)
  FILE2  = 2,  // list<str>, list<list<str>>, list<list<list<int64>>> (plain encoding)
  FILE3  = 3,  // structs (plain encoding)
  FILE4  = 4,  // FLBA flat (byte stream split encoding)
  FILE5  = 5,
  FILE6  = 6,
  FILE7  = 7,
  FILE8  = 8,
  FILE9  = 9,
  FILE10 = 10,
  FILE11 = 11,
  FILE12 = 12,
  FILE13 = 13,
  FILE14 = 14,
};

static constexpr testfile _file = testfile::NONE;

}  // namespace cudf::io::parquet::detail
