/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cstdint>

namespace cudf {
namespace strings {

/**
 * @addtogroup strings_contains
 * @{
 */

/**
 * @brief Regex flags.
 *
 * These types can be or'd to combine them.
 */
enum regex_flags : uint32_t {
  DEFAULT     = 0,  /// default
  SINGLE_LINE = 1,  /// the '^' and '$' ignore new-line characters
  DOT_ALL     = 2   /// the '.' matching includes new-line characters
};

#define IS_SINGLE_LINE(f) ((f & regex_flags::SINGLE_LINE) == regex_flags::SINGLE_LINE)
#define IS_DOT_ALL(f)     ((f & regex_flags::DOT_ALL) == regex_flags::DOT_ALL)

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
