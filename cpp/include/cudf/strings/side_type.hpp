/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_modify
 * @{
 * @file
 */

/**
 * @brief Direction identifier for cudf::strings::strip and cudf::strings::pad functions.
 */
enum class side_type {
  LEFT,   ///< strip/pad characters from the beginning of the string
  RIGHT,  ///< strip/pad characters from the end of the string
  BOTH    ///< strip/pad characters from the beginning and end of the string
};

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
