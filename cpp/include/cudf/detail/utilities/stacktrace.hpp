/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/utilities/export.hpp>

#include <string>

namespace CUDF_EXPORT cudf {
namespace detail {
/**
 * @addtogroup utility_stacktrace
 * @{
 * @file
 */

/**
 * @brief Specify whether the last stackframe is included in the stacktrace.
 */
enum class capture_last_stackframe : bool { YES, NO };

/**
 * @brief Query the current stacktrace and return the whole stacktrace as one string.
 *
 * Depending on the value of the flag `capture_last_frame`, the caller that executes stacktrace
 * retrieval can be included in the output result.
 *
 * @param capture_last_frame Flag to specify if the current stackframe will be included into
 *        the output
 * @return A string storing the whole current stacktrace
 */
std::string get_stacktrace(capture_last_stackframe capture_last_frame);

/** @} */  // end of group

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
