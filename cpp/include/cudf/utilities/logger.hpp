/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

// If using GCC, temporary workaround for older libcudacxx defining _LIBCPP_VERSION
// undefine it before including spdlog, due to fmtlib checking if it is defined
// TODO: remove once libcudacxx is on Github and RAPIDS depends on it
#ifdef __GNUG__
#undef _LIBCPP_VERSION
#endif
#include <spdlog/spdlog.h>

namespace cudf {

/**
 * @brief Returns the global logger
 *
 * This is a spdlog logger. The easiest way to log messages is to use the `CUDF_LOG_*` macros.
 *
 * @return spdlog::logger& The logger.
 */
spdlog::logger& logger();

// The default is INFO, but it should be used sparingly, so that by default a log file is only
// output if there is important information, warnings, errors, and critical failures
// Log messages that require computation should only be used at level TRACE and DEBUG
#define CUDF_LOG_TRACE(...)    SPDLOG_LOGGER_TRACE(&cudf::logger(), __VA_ARGS__)
#define CUDF_LOG_DEBUG(...)    SPDLOG_LOGGER_DEBUG(&cudf::logger(), __VA_ARGS__)
#define CUDF_LOG_INFO(...)     SPDLOG_LOGGER_INFO(&cudf::logger(), __VA_ARGS__)
#define CUDF_LOG_WARN(...)     SPDLOG_LOGGER_WARN(&cudf::logger(), __VA_ARGS__)
#define CUDF_LOG_ERROR(...)    SPDLOG_LOGGER_ERROR(&cudf::logger(), __VA_ARGS__)
#define CUDF_LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(&cudf::logger(), __VA_ARGS__)

}  // namespace cudf
