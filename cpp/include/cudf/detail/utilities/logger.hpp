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

#include <cudf/utilities/logger.hpp>

// Log messages that require computation should only be used at level TRACE and DEBUG
#define CUDF_LOG_TRACE(...)    SPDLOG_LOGGER_TRACE(&cudf::detail::logger(), __VA_ARGS__)
#define CUDF_LOG_DEBUG(...)    SPDLOG_LOGGER_DEBUG(&cudf::detail::logger(), __VA_ARGS__)
#define CUDF_LOG_INFO(...)     SPDLOG_LOGGER_INFO(&cudf::detail::logger(), __VA_ARGS__)
#define CUDF_LOG_WARN(...)     SPDLOG_LOGGER_WARN(&cudf::detail::logger(), __VA_ARGS__)
#define CUDF_LOG_ERROR(...)    SPDLOG_LOGGER_ERROR(&cudf::detail::logger(), __VA_ARGS__)
#define CUDF_LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(&cudf::detail::logger(), __VA_ARGS__)
