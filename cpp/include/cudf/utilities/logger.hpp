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

#include <spdlog/spdlog.h>

namespace CUDF_EXPORT cudf {

namespace detail {
spdlog::logger& logger();
}

/**
 * @brief Returns the global logger.
 *
 * This is a global instance of a spdlog logger. It can be used to configure logging behavior in
 * libcudf.
 *
 * Examples:
 * @code{.cpp}
 * // Turn off logging at runtime
 * cudf::logger().set_level(spdlog::level::off);
 * // Add a stdout sink to the logger
 * cudf::logger().sinks().push_back(std::make_shared<spdlog::sinks::stdout_sink_mt>());
 * // Replace the default sink
 * cudf::logger().sinks() ={std::make_shared<spdlog::sinks::stderr_sink_mt>()};
 * @endcode
 *
 * Note: Changes to the sinks are not thread safe and should only be done during global
 * initialization.
 *
 * @return spdlog::logger& The logger.
 */
[[deprecated(
  "Support for direct access to spdlog loggers in cudf is planned for removal")]] spdlog::logger&
logger();

}  // namespace CUDF_EXPORT cudf
