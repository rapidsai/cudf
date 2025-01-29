/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/logger_macros.hpp>
#include <cudf/utilities/export.hpp>

#include <rapids_logger/logger.hpp>

namespace CUDF_EXPORT cudf {

/**
 * @brief Returns the default sink for the global logger.
 *
 * If the environment variable `CUDF_DEBUG_LOG_FILE` is defined, the default sink is a sink to that
 * file. Otherwise, the default is to dump to stderr.
 *
 * @return sink_ptr The sink to use
 */
inline rapids_logger::sink_ptr default_sink()
{
  auto* filename = std::getenv("CUDF_DEBUG_LOG_FILE");
  return (filename == nullptr)
           ? static_cast<rapids_logger::sink_ptr>(std::make_shared<rapids_logger::stderr_sink_mt>())
           : static_cast<rapids_logger::sink_ptr>(
               std::make_shared<rapids_logger::basic_file_sink_mt>(filename, true));
}

/**
 * @brief Returns the default log pattern for the global logger.
 *
 * @return std::string The default log pattern.
 */
inline std::string default_pattern() { return "[%6t][%H:%M:%S:%f][%-6l] %v"; }

/**
 * @brief Get the default logger.
 *
 * @return logger& The default logger
 */
inline rapids_logger::logger& default_logger()
{
  static rapids_logger::logger logger_ = [] {
    rapids_logger::logger logger_{"CUDF", {default_sink()}};
    logger_.set_pattern(default_pattern());
    logger_.set_level(rapids_logger::level_enum::warn);
    return logger_;
  }();
  return logger_;
}

}  // namespace CUDF_EXPORT cudf
