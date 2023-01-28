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
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <string>

namespace cudf {

namespace detail {

/**
 * @brief Returns the default log filename for the global logger.
 *
 * If the environment variable `CUDF_DEBUG_LOG_FILE` is defined, its value is used as the path and
 * name of the log file. Otherwise, the file `cudf_log.txt` in the current working directory is
 * used.
 *
 * @return std::string The default log file name.
 */
inline std::string default_log_filename()
{
  auto* filename = std::getenv("LIBCUDF_DEBUG_LOG_FILE");
  return (filename == nullptr) ? std::string{"cudf_log.txt"} : std::string{filename};
}

/**
 * @brief Simple wrapper around a spdlog::logger that performs cuDF-specific initialization.
 */
struct logger_wrapper {
  spdlog::logger logger_;  ///< wrapped `spdlog` logger

  /**
   * @brief Converts the level name into the `spdlog` level enum.
   *
   * @param str logging level name
   * @return corresponding`spdlog` level enum; info if input string is invalid
   */
  spdlog::level::level_enum spdlog_level_from_string(std::string_view str)
  {
    if (str == "TRACE") return spdlog::level::trace;
    if (str == "DEBUG") return spdlog::level::debug;
    if (str == "INFO") return spdlog::level::info;
    if (str == "WARN") return spdlog::level::warn;
    if (str == "ERR") return spdlog::level::err;
    if (str == "CRITICAL") return spdlog::level::critical;
    if (str == "OFF") return spdlog::level::off;

    // keep the default if the env var is invalid
    return spdlog::level::info;
  }

  logger_wrapper()
    : logger_{"CUDF",
              std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                default_log_filename(), true  // truncate file
                )}
  {
    logger_.set_pattern("[%6t][%H:%M:%S:%f][%-6l] %v");
    logger_.flush_on(spdlog::level::warn);

    auto const env_level = std::getenv("LIBCUDF_LOGGING_LEVEL");
    if (env_level != nullptr) { logger_.set_level(spdlog_level_from_string(env_level)); }
  }
};

}  // namespace detail

/**
 * @brief Returns the global logger
 *
 * This is a spdlog logger. The easiest way to log messages is to use the `CUDF_LOG_*` macros.
 *
 * @return spdlog::logger& The logger.
 */
inline spdlog::logger& logger()
{
  static detail::logger_wrapper wrapped{};
  return wrapped.logger_;
}

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
