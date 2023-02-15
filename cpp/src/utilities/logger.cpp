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

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/logger.hpp>

#include <spdlog/sinks/basic_file_sink.h>

#include <string>

namespace {

/**
 * @brief Returns the default log filename for the global logger.
 *
 * If the environment variable `CUDF_DEBUG_LOG_FILE` is defined, its value is used as the path and
 * name of the log file. Otherwise, the file `cudf_log.txt` in the current working directory is
 * used.
 *
 * @return std::string The default log file name.
 */
std::string default_log_filename()
{
  auto* filename = std::getenv("LIBCUDF_DEBUG_LOG_FILE");
  return (filename == nullptr) ? std::string{"cudf_log.txt"} : std::string{filename};
}

/**
 * @brief Simple wrapper around a spdlog::logger that performs cuDF-specific initialization.
 */
struct logger_wrapper {
  spdlog::logger logger_;

  /**
   * @brief Converts the level name into the `spdlog` level enum.
   */
  spdlog::level::level_enum spdlog_level_from_string(std::string_view str)
  {
    if (str == "TRACE") return spdlog::level::trace;
    if (str == "DEBUG") return spdlog::level::debug;
    if (str == "INFO") return spdlog::level::info;
    if (str == "WARN") return spdlog::level::warn;
    if (str == "ERROR") return spdlog::level::err;
    if (str == "CRITICAL") return spdlog::level::critical;
    if (str == "OFF") return spdlog::level::off;

    CUDF_FAIL("Invalid value for LIBCUDF_LOGGING_LEVEL environment variable");
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

}  // namespace

spdlog::logger& cudf::logger()
{
  static logger_wrapper wrapped{};
  return wrapped.logger_;
}
