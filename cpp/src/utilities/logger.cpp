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

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/logger.hpp>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>

#include <string>

namespace {

/**
 * @brief Creates a sink for libcudf logging.
 *
 * Returns a file sink if the file name has been specified, otherwise returns a stderr sink.
 */
[[nodiscard]] spdlog::sink_ptr make_libcudf_sink()
{
  if (auto filename = std::getenv("LIBCUDF_DEBUG_LOG_FILE"); filename != nullptr) {
    return std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, true);
  } else {
    return std::make_shared<spdlog::sinks::stderr_sink_mt>();
  }
}

/**
 * @brief Converts the level name into the `spdlog` level enum.
 */
[[nodiscard]] spdlog::level::level_enum libcudf_log_level()
{
  auto const env_level = std::getenv("LIBCUDF_LOGGING_LEVEL");
  if (env_level == nullptr) { return spdlog::level::warn; }

  auto const env_lvl_str = std::string(env_level);
  if (env_lvl_str == "TRACE") return spdlog::level::trace;
  if (env_lvl_str == "DEBUG") return spdlog::level::debug;
  if (env_lvl_str == "INFO") return spdlog::level::info;
  if (env_lvl_str == "WARN") return spdlog::level::warn;
  if (env_lvl_str == "ERROR") return spdlog::level::err;
  if (env_lvl_str == "CRITICAL") return spdlog::level::critical;
  if (env_lvl_str == "OFF") return spdlog::level::off;

  CUDF_FAIL("Invalid value for LIBCUDF_LOGGING_LEVEL environment variable");
}

/**
 * @brief Simple wrapper around a spdlog::logger that performs cuDF-specific initialization.
 */
struct logger_wrapper {
  spdlog::logger logger_;

  logger_wrapper() : logger_{"CUDF", make_libcudf_sink()}
  {
    logger_.set_pattern("[%6t][%H:%M:%S:%f][%-6l] %v");
    logger_.set_level(libcudf_log_level());
    logger_.flush_on(spdlog::level::warn);
  }
};

}  // namespace

spdlog::logger& cudf::detail::logger()
{
  static logger_wrapper wrapped{};
  return wrapped.logger_;
}

spdlog::logger& cudf::logger() { return cudf::detail::logger(); }
