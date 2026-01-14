/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/logger.hpp>
#include <cudf/utilities/export.hpp>

#include <rapids_logger/logger.hpp>

namespace CUDF_EXPORT cudf {

rapids_logger::sink_ptr default_logger_sink()
{
  auto* filename = std::getenv("CUDF_DEBUG_LOG_FILE");
  if (filename != nullptr) {
    return std::make_shared<rapids_logger::basic_file_sink_mt>(filename, true);
  }
  return std::make_shared<rapids_logger::stderr_sink_mt>();
}

std::string default_logger_pattern() { return "[%6t][%H:%M:%S:%f][%-6l] %v"; }

rapids_logger::logger& default_logger()
{
  static rapids_logger::logger logger_ = [] {
    rapids_logger::logger logger_{"CUDF", {default_logger_sink()}};
    logger_.set_pattern(default_logger_pattern());
    logger_.set_level(rapids_logger::level_enum::warn);
    return logger_;
  }();
  return logger_;
}

}  // namespace CUDF_EXPORT cudf
