/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
rapids_logger::sink_ptr default_logger_sink();

/**
 * @brief Returns the default log pattern for the global logger.
 *
 * @return std::string The default log pattern.
 */
std::string default_logger_pattern();

/**
 * @brief Get the default logger.
 *
 * @return logger& The default logger
 */
rapids_logger::logger& default_logger();

}  // namespace CUDF_EXPORT cudf
