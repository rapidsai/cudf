/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <memory>

namespace cudf {

/**
 * @brief Tracks peak device memory usage for the lifetime of this object
 *
 * Wraps the current device resource in an `rmm::mr::statistics_resource_adaptor` on construction
 * and restores the upstream resource on destruction.
 *
 * The implementation is kept out of this header so it can be included from `.cu` translation
 * units without pulling in `<rmm/mr/statistics_resource_adaptor.hpp>`.
 */
class memory_stats_logger {
 public:
  memory_stats_logger();
  ~memory_stats_logger();

  memory_stats_logger(memory_stats_logger const&)            = delete;
  memory_stats_logger& operator=(memory_stats_logger const&) = delete;
  memory_stats_logger(memory_stats_logger&&)                 = delete;
  memory_stats_logger& operator=(memory_stats_logger&&)      = delete;

  [[nodiscard]] std::size_t peak_memory_usage() const noexcept;

 private:
  class impl;
  std::unique_ptr<impl> _impl;
};

}  // namespace cudf
