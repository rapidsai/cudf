/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "memory_stats.hpp"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

namespace cudf {

class memory_stats_logger::impl {
 public:
  impl() : statistics_mr(cudf::get_current_device_resource_ref())
  {
    cudf::set_current_device_resource(statistics_mr);
  }

  ~impl() { cudf::set_current_device_resource(statistics_mr.get_upstream_resource()); }

  impl(impl const&)            = delete;
  impl& operator=(impl const&) = delete;
  impl(impl&&)                 = delete;
  impl& operator=(impl&&)      = delete;

  [[nodiscard]] std::size_t peak_memory_usage() const noexcept
  {
    return statistics_mr.get_bytes_counter().peak;
  }

 private:
  rmm::mr::statistics_resource_adaptor statistics_mr;
};

memory_stats_logger::memory_stats_logger() : _impl(std::make_unique<impl>()) {}

memory_stats_logger::~memory_stats_logger() = default;

std::size_t memory_stats_logger::peak_memory_usage() const noexcept
{
  return _impl->peak_memory_usage();
}

}  // namespace cudf
