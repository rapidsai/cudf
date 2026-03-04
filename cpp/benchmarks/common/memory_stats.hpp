/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

namespace cudf {

class memory_stats_logger {
 public:
  memory_stats_logger()
    : existing_mr(cudf::get_current_device_resource_ref()),
      statistics_mr(
        rmm::mr::statistics_resource_adaptor<rmm::device_async_resource_ref>(existing_mr))
  {
    cudf::set_current_device_resource_ref(&statistics_mr);
  }

  ~memory_stats_logger() { cudf::set_current_device_resource_ref(existing_mr); }

  [[nodiscard]] size_t peak_memory_usage() const noexcept
  {
    return statistics_mr.get_bytes_counter().peak;
  }

 private:
  rmm::device_async_resource_ref existing_mr;
  rmm::mr::statistics_resource_adaptor<rmm::device_async_resource_ref> statistics_mr;
};

}  // namespace cudf
