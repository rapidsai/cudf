/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <string>
#include <variant>

using memory_resource_type =
  std::variant<rmm::mr::cuda_memory_resource, rmm::mr::pool_memory_resource>;

/**
 * @brief Create memory resource for libcudf functions
 */
memory_resource_type create_memory_resource(std::string const& name)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  if (name == "pool") {
    return rmm::mr::pool_memory_resource{cuda_mr, rmm::percent_of_free_device_memory(50)};
  }
  return cuda_mr;
}
