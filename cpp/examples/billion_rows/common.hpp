/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/owning_wrapper.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <string>

/**
 * @brief Create CUDA memory resource
 */
auto make_cuda_mr() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

/**
 * @brief Create a pool device memory resource
 */
auto make_pool_mr()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
    make_cuda_mr(), rmm::percent_of_free_device_memory(50));
}

/**
 * @brief Create memory resource for libcudf functions
 */
std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(std::string const& name)
{
  if (name == "pool") { return make_pool_mr(); }
  return make_cuda_mr();
}
