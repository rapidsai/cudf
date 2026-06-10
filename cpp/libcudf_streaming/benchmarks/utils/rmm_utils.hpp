/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>

#include <string>

/**
 * @brief Create and set a RMM memory resource as the current device resource.
 *
 * @param name The name of the resource:
 *  - `cuda`: use the default CUDA memory resource.
 *  - `async`: use a CUDA async memory resource.
 *  - `pool`: use a memory pool backed by a CUDA memory resource.
 *  - `managed`: use a CUDA managed memory resource.
 */
inline void set_current_rmm_resource(std::string const& name)
{
  if (name == "cuda") {
    rmm::mr::set_current_device_resource(rmm::mr::cuda_memory_resource{});
  } else if (name == "async") {
    rmm::mr::set_current_device_resource(rmm::mr::cuda_async_memory_resource{});
  } else if (name == "managed") {
    rmm::mr::set_current_device_resource(rmm::mr::managed_memory_resource{});
  } else if (name == "pool") {
    rmm::mr::set_current_device_resource(
      rmm::mr::pool_memory_resource{rmm::mr::cuda_memory_resource{},
                                    rmm::percent_of_free_device_memory(80),
                                    rmm::percent_of_free_device_memory(80)});
  } else {
    RAPIDSMPF_FAIL("unknown RMM resource name: " + name);
  }
}

/**
 * @brief Create a statistics-enabled device memory resource wrapping the current
 * device resource, and set it as the current device resource.
 *
 * @return A RmmResourceAdaptor (shared ownership) for accessing statistics.
 */
[[nodiscard]] inline rapidsmpf::RmmResourceAdaptor set_device_mem_resource_with_stats()
{
  rapidsmpf::RmmResourceAdaptor adaptor{rmm::mr::get_current_device_resource_ref()};
  rmm::mr::set_current_device_resource(adaptor);
  return adaptor;
}
