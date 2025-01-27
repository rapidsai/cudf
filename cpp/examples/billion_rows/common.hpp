/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

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
