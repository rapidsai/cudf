/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <string>

namespace cudf::examples {

namespace detail {

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
 * @brief Create a async device memory resource
 */
auto make_async_mr() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

}  // namespace detail

/**
 * @brief Create memory resource for libcudf functions
 *
 * @param name Memory resource name
 * @return Memory resource instance
 */
std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(std::string const& name)
{
  if (name == "pool" || name == "pool-stats") {
    return detail::make_pool_mr();
  } else if (name == "async" || name == "async-stats") {
    return detail::make_async_mr();
  } else if (name == "cuda" || name == "cuda-stats") {
    return detail::make_cuda_mr();
  }
  throw std::invalid_argument("Unrecognized memory resource name: " + name);
}

/**
 * @brief Create memory resource for libcudf functions
 *
 * @param is_pool_used Whether to use a pool memory resource
 * @return Memory resource instance
 */
std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(bool is_pool_used)
{
  if (is_pool_used) { return detail::make_pool_mr(); }
  return detail::make_cuda_mr();
}

/**
 * @brief Create and return a reference to a static pinned memory pool
 *
 * @return Reference to a static pinned memory pool
 */
rmm::host_async_resource_ref create_pinned_memory_resource()
{
  static auto mr = rmm::mr::pinned_host_memory_resource{};
  return mr;
}

}  // namespace cudf::examples
