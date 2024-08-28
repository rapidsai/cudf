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
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf {

/**
 * @brief Alias for a `cuda::mr::resource_ref` with the property
 * `cuda::mr::device_accessible`.
 */
using device_resource_ref = rmm::device_resource_ref;

/**
 * @brief Alias for a `cuda::mr::async_resource_ref` with the property
 * `cuda::mr::device_accessible`.
 */
using device_async_resource_ref = rmm::device_async_resource_ref;

/**
 * @brief Alias for a `cuda::mr::resource_ref` with the property
 * `cuda::mr::host_accessible`.
 */
using host_resource_ref = rmm::host_resource_ref;

/**
 * @brief Alias for a `cuda::mr::async_resource_ref` with the property
 * `cuda::mr::host_accessible`.
 */
using host_async_resource_ref = rmm::host_async_resource_ref;

/**
 * @brief Alias for a `cuda::mr::resource_ref` with the properties
 * `cuda::mr::host_accessible` and `cuda::mr::device_accessible`.
 */
using host_device_resource_ref = rmm::host_device_resource_ref;

/**
 * @brief Alias for a `cuda::mr::async_resource_ref` with the properties
 * `cuda::mr::host_accessible` and `cuda::mr::device_accessible`.
 */
using host_device_async_resource_ref = rmm::host_device_async_resource_ref;

/**
 * @brief Alias for a `rmm::mr::device_memory_resource`.
 */
using device_memory_resource = rmm::mr::device_memory_resource;

/**
 * @brief Get the current device memory resource.
 *
 * @return The current device memory resource.
 */
inline device_memory_resource* get_current_device_resource()
{
  return rmm::mr::get_current_device_resource();
}

/**
 * @brief Get the current device memory resource reference.
 *
 * @return The current device memory resource reference.
 */
inline device_async_resource_ref get_current_device_resource_ref()
{
  // For now, match current behavior which is to return current resource pointer
  return rmm::mr::get_current_device_resource();
}

/**
 * @brief Set the current device memory resource.
 *
 * @param mr The new device memory resource.
 * @return The previous device memory resource.
 */
inline device_memory_resource* set_current_device_resource(device_memory_resource* mr)
{
  return rmm::mr::set_current_device_resource(mr);
}

/**
 * @brief Set the current device memory resource reference.
 *
 * @param mr The new device memory resource reference.
 * @return The previous device memory resource reference.
 */
/*inline device_async_resource_ref set_current_device_resource_ref(device_async_resource_ref mr)
{
  return rmm::mr::set_current_device_resource_ref(mr);
}*/

/**
 * @brief Reset the current device memory resource reference to the initial resource.
 *
 * @return The previous device memory resource reference.
 */
/*inline device_async_resource_ref reset_current_device_resource_ref()
{
  return rmm::mr::reset_current_device_resource_ref();
}*/

}  // namespace cudf
