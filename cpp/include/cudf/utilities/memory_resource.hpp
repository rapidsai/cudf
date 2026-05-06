/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

namespace cudf {

/**
 * @addtogroup memory_resource
 * @{
 * @file
 */

/**
 * @brief Get the current device memory resource reference.
 *
 * @return The current device memory resource reference.
 */
inline rmm::device_async_resource_ref get_current_device_resource_ref()
{
  return rmm::mr::get_current_device_resource_ref();
}

/**
 * @brief Set the current device memory resource.
 *
 * @param mr The new device memory resource.
 * @return An owning any_resource holding the previous resource.
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> set_current_device_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> mr)
{
  return rmm::mr::set_current_device_resource(std::move(mr));
}

/**
 * @brief Set the current device memory resource reference.
 *
 * @deprecated Use `set_current_device_resource` instead.
 *
 * The object referenced by `mr` must outlive the last use of the resource, otherwise behavior is
 * undefined. It is the caller's responsibility to maintain the lifetime of the resource object.
 *
 * @param mr The new device memory resource reference.
 * @return An owning any_resource holding the previous resource.
 */
[[deprecated("Use set_current_device_resource instead.")]]  //
inline cuda::mr::any_resource<cuda::mr::device_accessible>
set_current_device_resource_ref(rmm::device_async_resource_ref mr)
{
  return set_current_device_resource(cuda::mr::any_resource<cuda::mr::device_accessible>{mr});
}

/**
 * @brief Reset the current device memory resource to the initial resource.
 *
 * @return An owning any_resource holding the previous resource.
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> reset_current_device_resource()
{
  return rmm::mr::reset_current_device_resource();
}

/**
 * @brief Reset the current device memory resource reference to the initial resource.
 *
 * @deprecated Use `reset_current_device_resource` instead.
 *
 * @return An owning any_resource holding the previous resource.
 */
[[deprecated("Use reset_current_device_resource instead.")]]  //
inline cuda::mr::any_resource<cuda::mr::device_accessible>
reset_current_device_resource_ref()
{
  return reset_current_device_resource();
}

/** @} */  // end of group
}  // namespace cudf
