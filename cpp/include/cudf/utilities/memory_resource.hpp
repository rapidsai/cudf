/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

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
  // For now, match current behavior which is to return current resource pointer
  return rmm::mr::get_current_device_resource();
}

/**
 * @brief Set the current device memory resource.
 *
 * @param mr The new device memory resource.
 * @return The previous device memory resource.
 */
inline rmm::mr::device_memory_resource* set_current_device_resource(
  rmm::mr::device_memory_resource* mr)
{
  return rmm::mr::set_current_device_resource(mr);
}

/**
 * @brief Set the current device memory resource reference.
 *
 * @param mr The new device memory resource reference.
 * @return The previous device memory resource reference.
 */
inline rmm::device_async_resource_ref set_current_device_resource_ref(
  rmm::device_async_resource_ref mr)
{
  return rmm::mr::set_current_device_resource_ref(mr);
}

/**
 * @brief Reset the current device memory resource reference to the initial resource.
 *
 * @return The previous device memory resource reference.
 */
inline rmm::device_async_resource_ref reset_current_device_resource_ref()
{
  return rmm::mr::reset_current_device_resource_ref();
}

/** @} */  // end of group
}  // namespace cudf
