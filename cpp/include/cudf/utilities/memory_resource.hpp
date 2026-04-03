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
 * @brief Set the current device memory resource reference.
 *
 * @param mr The new device memory resource reference.
 * @return An owning any_resource holding the previous resource.
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> set_current_device_resource_ref(
  rmm::device_async_resource_ref mr)
{
  return rmm::mr::set_current_device_resource_ref(mr);
}

/**
 * @brief Reset the current device memory resource reference to the initial resource.
 *
 * @return An owning any_resource holding the previous resource.
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> reset_current_device_resource_ref()
{
  return rmm::mr::reset_current_device_resource_ref();
}

/** @} */  // end of group
}  // namespace cudf
