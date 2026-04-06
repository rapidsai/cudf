/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
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
  return rmm::mr::get_current_device_resource();
}

/**
 * @brief Get the current device memory resource reference, bypassing default-argument detection.
 *
 * Calls the underlying RMM function directly, bypassing the `__attribute__((error))`
 * redeclaration that is active when `CUDF_CATCH_DEFAULT_MR` is defined.
 * Use this at call sites that intentionally want the current default device memory resource.
 *
 * @return The current device memory resource reference.
 */
inline rmm::device_async_resource_ref get_current_device_resource_ref_unsafe()
{
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

#ifdef CUDF_CATCH_DEFAULT_MR
#if defined(__GNUC__) && !defined(__clang__)
/**
 * Redeclare get_current_device_resource_ref() with __attribute__((error)) so that
 * any call emitted by the compiler — including implicit calls from default argument
 * evaluation — produces a hard compile error. The noinline attribute prevents the
 * inline body above from being substituted, ensuring the call survives to trigger
 * the error diagnostic.
 *
 * For intentional use of the current device resource, call
 * get_current_device_resource_ref_unsafe() instead.
 */
__attribute__((error("cudf default memory resource argument used. Pass mr explicitly.")))
rmm::device_async_resource_ref
get_current_device_resource_ref();
#endif  // __GNUC__ && !__clang__
#endif  // CUDF_CATCH_DEFAULT_MR

}  // namespace cudf
