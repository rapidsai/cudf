/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdlib>
#include <stdexcept>

namespace cudf {

/**
 * @addtogroup memory_resource
 * @{
 * @file
 */

// Forward declare get_current_device_resource_ref() for use in memory_resources constructor
inline rmm::device_async_resource_ref get_current_device_resource_ref();

/**
 * @brief Container for output and temporary memory resources
 *
 * This class enables separate control over output allocations (data returned to caller)
 * and temporary allocations (intermediate computations not returned).
 *
 * When a libcudf function accepts a `memory_resources` parameter, the output memory resource
 * is used for allocations that will be returned to the caller, while the temporary memory
 * resource is used for internal intermediate allocations that are freed before the function
 * returns.
 *
 * This provides fine-grained control over memory allocation, allowing users to:
 * - Use different memory pools for output vs temporary allocations
 * - Track memory usage separately for outputs and temporaries
 * - Optimize memory management based on allocation lifetime
 */
class memory_resources {
 public:
  /**
   * @brief Construct with explicit output and temporary memory resources
   *
   * @param output_mr Memory resource for output allocations (data returned to caller)
   * @param temporary_mr Memory resource for temporary allocations (internal intermediates)
   */
  memory_resources(rmm::device_async_resource_ref output_mr,
                   rmm::device_async_resource_ref temporary_mr)
    : output_mr{output_mr}, temporary_mr{temporary_mr}
  {
  }

  /**
   * @brief Construct with single memory resource (for API compatibility)
   *
   * Output allocations use the provided memory resource.
   * Temporary allocations use the current device resource.
   *
   * This constructor provides implicit conversion from `rmm::device_async_resource_ref` for
   * backward API compatibility.
   *
   * @param output_mr Memory resource for output allocations
   */
  memory_resources(rmm::device_async_resource_ref output_mr)
    : output_mr{output_mr}, temporary_mr{get_current_device_resource_ref()}
  {
  }

  /**
   * @brief Get the memory resource for temporary allocations
   *
   * @return Memory resource reference for temporary allocations
   */
  [[nodiscard]] rmm::device_async_resource_ref get_temporary_mr() const noexcept
  {
    return temporary_mr;
  }

  /**
   * @brief Get the memory resource for output allocations
   *
   * @return Memory resource reference for output allocations
   */
  [[nodiscard]] rmm::device_async_resource_ref get_output_mr() const noexcept
  {
    return output_mr;
  }

 private:
  rmm::device_async_resource_ref output_mr;
  rmm::device_async_resource_ref temporary_mr;
};

/**
 * @brief Get the current device memory resource reference.
 *
 * When the environment variable LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF is set,
 * this function throws an error. This is used during validation to ensure all code paths
 * properly thread memory resources through function calls instead of relying on the
 * global default.
 *
 * @throw std::runtime_error if LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF is set
 * @return The current device memory resource reference.
 */
inline rmm::device_async_resource_ref get_current_device_resource_ref()
{
  static bool const validation_enabled =
    std::getenv("LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF") != nullptr;

  if (validation_enabled) {
    throw std::runtime_error(
      "cudf::get_current_device_resource_ref() called during validation mode. "
      "All code paths should use resources.get_temporary_mr() instead of calling this function. "
      "Set LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF=0 to disable this check.");
  }

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
  cudf::memory_resources resources)
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
