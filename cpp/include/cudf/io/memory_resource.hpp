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

#include <rmm/resource_ref.hpp>

#include <optional>

namespace cudf::io {

/**
 * @brief Set the rmm resource to be used for host memory allocations by
 * cudf::detail::hostdevice_vector
 *
 * hostdevice_vector is a utility class that uses a pair of host and device-side buffers for
 * bouncing state between the cpu and the gpu. The resource set with this function (typically a
 * pinned memory allocator) is what it uses to allocate space for it's host-side buffer.
 *
 * @param mr The rmm resource to be used for host-side allocations
 * @return The previous resource that was in use
 */
rmm::host_async_resource_ref set_host_memory_resource(rmm::host_async_resource_ref mr);

/**
 * @brief Get the rmm resource being used for host memory allocations by
 * cudf::detail::hostdevice_vector
 *
 * @return The rmm resource used for host-side allocations
 */
rmm::host_async_resource_ref get_host_memory_resource();

/**
 * @brief Options to configure the default host memory resource
 */
struct host_mr_options {
  std::optional<size_t> pool_size;  ///< The size of the pool to use for the default host memory
                                    ///< resource. If not set, the default pool size is used.
};

/**
 * @brief Configure the size of the default host memory resource.
 *
 * @throws cudf::logic_error if called after the default host memory resource has been created
 *
 * @param opts Options to configure the default host memory resource
 */
void config_default_host_memory_resource(host_mr_options const& opts);

}  // namespace cudf::io
