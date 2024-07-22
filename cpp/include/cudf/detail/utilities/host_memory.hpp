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

#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/resource_ref.hpp>

#include <cstddef>

namespace cudf::detail {
/**
 * @brief Get the memory resource to be used for pageable memory allocations.
 *
 * @return Reference to the pageable memory resource
 */
CUDF_EXPORT rmm::host_async_resource_ref get_pageable_memory_resource();

/**
 * @brief Get the allocator to be used for the host memory allocation.
 *
 * @param size The number of elements of type T to allocate
 * @param stream The stream to use for the allocation
 * @return The allocator to be used for the host memory allocation
 */
template <typename T>
rmm_host_allocator<T> get_host_allocator(std::size_t size, rmm::cuda_stream_view stream)
{
  return {size * sizeof(T) <= get_allocate_host_as_pinned_threshold()
            ? get_pinned_memory_resource()
            : get_pageable_memory_resource(),
          stream};
}

}  // namespace cudf::detail
