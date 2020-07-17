/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

#include <vector>

namespace cudf {
namespace detail {
/**
 * @copydoc cudf::segmented_count_set_bits
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::vector<size_type> segmented_count_set_bits(bitmask_type const* bitmask,
                                                std::vector<size_type> const& indices,
                                                cudaStream_t stream = 0);

/**
 * @copydoc cudf::segmented_count_unset_bits
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::vector<size_type> segmented_count_unset_bits(bitmask_type const* bitmask,
                                                  std::vector<size_type> const& indices,
                                                  cudaStream_t stream = 0);

/**
 * @brief Returns a bitwise AND of the specified bitmasks
 * 
 * @param masks The list of data pointers of the bitmasks to be ANDed
 * @param begin_bits The bit offsets from which each mask is to be ANDed
 * @param mask_size The number of bits to be ANDed in each mask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return rmm::device_buffer Output bitmask 
 */
rmm::device_buffer bitmask_and(std::vector<bitmask_type const *> const &masks,
                               std::vector<size_type> const &begin_bits,
                               size_type mask_size,
                               cudaStream_t stream,
                               rmm::mr::device_memory_resource *mr);
}  // namespace detail

}  // namespace cudf
