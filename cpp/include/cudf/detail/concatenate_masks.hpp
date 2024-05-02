/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf {
//! Inner interfaces and implementations
namespace detail {

/**
 * @brief Concatenates the null mask bits of all the column device views in the
 * `views` array to the destination bitmask.
 *
 * @param d_views Column device views whose null masks will be concatenated
 * @param d_offsets Prefix sum of sizes of elements of `d_views`
 * @param dest_mask The output buffer to copy null masks into
 * @param output_size The total number of null masks bits that are being copied
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return The number of nulls
 */
size_type concatenate_masks(device_span<column_device_view const> d_views,
                            device_span<size_t const> d_offsets,
                            bitmask_type* dest_mask,
                            size_type output_size,
                            rmm::cuda_stream_view stream);

/**
 * @brief Concatenates `views[i]`'s bitmask from the bits
 * `[views[i].offset(), views[i].offset() + views[i].size())` for all elements
 * views[i] in views into a destination bitmask pointer.
 *
 * @param views Column views whose bitmasks will be concatenated
 * @param dest_mask The output buffer to copy null masks into
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return The number of nulls
 */
size_type concatenate_masks(host_span<column_view const> views,
                            bitmask_type* dest_mask,
                            rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::concatenate_masks(host_span<column_view const>, rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
rmm::device_buffer concatenate_masks(host_span<column_view const> views,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
