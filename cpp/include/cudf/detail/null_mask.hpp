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
 * @brief Creates a `device_buffer` for use as a null value indicator bitmask of
 * a `column`.
 *
 * @param size The number of elements to be represented by the mask
 * @param state The desired state of the mask
 * @param stream Optional, stream on which all memory allocations/operations
 * will be submitted
 * @param mr Device memory resource to use for device memory allocation
 * @return rmm::device_buffer A `device_buffer` for use as a null bitmask
 * satisfying the desired size and state
 **/
rmm::device_buffer create_null_mask(
  size_type size,
  mask_state state,
  stream_t stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @copydoc cudf::set_null_mask
 *
 * @param stream Optional, stream on which all memory allocations/operations
 * will be submitted
 **/
void set_null_mask(
  bitmask_type* bitmask, size_type begin_bit, size_type end_bit, bool valid, stream_t stream);

/**
 * @copydoc cudf::segmented_count_set_bits
 *
 * @param[in] stream Optional CUDA stream on which to execute kernels
 */
std::vector<size_type> segmented_count_set_bits(bitmask_type const* bitmask,
                                                std::vector<size_type> const& indices,
                                                stream_t stream = stream_t{});

/**
 * @copydoc cudf::segmented_count_unset_bits
 *
 * @param[in] stream Optional CUDA stream on which to execute kernels
 */
std::vector<size_type> segmented_count_unset_bits(bitmask_type const* bitmask,
                                                  std::vector<size_type> const& indices,
                                                  stream_t stream = stream_t{});

/**
 * @copydoc cudf::copy_bitmask(bitmask_type
 *const*,size_type,size_type,rmm::mr::device_memory_resource*)
 *
 * @param stream Optional, stream on which all memory allocations and copies
 * will be performed
 **/
rmm::device_buffer copy_bitmask(
  bitmask_type const* mask,
  size_type begin_bit,
  size_type end_bit,
  stream_t stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief cudf::copy_bitmask(column_view const&,rmm::mr::device_memory_resource*)
 *
 * @param stream Optional, stream on which all memory allocations and copies
 * will be performed
 **/
rmm::device_buffer copy_bitmask(
  column_view const& view,
  stream_t stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @copydoc cudf::bitmask_and(table_view const&,rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream on which to execute kernels
 */
rmm::device_buffer bitmask_and(
  table_view const& view,
  stream_t stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace detail

}  // namespace cudf
