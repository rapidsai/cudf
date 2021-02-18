/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>

#include <vector>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::create_null_mask(size_type, mask_state, rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
rmm::device_buffer create_null_mask(
  size_type size,
  mask_state state,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::set_null_mask(bitmask_type*, size_type, size_type, bool)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
void set_null_mask(bitmask_type *bitmask,
                   size_type begin_bit,
                   size_type end_bit,
                   bool valid,
                   rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @copydoc cudf::segmented_count_set_bits
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::vector<size_type> segmented_count_set_bits(bitmask_type const *bitmask,
                                                std::vector<size_type> const &indices,
                                                rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::segmented_count_unset_bits
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::vector<size_type> segmented_count_unset_bits(bitmask_type const *bitmask,
                                                  std::vector<size_type> const &indices,
                                                  rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::copy_bitmask(bitmask_type const*, size_type, size_type,
 *rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
rmm::device_buffer copy_bitmask(
  bitmask_type const *mask,
  size_type begin_bit,
  size_type end_bit,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::copy_bitmask(column_view const& view, rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
rmm::device_buffer copy_bitmask(
  column_view const &view,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc bitmask_binop(Binop op, std::vector<bitmask_type const*>, std::vector<size_type> const&,
 * size_type, rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <typename Binop>
rmm::device_buffer bitmask_binop(
  Binop op,
  std::vector<bitmask_type const *> const &masks,
  std::vector<size_type> const &begin_bits,
  size_type mask_size,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc bitmask_and(std::vector<bitmask_type const*>, std::vector<size_type> const&, size_type,
 * rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
rmm::device_buffer bitmask_and(
  std::vector<bitmask_type const *> const &masks,
  std::vector<size_type> const &begin_bits,
  size_type mask_size,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::bitmask_binop
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename Binop>
rmm::device_buffer bitmask_binop(
  Binop op,
  table_view const &view,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::bitmask_and
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
rmm::device_buffer bitmask_and(
  table_view const &view,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());

/**
 * @brief Performs a merger of the specified bitmasks using the binary operator
 *        provided, and writes in place to destination
 *
 * @param op The binary operator used to combine the bitmasks
 * @param dest_mask Destination to which the AND result is written
 * @param masks The list of data pointers of the bitmasks to be ANDed
 * @param begin_bits The bit offsets from which each mask is to be ANDed
 * @param mask_size The number of bits to be ANDed in each mask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return rmm::device_buffer Output bitmask
 */
template <typename Binop>
void inplace_bitmask_binop(
  Binop op,
  bitmask_type *dest_mask,
  std::vector<bitmask_type const *> const &masks,
  std::vector<size_type> const &begin_bits,
  size_type mask_size,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());

/**
 * @brief Performs a bitwise AND of the specified bitmasks,
 *        and writes in place to destination
 *
 * @param dest_mask Destination to which the AND result is written
 * @param masks The list of data pointers of the bitmasks to be ANDed
 * @param begin_bits The bit offsets from which each mask is to be ANDed
 * @param mask_size The number of bits to be ANDed in each mask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 * @return rmm::device_buffer Output bitmask
 */
void inplace_bitmask_and(
  bitmask_type *dest_mask,
  std::vector<bitmask_type const *> const &masks,
  std::vector<size_type> const &begin_bits,
  size_type mask_size,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());

}  // namespace detail

}  // namespace cudf
