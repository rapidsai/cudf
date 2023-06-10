/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <vector>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::create_null_mask(size_type, mask_state, rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
rmm::device_buffer create_null_mask(size_type size,
                                    mask_state state,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr);

/**
 * @copydoc cudf::set_null_mask(bitmask_type*, size_type, size_type, bool)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
void set_null_mask(bitmask_type* bitmask,
                   size_type begin_bit,
                   size_type end_bit,
                   bool valid,
                   rmm::cuda_stream_view stream);

/**
 * @brief Given a bitmask, counts the number of set (1) bits in the range
 * `[start, stop)`.
 *
 * @throws cudf::logic_error if `bitmask == nullptr`
 * @throws cudf::logic_error if `start > stop`
 * @throws cudf::logic_error if `start < 0`
 *
 * @param bitmask Bitmask residing in device memory whose bits will be counted.
 * @param start Index of the first bit to count (inclusive).
 * @param stop Index of the last bit to count (exclusive).
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return The number of non-zero bits in the specified range.
 */
cudf::size_type count_set_bits(bitmask_type const* bitmask,
                               size_type start,
                               size_type stop,
                               rmm::cuda_stream_view stream);

/**
 * @brief Given a bitmask, counts the number of unset (0) bits in the range
 * `[start, stop)`.
 *
 * @throws cudf::logic_error if `bitmask == nullptr`
 * @throws cudf::logic_error if `start > stop`
 * @throws cudf::logic_error if `start < 0`
 *
 * @param bitmask Bitmask residing in device memory whose bits will be counted.
 * @param start Index of the first bit to count (inclusive).
 * @param stop Index of the last bit to count (exclusive).
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return The number of zero bits in the specified range.
 */
cudf::size_type count_unset_bits(bitmask_type const* bitmask,
                                 size_type start,
                                 size_type stop,
                                 rmm::cuda_stream_view stream);

/**
 * @brief Given a bitmask, counts the number of set (1) bits in every range
 * `[indices[2*i], indices[(2*i)+1])` (where 0 <= i < indices.size() / 2).
 *
 * @throws cudf::logic_error if `bitmask == nullptr`
 * @throws cudf::logic_error if `indices.size() % 2 != 0`
 * @throws cudf::logic_error if `indices[2*i] < 0 or indices[2*i] > indices[(2*i)+1]`
 *
 * @param[in] bitmask Bitmask residing in device memory whose bits will be counted.
 * @param[in] indices A host_span of indices specifying ranges to count the number of set bits.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @return A vector storing the number of non-zero bits in the specified ranges.
 */
std::vector<size_type> segmented_count_set_bits(bitmask_type const* bitmask,
                                                host_span<size_type const> indices,
                                                rmm::cuda_stream_view stream);

/**
 * @brief Given a bitmask, counts the number of unset (0) bits in every range
 * `[indices[2*i], indices[(2*i)+1])` (where 0 <= i < indices.size() / 2).
 *
 * @throws cudf::logic_error if `bitmask == nullptr`
 * @throws cudf::logic_error if `indices.size() % 2 != 0`
 * @throws cudf::logic_error if `indices[2*i] < 0 or indices[2*i] > indices[(2*i)+1]`
 *
 * @param[in] bitmask Bitmask residing in device memory whose bits will be counted.
 * @param[in] indices A host_span of indices specifying ranges to count the number of unset bits.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @return A vector storing the number of zero bits in the specified ranges.
 */
std::vector<size_type> segmented_count_unset_bits(bitmask_type const* bitmask,
                                                  host_span<size_type const> indices,
                                                  rmm::cuda_stream_view stream);

/**
 * @brief Given a validity bitmask, counts the number of valid elements (set bits)
 * in the range `[start, stop)`.
 *
 * If `bitmask == nullptr`, all elements are assumed to be valid and the
 * function returns `stop-start`.
 *
 * @throws cudf::logic_error if `start > stop`
 * @throws cudf::logic_error if `start < 0`
 *
 * @param[in] bitmask Validity bitmask residing in device memory.
 * @param[in] start Index of the first bit to count (inclusive).
 * @param[in] stop Index of the last bit to count (exclusive).
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @return The number of valid elements in the specified range.
 */
cudf::size_type valid_count(bitmask_type const* bitmask,
                            size_type start,
                            size_type stop,
                            rmm::cuda_stream_view stream);

/**
 * @copydoc null_count(bitmask_type const* bitmask, size_type start, size_type stop)
 *
 * @param stream Stream view on which to allocate resources and queue execution.
 */
cudf::size_type null_count(bitmask_type const* bitmask,
                           size_type start,
                           size_type stop,
                           rmm::cuda_stream_view stream);

/**
 * @brief Given a validity bitmask, counts the number of valid elements (set
 * bits) in every range `[indices[2*i], indices[(2*i)+1])` (where 0 <= i <
 * indices.size() / 2).
 *
 * If `bitmask == nullptr`, all elements are assumed to be valid and a vector of
 * length `indices.size()` containing segment lengths is returned.
 *
 * @throws cudf::logic_error if `indices.size() % 2 != 0`.
 * @throws cudf::logic_error if `indices[2*i] < 0 or indices[2*i] > indices[(2*i)+1]`.
 *
 * @param[in] bitmask Validity bitmask residing in device memory.
 * @param[in] indices A host_span of indices specifying ranges to count the number of valid
 * elements.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @return A vector storing the number of valid elements in each specified range.
 */
std::vector<size_type> segmented_valid_count(bitmask_type const* bitmask,
                                             host_span<size_type const> indices,
                                             rmm::cuda_stream_view stream);

/**
 * @brief Given a validity bitmask, counts the number of null elements (unset
 * bits) in every range `[indices[2*i], indices[(2*i)+1])` (where 0 <= i <
 * indices.size() / 2).
 *
 * If `bitmask == nullptr`, all elements are assumed to be valid and a vector of
 * length `indices.size()` containing all zeros is returned.
 *
 * @throws cudf::logic_error if `indices.size() % 2 != 0`
 * @throws cudf::logic_error if `indices[2*i] < 0 or indices[2*i] > indices[(2*i)+1]`
 *
 * @param[in] bitmask Validity bitmask residing in device memory.
 * @param[in] indices A host_span of indices specifying ranges to count the number of null elements.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @return A vector storing the number of null elements in each specified range.
 */
std::vector<size_type> segmented_null_count(bitmask_type const* bitmask,
                                            host_span<size_type const> indices,
                                            rmm::cuda_stream_view stream);

/**
 * @copydoc cudf::copy_bitmask(bitmask_type const*, size_type, size_type,
 *rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
rmm::device_buffer copy_bitmask(bitmask_type const* mask,
                                size_type begin_bit,
                                size_type end_bit,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr);

/**
 * @copydoc cudf::copy_bitmask(column_view const& view, rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
rmm::device_buffer copy_bitmask(column_view const& view,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr);

/**
 * @copydoc bitmask_and(host_span<bitmask_type const* const>, host_span<size_type> const,
 * size_type, rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::pair<rmm::device_buffer, size_type> bitmask_and(host_span<bitmask_type const* const> masks,
                                                     host_span<size_type const> masks_begin_bits,
                                                     size_type mask_size_bits,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource* mr);

/**
 * @copydoc cudf::bitmask_and
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<rmm::device_buffer, size_type> bitmask_and(table_view const& view,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource* mr);

/**
 * @copydoc cudf::bitmask_or
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<rmm::device_buffer, size_type> bitmask_or(table_view const& view,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr);

/**
 * @brief Performs a bitwise AND of the specified bitmasks,
 *        and writes in place to destination
 *
 * @param dest_mask Destination to which the AND result is written
 * @param masks The list of data pointers of the bitmasks to be ANDed
 * @param masks_begin_bits The bit offsets from which each mask is to be ANDed
 * @param mask_size_bits The number of bits to be ANDed in each mask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Count of set bits
 */
cudf::size_type inplace_bitmask_and(device_span<bitmask_type> dest_mask,
                                    host_span<bitmask_type const* const> masks,
                                    host_span<size_type const> masks_begin_bits,
                                    size_type mask_size_bits,
                                    rmm::cuda_stream_view stream);

}  // namespace detail

}  // namespace cudf
