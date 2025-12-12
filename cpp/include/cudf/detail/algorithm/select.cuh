/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cub/device/device_select.cuh>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Helper to copy elements satisfying a predicate using CUB with pinned memory
 *
 * This function copies elements from the input range that satisfy the given predicate
 * to the output range, using CUB's DeviceSelect::If implementation with pinned memory
 * for efficient device-to-host transfer of the number of selected elements.
 *
 * @tparam Predicate **[inferred]** Type of the unary predicate
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 * @tparam OutputIterator **[inferred]** Type of device-accessible output iterator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param output Device-accessible iterator to start of output values
 * @param predicate Unary predicate that returns true for elements to copy
 * @param stream CUDA stream to use
 * @return Iterator pointing to the end of the output range
 */
template <typename Predicate, typename InputIterator, typename OutputIterator>
OutputIterator copy_if(InputIterator begin,
                       InputIterator end,
                       OutputIterator output,
                       Predicate predicate,
                       rmm::cuda_stream_view stream)
{
  auto const num_items = cuda::std::distance(begin, end);

  // Device memory to store the number of selected elements
  auto d_num_selected =
    rmm::device_uvector<cuda::std::size_t>(1, stream, cudf::get_current_device_resource_ref());

  // First call to get temporary storage size
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::If(nullptr,
                        temp_storage_bytes,
                        begin,
                        output,
                        d_num_selected.begin(),
                        num_items,
                        predicate,
                        stream.value());

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());

  // Run copy_if
  cub::DeviceSelect::If(d_temp_storage.data(),
                        temp_storage_bytes,
                        begin,
                        output,
                        d_num_selected.begin(),
                        num_items,
                        predicate,
                        stream.value());

  // Copy number of selected elements back to host via pinned memory
  auto num_selected = cudf::detail::make_pinned_vector<cuda::std::size_t>(size_t{1}, stream);
  cudf::detail::cuda_memcpy(
    cudf::host_span<cuda::std::size_t>{&num_selected.front(), size_t{1}},
    cudf::device_span<cuda::std::size_t const>{d_num_selected.begin(), size_t{1}},
    stream);

  return output + num_selected.front();
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
