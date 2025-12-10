/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda/functional>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Helper to reduce a device-accessible iterator using a binary operation
 *
 * This function performs a reduction operation on device data and returns the result
 * to the host using pinned memory for efficient transfer.
 *
 * @tparam InputIterator **[inferred]** The type of device-accessible input iterator
 * @tparam OutputType **[inferred]** The type of the reduction result
 * @tparam BinaryOp **[inferred]** The type of the binary reduction operator
 *
 * @param input Device-accessible iterator to input values
 * @param num_items Number of items to reduce
 * @param init Initial value for the reduction
 * @param op Binary reduction operator
 * @param stream CUDA stream to use
 * @return The reduction result
 */
template <typename InputIterator, typename OutputType, typename BinaryOp>
OutputType reduce(InputIterator begin,
                  InputIterator end,
                  OutputType init,
                  BinaryOp op,
                  rmm::cuda_stream_view stream)
{
  // Device memory for the result
  rmm::device_buffer d_result(sizeof(OutputType), stream);

  auto const num_items = std::distance(begin, end);

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(nullptr,
                            temp_storage_bytes,
                            begin,
                            static_cast<OutputType*>(d_result.data()),
                            num_items,
                            op,
                            init,
                            stream.value());

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);

  // Run reduction
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            begin,
                            static_cast<OutputType*>(d_result.data()),
                            num_items,
                            op,
                            init,
                            stream.value());

  // Copy result back to pinned vector
  auto h_result = cudf::detail::make_pinned_vector_async<OutputType>(1, stream);
  cudf::detail::cuda_memcpy(
    cudf::host_span<OutputType>{static_cast<OutputType*>(h_result.data()), size_t{1}},
    cudf::device_span<OutputType const>{static_cast<OutputType const*>(d_result.data()), size_t{1}},
    stream);

  return h_result.front();
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
