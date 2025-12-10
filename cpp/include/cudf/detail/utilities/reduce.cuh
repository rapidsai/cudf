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
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
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
  // Pinned vector store two values: initial value and reduced result
  auto host_data       = cudf::detail::make_pinned_vector_async<OutputType>(size_t{2}, stream);
  host_data.front()    = init;
  auto const num_items = std::distance(begin, end);
  // Device memory to store the result
  rmm::device_buffer d_result(sizeof(OutputType), stream, cudf::get_current_device_resource_ref());
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Reduce(nullptr,
                            temp_storage_bytes,
                            begin,
                            static_cast<OutputType*>(d_result.data()),
                            num_items,
                            op,
                            host_data.front(),
                            stream.value());

  rmm::device_buffer d_temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            begin,
                            static_cast<OutputType*>(d_result.data()),
                            num_items,
                            op,
                            host_data.front(),
                            stream.value());

  // Copy result back to host via pinned memory
  cudf::detail::cuda_memcpy(
    cudf::host_span<OutputType>{&host_data.back(), size_t{1}},
    cudf::device_span<OutputType const>{static_cast<OutputType const*>(d_result.data()), size_t{1}},
    stream);

  return host_data.back();
}

/**
 * @brief Helper to transform device-accessible iterator using a unary operation and then reduce the
 * transformed values using a binary operation
 *
 * This function applies a unary transformation to each element in the input range, then
 * performs a reduction operation on the transformed values. The result is returned to the
 * host using pinned memory for efficient transfer.
 *
 * @tparam InputIterator **[inferred]** The type of device-accessible input iterator
 * @tparam OutputType **[inferred]** The type of the reduction result
 * @tparam UnaryOp **[inferred]** The type of the unary transformation operator
 * @tparam BinaryOp **[inferred]** The type of the binary reduction operator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param transform_op Unary transformation operator to apply to each element
 * @param init Initial value for the reduction
 * @param reduce_op Binary reduction operator
 * @param stream CUDA stream to use
 * @return The reduction result
 */
template <typename InputIterator, typename OutputType, typename UnaryOp, typename BinaryOp>
OutputType transform_reduce(InputIterator begin,
                            InputIterator end,
                            UnaryOp transform_op,
                            OutputType init,
                            BinaryOp reduce_op,
                            rmm::cuda_stream_view stream)
{
  auto const num_items = std::distance(begin, end);
  auto transform_iter  = thrust::make_transform_iterator(begin, transform_op);
  return reduce(transform_iter, transform_iter + num_items, init, reduce_op, stream);
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
