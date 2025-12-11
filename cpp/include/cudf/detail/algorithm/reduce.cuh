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
#include <cuda/stream_ref>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Helper to reduce a device-accessible iterator using a binary operation
 *
 * This function performs a reduction operation on device data and returns the result
 * to the host using pinned memory for efficient transfer.
 *
 * @tparam Op **[inferred]** The type of the binary reduction operator
 * @tparam InputIterator **[inferred]** The type of device-accessible input iterator
 * @tparam OutputType **[inferred]** The type of the reduction result
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param init Initial value for the reduction
 * @param op Binary reduction operator
 * @param stream CUDA stream to use
 * @return The reduction result
 */
template <typename Op,
          typename InputIterator,
          typename OutputType = cuda::std::iter_value_t<InputIterator>>
OutputType reduce(InputIterator begin,
                  InputIterator end,
                  OutputType init,
                  Op binary_op,
                  rmm::cuda_stream_view stream)
{
  auto const num_items = std::distance(begin, end);

  // Device memory to store the result
  rmm::device_buffer d_result(sizeof(OutputType), stream, cudf::get_current_device_resource_ref());

  // Build environment with stream and memory resource for cub::DeviceReduce::Reduce
  auto env = cuda::std::execution::env{
    cuda::std::execution::prop{cuda::get_stream_t{}, cuda::stream_ref{stream.value()}},
    cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                               cudf::get_current_device_resource_ref()}};
  cub::DeviceReduce::Reduce(
    begin, static_cast<OutputType*>(d_result.data()), num_items, binary_op, init, env);

  // Copy result back to host via pinned memory
  auto result = cudf::detail::make_pinned_vector<OutputType>(size_t{1}, stream);
  cudf::detail::cuda_memcpy(
    cudf::host_span<OutputType>{&result.front(), size_t{1}},
    cudf::device_span<OutputType const>{static_cast<OutputType const*>(d_result.data()), size_t{1}},
    stream);

  return result.front();
}

/**
 * @brief Helper to transform device-accessible iterator using a unary operation and then reduce the
 * transformed values using a binary operation
 *
 * This function applies a unary transformation to each element in the input range, then
 * performs a reduction operation on the transformed values. The result is returned to the
 * host using pinned memory for efficient transfer.
 *
 * @tparam TransformationOp **[inferred]** The type of the unary transformation operator
 * @tparam ReductionOp **[inferred]** The type of the binary reduction operator
 * @tparam InputIterator **[inferred]** The type of device-accessible input iterator
 * @tparam OutputType **[inferred]** The type of the reduction result
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param transform_op Unary transformation operator to apply to each element
 * @param init Initial value for the reduction
 * @param reduce_op Binary reduction operator
 * @param stream CUDA stream to use
 * @return The reduction result
 */
template <typename TransformationOp,
          typename ReductionOp,
          typename InputIterator,
          typename OutputType = cuda::std::iter_value_t<InputIterator>>
OutputType transform_reduce(InputIterator begin,
                            InputIterator end,
                            TransformationOp transform_op,
                            OutputType init,
                            ReductionOp reduce_op,
                            rmm::cuda_stream_view stream)
{
  auto const num_items = std::distance(begin, end);

  // Device memory to store the result
  rmm::device_buffer d_result(sizeof(OutputType), stream, cudf::get_current_device_resource_ref());

  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::TransformReduce(nullptr,
                                     temp_storage_bytes,
                                     begin,
                                     static_cast<OutputType*>(d_result.data()),
                                     num_items,
                                     reduce_op,
                                     transform_op,
                                     init,
                                     stream.value());

  rmm::device_buffer d_temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
  cub::DeviceReduce::TransformReduce(d_temp_storage.data(),
                                     temp_storage_bytes,
                                     begin,
                                     static_cast<OutputType*>(d_result.data()),
                                     num_items,
                                     reduce_op,
                                     transform_op,
                                     init,
                                     stream.value());

  // Copy result back to host via pinned memory
  auto result = cudf::detail::make_pinned_vector<OutputType>(size_t{1}, stream);
  cudf::detail::cuda_memcpy(
    cudf::host_span<OutputType>{&result.front(), size_t{1}},
    cudf::device_span<OutputType const>{static_cast<OutputType const*>(d_result.data()), size_t{1}},
    stream);

  return result.front();
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
