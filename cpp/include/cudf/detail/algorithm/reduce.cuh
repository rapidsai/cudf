/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda/std/functional>
#include <cuda/stream_ref>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Helper to reduce a device-accessible iterator using a binary operation
 *
 * This function performs a reduction operation on device data and returns the result
 * to the host using pinned memory for efficient transfer.
 *
 * @tparam Op **[inferred]** Type of the binary reduction operator
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 * @tparam OutputType **[inferred]** Type of the reduction result
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
  auto const num_items = cuda::std::distance(begin, end);

  // Device memory to store the result
  auto d_result =
    rmm::device_uvector<OutputType>(1, stream, cudf::get_current_device_resource_ref());

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
 * @tparam TransformationOp **[inferred]** Type of the unary transformation operator
 * @tparam ReductionOp **[inferred]** Type of the binary reduction operator
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 * @tparam OutputType **[inferred]** Type of the reduction result
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
  auto const num_items = cuda::std::distance(begin, end);

  // Device memory to store the result
  auto d_result =
    rmm::device_uvector<OutputType>(1, stream, cudf::get_current_device_resource_ref());

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

/**
 * @brief Helper to perform reduce-by-key operation using CUB with pinned memory for result transfer
 *
 * This function reduces values by consecutive equal keys, similar to thrust::reduce_by_key,
 * but uses CUB's implementation with pinned memory for efficient device-to-host transfer
 * of the number of unique keys.
 *
 * @tparam Op **[inferred]** Type of the binary reduction operator
 * @tparam KeysInputIterator **[inferred]** Type of device-accessible input iterator for keys
 * @tparam KeysOutputIterator **[inferred]** Type of device-accessible output iterator for unique
 * keys
 * @tparam ValuesInputIterator **[inferred]** Type of device-accessible input iterator for values
 * @tparam ValuesOutputIterator **[inferred]** Type of device-accessible output iterator for reduced
 * values
 *
 * @param keys_begin Device-accessible iterator to start of input keys
 * @param keys_end Device-accessible iterator to end of input keys
 * @param values_begin Device-accessible iterator to start of input values
 * @param keys_output Device-accessible iterator to start of output unique keys
 * @param values_output Device-accessible iterator to start of output reduced values
 * @param op Binary reduction operator
 * @param stream CUDA stream to use
 * @return A pair of iterators pointing to the end of the output key and value ranges
 */
template <typename Op,
          typename KeysInputIterator,
          typename KeysOutputIterator,
          typename ValuesInputIterator,
          typename ValuesOutputIterator>
cuda::std::pair<KeysOutputIterator, ValuesOutputIterator> reduce_by_key(
  KeysInputIterator keys_begin,
  KeysInputIterator keys_end,
  ValuesInputIterator values_begin,
  KeysOutputIterator keys_output,
  ValuesOutputIterator values_output,
  Op op,
  rmm::cuda_stream_view stream)
{
  auto const num_items = cuda::std::distance(keys_begin, keys_end);

  // Device memory to store the number of runs (unique keys)
  auto d_num_runs =
    rmm::device_uvector<cuda::std::size_t>(1, stream, cudf::get_current_device_resource_ref());

  // First call to get temporary storage size
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::ReduceByKey(nullptr,
                                 temp_storage_bytes,
                                 keys_begin,
                                 keys_output,
                                 values_begin,
                                 values_output,
                                 d_num_runs.begin(),
                                 op,
                                 num_items,
                                 stream.value());

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());

  // Run reduce-by-key
  cub::DeviceReduce::ReduceByKey(d_temp_storage.data(),
                                 temp_storage_bytes,
                                 keys_begin,
                                 keys_output,
                                 values_begin,
                                 values_output,
                                 d_num_runs.begin(),
                                 op,
                                 num_items,
                                 stream.value());

  // Copy number of runs back to host via pinned memory
  auto num_runs = cudf::detail::make_pinned_vector<size_t>(size_t{1}, stream);
  cudf::detail::cuda_memcpy(
    cudf::host_span<cuda::std::size_t>{&num_runs.front(), size_t{1}},
    cudf::device_span<cuda::std::size_t const>{d_num_runs.begin(), size_t{1}},
    stream);

  return {keys_output + num_runs.front(), values_output + num_runs.front()};
}

/**
 * @brief Helper to count elements satisfying a predicate using CUB with pinned memory
 *
 * This function counts elements in the input range that satisfy the given predicate,
 * using CUB's transform_reduce implementation with pinned memory for efficient
 * device-to-host transfer of the count.
 *
 * @tparam Predicate **[inferred]** Type of the unary predicate
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param predicate Unary predicate that returns true for elements to count
 * @param stream CUDA stream to use
 * @return The count of elements satisfying the predicate
 */
template <typename Predicate, typename InputIterator>
cuda::std::size_t count_if(InputIterator begin,
                           InputIterator end,
                           Predicate predicate,
                           rmm::cuda_stream_view stream)
{
  // Transform each element to 0 or 1 based on predicate, then sum
  auto transform_op = [predicate] __device__(auto const& val) -> cuda::std::size_t {
    return predicate(val) ? cuda::std::size_t{1} : cuda::std::size_t{0};
  };

  return transform_reduce(
    begin, end, transform_op, cuda::std::size_t{0}, cuda::std::plus<>{}, stream);
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
