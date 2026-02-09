/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/device_scalar.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda/std/functional>
#include <cuda/stream_ref>
#include <thrust/copy.h>
#include <thrust/iterator/discard_iterator.h>

namespace cudf::detail {

template <typename Iterator, typename T, typename BinaryOp>
__device__ __forceinline__ T accumulate(Iterator first, Iterator last, T init, BinaryOp op)
{
  for (; first != last; ++first) {
    init = op(std::move(init), *first);
  }
  return init;
}

/**
 * @copydoc cudf::detail::copy_if_safe(rmm::exec_policy, InputIterator, InputIterator,
 * OutputIterator, Predicate, rmm::cuda_stream_view)
 *
 * @tparam StencilIterator Type of the stencil iterator
 * @param stencil The beginning of the stencil sequence
 */
template <typename InputIterator,
          typename StencilIterator,
          typename OutputIterator,
          typename Predicate>
[[deprecated("Use cudf::detail::copy_if instead")]] OutputIterator copy_if_safe(
  InputIterator first,
  InputIterator last,
  StencilIterator stencil,
  OutputIterator result,
  Predicate pred,
  rmm::cuda_stream_view stream)
{
  auto const copy_size = std::min(static_cast<std::size_t>(std::distance(first, last)),
                                  static_cast<std::size_t>(std::numeric_limits<int>::max()));

  auto itr = first;
  while (itr != last) {
    auto const copy_end =
      static_cast<std::size_t>(std::distance(itr, last)) <= copy_size ? last : itr + copy_size;
    result = thrust::copy_if(rmm::exec_policy_nosync(stream), itr, copy_end, stencil, result, pred);
    stencil += std::distance(itr, copy_end);
    itr = copy_end;
  }
  return result;
}

/**
 * @brief Helper to copy elements satisfying a predicate/stencil using CUB with pinned memory
 *
 * This function copies elements from the input range that satisfy the given predicate/stencil
 * to the output range, using CUB's DeviceSelect::FlaggedIf implementation with pinned memory
 * for efficient device-to-host transfer of the number of selected elements.
 *
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 * @tparam StencilIterator **[inferred]** Type of device-accessible stencil iterator
 * @tparam OutputIterator **[inferred]** Type of device-accessible output iterator
 * @tparam Predicate **[inferred]** Type of the unary predicate
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param stencil Device-accessible iterator to start of stencil values
 * @param result Device-accessible iterator to start of output values
 * @param predicate Unary predicate that returns true for elements to copy
 * @param stream CUDA stream to use
 * @return Iterator pointing to the end of the output range
 */
template <typename InputIterator,
          typename StencilIterator,
          typename OutputIterator,
          typename Predicate>
OutputIterator copy_if(InputIterator begin,
                       InputIterator end,
                       StencilIterator stencil,
                       OutputIterator result,
                       Predicate predicate,
                       rmm::cuda_stream_view stream)
{
  auto const num_items = cuda::std::distance(begin, end);

  auto num_selected =
    cudf::detail::device_scalar<std::size_t>(stream, cudf::get_current_device_resource_ref());

  auto temp_storage_bytes = std::size_t{0};
  CUDF_CUDA_TRY(cub::DeviceSelect::FlaggedIf(nullptr,
                                             temp_storage_bytes,
                                             begin,
                                             stencil,
                                             result,
                                             num_selected.data(),
                                             num_items,
                                             predicate,
                                             stream.value()));

  auto d_temp_storage =
    rmm::device_buffer(temp_storage_bytes, stream, cudf::get_current_device_resource_ref());

  CUDF_CUDA_TRY(cub::DeviceSelect::FlaggedIf(d_temp_storage.data(),
                                             temp_storage_bytes,
                                             begin,
                                             stencil,
                                             result,
                                             num_selected.data(),
                                             num_items,
                                             predicate,
                                             stream.value()));

  return result + num_selected.value(stream);
}

/**
 * @brief Utility for calling `thrust::copy_if`.
 *
 * @deprecated in 26.02 and to be removed in a future release. Use `cudf::detail::copy_if` instead.
 *
 * This is a proxy for `thrust::copy_if` which is a workaround for its bug
 * (https://github.com/NVIDIA/thrust/issues/1302) where it cannot iterate over int-max values
 * `distance(first,last) > int-max` This calls thrust::copy_if in 2B chunks instead.
 *
 * @tparam InputIterator Type of the input iterator
 * @tparam OutputIterator Type of the output iterator
 * @tparam Predicate Type of the binary predicate used to determine elements to copy
 *
 * @param first The beginning of the sequence from which to copy
 * @param last The end of the sequence from which to copy
 * @param result The beginning of the sequence into which to copy
 * @param pred The predicate to test on every value of the range `[first, last)`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return An iterator pointing to the position `result + n`, where `n` is equal to the number of
 *         times `pred` evaluated to `true` in the range `[first, last)`.
 */
template <typename InputIterator, typename OutputIterator, typename Predicate>
[[deprecated("Use cudf::detail::copy_if instead")]] OutputIterator copy_if_safe(
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  Predicate pred,
  rmm::cuda_stream_view stream)
{
  auto const copy_size = std::min(static_cast<std::size_t>(std::distance(first, last)),
                                  static_cast<std::size_t>(std::numeric_limits<int>::max()));

  auto itr = first;
  while (itr != last) {
    auto const copy_end =
      static_cast<std::size_t>(std::distance(itr, last)) <= copy_size ? last : itr + copy_size;
    result = thrust::copy_if(rmm::exec_policy_nosync(stream), itr, copy_end, result, pred);
    itr    = copy_end;
  }
  return result;
}

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

  // Device scalar to store the number of selected elements
  auto num_selected =
    cudf::detail::device_scalar<cuda::std::size_t>(stream, cudf::get_current_device_resource_ref());

  // First call to get temporary storage size
  size_t temp_storage_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceSelect::If(nullptr,
                                      temp_storage_bytes,
                                      begin,
                                      output,
                                      num_selected.data(),
                                      num_items,
                                      predicate,
                                      stream.value()));

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());

  // Run copy_if
  CUDF_CUDA_TRY(cub::DeviceSelect::If(d_temp_storage.data(),
                                      temp_storage_bytes,
                                      begin,
                                      output,
                                      num_selected.data(),
                                      num_items,
                                      predicate,
                                      stream.value()));

  // Copy number of selected elements back to host via pinned memory
  return output + num_selected.value(stream);
}

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

  // Device scalar to store the result
  auto result =
    cudf::detail::device_scalar<OutputType>(stream, cudf::get_current_device_resource_ref());

  // Build environment with stream and memory resource for cub::DeviceReduce::Reduce
  auto env = cuda::std::execution::env{
    cuda::std::execution::prop{cuda::get_stream_t{}, cuda::stream_ref{stream.value()}},
    cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                               cudf::get_current_device_resource_ref()}};
  CUDF_CUDA_TRY(cub::DeviceReduce::Reduce(begin, result.data(), num_items, binary_op, init, env));

  // Copy result back to host via pinned memory
  return result.value(stream);
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

  // Device scalar to store the number of runs (unique keys)
  auto d_num_runs =
    cudf::detail::device_scalar<cuda::std::size_t>(stream, cudf::get_current_device_resource_ref());

  // First call to get temporary storage size
  size_t temp_storage_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceReduce::ReduceByKey(nullptr,
                                               temp_storage_bytes,
                                               keys_begin,
                                               keys_output,
                                               values_begin,
                                               values_output,
                                               d_num_runs.data(),
                                               op,
                                               num_items,
                                               stream.value()));

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());

  // Run reduce-by-key
  CUDF_CUDA_TRY(cub::DeviceReduce::ReduceByKey(d_temp_storage.data(),
                                               temp_storage_bytes,
                                               keys_begin,
                                               keys_output,
                                               values_begin,
                                               values_output,
                                               d_num_runs.data(),
                                               op,
                                               num_items,
                                               stream.value()));

  // Copy number of runs back to host via pinned memory
  auto const num_runs = d_num_runs.value(stream);

  return {keys_output + num_runs, values_output + num_runs};
}

/**
 * @copydoc cudf::detail::reduce_by_key
 *
 * This function performs the reduce-by-key operation asynchronously.
 * It is useful when the calling function does not need the returned result
 * and therefore prevents a stream synchronization.
 *
 */
template <typename Op,
          typename KeysInputIterator,
          typename KeysOutputIterator,
          typename ValuesInputIterator,
          typename ValuesOutputIterator>
void reduce_by_key_async(KeysInputIterator keys_begin,
                         KeysInputIterator keys_end,
                         ValuesInputIterator values_begin,
                         KeysOutputIterator keys_output,
                         ValuesOutputIterator values_output,
                         Op op,
                         rmm::cuda_stream_view stream)
{
  auto const num_items = cuda::std::distance(keys_begin, keys_end);

  size_t temp_storage_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceReduce::ReduceByKey(nullptr,
                                               temp_storage_bytes,
                                               keys_begin,
                                               keys_output,
                                               values_begin,
                                               values_output,
                                               thrust::make_discard_iterator(),
                                               op,
                                               num_items,
                                               stream.value()));

  rmm::device_buffer d_temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());

  CUDF_CUDA_TRY(cub::DeviceReduce::ReduceByKey(d_temp_storage.data(),
                                               temp_storage_bytes,
                                               keys_begin,
                                               keys_output,
                                               values_begin,
                                               values_output,
                                               thrust::make_discard_iterator(),
                                               op,
                                               num_items,
                                               stream.value()));
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

  // Device scalar to store the result
  auto result =
    cudf::detail::device_scalar<OutputType>(stream, cudf::get_current_device_resource_ref());

  size_t temp_storage_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceReduce::TransformReduce(nullptr,
                                                   temp_storage_bytes,
                                                   begin,
                                                   result.data(),
                                                   num_items,
                                                   reduce_op,
                                                   transform_op,
                                                   init,
                                                   stream.value()));

  rmm::device_buffer d_temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
  CUDF_CUDA_TRY(cub::DeviceReduce::TransformReduce(d_temp_storage.data(),
                                                   temp_storage_bytes,
                                                   begin,
                                                   result.data(),
                                                   num_items,
                                                   reduce_op,
                                                   transform_op,
                                                   init,
                                                   stream.value()));

  // Copy result back to host via pinned memory
  return result.value(stream);
}

/**
 * @brief Check if a predicate is true for all elements in a device-accessible range
 *
 * This function applies a predicate to all elements in the range and returns true
 * if the predicate returns true for all elements.
 *
 * @tparam TransformOp **[inferred]** Type of the unary transformation operator
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param op Predicate operator to apply to each element
 * @param stream CUDA stream to use
 * @return true if the predicate is true for all elements, false otherwise
 */
template <typename TransformOp, typename InputIterator>
bool all_of(InputIterator begin, InputIterator end, TransformOp op, rmm::cuda_stream_view stream)
{
  return transform_reduce(begin, end, op, true, cuda::std::logical_and<bool>{}, stream);
}

/**
 * @brief Check if a predicate is true for any element in a device-accessible range
 *
 * This function applies a predicate to all elements in the range and returns true
 * if the predicate returns true for at least one element.
 *
 * @tparam TransformOp **[inferred]** Type of the unary transformation operator
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param op Predicate operator to apply to each element
 * @param stream CUDA stream to use
 * @return true if the predicate is true for any element, false otherwise
 */
template <typename TransformOp, typename InputIterator>
bool any_of(InputIterator begin, InputIterator end, TransformOp op, rmm::cuda_stream_view stream)
{
  return transform_reduce(begin, end, op, false, cuda::std::logical_or<bool>{}, stream);
}

/**
 * @brief Check if a predicate is false for all elements in a device-accessible range
 *
 * This function applies a predicate to all elements in the range and returns true
 * if the predicate returns false for all elements (i.e., no element satisfies the predicate).
 *
 * @tparam TransformOp **[inferred]** Type of the predicate operator
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param init Initial value (unused, for API compatibility)
 * @param op Predicate operator to apply to each element
 * @param stream CUDA stream to use
 * @return true if the predicate is false for all elements, false otherwise
 */
template <typename TransformOp, typename InputIterator>
bool none_of(InputIterator begin, InputIterator end, TransformOp op, rmm::cuda_stream_view stream)
{
  return not any_of(begin, end, op, stream);
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

}  // namespace cudf::detail
