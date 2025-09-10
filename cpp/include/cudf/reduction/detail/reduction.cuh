/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include "reduction_operators.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/utilities/cast_functor.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda/std/iterator>
#include <thrust/for_each.h>

#include <optional>

namespace cudf {
namespace reduction {
namespace detail {
/**
 * @brief Compute the specified simple reduction over the input range of elements.
 *
 * @tparam Op               the reduction operator with device binary operator
 * @tparam InputIterator    the input column iterator
 * @tparam OutputType       the output type of reduction
 *
 * @param d_in      the begin iterator
 * @param num_items the number of items
 * @param op        the reduction operator
 * @param init      Optional initial value of the reduction
 * @param stream    CUDA stream used for device memory operations and kernel launches
 * @param mr        Device memory resource used to allocate the returned scalar's device memory
 * @returns Output scalar in device memory
 */
template <typename Op,
          typename InputIterator,
          typename OutputType = cuda::std::iter_value_t<InputIterator>>
std::unique_ptr<scalar> reduce(InputIterator d_in,
                               cudf::size_type num_items,
                               op::simple_op<Op> op,
                               std::optional<OutputType> init,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
  requires(is_fixed_width<OutputType>() && not cudf::is_fixed_point<OutputType>())
{
  auto const binary_op     = cudf::detail::cast_functor<OutputType>(op.get_binary_op());
  auto const initial_value = init.value_or(op.template get_identity<OutputType>());
  using ScalarType         = cudf::scalar_type_t<OutputType>;
  
  // Use pinned memory for initial_value to optimize host-to-device transfer
  auto pinned_initial = cudf::detail::make_pinned_vector<OutputType>(1, stream);
  pinned_initial[0] = initial_value;  // Assign to element zero as suggested
  
  // Create scalar from pinned memory using rmm::device_scalar constructor
  rmm::device_scalar<OutputType> device_data(stream, mr);
  cudf::detail::cuda_memcpy_async<OutputType>(cudf::device_span<OutputType>{device_data.data(), 1}, pinned_initial, stream);
  auto result = std::make_unique<ScalarType>(std::move(device_data), true, stream, mr);

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_in,
                            result->data(),
                            num_items,
                            binary_op,
                            initial_value,
                            stream.value());
  d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};

  // Run reduction
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_in,
                            result->data(),
                            num_items,
                            binary_op,
                            initial_value,
                            stream.value());
  return result;
}

template <typename Op,
          typename InputIterator,
          typename OutputType = cuda::std::iter_value_t<InputIterator>>
std::unique_ptr<scalar> reduce(InputIterator d_in,
                               cudf::size_type num_items,
                               op::simple_op<Op> op,
                               std::optional<OutputType> init,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
  requires(is_fixed_point<OutputType>())
{
  CUDF_FAIL(
    "This function should never be called. fixed_point reduce should always go through the reduce "
    "for the corresponding device_storage_type_t");
}

// @brief string_view specialization of simple reduction
template <typename Op,
          typename InputIterator,
          typename OutputType = cuda::std::iter_value_t<InputIterator>>
std::unique_ptr<scalar> reduce(InputIterator d_in,
                               cudf::size_type num_items,
                               op::simple_op<Op> op,
                               std::optional<OutputType> init,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
  requires(std::is_same_v<OutputType, string_view>)
{
  auto const binary_op     = cudf::detail::cast_functor<OutputType>(op.get_binary_op());
  auto const initial_value = init.value_or(op.template get_identity<OutputType>());
  auto dev_result          = cudf::detail::device_scalar<OutputType>{initial_value, stream};

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_in,
                            dev_result.data(),
                            num_items,
                            binary_op,
                            initial_value,
                            stream.value());
  d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};

  // Run reduction
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_in,
                            dev_result.data(),
                            num_items,
                            binary_op,
                            initial_value,
                            stream.value());

  return std::make_unique<cudf::string_scalar>(dev_result, true, stream, mr);
}

/**
 * @brief compute reduction by the compound operator (reduce and transform)
 *
 * The reduction operator must have `intermediate::compute_result()` method.
 * This method performs reduction using binary operator `Op::Op` and transforms the
 * result to `OutputType` using `compute_result()` transform method.
 *
 * @tparam Op               the reduction operator with device binary operator
 * @tparam InputIterator    the input column iterator
 * @tparam OutputType       the output type of reduction
 *
 * @param d_in        the begin iterator
 * @param num_items   the number of items
 * @param op          the reduction operator
 * @param valid_count Number of valid items
 * @param ddof        Delta degrees of freedom used for standard deviation and variance
 * @param stream      CUDA stream used for device memory operations and kernel launches
 * @param mr          Device memory resource used to allocate the returned scalar's device memory
 * @returns Output scalar in device memory
 */
template <typename Op,
          typename InputIterator,
          typename OutputType,
          typename IntermediateType = cuda::std::iter_value_t<InputIterator>>
std::unique_ptr<scalar> reduce(InputIterator d_in,
                               cudf::size_type num_items,
                               op::compound_op<Op> op,
                               cudf::size_type valid_count,
                               cudf::size_type ddof,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  auto const binary_op     = cudf::detail::cast_functor<IntermediateType>(op.get_binary_op());
  auto const initial_value = op.template get_identity<IntermediateType>();

  cudf::detail::device_scalar<IntermediateType> intermediate_result{initial_value, stream};

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_in,
                            intermediate_result.data(),
                            num_items,
                            binary_op,
                            initial_value,
                            stream.value());
  d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};

  // Run reduction
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_in,
                            intermediate_result.data(),
                            num_items,
                            binary_op,
                            initial_value,
                            stream.value());

  // compute the result value from intermediate value in device
  using ScalarType = cudf::scalar_type_t<OutputType>;
  
  // Use pinned memory for initial OutputType{0} to optimize host-to-device transfer
  auto pinned_initial = cudf::detail::make_pinned_vector<OutputType>(1, stream);
  pinned_initial[0] = OutputType{0};  // Assign to element zero as suggested
  auto result = std::make_unique<ScalarType>(stream, mr);
  cudf::detail::cuda_memcpy_async<OutputType>(cudf::device_span<OutputType>{result->data(), 1}, pinned_initial, stream);
  
  thrust::for_each_n(rmm::exec_policy(stream),
                     intermediate_result.data(),
                     1,
                     [dres = result->data(), op, valid_count, ddof] __device__(auto i) {
                       *dres = op.template compute_result<OutputType>(i, valid_count, ddof);
                     });
  return result;
}

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
