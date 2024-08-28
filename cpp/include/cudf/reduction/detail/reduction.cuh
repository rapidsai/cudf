/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/utilities/cast_functor.cuh>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_reduce.cuh>
#include <thrust/for_each.h>
#include <thrust/iterator/iterator_traits.h>

#include <optional>

namespace cudf {
namespace reduction {
namespace detail {
/**
 * @brief Compute the specified simple reduction over the input range of elements.
 *
 * @param[in] d_in      the begin iterator
 * @param[in] num_items the number of items
 * @param[in] op        the reduction operator
 * @param[in] init      Optional initial value of the reduction
 * @param[in] stream    CUDA stream used for device memory operations and kernel launches
 * @param[in] mr        Device memory resource used to allocate the returned scalar's device
 * memory
 * @returns   Output scalar in device memory
 *
 * @tparam Op               the reduction operator with device binary operator
 * @tparam InputIterator    the input column iterator
 * @tparam OutputType       the output type of reduction
 */
template <typename Op,
          typename InputIterator,
          typename OutputType = typename thrust::iterator_value<InputIterator>::type,
          std::enable_if_t<is_fixed_width<OutputType>() &&
                           not cudf::is_fixed_point<OutputType>()>* = nullptr>
std::unique_ptr<scalar> reduce(InputIterator d_in,
                               cudf::size_type num_items,
                               op::simple_op<Op> op,
                               std::optional<OutputType> init,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  auto const binary_op     = cudf::detail::cast_functor<OutputType>(op.get_binary_op());
  auto const initial_value = init.value_or(op.template get_identity<OutputType>());
  auto dev_result          = rmm::device_scalar<OutputType>{initial_value, stream, mr};

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

  // only for string_view, data is copied
  auto s = new cudf::scalar_type_t<OutputType>(std::move(dev_result), true, stream, mr);
  return std::unique_ptr<scalar>(s);
}

template <typename Op,
          typename InputIterator,
          typename OutputType = typename thrust::iterator_value<InputIterator>::type,
          std::enable_if_t<is_fixed_point<OutputType>()>* = nullptr>
std::unique_ptr<scalar> reduce(InputIterator d_in,
                               cudf::size_type num_items,
                               op::simple_op<Op> op,
                               std::optional<OutputType> init,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FAIL(
    "This function should never be called. fixed_point reduce should always go through the reduce "
    "for the corresponding device_storage_type_t");
}

// @brief string_view specialization of simple reduction
template <typename Op,
          typename InputIterator,
          typename OutputType = typename thrust::iterator_value<InputIterator>::type,
          std::enable_if_t<std::is_same_v<OutputType, string_view>>* = nullptr>
std::unique_ptr<scalar> reduce(InputIterator d_in,
                               cudf::size_type num_items,
                               op::simple_op<Op> op,
                               std::optional<OutputType> init,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  auto const binary_op     = cudf::detail::cast_functor<OutputType>(op.get_binary_op());
  auto const initial_value = init.value_or(op.template get_identity<OutputType>());
  auto dev_result          = rmm::device_scalar<OutputType>{initial_value, stream};

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

  using ScalarType = cudf::scalar_type_t<OutputType>;
  auto s = new ScalarType(dev_result, true, stream, mr);  // only for string_view, data is copied
  return std::unique_ptr<scalar>(s);
}

/**
 * @brief compute reduction by the compound operator (reduce and transform)
 *
 * @param[in] d_in        the begin iterator
 * @param[in] num_items   the number of items
 * @param[in] op          the reduction operator
 * @param[in] valid_count Number of valid items
 * @param[in] ddof        Delta degrees of freedom used for standard deviation and variance
 * @param[in] init        Optional initial value of the reduction
 * @param[in] stream      CUDA stream used for device memory operations and kernel launches
 * @param[in] mr          Device memory resource used to allocate the returned scalar's device
 * memory
 * @returns   Output scalar in device memory
 *
 * The reduction operator must have `intermediate::compute_result()` method.
 * This method performs reduction using binary operator `Op::Op` and transforms the
 * result to `OutputType` using `compute_result()` transform method.
 *
 * @tparam Op               the reduction operator with device binary operator
 * @tparam InputIterator    the input column iterator
 * @tparam OutputType       the output type of reduction
 */
template <typename Op,
          typename InputIterator,
          typename OutputType,
          typename IntermediateType = typename thrust::iterator_value<InputIterator>::type>
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

  rmm::device_scalar<IntermediateType> intermediate_result{initial_value, stream};

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
  auto result      = new ScalarType(OutputType{0}, true, stream, mr);
  thrust::for_each_n(rmm::exec_policy(stream),
                     intermediate_result.data(),
                     1,
                     [dres = result->data(), op, valid_count, ddof] __device__(auto i) {
                       *dres = op.template compute_result<OutputType>(i, valid_count, ddof);
                     });
  return std::unique_ptr<scalar>(result);
}

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
