/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/cudf.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cub/device/device_reduce.cuh>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/for_each.h>
#include <rmm/thrust_rmm_allocator.h>

namespace cudf {
namespace experimental {
namespace reduction {
namespace detail {

/** --------------------------------------------------------------------------*
 * @brief compute reduction by the operator
 *
 * @param[in] d_in      the begin iterator
 * @param[in] num_items the number of items
 * @param[in] op        the device binary operator
 * @param[in] stream    cuda stream
 * @returns   scalar    output scalar in device memory
 *
 * @tparam Op               the device binary operator
 * @tparam InputIterator    the input column iterator
 * @tparam OutputType       the output type of reduction
 * ----------------------------------------------------------------------------**/
template <typename Op, typename InputIterator, typename OutputType=typename thrust::iterator_value<InputIterator>::type>
std::unique_ptr<scalar> reduce(InputIterator d_in, cudf::size_type num_items, Op op,
  rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  OutputType identity = Op::template identity<OutputType>();
  rmm::device_scalar<OutputType> dev_result{identity, stream, mr};

  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;

  // Allocate temporary storage
  cub::DeviceReduce::Reduce(d_temp_storage.data(), temp_storage_bytes, d_in, dev_result.data(), num_items, op, identity, stream);
  d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream, mr};

  // Run reduction
  cub::DeviceReduce::Reduce(d_temp_storage.data(), temp_storage_bytes, d_in, dev_result.data(), num_items, op, identity, stream);

  using ScalarType = cudf::experimental::scalar_type_t<OutputType>;
  auto s = new ScalarType(std::move(dev_result), true, stream, mr); //only for string_view, data is copied
  return std::unique_ptr<scalar>(s);
}

/** --------------------------------------------------------------------------*
 * @brief compute reduction by the compound operator (reduce and transform)
 *
 * @param[in] d_in      the begin iterator
 * @param[in] num_items the number of items
 * @param[in] op        the device binary operator
 * @param[in] iop       the intermediate operator
 * @param[in] valid_count   the intermediate operator argument 1
 * @param[in] ddof      the intermediate operator argument 2
 * @param[in] stream    cuda stream
 * @returns   scalar    output scalar in device memory
 *
 * @tparam Op               the device binary operator
 * @tparam InputIterator    the input column iterator
 * @tparam OutputType       the output type of reduction
 * ----------------------------------------------------------------------------**/
template <typename Op, typename InputIterator, typename OutputType, 
  template<typename> class intermediate, 
  typename IntermediateType=typename thrust::iterator_value<InputIterator>::type>
//  typename std::enable_if_t<!std::is_base_of<compound, Op>, int> = 0
std::unique_ptr<scalar> reduce(InputIterator d_in, cudf::size_type num_items, Op op, 
  intermediate<OutputType> iop,
  cudf::size_type valid_count,
  cudf::size_type ddof,
  rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  using intermediateOp = intermediate<OutputType>; 
  IntermediateType identity = Op::template identity<IntermediateType>();
  rmm::device_scalar<IntermediateType> intermediate_result{identity, stream, mr};

  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;

  // Allocate temporary storage
  cub::DeviceReduce::Reduce(d_temp_storage.data(), temp_storage_bytes, d_in, intermediate_result.data(), num_items, op, identity, stream);
  d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream, mr};

  // Run reduction
  cub::DeviceReduce::Reduce(d_temp_storage.data(), temp_storage_bytes, d_in, intermediate_result.data(), num_items, op, identity, stream);

  // compute the result value from intermediate value in device
  using ScalarType = cudf::experimental::scalar_type_t<OutputType>;
  auto result = new ScalarType(OutputType{0}, true, stream, mr);
  thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
   intermediate_result.data(), 1,
   [dres=result->data(), valid_count, ddof] __device__ (auto i)
   { *dres = intermediateOp::compute_result(i, valid_count, ddof); } );
  return std::unique_ptr<scalar>(result);
}

} // namespace detail
} // namespace reduction
} // namespace experimental
} // namespace cudf

