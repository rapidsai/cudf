/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cub/device/device_copy.cuh>
#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief A helper function that takes in a vector of device spans and memsets them to the
 * value provided using batches sent to the GPU.
 *
 * @param bufs Vector with device spans of data
 * @param value Value to memset all device spans to
 * @param _stream Stream used for device memory operations and kernel launches
 *
 * @return The data in device spans all set to value
 */
template <typename T>
void batched_memset(std::vector<cudf::device_span<T>> const& bufs,
                    T const value,
                    rmm::cuda_stream_view stream)
{
  // define task and bytes parameters
  auto const num_bufs = bufs.size();

  // copy bufs into device memory and then get sizes
  auto gpu_bufs =
    cudf::detail::make_device_uvector_async(bufs, stream, cudf::get_current_device_resource_ref());

  // get a vector with the sizes of all buffers
  auto sizes = thrust::make_transform_iterator(
    thrust::counting_iterator<std::size_t>(0),
    cuda::proclaim_return_type<std::size_t>(
      [gpu_bufs = gpu_bufs.data()] __device__(std::size_t i) { return gpu_bufs[i].size(); }));

  // get an iterator with a constant value to memset
  auto iter_in = thrust::make_constant_iterator(thrust::make_constant_iterator(value));

  // get an iterator pointing to each device span
  auto iter_out = thrust::make_transform_iterator(
    thrust::counting_iterator<std::size_t>(0),
    cuda::proclaim_return_type<T*>(
      [gpu_bufs = gpu_bufs.data()] __device__(std::size_t i) { return gpu_bufs[i].data(); }));

  size_t temp_storage_bytes = 0;

  cub::DeviceCopy::Batched(nullptr, temp_storage_bytes, iter_in, iter_out, sizes, num_bufs, stream);

  rmm::device_buffer d_temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());

  cub::DeviceCopy::Batched(
    d_temp_storage.data(), temp_storage_bytes, iter_in, iter_out, sizes, num_bufs, stream);
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
