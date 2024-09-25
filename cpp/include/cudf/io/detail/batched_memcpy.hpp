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
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/functional>
#include <thrust/iterator/constant_iterator.h>

namespace CUDF_EXPORT cudf {
namespace io::detail {

/**
 * @brief A helper function that copies a vector of host scalar data to the corresponding device
 * addresses in a batched manner.
 *
 *
 * @param[in] src_data A vector of host scalar data
 * @param[in] dst_addrs A vector of device destination addresses
 * @param[in] mr Device memory resource to allocate temporary memory
 * @param[in] stream CUDA stream to use
 */
template <typename T>
void batched_memcpy(std::vector<T> const& src_data,
                    std::vector<T*> const& dst_addrs,
                    rmm::device_async_resource_ref mr,
                    rmm::cuda_stream_view stream)
{
  // Number of elements to copy
  auto const num_elems = src_data.size();

  // Copy src data to device and create an iterator
  auto d_src_data = cudf::detail::make_device_uvector_async(src_data, stream, mr);
  auto src_iter   = cudf::detail::make_counting_transform_iterator(
    static_cast<std::size_t>(0),
    cuda::proclaim_return_type<T*>(
      [src = d_src_data.data()] __device__(std::size_t i) { return src + i; }));

  // Copy dst addresses to device and create an iterator
  auto d_dst_addrs = cudf::detail::make_device_uvector_async(dst_addrs, stream, mr);
  auto dst_iter    = cudf::detail::make_counting_transform_iterator(
    static_cast<std::size_t>(0),
    cuda::proclaim_return_type<T*>(
      [dst = d_dst_addrs.data()] __device__(std::size_t i) { return dst[i]; }));

  // Scalar src data so size_iter is simply a constant iterator.
  auto size_iter = thrust::make_constant_iterator(sizeof(T));

  // Get temp storage needed for cub::DeviceMemcpy::Batched
  size_t temp_storage_bytes = 0;
  cub::DeviceMemcpy::Batched(
    nullptr, temp_storage_bytes, src_iter, dst_iter, size_iter, num_elems, stream.value());

  // Allocate temporary storage
  auto d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream.value(), mr};

  // Run cub::DeviceMemcpy::Batched
  cub::DeviceMemcpy::Batched(d_temp_storage.data(),
                             temp_storage_bytes,
                             src_iter,
                             dst_iter,
                             size_iter,
                             num_elems,
                             stream.value());
}

}  // namespace io::detail
}  // namespace CUDF_EXPORT cudf
