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
namespace detail {

/**
 * @brief A helper function that copies a vector of vectors from source to destination addresses in
 * a batched manner.
 *
 * @tparam SrcIterator **[inferred]** The type of device-accessible source addresses iterator
 * @tparam DstIterator **[inferred]** The type of device-accessible destination address iterator
 * @tparam SizeIterator **[inferred]** The type of device-accessible buffer size iterator
 *
 * @param src_iter Device-accessible iterator to source addresses
 * @param dst_iter Device-accessible iterator to destination addresses
 * @param size_iter Device-accessible iterator to the buffer sizes (in bytes)
 * @param num_buffs Number of buffers to be copied
 * @param stream CUDA stream to use
 */
template <typename SrcIterator, typename DstIterator, typename SizeIterator>
void batched_memcpy_async(SrcIterator src_iter,
                          DstIterator dst_iter,
                          SizeIterator size_iter,
                          size_t num_buffs,
                          rmm::cuda_stream_view stream)
{
  size_t temp_storage_bytes = 0;
  cub::DeviceMemcpy::Batched(
    nullptr, temp_storage_bytes, src_iter, dst_iter, size_iter, num_buffs, stream.value());

  rmm::device_buffer d_temp_storage{temp_storage_bytes, stream.value()};

  cub::DeviceMemcpy::Batched(d_temp_storage.data(),
                             temp_storage_bytes,
                             src_iter,
                             dst_iter,
                             size_iter,
                             num_buffs,
                             stream.value());
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
