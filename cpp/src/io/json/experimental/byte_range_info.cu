/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

// #include "byte_range_info.hpp"
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <limits>

namespace cudf::io::detail::json::experimental {

// custom kernel
template <int BlockSize, typename InputIteratorT, typename PredicateOp, typename IndexT>
__global__ void __launch_bounds__(BlockSize)
  argmin_if(InputIteratorT input, PredicateOp predicate, IndexT* output, int size)
{
  IndexT tid       = threadIdx.x + blockIdx.x * blockDim.x;
  IndexT step_size = blockDim.x * gridDim.x;
  IndexT min_index = std::numeric_limits<IndexT>::max();

  while (tid < size) {
    if (predicate(input[tid])) {
      if (tid < min_index) { min_index = tid; }
    }
    tid += step_size;
  }

  using BlockReduce = cub::BlockReduce<IndexT, BlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  IndexT block_min_index = BlockReduce(temp_storage).Reduce(min_index, cub::Min());

  if (threadIdx.x == 0) { atomicMin(output, block_min_index); }
}

// Extract the first and last character positions in the string.
size_type find_first_delimiter(device_span<char const> d_data,
                               char const delimiter,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  size_type result{std::numeric_limits<size_type>::max()};
  rmm::device_scalar<size_type> d_result(result, stream, mr);
  auto const is_delimiter = [delimiter] __device__(char c) { return c == delimiter; };

  {
    // Get device ordinal
    int device_ordinal;
    CUDF_CUDA_TRY(cudaGetDevice(&device_ordinal));

    // Get SM count
    int sm_count;
    CUDF_CUDA_TRY(
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal));

    constexpr int block_size = 128;
    auto const grid_size =
      std::min(sm_count, static_cast<int>((d_data.size() + block_size - 1) / block_size));

    argmin_if<block_size><<<grid_size, block_size, 0, stream.value()>>>(
      d_data.data(), is_delimiter, d_result.data(), d_data.size());
  }

  return d_result.value(stream);
}

}  // namespace cudf::io::detail::json::experimental
