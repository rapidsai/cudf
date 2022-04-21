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
#include "nvcomp_adapter.cuh"

#include <cudf/detail/utilities/integer_utils.hpp>

#include <rmm/exec_policy.hpp>

namespace cudf::io::nvcomp {
__global__ void convert_status_kernel(
  device_span<nvcompStatus_t const> nvcomp_stats,
  device_span<size_t const> actual_uncompressed_sizes,  // TODO optional
  device_span<decompress_status> cudf_stats)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < cudf_stats.size()) {
    cudf_stats[tid].status        = nvcomp_stats[tid] == nvcompStatus_t::nvcompSuccess ? 0 : 1;
    cudf_stats[tid].bytes_written = actual_uncompressed_sizes[tid];
  }
}

__host__ void convert_status(device_span<nvcompStatus_t const> nvcomp_stats,
                             device_span<size_t const> actual_uncompressed_sizes,
                             device_span<decompress_status> cudf_stats,
                             rmm::cuda_stream_view stream)
{
  dim3 block(128);
  dim3 grid(cudf::util::div_rounding_up_safe(nvcomp_stats.size(), static_cast<size_t>(block.x)));
  convert_status_kernel<<<grid, block, 0, stream.value()>>>(
    nvcomp_stats, actual_uncompressed_sizes, cudf_stats);
}

batched_inputs create_batched_inputs(device_span<device_decompress_input const> cudf_comp_in,
                                     rmm::cuda_stream_view stream)
{
  size_t num_comp_pages = cudf_comp_in.size();
  // Analogous to cudf_comp_in.srcDevice
  rmm::device_uvector<void const*> compressed_data_ptrs(num_comp_pages, stream);
  // Analogous to cudf_comp_in.srcSize
  rmm::device_uvector<size_t> compressed_data_sizes(num_comp_pages, stream);
  // Analogous to cudf_comp_in.dstDevice
  rmm::device_uvector<void*> uncompressed_data_ptrs(num_comp_pages, stream);
  // Analogous to cudf_comp_in.dstSize
  rmm::device_uvector<size_t> uncompressed_data_sizes(num_comp_pages, stream);

  // Prepare the vectors
  auto comp_it = thrust::make_zip_iterator(compressed_data_ptrs.begin(),
                                           compressed_data_sizes.begin(),
                                           uncompressed_data_ptrs.begin(),
                                           uncompressed_data_sizes.begin());
  thrust::transform(rmm::exec_policy(stream),
                    cudf_comp_in.begin(),
                    cudf_comp_in.end(),
                    comp_it,
                    [] __device__(device_decompress_input in) {
                      return thrust::make_tuple(
                        in.src.data(), in.src.size(), in.dstDevice, in.dstSize);
                    });

  return {std::move(compressed_data_ptrs),
          std::move(compressed_data_sizes),
          std::move(uncompressed_data_ptrs),
          std::move(uncompressed_data_sizes)};
}

}  // namespace cudf::io::nvcomp
