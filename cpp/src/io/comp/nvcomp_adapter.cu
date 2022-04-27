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

batched_args create_batched_nvcomp_args(device_span<device_span<uint8_t const> const> inputs,
                                        device_span<device_span<uint8_t> const> outputs,
                                        rmm::cuda_stream_view stream)
{
  size_t num_comp_pages = inputs.size();
  rmm::device_uvector<void const*> compressed_data_ptrs(num_comp_pages, stream);
  rmm::device_uvector<size_t> compressed_data_sizes(num_comp_pages, stream);
  rmm::device_uvector<void*> uncompressed_data_ptrs(num_comp_pages, stream);
  rmm::device_uvector<size_t> uncompressed_data_sizes(num_comp_pages, stream);

  // Prepare the input vectors
  auto ins_it =
    thrust::make_zip_iterator(compressed_data_ptrs.begin(), compressed_data_sizes.begin());
  thrust::transform(
    rmm::exec_policy(stream), inputs.begin(), inputs.end(), ins_it, [] __device__(auto const& in) {
      return thrust::make_tuple(in.data(), in.size());
    });

  // Prepare the output vectors
  auto outs_it =
    thrust::make_zip_iterator(uncompressed_data_ptrs.begin(), uncompressed_data_sizes.begin());
  thrust::transform(
    rmm::exec_policy(stream),
    outputs.begin(),
    outputs.end(),
    outs_it,
    [] __device__(auto const& out) { return thrust::make_tuple(out.data(), out.size()); });

  return {std::move(compressed_data_ptrs),
          std::move(compressed_data_sizes),
          std::move(uncompressed_data_ptrs),
          std::move(uncompressed_data_sizes)};
}

void convert_status(device_span<nvcompStatus_t const> nvcomp_stats,
                    device_span<size_t const> actual_uncompressed_sizes,
                    device_span<decompress_status> cudf_stats,
                    rmm::cuda_stream_view stream)
{
  thrust::transform(
    rmm::exec_policy(stream),
    nvcomp_stats.begin(),
    nvcomp_stats.end(),
    actual_uncompressed_sizes.begin(),
    cudf_stats.begin(),
    [] __device__(auto const& status, auto const& size) {
      return decompress_status{size, status == nvcompStatus_t::nvcompSuccess ? 0u : 1u};
    });
}
}  // namespace cudf::io::nvcomp
