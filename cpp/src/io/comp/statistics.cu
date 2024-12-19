/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "gpuinflate.hpp"

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/transform_reduce.h>

namespace cudf::io::detail {

writer_compression_statistics collect_compression_statistics(
  device_span<device_span<uint8_t const> const> inputs,
  device_span<compression_result const> results,
  rmm::cuda_stream_view stream)
{
  // bytes_written on success
  auto const output_size_successful = thrust::transform_reduce(
    rmm::exec_policy(stream),
    results.begin(),
    results.end(),
    cuda::proclaim_return_type<size_t>([] __device__(compression_result const& res) {
      return res.status == compression_status::SUCCESS ? res.bytes_written : 0;
    }),
    0ul,
    thrust::plus<size_t>());

  auto input_size_with_status = [inputs, results, stream](compression_status status) {
    auto const zipped_begin =
      thrust::make_zip_iterator(thrust::make_tuple(inputs.begin(), results.begin()));
    auto const zipped_end = zipped_begin + inputs.size();

    return thrust::transform_reduce(
      rmm::exec_policy(stream),
      zipped_begin,
      zipped_end,
      cuda::proclaim_return_type<size_t>([status] __device__(auto tup) {
        return thrust::get<1>(tup).status == status ? thrust::get<0>(tup).size() : 0;
      }),
      0ul,
      thrust::plus<size_t>());
  };

  return writer_compression_statistics{input_size_with_status(compression_status::SUCCESS),
                                       input_size_with_status(compression_status::FAILURE),
                                       input_size_with_status(compression_status::SKIPPED),
                                       output_size_successful};
}

}  // namespace cudf::io::detail
