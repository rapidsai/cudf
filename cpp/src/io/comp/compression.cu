/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compression.hpp"

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/transform_reduce.h>

namespace cudf::io::detail {

writer_compression_statistics collect_compression_statistics(
  device_span<device_span<uint8_t const> const> inputs,
  device_span<codec_exec_result const> results,
  rmm::cuda_stream_view stream)
{
  // bytes_written on success
  auto const output_size_successful = thrust::transform_reduce(
    rmm::exec_policy(stream),
    results.begin(),
    results.end(),
    cuda::proclaim_return_type<size_t>([] __device__(codec_exec_result const& res) {
      return res.status == codec_status::SUCCESS ? res.bytes_written : 0;
    }),
    0ul,
    cuda::std::plus<size_t>());

  auto input_size_with_status = [inputs, results, stream](codec_status status) {
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
      cuda::std::plus<size_t>());
  };

  return writer_compression_statistics{input_size_with_status(codec_status::SUCCESS),
                                       input_size_with_status(codec_status::FAILURE),
                                       input_size_with_status(codec_status::SKIPPED),
                                       output_size_successful};
}

}  // namespace cudf::io::detail
