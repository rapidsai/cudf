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
#include "nvcomp_adapter.hpp"
#include "nvcomp_adapter.cuh"

#include <cudf/utilities/error.hpp>

#include <nvcomp/snappy.h>

namespace cudf::io::nvcomp {

template <typename... Args>
auto batched_decompress_get_temp_size(compression_type compression, Args&&... args)
{
  switch (compression) {
    case compression_type::SNAPPY:
      return nvcompBatchedSnappyDecompressGetTempSize(std::forward<Args>(args)...);
    default: CUDF_FAIL("Unsupported compression type");
  }
};

template <typename... Args>
auto batched_decompress_async(compression_type compression, Args&&... args)
{
  switch (compression) {
    case compression_type::SNAPPY:
      return nvcompBatchedSnappyDecompressAsync(std::forward<Args>(args)...);
    default: CUDF_FAIL("Unsupported compression type");
  }
};

size_t get_temp_size(compression_type compression, size_t num_chunks, size_t max_uncomp_chunk_size)
{
  size_t temp_size = 0;
  nvcompStatus_t nvcomp_status =
    batched_decompress_get_temp_size(compression, num_chunks, max_uncomp_chunk_size, &temp_size);
  CUDF_EXPECTS(nvcomp_status == nvcompStatus_t::nvcompSuccess,
               "Unable to get scratch size for decompression");

  return temp_size;
}

void batched_decompress(compression_type compression,
                        device_span<device_span<uint8_t const> const> inputs,
                        device_span<device_span<uint8_t> const> outputs,
                        device_span<decompress_status> statuses,
                        size_t max_uncomp_chunk_size,
                        rmm::cuda_stream_view stream)
{
  auto const num_chunks = inputs.size();

  // cuDF inflate inputs converted to nvcomp inputs
  auto const nvcomp_args = create_batched_nvcomp_args(inputs, outputs, stream);
  rmm::device_uvector<size_t> actual_uncompressed_data_sizes(num_chunks, stream);
  rmm::device_uvector<nvcompStatus_t> nvcomp_statuses(num_chunks, stream);
  // Temporary space required for decompression
  rmm::device_buffer scratch(get_temp_size(compression, num_chunks, max_uncomp_chunk_size), stream);
  auto const nvcomp_status = batched_decompress_async(compression,
                                                      nvcomp_args.compressed_data_ptrs.data(),
                                                      nvcomp_args.compressed_data_sizes.data(),
                                                      nvcomp_args.uncompressed_data_sizes.data(),
                                                      actual_uncompressed_data_sizes.data(),
                                                      num_chunks,
                                                      scratch.data(),
                                                      scratch.size(),
                                                      nvcomp_args.uncompressed_data_ptrs.data(),
                                                      nvcomp_statuses.data(),
                                                      stream.value());
  CUDF_EXPECTS(nvcomp_status == nvcompStatus_t::nvcompSuccess, "unable to perform decompression");

  convert_status(nvcomp_statuses, actual_uncompressed_data_sizes, statuses, stream);
}
}  // namespace cudf::io::nvcomp
