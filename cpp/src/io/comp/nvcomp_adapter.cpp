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
auto batched_decompress_get_temp_size(compression_type type, Args&&... args)
{
  switch (type) {
    case compression_type::SNAPPY:
      return nvcompBatchedSnappyDecompressGetTempSize(std::forward<Args>(args)...);
    default: CUDF_FAIL("Unsupported compression type");
  }
};

template <typename... Args>
auto batched_decompress_async(compression_type type, Args&&... args)
{
  switch (type) {
    case compression_type::SNAPPY:
      return nvcompBatchedSnappyDecompressAsync(std::forward<Args>(args)...);
    default: CUDF_FAIL("Unsupported compression type");
  }
};

size_t get_temp_size(compression_type type, size_t num_chunks, size_t max_uncomp_chunk_size)
{
  size_t temp_size = 0;
  nvcompStatus_t nvcomp_status =
    batched_decompress_get_temp_size(type, num_chunks, max_uncomp_chunk_size, &temp_size);
  CUDF_EXPECTS(nvcomp_status == nvcompStatus_t::nvcompSuccess,
               "Unable to get scratch size for decompression");

  return temp_size;
}

void batched_decompress(compression_type type,
                        device_span<device_decompress_input const> comp_in,
                        device_span<decompress_status> comp_stat,
                        size_t max_uncomp_chunk_size,
                        rmm::cuda_stream_view stream)
{
  auto const num_chunks = comp_in.size();

  // cuDF inflate inputs converted to nvcomp inputs
  auto const inputs = create_batched_inputs(comp_in, stream);
  // Analogous to comp_stat.bytes_written
  rmm::device_uvector<size_t> actual_uncompressed_data_sizes(num_chunks, stream);
  // Convertible to comp_stat.status
  rmm::device_uvector<nvcompStatus_t> statuses(num_chunks, stream);
  // Temporary space required for decompression
  rmm::device_buffer scratch(get_temp_size(type, num_chunks, max_uncomp_chunk_size), stream);
  auto const nvcomp_status = batched_decompress_async(type,
                                                      inputs.compressed_data_ptrs.data(),
                                                      inputs.compressed_data_sizes.data(),
                                                      inputs.uncompressed_data_sizes.data(),
                                                      actual_uncompressed_data_sizes.data(),
                                                      num_chunks,
                                                      scratch.data(),
                                                      scratch.size(),
                                                      inputs.uncompressed_data_ptrs.data(),
                                                      statuses.data(),
                                                      stream.value());
  CUDF_EXPECTS(nvcomp_status == nvcompStatus_t::nvcompSuccess, "unable to perform decompression");

  convert_status(statuses, actual_uncompressed_data_sizes, comp_stat, stream);
}
}  // namespace cudf::io::nvcomp
