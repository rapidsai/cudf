/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.
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

#pragma once

#include "common.hpp"

#include <cudf/io/types.hpp>
#include <cudf/utilities/span.hpp>

#include <vector>

namespace CUDF_EXPORT cudf {
namespace io::detail {

/**
 * @brief Decompresses a system memory buffer.
 *
 * @param compression Type of compression of the input data
 * @param src Compressed host buffer
 *
 * @return Vector containing the Decompressed output
 */
[[nodiscard]] std::vector<uint8_t> decompress(compression_type compression,
                                              host_span<uint8_t const> src);

/**
 * @brief Decompresses a system memory buffer.
 *
 * @param compression Type of compression of the input data
 * @param src         Compressed host buffer
 * @param dst         Destination host span to place decompressed buffer
 * @param stream      CUDA stream used for device memory operations and kernel launches
 *
 * @return Size of decompressed output
 */
size_t decompress(compression_type compression,
                  host_span<uint8_t const> src,
                  host_span<uint8_t> dst,
                  rmm::cuda_stream_view stream);

/**
 * @brief Decompresses device memory buffers.
 *
 * @param compression Type of compression of the output data
 * @param inputs      Device memory buffers to decompress
 * @param outputs     Device memory buffers to store the decompressed output
 * @param results     Compression results
 * @param max_uncomp_chunk_size Maximum size of any single uncompressed chunk
 * @param max_total_uncomp_size Maximum size of the total uncompressed data
 * @param stream      CUDA stream used for device memory operations and kernel launches
 */
void decompress(compression_type compression,
                device_span<device_span<uint8_t const> const> inputs,
                device_span<device_span<uint8_t> const> outputs,
                device_span<compression_result> results,
                size_t max_uncomp_chunk_size,
                size_t max_total_uncomp_size,
                rmm::cuda_stream_view stream);

/**
 * @brief Without actually decompressing the compressed input buffer passed, return the size of
 * decompressed output. If the decompressed size cannot be extracted apriori, return zero.
 *
 * @param compression Type of compression of the input data
 * @param src         Compressed host buffer
 *
 * @return Size of decompressed output
 */
size_t get_uncompressed_size(compression_type compression, host_span<uint8_t const> src);

/**
 * @brief Struct to hold information about decompression.
 *
 * This struct contains details about the decompression process, including
 * the type of compression, the number of pages, the maximum size
 * of a decompressed page, and the total decompressed size.
 */
struct decompression_info {
  compression_type type;
  size_t num_pages;
  size_t max_page_decompressed_size;
  size_t total_decompressed_size;
};

/**
 * @brief Functor which returns total scratch space required based on computed decompression_info
 * data.
 *
 */
[[nodiscard]] size_t get_decompression_scratch_size(decompression_info const& di);

}  // namespace io::detail
}  // namespace CUDF_EXPORT cudf
