/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

/**
 * @file codec.hpp
 * @brief cuDF-IO compression and decompression API definitions
 */

#pragma once

#include <cudf/io/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <vector>

namespace CUDF_EXPORT cudf::io {
/**
 * @addtogroup io_codec
 * @{
 * @file
 */

/**
 * @brief Status of a compression/decompression operation.
 */
enum class compression_status : uint8_t {
  SUCCESS,          ///< Successful, output is valid
  FAILURE,          ///< Failed, output is invalid (e.g. input is unsupported in some way)
  SKIPPED,          ///< Operation skipped (if conversion, uncompressed data can be used)
  OUTPUT_OVERFLOW,  ///< Output buffer is too small; operation can succeed with larger output
};

/**
 * @brief Descriptor of compression/decompression result.
 */
struct compression_result {
  uint64_t bytes_written;
  compression_status status;
};

/**
 * @brief Check if compression is supported for the given compression type.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported
 */
[[nodiscard]] bool is_compression_supported(compression_type compression);

/**
 * @brief Check if decompression is supported for the given compression type.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported
 */
[[nodiscard]] bool is_decompression_supported(compression_type compression);

/**
 * @brief Check if host compression is supported for the given compression type.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported on host
 */
[[nodiscard]] bool is_host_compression_supported(compression_type compression);

/**
 * @brief Check if device compression is supported for the given compression type.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported on device
 */
[[nodiscard]] bool is_device_compression_supported(compression_type compression);

/**
 * @brief Check if host decompression is supported for the given compression type.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported on host
 */
[[nodiscard]] bool is_host_decompression_supported(compression_type compression);

/**
 * @brief Check if device decompression is supported for the given compression type.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported on device
 */
[[nodiscard]] bool is_device_decompression_supported(compression_type compression);

/**
 * @brief Compresses a system memory buffer.
 *
 * @param compression Type of compression of the input data
 * @param src Decompressed host buffer
 *
 * @return Vector containing the Compressed output
 */
std::vector<uint8_t> compress(compression_type compression, host_span<uint8_t const> src);

/**
 * @brief Compresses device memory buffers.
 *
 * @param compression Type of compression of the input data
 * @param inputs Device memory buffers to compress
 * @param outputs Device memory buffers to store the compressed output
 * @param results Compression results
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void compress(compression_type compression,
              device_span<device_span<uint8_t const> const> inputs,
              device_span<device_span<uint8_t> const> outputs,
              device_span<compression_result> results,
              rmm::cuda_stream_view stream);

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
 * @param src Compressed host buffer
 * @param dst Destination host span to place decompressed buffer
 *
 * @return Size of decompressed output
 */
size_t decompress(compression_type compression,
                  host_span<uint8_t const> src,
                  host_span<uint8_t> dst);

/**
 * @brief Decompresses device memory buffers.
 *
 * @param compression Type of compression of the output data
 * @param inputs Device memory buffers to decompress
 * @param outputs Device memory buffers to store the decompressed output
 * @param results Compression results
 * @param max_uncomp_chunk_size Maximum size of any single uncompressed chunk
 * @param max_total_uncomp_size Maximum size of the total uncompressed data
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void decompress(compression_type compression,
                device_span<device_span<uint8_t const> const> inputs,
                device_span<device_span<uint8_t> const> outputs,
                device_span<compression_result> results,
                size_t max_uncomp_chunk_size,
                size_t max_total_uncomp_size,
                rmm::cuda_stream_view stream);

/** @} */  // end of group
}  // namespace cudf::io
