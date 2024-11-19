/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <cudf/io/types.hpp>
#include <cudf/utilities/span.hpp>

#include <memory>
#include <string>
#include <vector>

using cudf::host_span;

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
 * @brief GZIP header flags
 * See https://tools.ietf.org/html/rfc1952
 */
namespace GZIPHeaderFlag {
constexpr uint8_t ftext    = 0x01;  // ASCII text hint
constexpr uint8_t fhcrc    = 0x02;  // Header CRC present
constexpr uint8_t fextra   = 0x04;  // Extra fields present
constexpr uint8_t fname    = 0x08;  // Original file name present
constexpr uint8_t fcomment = 0x10;  // Comment present
};                                  // namespace GZIPHeaderFlag

}  // namespace io::detail
}  // namespace CUDF_EXPORT cudf
