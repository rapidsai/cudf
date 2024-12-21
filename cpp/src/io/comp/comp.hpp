/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
 * @brief Compresses a system memory buffer.
 *
 * @param compression Type of compression of the input data
 * @param src         Decompressed host buffer
 * @param stream      CUDA stream used for device memory operations and kernel launches
 *
 * @return Vector containing the Compressed output
 */
std::vector<uint8_t> compress(compression_type compression,
                              host_span<uint8_t const> src,
                              rmm::cuda_stream_view stream);

}  // namespace io::detail
}  // namespace CUDF_EXPORT cudf
