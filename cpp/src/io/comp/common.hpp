/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cstddef>
#include <cstdint>
#include <string>

namespace cudf::io::detail {

/**
 * @brief The size used for padding a data buffer's size to a multiple of the padding.
 *
 * Padding is necessary for input/output buffers of several compression/decompression kernels
 * (inflate_kernel and nvcomp snappy). Such kernels operate on aligned data pointers, which require
 * padding to the buffers so that the pointers can shift along the address space to satisfy their
 * alignment requirement.
 *
 * In the meantime, it is not entirely clear why such padding is needed. We need to further
 * investigate and implement a better fix rather than just padding the buffer.
 * See https://github.com/rapidsai/cudf/issues/13605.
 */
constexpr std::size_t BUFFER_PADDING_MULTIPLE{8};

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

[[nodiscard]] std::string compression_type_name(compression_type compression);

}  // namespace cudf::io::detail
