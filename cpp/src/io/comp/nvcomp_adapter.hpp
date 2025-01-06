/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "io/comp/comp.hpp"

#include <cudf/io/nvcomp_adapter.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <optional>

namespace cudf::io::detail::nvcomp {
/**
 * @brief Device batch decompression of given type.
 *
 * @param[in] compression Compression type
 * @param[in] inputs List of input buffers
 * @param[out] outputs List of output buffers
 * @param[out] results List of output status structures
 * @param[in] max_uncomp_chunk_size Maximum size of any single uncompressed chunk
 * @param[in] max_total_uncomp_size Maximum total size of uncompressed data
 * @param[in] stream CUDA stream to use
 */
void batched_decompress(compression_type compression,
                        device_span<device_span<uint8_t const> const> inputs,
                        device_span<device_span<uint8_t> const> outputs,
                        device_span<compression_result> results,
                        size_t max_uncomp_chunk_size,
                        size_t max_total_uncomp_size,
                        rmm::cuda_stream_view stream);

/**
 * @brief Return the amount of temporary space required in bytes for a given decompression
 * operation.
 *
 * The size returned reflects the size of the scratch buffer to be passed to
 * `batched_decompress_async`
 *
 * @param[in] compression Compression type
 * @param[in] num_chunks The number of decompression chunks to be processed
 * @param[in] max_uncomp_chunk_size Maximum size of any single uncompressed chunk
 * @param[in] max_total_uncomp_size Maximum total size of uncompressed data
 * @returns The total required size in bytes
 */
size_t batched_decompress_temp_size(compression_type compression,
                                    size_t num_chunks,
                                    size_t max_uncomp_chunk_size,
                                    size_t max_total_uncomp_size);

/**
 * @brief Gets the maximum size any chunk could compress to in the batch.
 *
 * @param compression Compression type
 * @param max_uncomp_chunk_size Size of the largest uncompressed chunk in the batch
 */
[[nodiscard]] size_t compress_max_output_chunk_size(compression_type compression,
                                                    uint32_t max_uncomp_chunk_size);

/**
 * @brief Gets input and output alignment requirements for the given compression type.
 *
 * @param compression Compression type
 * @returns required alignment
 */
[[nodiscard]] size_t required_alignment(compression_type compression);

/**
 * @brief Maximum size of uncompressed chunks that can be compressed with nvCOMP.
 *
 * @param compression Compression type
 * @returns maximum chunk size
 */
[[nodiscard]] std::optional<size_t> compress_max_allowed_chunk_size(compression_type compression);

/**
 * @brief Device batch compression of given type.
 *
 * @param[in] compression Compression type
 * @param[in] inputs List of input buffers
 * @param[out] outputs List of output buffers
 * @param[out] results List of output status structures
 * @param[in] stream CUDA stream to use
 */
void batched_compress(compression_type compression,
                      device_span<device_span<uint8_t const> const> inputs,
                      device_span<device_span<uint8_t> const> outputs,
                      device_span<compression_result> results,
                      rmm::cuda_stream_view stream);

}  // namespace cudf::io::detail::nvcomp
