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

#pragma once

#include "gpuinflate.hpp"

#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf::io::nvcomp {

enum class compression_type { SNAPPY, ZSTD, DEFLATE };

/**
 * @brief Device batch decompression of given type.
 *
 * @param[in] compression Compression type
 * @param[in] inputs List of input buffers
 * @param[out] outputs List of output buffers
 * @param[out] statuses List of output status structures
 * @param[in] max_uncomp_chunk_size maximum size of uncompressed chunk
 * @param[in] max_total_uncomp_size maximum total size of uncompressed data
 * @param[in] stream CUDA stream to use
 */
void batched_decompress(compression_type compression,
                        device_span<device_span<uint8_t const> const> inputs,
                        device_span<device_span<uint8_t> const> outputs,
                        device_span<decompress_status> statuses,
                        size_t max_uncomp_chunk_size,
                        size_t max_total_uncomp_size,
                        rmm::cuda_stream_view stream);

/**
 * @brief Gets the maximum size any chunk could compress to in the batch.
 *
 * @param compression Compression type
 * @param max_uncomp_chunk_size Size of the largest uncompressed chunk in the batch
 */
size_t batched_compress_get_max_output_chunk_size(compression_type compression,
                                                  uint32_t max_uncomp_chunk_size);

/**
 * @brief Device batch compression of given type.
 *
 * @param[in] compression Compression type
 * @param[in] inputs List of input buffers
 * @param[out] outputs List of output buffers
 * @param[out] statuses List of output status structures
 * @param[in] max_uncomp_chunk_size Size of the largest uncompressed chunk in the batch
 * @param[in] stream CUDA stream to use
 */
void batched_compress(compression_type compression,
                      device_span<device_span<uint8_t const> const> inputs,
                      device_span<device_span<uint8_t> const> outputs,
                      device_span<decompress_status> statuses,
                      uint32_t max_uncomp_chunk_size,
                      rmm::cuda_stream_view stream);

}  // namespace cudf::io::nvcomp
