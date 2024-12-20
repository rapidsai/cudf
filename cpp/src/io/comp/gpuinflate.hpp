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

#include "io/comp/comp.hpp"

#include <cudf/io/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

namespace cudf::io::detail {

enum class gzip_header_included { NO, YES };

/**
 * @brief Interface for decompressing GZIP-compressed data
 *
 * Multiple, independent chunks of compressed data can be decompressed by using
 * separate input/output/status for each chunk.
 *
 * @param[in] inputs List of input buffers
 * @param[out] outputs List of output buffers
 * @param[out] results List of output status structures
 * @param[in] parse_hdr Whether or not to parse GZIP header
 * @param[in] stream CUDA stream to use
 */
CUDF_EXPORT
void gpuinflate(device_span<device_span<uint8_t const> const> inputs,
                device_span<device_span<uint8_t> const> outputs,
                device_span<compression_result> results,
                gzip_header_included parse_hdr,
                rmm::cuda_stream_view stream);

/**
 * @brief Interface for copying uncompressed byte blocks
 *
 * @param[in] inputs List of input buffers
 * @param[out] outputs List of output buffers
 * @param[in] stream CUDA stream to use
 */
void gpu_copy_uncompressed_blocks(device_span<device_span<uint8_t const> const> inputs,
                                  device_span<device_span<uint8_t> const> outputs,
                                  rmm::cuda_stream_view stream);

/**
 * @brief Interface for decompressing Snappy-compressed data
 *
 * Multiple, independent chunks of compressed data can be decompressed by using
 * separate input/output/status for each chunk.
 *
 * @param[in] inputs List of input buffers
 * @param[out] outputs List of output buffers
 * @param[out] results List of output status structures
 * @param[in] stream CUDA stream to use
 */
CUDF_EXPORT
void gpu_unsnap(device_span<device_span<uint8_t const> const> inputs,
                device_span<device_span<uint8_t> const> outputs,
                device_span<compression_result> results,
                rmm::cuda_stream_view stream);

/**
 * @brief Computes the size of temporary memory for Brotli decompression
 *
 * @param[in] max_num_inputs The maximum number of compressed input chunks
 *
 * @return The size in bytes of required temporary memory
 */
CUDF_EXPORT
size_t get_gpu_debrotli_scratch_size(int max_num_inputs = 0);

/**
 * @brief Interface for decompressing Brotli-compressed data
 *
 * Multiple, independent chunks of compressed data can be decompressed by using
 * separate input/output/status pairs for each chunk.
 *
 * @param[in] inputs List of input buffers
 * @param[out] outputs List of output buffers
 * @param[out] results List of output status structures
 * @param[in] scratch Temporary memory for intermediate work
 * @param[in] scratch_size Size in bytes of the temporary memory
 * @param[in] stream CUDA stream to use
 */
CUDF_EXPORT
void gpu_debrotli(device_span<device_span<uint8_t const> const> inputs,
                  device_span<device_span<uint8_t> const> outputs,
                  device_span<compression_result> results,
                  void* scratch,
                  size_t scratch_size,
                  rmm::cuda_stream_view stream);

/**
 * @brief Interface for compressing data with Snappy
 *
 * Multiple, independent chunks of compressed data can be compressed by using
 * separate input/output/status for each chunk.
 *
 * @param[in] inputs List of input buffers
 * @param[out] outputs List of output buffers
 * @param[out] results List of output status structures
 * @param[in] stream CUDA stream to use
 */
void gpu_snap(device_span<device_span<uint8_t const> const> inputs,
              device_span<device_span<uint8_t> const> outputs,
              device_span<compression_result> results,
              rmm::cuda_stream_view stream);

/**
 * @brief Aggregate results of compression into a single statistics object.
 *
 * @param inputs List of uncompressed input buffers
 * @param results List of compression results
 * @param stream CUDA stream to use
 * @return writer_compression_statistics
 */
[[nodiscard]] writer_compression_statistics collect_compression_statistics(
  device_span<device_span<uint8_t const> const> inputs,
  device_span<compression_result const> results,
  rmm::cuda_stream_view stream);

}  // namespace cudf::io::detail
