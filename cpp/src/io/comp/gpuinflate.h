/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include <cstdint>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace io {
/**
 * @brief Input parameters for the decompression interface
 */
struct device_decompress_input {
  const void* srcDevice;
  uint64_t srcSize;
  void* dstDevice;
  uint64_t dstSize;
};

/**
 * @brief Output parameters for the decompression interface
 */
struct decompress_status {
  uint64_t bytes_written;
  uint32_t status;
  uint32_t reserved;
};

/**
 * @brief Interface for decompressing GZIP-compressed data
 *
 * Multiple, independent chunks of compressed data can be decompressed by using
 * separate device_decompress_input/decompress_status pairs for each chunk.
 *
 * @param[in] inputs List of input argument structures
 * @param[out] outputs List of output status structures
 * @param[in] count Number of input/output structures
 * @param[in] parse_hdr Whether or not to parse GZIP header
 * @param[in] stream CUDA stream to use
 */
cudaError_t gpuinflate(device_decompress_input* inputs,
                       decompress_status* outputs,
                       int count,
                       int parse_hdr,
                       rmm::cuda_stream_view stream);

/**
 * @brief Interface for copying uncompressed byte blocks
 *
 * @param[in] inputs List of input argument structures
 * @param[in] count Number of input structures
 * @param[in] stream CUDA stream to use
 */
cudaError_t gpu_copy_uncompressed_blocks(device_decompress_input* inputs,
                                         int count,
                                         rmm::cuda_stream_view stream);

/**
 * @brief Interface for decompressing Snappy-compressed data
 *
 * Multiple, independent chunks of compressed data can be decompressed by using
 * separate device_decompress_input/decompress_status pairs for each chunk.
 *
 * @param[in] inputs List of input argument structures
 * @param[out] outputs List of output status structures
 * @param[in] count Number of input/output structures
 * @param[in] stream CUDA stream to use
 */
cudaError_t gpu_unsnap(device_decompress_input* inputs,
                       decompress_status* outputs,
                       int count,
                       rmm::cuda_stream_view stream);

/**
 * @brief Computes the size of temporary memory for Brotli decompression
 *
 * @param[in] max_num_inputs The maximum number of compressed input chunks
 *
 * @return The size in bytes of required temporary memory
 */
size_t get_gpu_debrotli_scratch_size(int max_num_inputs = 0);

/**
 * @brief Interface for decompressing Brotli-compressed data
 *
 * Multiple, independent chunks of compressed data can be decompressed by using
 * separate device_decompress_input/decompress_status pairs for each chunk.
 *
 * @param[in] inputs List of input argument structures
 * @param[out] outputs List of output status structures
 * @param[in] scratch Temporary memory for intermediate work
 * @param[in] scratch_size Size in bytes of the temporary memory
 * @param[in] count Number of input/output structures
 * @param[in] stream CUDA stream to use
 */
cudaError_t gpu_debrotli(device_decompress_input* inputs,
                         decompress_status* outputs,
                         void* scratch,
                         size_t scratch_size,
                         int count,
                         rmm::cuda_stream_view stream);

/**
 * @brief Interface for compressing data with Snappy
 *
 * Multiple, independent chunks of compressed data can be compressed by using
 * separate device_decompress_input/decompress_status pairs for each chunk.
 *
 * @param[in] inputs List of input argument structures
 * @param[out] outputs List of output status structures
 * @param[in] count Number of input/output structures
 * @param[in] stream CUDA stream to use
 */
cudaError_t gpu_snap(device_decompress_input* inputs,
                     decompress_status* outputs,
                     int count,
                     rmm::cuda_stream_view stream);

}  // namespace io
}  // namespace cudf
