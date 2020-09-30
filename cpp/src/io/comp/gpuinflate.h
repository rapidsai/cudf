/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <stdint.h>

namespace cudf {
namespace io {
/**
 * @brief Input parameters for the decompression interface
 **/
struct gpu_inflate_input_s {
  const void *srcDevice;
  uint64_t srcSize;
  void *dstDevice;
  uint64_t dstSize;
};

/**
 * @brief Output parameters for the decompression interface
 **/
struct gpu_inflate_status_s {
  uint64_t bytes_written;
  uint32_t status;
  uint32_t reserved;
};

/**
 * @brief Interface for decompressing GZIP-compressed data
 *
 * Multiple, independent chunks of compressed data can be decompressed by using
 * separate gpu_inflate_input_s/gpu_inflate_status_s pairs for each chunk.
 *
 * @param[in] inputs List of input argument structures
 * @param[out] outputs List of output status structures
 * @param[in] count Number of input/output structures, default 1
 * @param[in] parse_hdr Whether or not to parse GZIP header, default false
 * @param[in] stream CUDA stream to use, default 0
 **/
cudaError_t gpuinflate(gpu_inflate_input_s *inputs,
                       gpu_inflate_status_s *outputs,
                       int count           = 1,
                       int parse_hdr       = 0,
                       cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Interface for copying uncompressed byte blocks
 *
 * @param[in] inputs List of input argument structures
 * @param[in] count Number of input structures, default 1
 * @param[in] stream CUDA stream to use, default 0
 **/
cudaError_t gpu_copy_uncompressed_blocks(gpu_inflate_input_s *inputs,
                                         int count           = 1,
                                         cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Interface for decompressing Snappy-compressed data
 *
 * Multiple, independent chunks of compressed data can be decompressed by using
 * separate gpu_inflate_input_s/gpu_inflate_status_s pairs for each chunk.
 *
 * @param[in] inputs List of input argument structures
 * @param[out] outputs List of output status structures
 * @param[in] count Number of input/output structures, default 1
 * @param[in] stream CUDA stream to use, default 0
 **/
cudaError_t gpu_unsnap(gpu_inflate_input_s *inputs,
                       gpu_inflate_status_s *outputs,
                       int count           = 1,
                       cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Computes the size of temporary memory for Brotli decompression
 *
 * @param[in] max_num_inputs The maximum number of compressed input chunks
 *
 * @return The size in bytes of required temporary memory
 **/
size_t get_gpu_debrotli_scratch_size(int max_num_inputs = 0);

/**
 * @brief Interface for decompressing Brotli-compressed data
 *
 * Multiple, independent chunks of compressed data can be decompressed by using
 * separate gpu_inflate_input_s/gpu_inflate_status_s pairs for each chunk.
 *
 * @param[in] inputs List of input argument structures
 * @param[out] outputs List of output status structures
 * @param[in] scratch Temporary memory for intermediate work
 * @param[in] scratch_size Size in bytes of the temporary memory
 * @param[in] count Number of input/output structures, default 1
 * @param[in] stream CUDA stream to use, default 0
 **/
cudaError_t gpu_debrotli(gpu_inflate_input_s *inputs,
                         gpu_inflate_status_s *outputs,
                         void *scratch,
                         size_t scratch_size,
                         int count           = 1,
                         cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Interface for compressing data with Snappy
 *
 * Multiple, independent chunks of compressed data can be compressed by using
 * separate gpu_inflate_input_s/gpu_inflate_status_s pairs for each chunk.
 *
 * @param[in] inputs List of input argument structures
 * @param[out] outputs List of output status structures
 * @param[in] count Number of input/output structures, default 1
 * @param[in] stream CUDA stream to use, default 0
 **/
cudaError_t gpu_snap(gpu_inflate_input_s *inputs,
                     gpu_inflate_status_s *outputs,
                     int count           = 1,
                     cudaStream_t stream = (cudaStream_t)0);

}  // namespace io
}  // namespace cudf
