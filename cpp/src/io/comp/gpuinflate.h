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

#ifndef _GPUINFLATE_H_
#define _GPUINFLATE_H_

#include <cstdint>

struct gpu_inflate_input_s
{
    void *srcDevice;
    uint64_t srcSize;
    void *dstDevice;
    uint64_t dstSize;
};

struct gpu_inflate_status_s
{
    uint64_t bytes_written;
    uint32_t status;
    uint32_t reserved;
};

// GZIP decompression
cudaError_t gpuinflate(gpu_inflate_input_s *inputs, gpu_inflate_status_s *outputs, int count = 1, int parse_hdr = 0, cudaStream_t stream = (cudaStream_t)0);

// SNAPPY decompression
cudaError_t gpu_unsnap(gpu_inflate_input_s *inputs, gpu_inflate_status_s *outputs, int count = 1, cudaStream_t stream = (cudaStream_t)0);

// BROTLI decompression
size_t get_gpu_debrotli_scratch_size(int max_num_inputs = 0);
cudaError_t gpu_debrotli(gpu_inflate_input_s *inputs, gpu_inflate_status_s *outputs, void *scratch, size_t scratch_size, int count = 1, cudaStream_t stream = (cudaStream_t)0);

#endif // _GPUINFLATE_H_

