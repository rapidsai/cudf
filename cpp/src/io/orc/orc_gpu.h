/*
* Copyright (c) 2019, NVIDIA CORPORATION.
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

#ifndef __IO_ORC_GPU_H__
#define __IO_ORC_GPU_H__

#include "io/comp/gpuinflate.h"

namespace orc { namespace gpu {

struct CompressedStreamInfo
{
    uint8_t *compressed_data;           // [in] base ptr to compressed stream data
    uint8_t *uncompressed_data;         // [in] base ptr to uncompressed stream data or NULL if not known yet
    size_t compressed_data_size;        // [in] compressed data size for this stream
    gpu_inflate_input_s *decctl;        // [in] base ptr to decompression structure to be filled
    gpu_inflate_status_s *decstatus;    // [in] results of decompression
    uint32_t max_compressed_blocks;     // [in] number of entries in decctl
    uint32_t num_compressed_blocks;     // [out] total number of compressed blocks in this stream
    uint64_t max_uncompressed_size;     // [out] maximum uncompressed data size
};


// stripe_init.cu
cudaError_t ParseCompressedStripeData(CompressedStreamInfo *strm_info, int32_t num_streams, uint32_t compression_block_size, uint32_t log2maxcr = 24, cudaStream_t stream = (cudaStream_t)0);
cudaError_t PostDecompressionReassemble(CompressedStreamInfo *strm_info, int32_t num_streams, cudaStream_t stream = (cudaStream_t)0);


};}; // orc::gpu namespace

#endif // __IO_ORC_GPU_H__
