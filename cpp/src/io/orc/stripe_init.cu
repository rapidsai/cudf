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

#include "orc_gpu.h"

#if (__CUDACC_VER_MAJOR__ >= 9)
#define SHFL0(v)    __shfl_sync(~0, v, 0)
#define SHFL(v, t)  __shfl_sync(~0, v, t)
#define SYNCWARP()  __syncwarp()
#define BALLOT(v)   __ballot_sync(~0, v)
#else
#define SHFL0(v)    __shfl(v, 0)
#define SHFL(v, t)  __shfl(v, t)
#define SYNCWARP()
#define BALLOT(v)   __ballot(v)
#endif

namespace orc { namespace gpu {


struct compressed_stream_s
{
    CompressedStreamInfo info;
};

// blockDim {128,1,1}
extern "C" __global__ void __launch_bounds__(128, 8)
gpuParseCompressedStripeData(CompressedStreamInfo *strm_info, int32_t num_streams, uint32_t block_size, uint32_t log2maxcr)
{
    __shared__ compressed_stream_s strm_g[4];

    volatile compressed_stream_s * const s = &strm_g[threadIdx.x >> 5];
    int strm_id = blockIdx.x * 4 + (threadIdx.x >> 5);
    int t = threadIdx.x & 0x1f;

    if (strm_id < num_streams && t < sizeof(CompressedStreamInfo) / sizeof(uint32_t))
    {
        // NOTE: Assumes that sizeof(CompressedStreamInfo) <= 128
        ((uint32_t *)&s->info)[t] = ((const uint32_t *)&strm_info[strm_id])[t];
    }
    __syncthreads();
    if (strm_id < num_streams)
    {
        // Walk through the compressed blocks
        const uint8_t *cur = s->info.compressed_data;
        const uint8_t *end = cur + s->info.compressed_data_size;
        gpu_inflate_input_s *decctl = s->info.decctl;
        uint8_t *uncompressed = s->info.uncompressed_data;
        bool setup_decompression = (decctl != NULL);
        size_t max_uncompressed_size = 0;
        uint32_t num_compressed_blocks = 0;
        while (cur + 3 < end)
        {
            uint32_t block_len = SHFL0((t == 0) ? cur[0] | (cur[1] << 8) | (cur[2] << 16) : 0);
            uint32_t is_uncompressed = block_len & 1;
            uint32_t uncompressed_size;
            block_len >>= 1;
            cur += 3;
            if (block_len > block_size || cur + block_len > end)
            {
                // Fatal
                num_compressed_blocks = 0;
                max_uncompressed_size = 0;
                break;
            }
            // TBD: For some codecs like snappy, it wouldn't be too difficult to get the actual uncompressed size and avoid waste due to block size alignment
            // For now, rely on the max compression ratio to limit waste for the most extreme cases (small single-block streams)
            uncompressed_size = (is_uncompressed) ? block_len : (block_len < (block_size >> log2maxcr)) ? block_len << log2maxcr : block_size;
            if (setup_decompression)
            {
                if (is_uncompressed)
                {
                    // Copy the uncompressed data to output (not very efficient, but should only occur for small blocks and probably faster than decompression)
                    if (max_uncompressed_size + uncompressed_size <= s->info.max_uncompressed_size)
                    {
                        for (int i = t; i < uncompressed_size; i += 32)
                        {
                            uncompressed[max_uncompressed_size + i] = cur[i];
                        }
                    }
                }
                else if (!t && num_compressed_blocks < s->info.max_compressed_blocks)
                {
                    decctl[num_compressed_blocks].srcDevice = const_cast<uint8_t *>(cur);
                    decctl[num_compressed_blocks].srcSize = block_len;
                    decctl[num_compressed_blocks].dstDevice = uncompressed + max_uncompressed_size;
                    decctl[num_compressed_blocks].dstSize = uncompressed_size;
                }
            }
            cur += block_len;
            max_uncompressed_size += uncompressed_size;
            num_compressed_blocks += 1 - is_uncompressed;
        }
        if (!t)
        {
            s->info.num_compressed_blocks = num_compressed_blocks;
            s->info.max_uncompressed_size = max_uncompressed_size;
        }
    }

    __syncthreads();
    if (strm_id < num_streams && t < sizeof(CompressedStreamInfo) / sizeof(uint32_t))
    {
        // NOTE: Assumes that sizeof(CompressedStreamInfo) <= 128
        ((uint32_t *)&strm_info[strm_id])[t] = ((uint32_t *)&s->info)[t];
    }
}


// blockDim {128,1,1}
extern "C" __global__ void __launch_bounds__(128, 8)
gpuPostDecompressionReassemble(CompressedStreamInfo *strm_info, int32_t num_streams)
{
    __shared__ compressed_stream_s strm_g[4];

    volatile compressed_stream_s * const s = &strm_g[threadIdx.x >> 5];
    int strm_id = blockIdx.x * 4 + (threadIdx.x >> 5);
    int t = threadIdx.x & 0x1f;

    if (strm_id < num_streams && t < sizeof(CompressedStreamInfo) / sizeof(uint32_t))
    {
        // NOTE: Assumes that sizeof(CompressedStreamInfo) <= 128
        ((uint32_t *)&s->info)[t] = ((const uint32_t *)&strm_info[strm_id])[t];
    }
    __syncthreads();
    if (strm_id < num_streams && s->info.max_compressed_blocks > 0 && s->info.max_uncompressed_size > 0)
    {
        // Walk through the compressed blocks
        const uint8_t *cur = s->info.compressed_data;
        const uint8_t *end = cur + s->info.compressed_data_size;
        const gpu_inflate_input_s *dec_in = s->info.decctl;
        const gpu_inflate_status_s *dec_out = s->info.decstatus;
        uint8_t *uncompressed_actual = s->info.uncompressed_data;
        uint8_t *uncompressed_estimated = uncompressed_actual;
        uint32_t num_compressed_blocks = 0;
        uint32_t max_compressed_blocks = min(s->info.num_compressed_blocks, s->info.max_compressed_blocks);

        while (cur + 3 < end)
        {
            uint32_t block_len = SHFL0((t == 0) ? cur[0] | (cur[1] << 8) | (cur[2] << 16) : 0);
            uint32_t is_uncompressed = block_len & 1;
            uint32_t uncompressed_size_est, uncompressed_size_actual;
            block_len >>= 1;
            cur += 3;
            if (cur + block_len > end)
            {
                break;
            }
            if (is_uncompressed)
            {
                uncompressed_size_est = block_len;
                uncompressed_size_actual = block_len;
            }
            else
            {
                if (num_compressed_blocks > max_compressed_blocks)
                {
                    break;
                }
                if (SHFL0((t == 0) ? dec_out[num_compressed_blocks].status : 0) != 0)
                {
                    // Decompression failed, not much point in doing anything else
                    break;
                }
                uncompressed_size_est = SHFL0((t == 0) ? *(const uint32_t *)&dec_in[num_compressed_blocks].dstSize : 0);
                uncompressed_size_actual = SHFL0((t == 0) ? *(const uint32_t *)&dec_out[num_compressed_blocks].bytes_written : 0);
            }
            // In practice, this should never happen with a well-behaved writer, as we would expect the uncompressed size to always be equal to
            // the compression block size except for the last block
            if (uncompressed_actual < uncompressed_estimated)
            {
                // warp-level memmove
                for (int i = t; i < (int)uncompressed_size_actual; i += 32)
                {
                    uncompressed_actual[i] = uncompressed_estimated[i];
                }
            }
            cur += block_len;
            num_compressed_blocks += 1 - is_uncompressed;
            uncompressed_estimated += uncompressed_size_est;
            uncompressed_actual += uncompressed_size_actual;
        }
        // Update info with actual uncompressed size
        if (!t)
        {
            size_t total_uncompressed_size = uncompressed_actual - s->info.uncompressed_data;
            // Set uncompressed size to zero if there were any errors
            strm_info[strm_id].max_uncompressed_size = (num_compressed_blocks == s->info.num_compressed_blocks) ? total_uncompressed_size : 0;
        }
    }
}


cudaError_t __host__ ParseCompressedStripeData(CompressedStreamInfo *strm_info, int32_t num_streams, uint32_t compression_block_size, uint32_t log2maxcr, cudaStream_t stream)
{
    dim3 dim_block(128, 1);
    dim3 dim_grid((num_streams + 3) >> 2, 1); // 1 stream per warp, 4 warps per block
    gpuParseCompressedStripeData <<< dim_grid, dim_block, 0, stream >>>(strm_info, num_streams, compression_block_size, log2maxcr);
    return cudaSuccess;
}


cudaError_t __host__ PostDecompressionReassemble(CompressedStreamInfo *strm_info, int32_t num_streams, cudaStream_t stream)
{
    dim3 dim_block(128, 1);
    dim3 dim_grid((num_streams + 3) >> 2, 1); // 1 stream per warp, 4 warps per block
    gpuPostDecompressionReassemble <<< dim_grid, dim_block, 0, stream >>>(strm_info, num_streams);
    return cudaSuccess;
}


};}; // orc::gpu namespace
