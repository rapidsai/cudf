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

#include "gpuinflate.h"

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

#if (__CUDA_ARCH__ >= 700)
#define NANOSLEEP(d)  __nanosleep(d)
#else
#define NANOSLEEP(d)  clock()
#endif

#define LOG2_BATCH_SIZE     4
#define BATCH_SIZE          (1 << LOG2_BATCH_SIZE)
#define LOG2_BATCH_COUNT    2
#define BATCH_COUNT         (1 << LOG2_BATCH_COUNT)


struct unsnap_queue_s
{
    int32_t batch_len[BATCH_COUNT];         // <0: exit, 0:finished/unfilled batch, >0: entries in batch
    uint8_t wlen[BATCH_COUNT * BATCH_SIZE]; // warp_length*2 + is_literal
    uint32_t offset[BATCH_COUNT * BATCH_SIZE];
};


struct unsnap_state_s
{
    const uint8_t *cur;
    const uint8_t *end;
    uint8_t *dst;
    uint32_t uncompressed_size;
    int32_t error;
    volatile unsnap_queue_s q;
    gpu_inflate_input_s in;
};


__device__ uint32_t snappy_decode_symbols(unsnap_state_s *s)
{
    const uint8_t *bs = s->cur;
    uint32_t cur = 0;
    uint32_t end = (uint32_t)min((size_t)(s->end - bs), (size_t)0xffffffffu);
    uint32_t bytes_left = s->uncompressed_size;
    uint32_t cur_len = 0;
    uint32_t offset = 0;
    uint32_t is_literal = 0;
    uint32_t dst_pos = 0;
    int32_t b = 0;
    while (bytes_left > 0)
    {
        uint32_t wlen;

        if (cur_len == 0)
        {
            if (cur >= end)
                break;
            cur_len = bs[cur++];
            if (cur_len & 2)
            {
                // xxxxxx1x: copy with 6-bit length, 2-byte or 4-byte offset
                if (2 * (1 + (cur_len & 1)) > end - cur)
                    break;
                offset = bs[cur] + (bs[cur+1] << 8);
                cur += 2;
                if (cur_len & 1) // 4-byte offset
                {
                    offset |= (bs[cur] << 16) | (bs[cur+1] << 24);
                    cur += 2;
                }
                cur_len = (cur_len >> 2) + 1;
                is_literal = 0;
                if (offset > dst_pos)
                    break;
            }
            else
            {
                if (cur_len & 1)
                {
                    // xxxxxx01.oooooooo: copy with 3-bit length, 11-bit offset
                    if (cur >= end)
                        break;
                    offset = ((cur_len & 0xe0) << 3) | bs[cur++];
                    cur_len = ((cur_len >> 2) & 7) + 4;
                    is_literal = 0;
                    if (offset > dst_pos)
                        break;
                }
                else
                {
                    // xxxxxx00: literal
                    cur_len >>= 2;
                    if (cur_len >= 60)
                    {
                        uint32_t num_bytes = cur_len - 59;
                        if (num_bytes >= end - cur)
                            break;
                        cur_len = bs[cur++];
                        if (num_bytes > 1)
                        {
                            cur_len |= bs[cur++] << 8;
                            if (num_bytes > 2)
                            {
                                cur_len |= bs[cur++] << 16;
                                if (num_bytes > 3)
                                {
                                    cur_len |= bs[cur++] << 24;
                                    if (cur_len >= end)
                                        break;
                                }
                            }
                        }
                    }
                    cur_len += 1;
                    offset = cur;
                    if (cur_len > end - cur)
                        break;
                    cur += cur_len;
                    is_literal = 1;
                }
            }
            dst_pos += cur_len;
        }
        wlen = min(cur_len, 32u);
        cur_len -= wlen;
        if (bytes_left < wlen)
        {
            break;
        }
        bytes_left -= wlen;
        s->q.wlen[b] = wlen * 2 + is_literal;
        s->q.offset[b] = offset;
        b = (b + 1) & (BATCH_COUNT * BATCH_SIZE - 1);
    }
    return bytes_left;
}


// blockDim {64,2,1}
extern "C" __global__ void __launch_bounds__(128)
unsnap_kernel(gpu_inflate_input_s *inputs, gpu_inflate_status_s *outputs, int count)
{
    __shared__ __align__(16) unsnap_state_s state_g[2];

    int t = threadIdx.x;
    unsnap_state_s *s = &state_g[threadIdx.y];
    int strm_id = blockIdx.x * 2 + threadIdx.y;

    if (strm_id < count && t < sizeof(gpu_inflate_input_s) / sizeof(uint32_t))
    {
        reinterpret_cast<uint32_t *>(&s->in)[t] = reinterpret_cast<const uint32_t *>(&inputs[strm_id])[t];
        __threadfence_block();
    }
    if (t < BATCH_COUNT)
    {
        s->q.batch_len[t] = 0;
    }
    __syncthreads();
    if (!t && strm_id < count)
    {
        const uint8_t *cur = reinterpret_cast<const uint8_t *>(s->in.srcDevice);
        const uint8_t *end = cur + s->in.srcSize;
        s->error = 0;
        if (cur < end)
        {
            // Read uncompressed size (varint), limited to 32-bit
            uint32_t uncompressed_size = *cur++;
            if (uncompressed_size > 0x7f)
            {
                uint32_t c = (cur < end) ? *cur++ : 0;
                uncompressed_size = (uncompressed_size & 0x7f) | (c << 7);
                if (uncompressed_size >= (0x80 << 7))
                {
                    c = (cur < end) ? *cur++ : 0;
                    uncompressed_size = (uncompressed_size & ((0x7f << 7) | 0x7f)) | (c << 14);
                    if (uncompressed_size >= (0x80 << 14))
                    {
                        c = (cur < end) ? *cur++ : 0;
                        uncompressed_size = (uncompressed_size & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | (c << 21);
                        if (uncompressed_size >= (0x80 << 21))
                        {
                            c = (cur < end) ? *cur++ : 0;
                            if (c <= 0xf)
                                uncompressed_size = (uncompressed_size & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) | (c << 28);
                            else
                                s->error = -1;
                        }
                    }
                }
            }
            s->uncompressed_size = uncompressed_size;
            s->cur = cur;
            s->end = end;
            if ((cur >= end && uncompressed_size != 0) || (uncompressed_size > s->in.dstSize))
            {
                s->error = -1;
            }
        }
        else
        {
            s->error = -1;
        }
    }
    __syncthreads();
    if (strm_id < count && !s->error)
    {
        if (t < 32)
        {
            // WARP0: decode lengths and offsets
            if (!t)
            {
                snappy_decode_symbols(s);
            }
        }
        else
        {

        }
    }
    __syncthreads();
    if (!t && strm_id < count)
    {
        outputs[strm_id].bytes_written = 0;
        outputs[strm_id].status = s->error;
        outputs[strm_id].reserved = s->uncompressed_size;
    }
}


cudaError_t __host__ gpu_unsnap(gpu_inflate_input_s *inputs, gpu_inflate_status_s *outputs, int count, cudaStream_t stream)
{
    uint32_t count32 = (count > 0) ? count : 0;
    dim3 dim_block(64, 2);      // 2 warps per stream, 2 streams per block
    dim3 dim_grid(count32, 1);  // TODO: Check max grid dimensions vs max expected count

    unsnap_kernel << < dim_grid, dim_block, 0, stream >> >(inputs, outputs, count32);

    return cudaSuccess;
}

