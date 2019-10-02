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

// Not supporting streams longer than this (not what snappy is intended for)
#define SNAPPY_MAX_STREAM_SIZE	0xefffffff

#define LOG2_BATCH_SIZE     5
#define BATCH_SIZE          (1 << LOG2_BATCH_SIZE)
#define LOG2_BATCH_COUNT    2
#define BATCH_COUNT         (1 << LOG2_BATCH_COUNT)
#define LOG2_PREFETCH_SIZE  9
#define PREFETCH_SIZE       (1 << LOG2_PREFETCH_SIZE)   // 512B, in 32B chunks

struct unsnap_batch_s
{
    int32_t len;        // 1..64 = Number of bytes to copy at given offset, 65..97 = Number of literal bytes
    uint32_t offset;    // copy distance or absolute literal offset in byte stream
};


struct unsnap_queue_s
{
    uint32_t prefetch_wrpos;
    uint32_t prefetch_rdpos;
    int32_t prefetch_end;
    int32_t batch_len[BATCH_COUNT];     // Length of each batch - <0:end, 0:not ready, >0:symbol count
    unsnap_batch_s batch[BATCH_COUNT * BATCH_SIZE];
    uint8_t buf[PREFETCH_SIZE];         // Prefetch buffer
};


struct unsnap_state_s
{
    const uint8_t *base;
    const uint8_t *end;
    uint32_t uncompressed_size;
    uint32_t bytes_left;
    int32_t error;
    volatile unsnap_queue_s q;
    gpu_inflate_input_s in;
};


__device__ void snappy_prefetch_bytestream(unsnap_state_s *s, int t)
{
    const uint8_t *base = s->base;
    uint32_t end = (uint32_t)min((size_t)(s->end - base), (size_t)0xffffffffu);
    uint32_t align_bytes = (uint32_t)(0x20 - (0x1f & reinterpret_cast<uintptr_t>(base)));
    int32_t pos = min(align_bytes, end);
    int32_t blen;
    // Start by prefetching up to the next a 32B-aligned location
    if (t < pos)
    {
        s->q.buf[t] = base[t];
    }
    blen = 0;
    do
    {
        SYNCWARP();
        if (!t)
        {
            uint32_t minrdpos;
            s->q.prefetch_wrpos = pos;
            minrdpos = pos - min(pos, PREFETCH_SIZE - 32u);
            blen = (int)min(32u, end - pos);
            for (;;)
            {
                uint32_t rdpos = s->q.prefetch_rdpos;
                if (rdpos >= minrdpos)
                    break;
                if (s->q.prefetch_end)
                {
                    blen = 0;
                    break;
                }
                NANOSLEEP(100);
            }
        }
        blen = SHFL0(blen);
        if (t < blen)
        {
            s->q.buf[(pos + t) & (PREFETCH_SIZE - 1)] = base[pos + t];
        }
        pos += blen;
    } while (blen > 0);
}


#define READ_BYTE(pos)  s->q.buf[(pos) & (PREFETCH_SIZE-1)]

__device__ uint32_t snappy_decode_symbols(unsnap_state_s *s)
{
    uint32_t cur = 0;
    uint32_t end = (uint32_t)min((size_t)(s->end - s->base), (size_t)0xffffffffu);
    uint32_t bytes_left = s->uncompressed_size;
    uint32_t dst_pos = 0;
    int32_t batch = 0;

    for (;;)
    {
        volatile unsnap_batch_s *b = &s->q.batch[batch * BATCH_SIZE];
        int32_t batch_len = 0;
        uint32_t min_wrpos;

        // Wait for prefetcher
        s->q.prefetch_rdpos = cur;
        min_wrpos = min(cur + 5 * BATCH_SIZE, end);
        #pragma unroll(1) // We don't want unrolling here
        while (s->q.prefetch_wrpos < min_wrpos)
        {
            NANOSLEEP(50);
        }

        while (bytes_left > 0)
        {
            uint32_t blen, offset;
            uint8_t b0 = READ_BYTE(cur);
            if (b0 & 3)
            {
                uint8_t b1 = READ_BYTE(cur+1);
                if (!(b0 & 2))
                {
                    // xxxxxx01.oooooooo: copy with 3-bit length, 11-bit offset
                    offset = ((b0 & 0xe0) << 3) | b1;
                    blen = ((b0 >> 2) & 7) + 4;
                    cur += 2;
                }
                else
                {
                    // xxxxxx1x: copy with 6-bit length, 2-byte or 4-byte offset
                    offset = b1 | (READ_BYTE(cur+2) << 8);
                    if (b0 & 1) // 4-byte offset
                    {
                        offset |= (READ_BYTE(cur+3) << 16) | (READ_BYTE(cur + 4) << 24);
                        cur += 5;
                    }
                    else
                    {
                        cur += 3;
                    }
                    blen = (b0 >> 2) + 1;
                }
                dst_pos += blen;
                if (offset - 1u >= dst_pos ||  bytes_left < blen)
                    break;
                bytes_left -= blen;
            }
            else
            {
                // xxxxxx00: literal
                blen = b0 >> 2;
                if (blen >= 60)
                {
                    uint32_t num_bytes = blen - 59;
                    blen = READ_BYTE(cur + 1);
                    if (num_bytes > 1)
                    {
                        blen |= READ_BYTE(cur + 2) << 8;
                        if (num_bytes > 2)
                        {
                            blen |= READ_BYTE(cur + 3) << 16;
                            if (num_bytes > 3)
                            {
                                blen |= READ_BYTE(cur + 4) << 24;
                                if (blen >= end)
                                    break;
                            }
                        }
                    }
                    cur += num_bytes;
                }
                cur += 1;
                blen += 1;
                offset = cur;
                cur += blen;
                // Wait for prefetcher
                s->q.prefetch_rdpos = cur;
                min_wrpos = min(cur + 5 * BATCH_SIZE, end);
                #pragma unroll(1) // We don't want unrolling here
                while (s->q.prefetch_wrpos < min_wrpos)
                {
                    NANOSLEEP(50);
                }
                dst_pos += blen;
                if (bytes_left < blen)
                {
                    break;
                }
                bytes_left -= blen;
                blen += 64;
            }
            b->len = blen;
            b->offset = offset;
            b++;
            if (++batch_len == BATCH_SIZE)
                break;
        }
        if (batch_len != 0)
        {
            s->q.batch_len[batch] = batch_len;
            batch = (batch + 1) & (BATCH_COUNT - 1);
        }
        while (s->q.batch_len[batch] != 0)
        {
            NANOSLEEP(100);
        }
        if (batch_len != BATCH_SIZE || bytes_left == 0)
        {
            break;
        }
    }
    s->q.batch_len[batch] = -1;
    return bytes_left;
}


// WARP1: process symbols and output uncompressed stream
// NOTE: No error checks at this stage (WARP0 responsible for not sending offsets and lengths that would result in out-of-bounds accesses)
__device__ void snappy_process_symbols(unsnap_state_s *s, int t)
{
    const uint8_t *literal_base = s->base;
    uint8_t *out = reinterpret_cast<uint8_t *>(s->in.dstDevice);
    int batch = 0;

    do
    {
        volatile unsnap_batch_s *b = &s->q.batch[batch * BATCH_SIZE];
        int32_t batch_len;

        if (t == 0)
        {
            while ((batch_len = s->q.batch_len[batch]) == 0)
            {
                NANOSLEEP(100);
            }
        }
        else
        {
            batch_len = 0;
        }
        batch_len = SHFL0(batch_len);
        if (batch_len <= 0)
        {
            break;
        }
        for (int i = 0; i < batch_len; i++, b++)
        {
            int blen = b->len;
            uint32_t dist = b->offset;
            if (blen <= 64)
            {
                uint8_t b0, b1;
                // Copy
                if (t < blen)
                {
                    uint32_t pos = t;
                    const uint8_t *src = out + ((pos >= dist) ? (pos % dist) : pos) - dist;
                    b0 = *src;
                }
                if (32 + t < blen)
                {
                    uint32_t pos = 32 + t;
                    const uint8_t *src = out + ((pos >= dist) ? (pos % dist) : pos) - dist;
                    b1 = *src;
                }
                if (t < blen)
                {
                    out[t] = b0;
                }
                if (32 + t < blen)
                {
                    out[32 + t] = b1;
                }
                SYNCWARP();
            }
            else
            {
                // Literal
                uint8_t b0, b1;
                blen -= 64;
                while (blen >= 64)
                {
                    b0 = literal_base[dist + t];
                    b1 = literal_base[dist + 32 + t];
                    out[t] = b0;
                    out[32 + t] = b1;
                    dist += 64;
                    out += 64;
                    blen -= 64;
                }
                if (t < blen)
                {
                    b0 = literal_base[dist + t];
                }
                if (32 + t < blen)
                {
                    b1 = literal_base[dist + 32 + t];
                }
                if (t < blen)
                {
                    out[t] = b0;
                }
                if (32 + t < blen)
                {
                    out[32 + t] = b1;
                }
            }
            out += blen;
        }
        SYNCWARP();
        if (t == 0)
        {
            s->q.batch_len[batch] = 0;
        }
        batch = (batch + 1) & (BATCH_COUNT - 1);
    } while (1);
}


// blockDim {128,1,1}
extern "C" __global__ void __launch_bounds__(128)
unsnap_kernel(gpu_inflate_input_s *inputs, gpu_inflate_status_s *outputs, int count)
{
    __shared__ __align__(16) unsnap_state_s state_g;

    int t = threadIdx.x;
    unsnap_state_s *s = &state_g;
    int strm_id = blockIdx.x;

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
                            if (c < 0xf)
                                uncompressed_size = (uncompressed_size & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) | (c << 28);
                            else
                                s->error = -1;
                        }
                    }
                }
            }
            s->uncompressed_size = uncompressed_size;
            s->bytes_left = uncompressed_size;
            s->base = cur;
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
        s->q.prefetch_end = 0;
        s->q.prefetch_wrpos = 0;
        s->q.prefetch_rdpos = 0;
    }
    __syncthreads();
    if (strm_id < count && !s->error)
    {
        if (t < 32)
        {
            // WARP0: decode lengths and offsets
            if (!t)
            {
                s->bytes_left = snappy_decode_symbols(s);
                if (s->bytes_left != 0)
                {
                    s->error = -2;
                }
                s->q.prefetch_end = 1;
            }
        }
        else if (t < 64)
        {
            // WARP1: prefetch byte stream for WARP0
            snappy_prefetch_bytestream(s, t & 0x1f);
        }
        else if (t < 96)
        {
            // WARP2: LZ77
            snappy_process_symbols(s, t & 0x1f);
        }
    }
    __syncthreads();
    if (!t && strm_id < count)
    {
        outputs[strm_id].bytes_written = s->uncompressed_size - s->bytes_left;
        outputs[strm_id].status = s->error;
        outputs[strm_id].reserved = 0;
    }
}


cudaError_t __host__ gpu_unsnap(gpu_inflate_input_s *inputs, gpu_inflate_status_s *outputs, int count, cudaStream_t stream)
{
    uint32_t count32 = (count > 0) ? count : 0;
    dim3 dim_block(128, 1);     // 4 warps per stream, 1 stream per block
    dim3 dim_grid(count32, 1);  // TODO: Check max grid dimensions vs max expected count

    unsnap_kernel << < dim_grid, dim_block, 0, stream >> >(inputs, outputs, count32);

    return cudaSuccess;
}

