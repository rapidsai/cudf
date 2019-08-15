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
#include "orc_common.h"
#include "orc_gpu.h"

namespace cudf {
namespace io {
namespace orc {
namespace gpu {

#if (__CUDACC_VER_MAJOR__ >= 9)
#define SHFL0(v)        __shfl_sync(~0, v, 0)
#define SHFL(v, t)      __shfl_sync(~0, v, t)
#define SHFL_XOR(v, m)  __shfl_xor_sync(~0, v, m)
#define SYNCWARP()      __syncwarp()
#define BALLOT(v)       __ballot_sync(~0, v)
#else
#define SHFL0(v)        __shfl(v, 0)
#define SHFL(v, t)      __shfl(v, t)
#define SHFL_XOR(v, m)  __shfl_xor(v, m)
#define SYNCWARP()
#define BALLOT(v)       __ballot(v)
#endif

#define OUTBUFSZ        (512*10+16)

struct orcenc_state_s
{
    uint32_t cur_row;       // Current row in group
    uint32_t present_rows;  // # of rows in present buffer
    uint32_t present_out;   // # of rows in present buffer that have been flushed
    uint32_t nrows;         // # of rows in current batch
    uint32_t numvals;       // # of non-zero values in current batch (<=nrows)
    uint32_t numlengths;    // # of non-zero values in DATA2 batch
    uint32_t nnz;           // Running count of non-null values
    EncChunk chunk;
    uint32_t strm_pos[CI_NUM_STREAMS];
    uint8_t valid_buf[512]; // valid map bits
    union {
        uint8_t u8[OUTBUFSZ];       // output scratch buffer
        uint32_t u32[OUTBUFSZ/4];
    } buf;
    union {
        uint8_t u8[1024];
        uint32_t u32[1024];
        int32_t i32[1024];
        uint64_t u64[1024];
        int64_t i64[1024];
    } vals;
    uint32_t lengths[1024];
};


/**
 * @brief In-place conversion from lengths to positions
 *
 * @param[in] vals input values
 * @param[in] numvals number of values
 * @param[in] t thread id
 *
 **/
template<class T>
inline __device__ void lengths_to_positions(volatile T *vals, uint32_t numvals, unsigned int t)
{
    for (uint32_t n = 1; n<numvals; n <<= 1)
    {
        __syncthreads();
        if ((t & n) && (t < numvals))
            vals[t] += vals[(t & ~n) | (n - 1)];
    }
}



/**
 * @brief Encode column data
 *
 * @param[in] chunks EncChunk device array [rowgroup][column]
 * @param[in] num_columns Number of columns
 * @param[in] num_rowgroups Number of row groups
 *
 **/
// blockDim {512,1,1}
extern "C" __global__ void __launch_bounds__(512)
gpuEncodeOrcColumnData(EncChunk *chunks, uint32_t num_columns, uint32_t num_rowgroups)
{
    __shared__ __align__(16) orcenc_state_s state_g;

    orcenc_state_s * const s = &state_g;
    uint32_t col_id = blockIdx.x;
    uint32_t group_id = blockIdx.y;
    int t = threadIdx.x;

    if (t < sizeof(EncChunk) / sizeof(uint32_t))
    {
        ((volatile uint32_t *)&s->chunk)[t] = ((const uint32_t *)&chunks[group_id * num_columns + col_id])[t];
    }
    if (t < CI_NUM_STREAMS)
    {
        s->strm_pos[t] = 0;
    }
    __syncthreads();
    if (!t)
    {
        s->cur_row = 0;
        s->present_rows = 0;
        s->present_out = 0;
        s->numvals = 0;
        s->numlengths = 0;
    }
    __syncthreads();
    while (s->cur_row < s->chunk.num_rows || s->numvals + s->numlengths != 0)
    {
        // Encode valid map
        if (s->present_rows < s->chunk.num_rows)
        {
            uint32_t present_rows = s->present_rows;
            uint32_t nrows = min(s->chunk.num_rows - present_rows, 512 * 8 - (present_rows - (min(s->cur_row, s->present_out) & ~7)));
            uint32_t nrows_out;
            if (t < nrows)
            {
                uint32_t row = s->chunk.start_row + present_rows + t * 8;
                uint8_t valid = 0;
                if (row < s->chunk.valid_rows)
                {
                    const uint8_t *valid_map_base = reinterpret_cast<const uint8_t *>(s->chunk.valid_map_base);
                    valid = (valid_map_base) ? valid_map_base[row >> 3] : 0xff;
                    if (row + 7 > s->chunk.valid_rows)
                    {
                        valid = valid & ((1 << (s->chunk.valid_rows & 7)) - 1);
                    }
                }
                s->valid_buf[(row >> 3) & 0x1ff] = valid;
            }
            __syncthreads();
            present_rows += nrows;
            if (!t)
            {
                s->present_rows = present_rows;
            }
            // RLE encode the present stream
            nrows_out = present_rows - s->present_out; // Should always be a multiple of 8 except at the end of the last row group
            if (nrows_out > ((present_rows <= s->chunk.num_rows) ? 131 * 8 : 0))
            {
                uint32_t present_out = s->present_out;
                if (s->chunk.strm_id[CI_PRESENT] >= 0)
                {

                }
                __syncthreads();
                if (!t)
                {
                    s->present_out = present_out + nrows_out;
                }
            }
            __syncthreads();
        }
        // Fetch non-null values
        if (!s->chunk.streams[CI_DATA])
        {
            // Pass-through
            __syncthreads();
            if (!t)
            {
                s->cur_row = s->present_rows;
                s->strm_pos[CI_DATA] = s->cur_row * s->chunk.dtype_len;
            }
            __syncthreads();
        }
        else if (s->cur_row < s->present_rows)
        {
            uint32_t maxnumvals = (s->chunk.type_kind == BOOLEAN) ? 2048 : 1024;
            uint32_t nrows = min(min(s->present_rows - s->cur_row, maxnumvals - max(s->numvals, s->numlengths)), 512);
            uint32_t row = s->chunk.start_row + s->cur_row + t;
            uint32_t valid = (t < nrows) ? (s->valid_buf[(row >> 3) & 0x1ff] >> (row & 7)) & 1 : 0;
            s->buf.u32[t] = valid;
            // TODO: Could use a faster reduction relying on _popc() for the initial phase
            lengths_to_positions(s->buf.u32, 512, t);
            __syncthreads();
            if (valid)
            {
                int nz_idx = (s->nnz + s->buf.u32[t] - 1) & (maxnumvals - 1);
                const uint8_t *base = reinterpret_cast<const uint8_t *>(s->chunk.column_data_base);
                switch (s->chunk.type_kind)
                {
                case INT:
                case DATE:
                case FLOAT:
                    s->vals.u32[nz_idx] = reinterpret_cast<const uint32_t *>(base)[row];
                    break;
                case DOUBLE:
                case LONG:
                    s->vals.u64[nz_idx] = reinterpret_cast<const uint64_t *>(base)[row];
                    break;
                case SHORT:
                    s->vals.u32[nz_idx] = reinterpret_cast<const uint16_t *>(base)[row];
                    break;
                case BOOLEAN:
                    s->vals.u8[nz_idx] = reinterpret_cast<const uint8_t *>(base)[row];
                    break;
                }
            }
            __syncthreads();
            if (!t)
            {
                uint32_t nz = s->buf.u32[511];
                s->nnz += nz;
                s->numvals += nz;
                s->cur_row += nrows;
            }
            __syncthreads();
            // Encode values
            if (!t)
            {
                // TODO
                s->numvals = 0;
                s->numlengths = 0;
            }
        }
        __syncthreads();
    }
    __syncthreads();
    if (t < CI_NUM_STREAMS && s->chunk.strm_id[t] >= 0)
    {
        // Update actual compressed length
        chunks[group_id * num_columns + col_id].strm_len[t] = s->strm_pos[t];
        if (!s->chunk.streams[t])
        {
            chunks[group_id * num_columns + col_id].streams[t] = reinterpret_cast<uint8_t *>(const_cast<void *>(s->chunk.column_data_base)) + s->chunk.start_row * s->chunk.dtype_len;
        }
    }
}


/**
 * @brief Launches kernel for encoding column data
 *
 * @param[in] chunks EncChunk device array [rowgroup][column]
 * @param[in] num_columns Number of columns
 * @param[in] num_rowgroups Number of row groups
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t EncodeOrcColumnData(EncChunk *chunks, uint32_t num_columns, uint32_t num_rowgroups, cudaStream_t stream)
{
    dim3 dim_block(512, 1); // 512 threads per chunk
    dim3 dim_grid(num_columns, num_rowgroups);
    gpuEncodeOrcColumnData <<< dim_grid, dim_block, 0, stream >>>(chunks, num_columns, num_rowgroups);
    return cudaSuccess;
}


} // namespace gpu
} // namespace orc
} // namespace io
} // namespace cudf
