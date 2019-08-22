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

#define SCRATCH_BFRSZ   (512*4)

struct byterle_enc_state_s
{
    uint32_t literal_run;
    uint32_t repeat_run;
    volatile uint32_t rpt_map[(512 / 32) + 1];
};

struct intrle_enc_state_s
{
    uint32_t literal_run;
    uint32_t delta_run;
    volatile uint32_t delta_map[(512 / 32) + 1];
    volatile uint8_t maxlen[512 / 32];
    volatile uint16_t maxpos[512 / 32];
};


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
        byterle_enc_state_s byterle;
        intrle_enc_state_s intrle;
    } u;
    union {
        uint8_t u8[SCRATCH_BFRSZ];  // general scratch buffer
        uint32_t u32[SCRATCH_BFRSZ /4];
    } buf;
    union {
        uint8_t u8[2048];
        uint32_t u32[1024];
        int32_t i32[1024];
        uint64_t u64[1024];
        int64_t i64[1024];
    } vals;
    union {
        uint8_t u8[2048];
        uint32_t u32[1024];
    } lengths;
};


/**
 * @brief Raw data output
 *
 * @param[in] cid stream type (strm_pos[cid] will be updated and output stored at streams[cid]+strm_pos[cid])
 * @param[in] inmask input buffer position mask for circular buffers
 * @param[in] s encoder state
 * @param[in] inbuf base input buffer
 * @param[in] inpos position in input buffer
 * @param[in] count number of bytes to encode
 * @param[in] t thread id
 *
 **/
template<StreamIndexType cid, uint32_t inmask>
static __device__ void StoreBytes(orcenc_state_s *s, const uint8_t *inbuf, uint32_t inpos, uint32_t count, int t)
{
    uint8_t *dst = s->chunk.streams[cid] + s->strm_pos[cid];
    while (count > 0)
    {
        uint32_t n = min(count, 512);
        if (t < n)
        {
            dst[t] = inbuf[(inpos + t) & inmask];
        }
        dst += n;
        inpos += n;
        count -= n;
    }
    __syncthreads();
    if (!t)
    {
        s->strm_pos[cid] = static_cast<uint32_t>(dst - s->chunk.streams[cid]);
    }
}


/**
 * @brief ByteRLE encoder
 *
 * @param[in] cid stream type (strm_pos[cid] will be updated and output stored at streams[cid]+strm_pos[cid])
 * @param[in] s encoder state
 * @param[in] inbuf base input buffer
 * @param[in] inpos position in input buffer
 * @param[in] inmask input buffer position mask for circular buffers
 * @param[in] numvals max number of values to encode
 * @param[in] flush encode all remaining values if nonzero
 * @param[in] t thread id
 *
 * @return number of input values encoded
 *
 **/
template<StreamIndexType cid, uint32_t inmask>
static __device__ uint32_t ByteRLE(orcenc_state_s *s, const uint8_t *inbuf, uint32_t inpos, uint32_t numvals, uint32_t flush, int t)
{
    uint8_t *dst = s->chunk.streams[cid] + s->strm_pos[cid];
    uint32_t out_cnt = 0;

    while (numvals > 0)
    {
        uint8_t v0 = (t < numvals) ? inbuf[(inpos + t) & inmask] : 0;
        uint8_t v1 = (t + 1 < numvals) ? inbuf[(inpos + t + 1) & inmask] : 0;
        uint32_t rpt_map = BALLOT(t + 1 < numvals && v0 == v1), literal_run, repeat_run, maxvals = min(numvals, 512);
        if (!(t & 0x1f))
            s->u.byterle.rpt_map[t >> 5] = rpt_map;
        __syncthreads();
        if (t == 0)
        {
            // Find the start of an identical 3-byte sequence
            // TBD: The two loops below could be eliminated using more ballot+ffs using warp0
            literal_run = 0;
            repeat_run = 0;
            while (literal_run < maxvals)
            {
                uint32_t next = s->u.byterle.rpt_map[(literal_run >> 5) + 1];
                uint32_t mask = rpt_map & __funnelshift_r(rpt_map, next, 1);
                if (mask)
                {
                    uint32_t literal_run_ofs = __ffs(mask) - 1;
                    literal_run += literal_run_ofs;
                    repeat_run = __ffs(~((rpt_map >> literal_run_ofs) >> 1));
                    if (repeat_run + literal_run_ofs == 32)
                    {
                        while (next == ~0)
                        {
                            uint32_t next_idx = ((literal_run + repeat_run) >> 5) + 1;
                            next = (next_idx < 512 / 32) ? s->u.byterle.rpt_map[next_idx] : 0;
                            repeat_run += 32;
                        }
                        repeat_run += __ffs(~next) - 1;
                    }
                    repeat_run = min(repeat_run + 1, maxvals - min(literal_run, maxvals));
                    if (repeat_run < 3)
                    {
                        literal_run += (flush && literal_run + repeat_run >= numvals) ? repeat_run : 0;
                        repeat_run = 0;
                    }
                    break;
                }
                rpt_map = next;
                literal_run += 32;
            }
            if (repeat_run >= 130)
            {
                // Limit large runs to multiples of 130
                repeat_run = (repeat_run >= 3*130) ? 3*130 : (repeat_run >= 2*130) ? 2*130 : 130;
            }
            else if (literal_run && literal_run + repeat_run == maxvals)
            {
                repeat_run = 0; // Try again at next iteration
            }
            s->u.byterle.repeat_run = repeat_run;
            s->u.byterle.literal_run = min(literal_run, maxvals);
        }
        __syncthreads();
        literal_run = s->u.byterle.literal_run;
        if (!flush && literal_run == numvals)
        {
            literal_run &= ~0x7f;
            if (!literal_run)
                break;
        }
        if (literal_run > 0)
        {
            uint32_t num_runs = (literal_run + 0x7f) >> 7;
            if (t < literal_run)
            {
                uint32_t run_id = t >> 7;
                uint32_t run = (run_id == num_runs - 1) ? literal_run & 0x7f : 0x80;
                if (!(t & 0x7f))
                    dst[run_id + t] = 0x100 - run;
                dst[run_id + t + 1] = (cid == CI_PRESENT) ? __brev(v0) >> 24 : v0;
            }
            dst += num_runs + literal_run;
            out_cnt += literal_run;
            numvals -= literal_run;
            inpos += literal_run;
        }
        repeat_run = s->u.byterle.repeat_run;
        if (repeat_run > 0)
        {
            while (repeat_run >= 130)
            {
                if (t == literal_run) // repeat_run follows literal_run
                {
                    dst[0] = 0x7f;
                    dst[1] = (cid == CI_PRESENT) ? __brev(v0) >> 24 : v0;
                }
                dst += 2;
                out_cnt += 130;
                numvals -= 130;
                inpos += 130;
                repeat_run -= 130;
            }
            if (!flush)
            {
                // Wait for more data in case we can continue the run later 
                if (repeat_run == numvals && !flush)
                    break;
            }
            if (repeat_run >= 3)
            {
                if (t == literal_run) // repeat_run follows literal_run
                {
                    dst[0] = repeat_run - 3;
                    dst[1] = (cid == CI_PRESENT) ? __brev(v0) >> 24 : v0;
                }
                dst += 2;
                out_cnt += repeat_run;
                numvals -= repeat_run;
                inpos += repeat_run;
            }
        }
    }
    if (!t)
    {
        s->strm_pos[cid] = static_cast<uint32_t>(dst - s->chunk.streams[cid]);
    }
    __syncthreads();
    return out_cnt;
}


/**
 * @brief Integer RLEv2 encoder
 *
 * @param[in] cid stream type (strm_pos[cid] will be updated and output stored at streams[cid]+strm_pos[cid])
 * @param[in] s encoder state
 * @param[in] inbuf base input buffer
 * @param[in] inpos position in input buffer
 * @param[in] inmask input buffer position mask for circular buffers
 * @param[in] numvals max number of values to encode
 * @param[in] flush encode all remaining values if nonzero
 * @param[in] t thread id
 *
 * @return number of input values encoded
 *
 **/
template<StreamIndexType cid, class T, bool is_signed, uint32_t inmask>
static __device__ uint32_t IntegerRLE(orcenc_state_s *s, const T *inbuf, uint32_t inpos, uint32_t numvals, uint32_t flush, int t)
{
    uint8_t *dst = s->chunk.streams[cid] + s->strm_pos[cid];
    uint32_t out_cnt = 0;

    while (numvals > 0)
    {
        T v0 = (t < numvals) ? inbuf[(inpos + t) & inmask] : 0;
        T v1 = (t + 1 < numvals) ? inbuf[(inpos + t + 1) & inmask] : 0;
        T v2 = (t + 2 < numvals) ? inbuf[(inpos + t + 2) & inmask] : 0;
        uint32_t delta_map = BALLOT(t + 2 < numvals && v1 - v0 == v2 - v1), maxvals = min(numvals, 512), literal_run, delta_run;
        if (!(t & 0x1f))
            s->u.intrle.delta_map[t >> 5] = delta_map;
        __syncthreads();
        if (!t)
        {
            // Find the start of the next delta run (2 consecutive values with the same delta)
            literal_run = delta_run = 0;
            while (literal_run < maxvals)
            {
                if (delta_map != 0)
                {
                    uint32_t literal_run_ofs = __ffs(delta_map) - 1;
                    literal_run += literal_run_ofs;
                    delta_run = __ffs(~((delta_map >> literal_run_ofs) >> 1));
                    if (literal_run_ofs + delta_run == 32)
                    {
                        for (;;)
                        {
                            uint32_t delta_idx = (literal_run + delta_run) >> 5;
                            delta_map = (delta_idx < 512/32) ? s->u.intrle.delta_map[delta_idx] : 0;
                            if (delta_map != ~0)
                                break;
                            delta_run += 32;
                        }
                        delta_run += __ffs(~delta_map) - 1;
                    }
                    delta_run += 2;
                    break;
                }
                literal_run += 32;
                delta_map = s->u.intrle.delta_map[(literal_run >> 5)];
            }
            literal_run = min(literal_run, maxvals);
            s->u.intrle.literal_run = literal_run;
            s->u.intrle.delta_run = min(delta_run, maxvals - literal_run);
        }
        __syncthreads();
        literal_run = s->u.intrle.literal_run;

        numvals -= literal_run;
        inpos += literal_run;
        out_cnt += literal_run;
        delta_run = s->u.intrle.delta_run;
        if (delta_run > 0)
        {
            __syncthreads();
            numvals -= delta_run;
            inpos += delta_run;
            out_cnt += delta_run;
        }
    }
    if (!t)
    {
        s->strm_pos[cid] = static_cast<uint32_t>(dst - s->chunk.streams[cid]);
    }
    __syncthreads();
    return out_cnt;
}

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
            if (nrows_out > ((present_rows < s->chunk.num_rows) ? 130 * 8 : 0))
            {
                uint32_t present_out = s->present_out;
                if (s->chunk.strm_id[CI_PRESENT] >= 0)
                {
                    uint32_t flush = (present_rows < s->chunk.num_rows) ? 0 : 7;
                    nrows_out = (nrows_out + flush) >> 3;
                    nrows_out = ByteRLE<CI_PRESENT, 0x1ff>(s, s->valid_buf, present_out, nrows_out, flush, t) * 8;
                }
                __syncthreads();
                if (!t)
                {
                    s->present_out = min(present_out + nrows_out, present_rows);
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
                case BYTE:
                    s->vals.u8[nz_idx] = reinterpret_cast<const uint8_t *>(base)[row];
                    break;
                }
            }
            __syncthreads();
            if (s->chunk.type_kind == BOOLEAN)
            {
                // bool8 -> 8x bool1
                uint32_t nz = s->buf.u32[511];
                uint8_t n = ((s->nnz + nz) - (s->nnz & ~7) + 7) >> 3;
                if (t < n)
                {
                    uint32_t idx8 = (s->nnz & ~7) + (t << 3);
                    s->lengths.u8[((s->nnz >> 3) + t) & 0x1ff] = ((s->vals.u8[(idx8 + 0) & 0x7ff] & 1) << 7)
                                                               | ((s->vals.u8[(idx8 + 1) & 0x7ff] & 1) << 6)
                                                               | ((s->vals.u8[(idx8 + 2) & 0x7ff] & 1) << 5)
                                                               | ((s->vals.u8[(idx8 + 3) & 0x7ff] & 1) << 4)
                                                               | ((s->vals.u8[(idx8 + 4) & 0x7ff] & 1) << 3)
                                                               | ((s->vals.u8[(idx8 + 5) & 0x7ff] & 1) << 2)
                                                               | ((s->vals.u8[(idx8 + 6) & 0x7ff] & 1) << 1)
                                                               | ((s->vals.u8[(idx8 + 6) & 0x7ff] & 1) << 0);
                }
                __syncthreads();
            }
            if (!t)
            {
                uint32_t nz = s->buf.u32[511];
                s->nnz += nz;
                s->numvals += nz;
                s->cur_row += nrows;
            }
            __syncthreads();
            // Encode values
            if (s->numvals > 0)
            {
                uint32_t flush = (s->cur_row == s->chunk.num_rows) ? 7 : 0, n;
                switch (s->chunk.type_kind)
                {
                case INT:
                case DATE:
                    n = IntegerRLE<CI_DATA, int32_t, true, 0x3ff>(s, s->vals.i32, s->nnz - s->numvals, s->numvals, flush, t);
                    break;
                case BYTE:
                    n = ByteRLE<CI_DATA, 0x3ff>(s, s->vals.u8, s->nnz - s->numvals, s->numvals, flush, t);
                    break;
                case BOOLEAN:
                    n = ByteRLE<CI_DATA, 0x1ff>(s, s->lengths.u8, (s->nnz - s->numvals + flush) >> 3, (s->numvals + flush) >> 3, flush, t) * 8;
                    break;
                case FLOAT:
                    StoreBytes<CI_DATA, 0xfff>(s, s->vals.u8, (s->nnz - s->numvals) * 4, s->numvals * 4, t);
                    n = s->numvals;
                    break;
                case DOUBLE:
                    StoreBytes<CI_DATA, 0x1fff>(s, s->vals.u8, (s->nnz - s->numvals) * 8, s->numvals * 8, t);
                    n = s->numvals;
                    break;
                default:
                    n = s->numvals;
                    break;
                }
                __syncthreads();
                if (!t)
                {
                    s->numvals -= min(n, s->numvals);
                }
            }
            if (s->numlengths > 0)
            {
                // TODO
                __syncthreads();
                if (!t)
                {
                    s->numlengths = 0;
                }
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
