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

namespace cudf {
namespace io {
namespace orc {
namespace gpu {

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
            if (is_uncompressed)
            {
                // Copy the uncompressed data to output (not very efficient, but should only occur for small blocks and probably faster than decompression)
                if (uncompressed && max_uncompressed_size + uncompressed_size <= s->info.max_uncompressed_size)
                {
                    for (int i = t; i < uncompressed_size; i += 32)
                    {
                        uncompressed[max_uncompressed_size + i] = cur[i];
                    }
                }
            }
            else if (decctl && !t && num_compressed_blocks < s->info.max_compressed_blocks)
            {
                decctl[num_compressed_blocks].srcDevice = const_cast<uint8_t *>(cur);
                decctl[num_compressed_blocks].srcSize = block_len;
                decctl[num_compressed_blocks].dstDevice = uncompressed + max_uncompressed_size;
                decctl[num_compressed_blocks].dstSize = uncompressed_size;
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


/**
 * @brief Shared mem state for gpuParseRowGroupIndex
 *
 */
struct rowindex_state_s
{
    ColumnDesc chunk;
    uint32_t rowgroup_start;
    uint32_t rowgroup_end;
    int is_compressed;
    uint32_t row_index_entry[3][CI_PRESENT]; // NOTE: Assumes CI_PRESENT follows CI_DATA and CI_DATA2
    CompressedStreamInfo strm_info[2];
    RowGroup rowgroups[128];
    uint32_t compressed_offset[128][2];
};


#define PB_ROWINDEXENTRY_ID     ((1*8) + PB_TYPE_FIXEDLEN)

enum row_entry_state_e {
    NOT_FOUND = 0,
    GET_LENGTH,
    SKIP_VARINT,
    SKIP_FIXEDLEN,
    STORE_INDEX0,
    STORE_INDEX1,
    STORE_INDEX2,
};

/**
 * @brief Decode a single row group index entry
 *
 * @param[in,out] s row group index state
 * @param[in] start start position in byte stream
 * @param[in] end end of byte stream
 * @return bytes consumed
 *
 **/
static uint32_t __device__ ProtobufParseRowIndexEntry(rowindex_state_s *s, const uint8_t *start, const uint8_t *end)
{
    const uint8_t *cur = start;
    row_entry_state_e state = NOT_FOUND;
    uint32_t length = 0, strm_idx_id = s->chunk.skip_count >> 8, idx_id = 1, ci_id = CI_PRESENT, pos_end = 0;
    while (cur < end)
    {
        uint32_t v = 0;
        for (uint32_t l = 0; l <= 28; l += 7)
        {
            uint32_t c = (cur < end) ? *cur++ : 0;
            v |= (c & 0x7f) << l;
            if (c <= 0x7f)
                break;
        }
        switch (state)
        {
        case NOT_FOUND:
            if (v == PB_ROWINDEXENTRY_ID)
            {
                state = GET_LENGTH;
            }
            else
            {
                v &= 7;
                if (v == PB_TYPE_FIXED64)
                    cur += 8;
                else if (v == PB_TYPE_FIXED32)
                    cur += 4;
                else if (v == PB_TYPE_VARINT)
                    state = SKIP_VARINT;
                else if (v == PB_TYPE_FIXEDLEN)
                    state = SKIP_FIXEDLEN;
            }
            break;
        case SKIP_VARINT:
            state = NOT_FOUND;
            break;
        case SKIP_FIXEDLEN:
            cur += v;
            state = NOT_FOUND;
            break;
        case GET_LENGTH:
            if (length == 0)
            {
                length = (uint32_t)(cur + v - start);
                state = NOT_FOUND; // Scan for positions (same field id & low-level type as RowIndexEntry entry)
            }
            else
            {
                pos_end = min((uint32_t)(cur + v - start), length);
                state = STORE_INDEX0;
            }
            break;
        case STORE_INDEX0:
            ci_id = (idx_id == (strm_idx_id & 0xff)) ? CI_DATA : (idx_id == ((strm_idx_id >> 8) & 0xff)) ? CI_DATA2 : CI_PRESENT;
            idx_id++;
            if (s->is_compressed)
            {
                if (ci_id < CI_PRESENT)
                    s->row_index_entry[0][ci_id] = v;
                if (cur >= start + pos_end)
                    return length;
                state = STORE_INDEX1;
                break;
            }
            else
            {
                if (ci_id < CI_PRESENT)
                    s->row_index_entry[0][ci_id] = 0;
                // Fall through to STORE_INDEX1 for uncompressed (always block0)
            }
        case STORE_INDEX1:
            if (ci_id < CI_PRESENT)
                s->row_index_entry[1][ci_id] = v;
            if (cur >= start + pos_end)
                return length;
            state = (ci_id == CI_DATA && s->chunk.encoding_kind != DICTIONARY && s->chunk.encoding_kind != DICTIONARY_V2
                 && (s->chunk.type_kind == STRING || s->chunk.type_kind == BINARY || s->chunk.type_kind == VARCHAR || s->chunk.type_kind == CHAR
                  || s->chunk.type_kind == DECIMAL || s->chunk.type_kind == FLOAT || s->chunk.type_kind == DOUBLE))
                ? STORE_INDEX0 : STORE_INDEX2;
            break;
        case STORE_INDEX2:
            if (ci_id < CI_PRESENT)
                s->row_index_entry[2][ci_id] = v;
            // Boolean columns have an extra byte to indicate the position of the bit within the byte
            // TODO: Currently assuming rowIndexStride is a multiple of 8 and ignoring this value
            if (ci_id == CI_PRESENT || s->chunk.type_kind == BOOLEAN)
                cur++;
            if (cur >= start + pos_end)
                return length;
            state = STORE_INDEX0;
            break;
        }
    }
    return (uint32_t)(end - start);
}

/**
 * @brief Decode row group index entries
 *
 * @param[in,out] s row group index state
 * @param[in] num_rowgroups Number of index entries to read
 *
 **/
static __device__ void gpuReadRowGroupIndexEntries(rowindex_state_s *s, int num_rowgroups)
{
    const uint8_t *index_data = s->chunk.streams[CI_INDEX];
    int index_data_len = s->chunk.strm_len[CI_INDEX];
    for (int i = 0; i < num_rowgroups; i++)
    {
        s->row_index_entry[0][0] = 0;
        s->row_index_entry[0][1] = 0;
        s->row_index_entry[1][0] = 0;
        s->row_index_entry[1][1] = 0;
        s->row_index_entry[2][0] = 0;
        s->row_index_entry[2][1] = 0;
        if (index_data_len > 0)
        {
            int len = ProtobufParseRowIndexEntry(s, index_data, index_data + index_data_len);
            index_data += len;
            index_data_len = max(index_data_len - len, 0);
            for (int j = 0; j < 2; j++)
            {
                s->rowgroups[i].strm_offset[j] = s->row_index_entry[1][j];
                s->rowgroups[i].run_pos[j] = s->row_index_entry[2][j];
                s->compressed_offset[i][j] = s->row_index_entry[0][j];
            }
        }
    }
    s->chunk.streams[CI_INDEX] = index_data;
    s->chunk.strm_len[CI_INDEX] = index_data_len;
}


/**
 * @brief Translate block+offset compressed position into an uncompressed offset
 *
 * @param[in,out] s row group index state
 * @param[in] ci_id index to convert (CI_DATA or CI_DATA2)
 * @param[in] num_rowgroups Number of index entries
 * @param[in] t thread id
 *
 **/
static __device__ void gpuMapRowIndexToUncompressed(rowindex_state_s *s, int ci_id, int num_rowgroups, int t)
{
    int32_t strm_len = s->chunk.strm_len[ci_id];
    if (strm_len > 0)
    {
        int32_t compressed_offset = (t < num_rowgroups) ? s->compressed_offset[t][ci_id] : 0;
        if (compressed_offset > 0)
        {
            const uint8_t *start = s->strm_info[ci_id].compressed_data;
            const uint8_t *cur = start;
            const uint8_t *end = cur + s->strm_info[ci_id].compressed_data_size;
            gpu_inflate_status_s *decstatus = s->strm_info[ci_id].decstatus;
            uint32_t uncomp_offset = 0;
            for (;;)
            {
                uint32_t block_len, is_uncompressed;

                if (cur + 3 > end || cur + 3 >= start + compressed_offset)
                {
                    break;
                }
                block_len = cur[0] | (cur[1] << 8) | (cur[2] << 16);
                cur += 3;
                is_uncompressed = block_len & 1;
                block_len >>= 1;
                cur += block_len;
                if (cur > end)
                {
                    break;
                }
                if (is_uncompressed)
                {
                    uncomp_offset += block_len;
                }
                else
                {
                    uncomp_offset += decstatus->bytes_written;
                    decstatus++;
                }
            }
            s->rowgroups[t].strm_offset[ci_id] += uncomp_offset;
        }
    }
}


/**
 * @brief Decode index streams
 *
 * @param[out] row_groups RowGroup device array [rowgroup][column]
 * @param[in] strm_info List of compressed streams (or NULL if uncompressed)
 * @param[in] chunks ColumnDesc device array [stripe][column]
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] num_rowgroups Number of row groups
 *
 **/
// blockDim {128,1,1}
extern "C" __global__ void __launch_bounds__(128, 8)
gpuParseRowGroupIndex(RowGroup *row_groups, CompressedStreamInfo *strm_info, ColumnDesc *chunks, uint32_t num_columns, uint32_t num_stripes, uint32_t num_rowgroups, uint32_t rowidx_stride)
{
    __shared__ __align__(16) rowindex_state_s state_g;
    rowindex_state_s * const s = &state_g;
    uint32_t chunk_id = blockIdx.y * num_columns + blockIdx.x;
    int t = threadIdx.x;

    if (t < sizeof(ColumnDesc) / sizeof(uint32_t))
    {
        // NOTE: Assumes that sizeof(ColumnDesc) <= 128x4
        ((volatile uint32_t *)&s->chunk)[t] = ((const uint32_t *)&chunks[chunk_id])[t];
    }
    __syncthreads();
    if (strm_info)
    {
        int strm_len = s->chunk.strm_len[t >> 6];
        int strm_id = s->chunk.strm_id[t >> 6];
        int t6 = t & 0x3f;
        if (strm_len > 0 && t6 < sizeof(CompressedStreamInfo) / sizeof(uint32_t))
        {
            ((volatile uint32_t *)&s->strm_info[t >> 6])[t6] = ((const uint32_t *)&strm_info[strm_id])[t6];
        }
    }
    if (t == 0)
    {
        uint32_t rowgroups_in_chunk = (rowidx_stride > 0) ? (s->chunk.num_rows + rowidx_stride - 1) / rowidx_stride : 1;
        s->rowgroup_start = s->chunk.rowgroup_id;
        s->rowgroup_end = s->rowgroup_start + rowgroups_in_chunk;
        s->is_compressed = (strm_info != NULL);
    }
    __syncthreads();
    while (s->rowgroup_start < s->rowgroup_end)
    {
        int num_rowgroups = min(s->rowgroup_end - s->rowgroup_start, 128);
        int rowgroup_size4, t4, t32;

        s->rowgroups[t].chunk_id = chunk_id;
        if (t == 0)
        {
            gpuReadRowGroupIndexEntries(s, num_rowgroups);
        }
        __syncthreads();
        if (s->is_compressed)
        {
            // Convert the block + blk_offset pair into a raw offset into the decompressed stream
            if (s->chunk.strm_len[CI_DATA] > 0)
            {
                gpuMapRowIndexToUncompressed(s, CI_DATA, num_rowgroups, t);
            }
            if (s->chunk.strm_len[CI_DATA2] > 0)
            {
                gpuMapRowIndexToUncompressed(s, CI_DATA2, num_rowgroups, t);
            }
            __syncthreads();
        }
        rowgroup_size4 = sizeof(RowGroup) / sizeof(uint32_t);
        t4 = t & 3;
        t32 = t >> 2;
        for (int i = t32; i < num_rowgroups; i += 32)
        {
            for (int j = t4; j < rowgroup_size4; j += 4)
            {
                ((uint32_t *)&row_groups[(s->rowgroup_start + i) * num_columns + blockIdx.x])[j] = ((volatile uint32_t *)&s->rowgroups[i])[j];
            }
        }
        __syncthreads();
        if (t == 0)
        {
            s->rowgroup_start += num_rowgroups;
        }
        __syncthreads();
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

/**
 * @brief Launches kernel for constructing rowgroup from index streams
 *
 * @param[out] row_groups RowGroup device array [rowgroup][column]
 * @param[in] strm_info List of compressed streams (or NULL if uncompressed)
 * @param[in] chunks ColumnDesc device array [stripe][column]
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] num_rowgroups Number of row groups
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t __host__ ParseRowGroupIndex(RowGroup *row_groups, CompressedStreamInfo *strm_info, ColumnDesc *chunks, uint32_t num_columns, uint32_t num_stripes, uint32_t num_rowgroups, uint32_t rowidx_stride, cudaStream_t stream)
{
    dim3 dim_block(128, 1);
    dim3 dim_grid(num_columns, num_stripes); // 1 column chunk per block
    gpuParseRowGroupIndex <<< dim_grid, dim_block, 0, stream >>>(row_groups, strm_info, chunks, num_columns, num_stripes, num_rowgroups, rowidx_stride);
    return cudaSuccess;
}

} // namespace gpu
} // namespace orc
} // namespace io
} // namespace cudf
