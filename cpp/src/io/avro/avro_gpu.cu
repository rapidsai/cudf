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
#include "avro_gpu.h"
#include <io/utilities/block_utils.cuh>

namespace cudf {
namespace io {
namespace avro {
namespace gpu {

#define NWARPS                  16
#define MAX_SHARED_SCHEMA_LEN   1000

/**
 * @brief Decode column data
 *
 * @param[in] blocks Data block descriptions
 * @param[in] schema Schema description
 * @param[in] global_Dictionary Global dictionary entries
 * @param[in] avro_data Raw block data
 * @param[in] num_blocks Number of blocks
 * @param[in] schema_len Number of entries in schema
 * @param[in] num_dictionary_entries Number of entries in global dictionary
 * @param[in] min_row_size Minimum size in bytes of a row
 * @param[in] max_rows Maximum number of rows to load
 * @param[in] first_row Crop all rows below first_row
 *
 **/
// blockDim {32,NWARPS,1}
extern "C" __global__ void __launch_bounds__(NWARPS * 32, 2)
gpuDecodeAvroColumnData(block_desc_s *blocks, schemadesc_s *schema_g, nvstrdesc_s *global_dictionary, const uint8_t *avro_data,
    uint32_t num_blocks, uint32_t schema_len, uint32_t num_dictionary_entries, uint32_t min_row_size, size_t max_rows, size_t first_row)
{
    __shared__ __align__(8) schemadesc_s g_shared_schema[MAX_SHARED_SCHEMA_LEN];
    __shared__ __align__(8) block_desc_s blk_g[NWARPS];

    schemadesc_s *schema;
    block_desc_s * const blk = &blk_g[threadIdx.y];
    uint32_t block_id = blockIdx.x * NWARPS + threadIdx.y;
    size_t cur_row;
    uint32_t rows_remaining;
    const uint8_t *cur, *end;

    // Fetch schema into shared mem if possible
    if (schema_len <= MAX_SHARED_SCHEMA_LEN)
    {
        for (int i = threadIdx.y * 32 + threadIdx.x; i < schema_len * sizeof(schemadesc_s) / sizeof(uint32_t); i += NWARPS * 32)
        {
            reinterpret_cast<uint32_t *>(&g_shared_schema)[i] = reinterpret_cast<const uint32_t *>(schema_g)[i];
        }
        __syncthreads();
        schema = g_shared_schema;
    }
    else
    {
        schema = schema_g;
    }
    if (block_id < num_blocks && threadIdx.x < sizeof(block_desc_s) / sizeof(uint32_t))
    {
        reinterpret_cast<volatile uint32_t *>(blk)[threadIdx.x] = reinterpret_cast<const uint32_t *>(&blocks[block_id])[threadIdx.x];
        __threadfence_block();
    }
    __syncthreads();
    if (block_id >= num_blocks)
    {
        return;
    }
    cur_row = blk->first_row;
    rows_remaining = blk->num_rows;
    cur = avro_data + blk->offset;
    end = cur + blk->size;
    while (rows_remaining > 0 && cur < end)
    {
        uint32_t nrows;
        const uint8_t *start = cur;

        if (cur_row > first_row + max_rows)
            break;
        if (cur + min_row_size * rows_remaining == end)
        {
            nrows = min(rows_remaining, 32);
            cur += threadIdx.x * min_row_size;
        }
        else
        {
            nrows = 1;
        }
        if (threadIdx.x < nrows)
        {
            for (uint32_t i = 0; i < schema_len; )
            {
                uint32_t kind = schema[i].kind;
                int skip = 0;
                uint8_t *dataptr;
                if (kind == type_union)
                {
                    int skip_after;
                    if (cur >= end)
                        break;
                    skip = (*cur++) >> 1;  // NOTE: Assumes 1-byte union member
                    skip_after = schema[i].count - skip - 1;
                    ++i;
                    while (skip > 0 && i < schema_len)
                    {
                        if (schema[i].kind >= type_record)
                        {
                            skip += schema[i].count;
                        }
                        ++i;
                        --skip;
                    }
                    if (i >= schema_len || skip_after < 0)
                        break;
                    kind = schema[i].kind;
                    skip = skip_after;
                }
                dataptr = reinterpret_cast<uint8_t *>(schema[i].dataptr);
                if (dataptr)
                {
                    size_t row = cur_row - first_row + threadIdx.x;
                    switch (kind)
                    {
                    case type_null:
                        if (row < max_rows)
                        {
                            atomicAnd(reinterpret_cast<uint32_t *>(dataptr) + (row >> 5), ~(1 << (row & 0x1f)));
                            atomicAdd(&schema_g[i].count, 1);
                        }
                        break;

                    case type_int:
                    case type_long:
                    case type_bytes:
                    case type_string:
                    case type_enum:
                        {
                            uint64_t u = 0;
                            int64_t v;
                            if (cur < end)
                            {
                                u = *cur++;
                                if (u > 0x7f)
                                {
                                    uint64_t scale = 128;
                                    u &= 0x7f;
                                    while (cur < end)
                                    {
                                        uint32_t c = *cur++;
                                        u += (c & 0x7f) * scale;
                                        scale <<= 7;
                                        if (c < 0x80)
                                            break;
                                    }
                                }
                            }
                            v = (int64_t)((u >> 1u) ^ -(int64_t)(u & 1));
                            if (kind == type_int)
                            {
                                if (row < max_rows)
                                {
                                    reinterpret_cast<int32_t *>(dataptr)[row] = static_cast<int32_t>(v);
                                }
                            }
                            else if (kind == type_long)
                            {
                                if (row < max_rows)
                                {
                                    reinterpret_cast<int64_t *>(dataptr)[row] = v;
                                }
                            }
                            else // string or enum
                            {
                                size_t count = 0;
                                const char *ptr = 0;
                                if (kind == type_enum) // dictionary
                                {
                                    size_t idx = schema[i].count + v;
                                    if (idx < num_dictionary_entries)
                                    {
                                        ptr = global_dictionary[idx].ptr;
                                        count = global_dictionary[idx].count;
                                    }
                                }
                                else if (v > 0 && cur + v <= end) // string
                                {
                                    ptr = reinterpret_cast<const char *>(cur);
                                    count = (size_t)v;
                                    cur += count;
                                }
                                if (row < max_rows)
                                {
                                    reinterpret_cast<nvstrdesc_s *>(dataptr)[row].ptr = ptr;
                                    reinterpret_cast<nvstrdesc_s *>(dataptr)[row].count = count;
                                }
                            }
                        }
                        break;

                    case type_float:
                        {
                            uint32_t v;
                            if (cur + 3 < end)
                            {
                                v = (cur[3] << 24) | (cur[2] << 16) | (cur[1] << 8) | cur[0];
                                cur += 4;
                            }
                            else
                            {
                                v = 0;
                            }
                            if (row < max_rows)
                            {
                                reinterpret_cast<uint32_t *>(dataptr)[row] = v;
                            }
                        }
                        break;

                    case type_double:
                        {
                            uint2 v;
                            if (cur + 7 < end)
                            {
                                v.x = (cur[3] << 24) | (cur[2] << 16) | (cur[1] << 8) | cur[0];
                                v.y = (cur[7] << 24) | (cur[6] << 16) | (cur[5] << 8) | cur[4];
                                cur += 8;
                            }
                            else
                            {
                                v.x = v.y = 0;
                            }
                            if (row < max_rows)
                            {
                                reinterpret_cast<uint2 *>(dataptr)[row] = v;
                            }
                        }
                        break;

                    case type_boolean:
                        {
                            uint8_t v = (cur < end) ? *cur++ : 0;
                            if (row < max_rows)
                            {
                                reinterpret_cast<uint8_t *>(dataptr)[row] = (v) ? 1 : 0;
                            }
                        }
                        break;
                    }
                }
                i++;
                while (skip > 0 && i < schema_len)
                {
                    if (schema[i].kind >= type_record)
                    {
                        skip += schema[i].count;
                    }
                    ++i;
                    --skip;
                }
            }
        }
        if (nrows <= 1)
        {
            cur = start + SHFL0(static_cast<uint32_t>(cur - start));
        }
        else
        {
            cur = start + nrows * min_row_size;
        }
        SYNCWARP();
        cur_row += nrows;
        rows_remaining -= nrows;
    }
}


/**
 * @brief Launches kernel for decoding column data
 *
 * @param[in] blocks Data block descriptions
 * @param[in] schema Schema description
 * @param[in] global_dictionary Global dictionary entries
 * @param[in] avro_data Raw block data
 * @param[in] num_blocks Number of blocks
 * @param[in] schema_len Number of entries in schema
 * @param[in] num_dictionary_entries Number of entries in global dictionary
 * @param[in] max_rows Maximum number of rows to load
 * @param[in] first_row Crop all rows below first_row
 * @param[in] min_row_size Minimum size in bytes of a row
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t __host__ DecodeAvroColumnData(block_desc_s *blocks, schemadesc_s *schema, nvstrdesc_s *global_dictionary, const uint8_t *avro_data,
    uint32_t num_blocks, uint32_t schema_len, uint32_t num_dictionary_entries, size_t max_rows, size_t first_row, uint32_t min_row_size, cudaStream_t stream)
{
    dim3 dim_block(32, NWARPS); // NWARPS warps per threadblock
    dim3 dim_grid((num_blocks + NWARPS - 1) / NWARPS, 1); // 1 warp per datablock, NWARPS datablocks per threadblock
    gpuDecodeAvroColumnData <<< dim_grid, dim_block, 0, stream >>>(blocks, schema, global_dictionary, avro_data, num_blocks, schema_len, num_dictionary_entries, min_row_size, max_rows, first_row);
    return cudaSuccess;
}


}}}} // cudf::io::avro::gpu namespace

