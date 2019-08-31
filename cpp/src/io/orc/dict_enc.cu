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

#define MAX_SHORT_DICT_ENTRIES      (10*1024)

struct dictbuild_state_s
{
    uint32_t nnz;
    DictionaryChunk chunk;
    volatile uint32_t scratch_red[32];
    uint16_t dict[MAX_SHORT_DICT_ENTRIES];
};


/**
 * @brief Fill dictionary with the indices of non-null rows
 *
 * @param[in,out] s dictionary builder state
 * @param[in] t thread id
 *
 **/
static __device__ void LoadNonNullIndices(volatile dictbuild_state_s *s, int t)
{
    if (t == 0)
    {
        s->nnz = 0;
    }
    for (uint32_t i = 0; i < s->chunk.num_rows; i += 512)
    {
        const uint32_t *valid_map = s->chunk.valid_map_base;
        uint32_t is_valid, nz_map, nz_pos;
        if (t < 16)
        {
            uint32_t row = s->chunk.start_row + i + t * 32;
            uint32_t v = (row < s->chunk.start_row + s->chunk.num_rows) ? valid_map[row >> 5] : 0;
            if (row & 0x1f)
            {
                uint32_t v1 = (row + 32 < s->chunk.start_row + s->chunk.num_rows) ? valid_map[(row >> 5) + 1] : 0;
                v = __funnelshift_r(v, v1, row & 0x1f);
            }
            s->scratch_red[t] = v;
        }
        __syncthreads();
        is_valid = (i + t < s->chunk.num_rows) ? (s->scratch_red[t >> 5] >> (t & 0x1f)) & 1 : 0;
        nz_map = BALLOT(is_valid);
        nz_pos = s->nnz + __popc(nz_map & (0x7fffffffu >> (0x1fu - ((uint32_t)t & 0x1f))));
        if (!(t & 0x1f))
        {
            s->scratch_red[16 + (t >> 5)] = __popc(nz_map);
        }
        __syncthreads();
        if (t < 32)
        {
            uint32_t nnz = s->scratch_red[16 + (t & 0xf)];
            uint32_t nnz_pos = nnz;
            for (uint32_t n = 1; n<16; n <<= 1)
            {
                uint32_t tmp = SHFL(nnz_pos, (t & ~n) | (n - 1));
                nnz_pos += (t & n) ? tmp : 0;
            }
            if (t == 0xf)
            {
                s->nnz += nnz_pos;
                __threadfence_block();
            }
            if (t <= 0xf)
                s->scratch_red[t] = nnz_pos - nnz;
        }
        __syncthreads();
        if (is_valid)
        {
            s->dict[nz_pos + s->scratch_red[t >> 5]] = i + t;
        }
        __syncthreads();
    }
}

/**
 * @brief Builds per-chunk string dictionaries
 *
 * @param[in] chunks DictionaryChunk device array [rowgroup][column]
 * @param[in] num_columns Number of columns
 * @param[in] num_rowgroups Number of row groups
 *
 **/
// blockDim {512,1,1}
extern "C" __global__ void __launch_bounds__(512)
gpuBuildOrcStringDictionaries(DictionaryChunk *chunks, uint32_t num_columns, uint32_t num_rowgroups)
{
    __shared__ __align__(16) dictbuild_state_s state_g;

    volatile dictbuild_state_s * const s = &state_g;
    uint32_t col_id = blockIdx.x;
    uint32_t group_id = blockIdx.y;
    int t = threadIdx.x;

    if (t < sizeof(DictionaryChunk) / sizeof(uint32_t))
    {
        ((volatile uint32_t *)&s->chunk)[t] = ((const uint32_t *)&chunks[group_id * num_columns + col_id])[t];
    }
    __syncthreads();
    // First, take care of NULLs, and count how many strings we have (TODO: bypass this step when there are no nulls)
    LoadNonNullIndices(s, t);
    // Sum the lengths of all the strings
    if (t == 0)
    {
        s->chunk.string_char_count = 0;
    }
    for (uint32_t i = 0; i < s->nnz; i += 512)
    {
        const nvstrdesc_s *ck_data = reinterpret_cast<const nvstrdesc_s *>(s->chunk.column_data_base) + s->chunk.start_row;
        uint32_t ck_row = (i + t < s->nnz) ? s->dict[i + t] : 0;
        uint32_t len = (i + t < s->nnz) ? ck_data[ck_row].count : 0;
        len += SHFL_XOR(len, 1);
        len += SHFL_XOR(len, 2);
        len += SHFL_XOR(len, 4);
        len += SHFL_XOR(len, 8);
        s->scratch_red[t >> 4] = len;
        __syncthreads();
        if (t < 32)
        {
            len = s->scratch_red[t];
            len += SHFL_XOR(len, 1);
            len += SHFL_XOR(len, 2);
            len += SHFL_XOR(len, 4);
            len += SHFL_XOR(len, 8);
            len += SHFL_XOR(len, 16);
            if (t == 0)
                s->chunk.string_char_count += len;
        }
        __syncthreads();
    }

    __syncthreads();
    if (!t)
    {
        chunks[group_id * num_columns + col_id].string_char_count = s->chunk.string_char_count;
    }
}


/**
 * @brief Launches kernel for building string dictionaries
 *
 * @param[in] chunks DictionaryChunk device array [rowgroup][column]
 * @param[in] num_columns Number of columns
 * @param[in] num_rowgroups Number of row groups
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t BuildOrcStringDictionaries(DictionaryChunk *chunks, uint32_t num_columns, uint32_t num_rowgroups, cudaStream_t stream)
{
    dim3 dim_block(512, 1); // 512 threads per chunk
    dim3 dim_grid(num_columns, num_rowgroups);
    gpuBuildOrcStringDictionaries <<< dim_grid, dim_block, 0, stream >>>(chunks, num_columns, num_rowgroups);
    return cudaSuccess;
}



} // namespace gpu
} // namespace orc
} // namespace io
} // namespace cudf
