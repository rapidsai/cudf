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
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

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


static inline bool __device__ nvstr_is_greater(const char *as, uint32_t alen, const char *bs, uint32_t blen)
{
    uint32_t len = min(alen, blen);
    for (uint32_t i = 0; i < len; i++)
    {
        uint8_t a = as[i];
        uint8_t b = bs[i];
        if (a != b)
        {
            return (a > b);
        }
    }
    return (alen > blen);
}

static inline bool __device__ nvstr_is_equal(const char *as, uint32_t alen, const char *bs, uint32_t blen)
{
    if (alen != blen)
        return false;
    for (uint32_t i = 0; i < alen; i++)
    {
        if (as[i] != bs[i])
        {
            return false;
        }
    }
    return true;
}


struct dictinit_state_s
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
static __device__ void LoadNonNullIndices(volatile dictinit_state_s *s, int t)
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
 * @brief Gather all non-NULL string rows and compute total character data size
 *
 * @param[in] chunks DictionaryChunk device array [rowgroup][column]
 * @param[in] num_columns Number of columns
 *
 **/
// blockDim {512,1,1}
extern "C" __global__ void __launch_bounds__(512)
gpuInitDictionaryIndices(DictionaryChunk *chunks, uint32_t num_columns)
{
    __shared__ __align__(16) dictinit_state_s state_g;

    volatile dictinit_state_s * const s = &state_g;
    uint32_t col_id = blockIdx.x;
    uint32_t group_id = blockIdx.y;
    const nvstrdesc_s *ck_data;
    uint32_t *dict_data;
    uint32_t nnz, start_row;
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
    nnz = s->nnz;
    dict_data = s->chunk.dict_data;
    start_row = s->chunk.start_row;
    ck_data = reinterpret_cast<const nvstrdesc_s *>(s->chunk.column_data_base) + start_row;
    for (uint32_t i = 0; i < s->nnz; i += 512)
    {
        uint32_t ck_row = (i + t < nnz) ? s->dict[i + t] : 0;
        uint32_t len = (i + t < nnz) ? ck_data[ck_row].count : 0;
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
        if (i + t < nnz)
        {
            dict_data[i + t] = start_row + ck_row;
        }
        __syncthreads();
    }
    if (!t)
    {
        chunks[group_id * num_columns + col_id].num_strings = nnz;
        chunks[group_id * num_columns + col_id].string_char_count = s->chunk.string_char_count;
    }
}


struct compact_state_s
{
    uint32_t *stripe_data;
    StripeDictionary stripe;
    DictionaryChunk chunk;
    volatile uint32_t scratch_red[32];
};


/**
 * @brief In-place concatenate dictionary data for all chunks in each stripe
 *
 * @param[in] stripes StripeDictionary device array [stripe][column]
 * @param[in] chunks DictionaryChunk device array [rowgroup][column]
 * @param[in] num_columns Number of columns
 *
 **/
// blockDim {1024,1,1}
extern "C" __global__ void __launch_bounds__(1024)
gpuCompactChunkDictionaries(StripeDictionary *stripes, DictionaryChunk *chunks, uint32_t num_columns)
{
    __shared__ __align__(16) compact_state_s state_g;

    volatile compact_state_s * const s = &state_g;
    uint32_t chunk_id = blockIdx.x;
    uint32_t col_id = blockIdx.y;
    uint32_t stripe_id = blockIdx.z;
    uint32_t chunk_len;
    int t = threadIdx.x;
    const uint32_t *src;
    uint32_t *dst;

    if (t < sizeof(StripeDictionary) / sizeof(uint32_t))
    {
        ((volatile uint32_t *)&s->stripe)[t] = ((const uint32_t *)&stripes[stripe_id * num_columns + col_id])[t];
    }
    __syncthreads();
    if (chunk_id >= s->stripe.num_chunks)
    {
        return;
    }
    if (t < sizeof(DictionaryChunk) / sizeof(uint32_t))
    {
        ((volatile uint32_t *)&s->chunk)[t] = ((const uint32_t *)&chunks[(s->stripe.start_chunk + chunk_id) * num_columns + col_id])[t];
    }
    chunk_len = (t < chunk_id) ? chunks[(s->stripe.start_chunk + t) * num_columns + col_id].num_strings : 0;
    if (chunk_id != 0)
    {
        chunk_len += SHFL_XOR(chunk_len, 1);
        chunk_len += SHFL_XOR(chunk_len, 2);
        chunk_len += SHFL_XOR(chunk_len, 4);
        chunk_len += SHFL_XOR(chunk_len, 8);
        chunk_len += SHFL_XOR(chunk_len, 16);
        if (!(t & 0x1f))
            s->scratch_red[t >> 5] = chunk_len;
        __syncthreads();
        if (t < 32)
        {
            chunk_len = s->scratch_red[t];
            chunk_len += SHFL_XOR(chunk_len, 1);
            chunk_len += SHFL_XOR(chunk_len, 2);
            chunk_len += SHFL_XOR(chunk_len, 4);
            chunk_len += SHFL_XOR(chunk_len, 8);
            chunk_len += SHFL_XOR(chunk_len, 16);
        }
    }
    if (!t)
    {
        s->stripe_data = s->stripe.dict_data + chunk_len;
    }
    __syncthreads();
    chunk_len = s->chunk.num_strings;
    src = s->chunk.dict_data;
    dst = s->stripe_data;
    if (src != dst)
    {
        for (uint32_t i = 0; i < chunk_len; i += 1024)
        {
            uint32_t idx = (i + t < chunk_len) ? src[i + t] : 0;
            __syncthreads();
            if (i + t < chunk_len)
                dst[i + t] = idx;
            __syncthreads();
        }
    }
}


struct build_state_s
{
    uint32_t total_dupes;
    StripeDictionary stripe;
    volatile uint32_t scratch_red[32];
};

/**
 * @brief Eliminate duplicates in-place and generate column dictionary index
 *
 * @param[in] stripes StripeDictionary device array [stripe][column]
 * @param[in] num_columns Number of string columns
 *
 **/
// NOTE: Prone to poor utilization on small datasets due to 1 block per dictionary
// blockDim {1024,1,1}
extern "C" __global__ void __launch_bounds__(1024)
gpuBuildStripeDictionaries(StripeDictionary *stripes, uint32_t num_columns)
{
    __shared__ __align__(16) build_state_s state_g;

    volatile build_state_s * const s = &state_g;
    uint32_t col_id = blockIdx.x;
    uint32_t stripe_id = blockIdx.y;
    uint32_t num_strings;
    uint32_t *dict_data, *dict_index;
    uint32_t dict_char_count;
    const nvstrdesc_s *str_data;
    int t = threadIdx.x;

    if (t < sizeof(StripeDictionary) / sizeof(uint32_t))
    {
        ((volatile uint32_t *)&s->stripe)[t] = ((const uint32_t *)&stripes[stripe_id * num_columns + col_id])[t];
    }
    if (t == 31 * 32)
    {
        s->total_dupes = 0;
    }
    __syncthreads();
    num_strings = s->stripe.num_strings;
    dict_data = s->stripe.dict_data;
    dict_index = s->stripe.dict_index;
    str_data = reinterpret_cast<const nvstrdesc_s *>(s->stripe.column_data_base);
    dict_char_count = 0;
    for (uint32_t i = 0; i < num_strings; i += 1024)
    {
        uint32_t cur = (i + t < num_strings) ? dict_data[i + t] : 0;
        uint32_t dupe_mask, dupes_before, cur_len = 0;
        const char *cur_ptr;
        bool is_dupe = false;
        if (i + t < num_strings)
        {
            cur_ptr = str_data[cur].ptr;
            cur_len = str_data[cur].count;
        }
        if (i + t != 0 && i + t < num_strings)
        {
            uint32_t prev = dict_data[i + t - 1];
            is_dupe = nvstr_is_equal(cur_ptr, cur_len, str_data[prev].ptr, str_data[prev].count);
        }
        dict_char_count += (is_dupe) ? 0 : cur_len;
        dupe_mask = BALLOT(is_dupe);
        dupes_before = s->total_dupes + __popc(dupe_mask & ((2 << (t & 0x1f)) - 1));
        if (!(t & 0x1f))
        {
            s->scratch_red[t >> 5] = __popc(dupe_mask);
        }
        __syncthreads();
        if (t < 32)
        {
            uint32_t warp_dupes = s->scratch_red[t];
            uint32_t warp_pos = warp_dupes;
            for (uint32_t n = 1; n<32; n <<= 1)
            {
                uint32_t tmp = SHFL(warp_pos, (t & ~n) | (n - 1));
                warp_pos += (t & n) ? tmp : 0;
            }
            if (t == 0x1f)
            {
                s->total_dupes += warp_pos;
            }
            s->scratch_red[t] = warp_pos - warp_dupes;
        }
        __syncthreads();
        if (i + t < num_strings)
        {
            dupes_before += s->scratch_red[t >> 5];
            dict_index[cur] = i + t - dupes_before;
            if (!is_dupe && dupes_before != 0)
            {
                dict_data[i + t - dupes_before] = cur;
            }
        }
        __syncthreads();
    }
    dict_char_count += SHFL_XOR(dict_char_count, 1);
    dict_char_count += SHFL_XOR(dict_char_count, 2);
    dict_char_count += SHFL_XOR(dict_char_count, 4);
    dict_char_count += SHFL_XOR(dict_char_count, 8);
    dict_char_count += SHFL_XOR(dict_char_count, 16);
    if (!(t & 0x1f))
    {
        s->scratch_red[t >> 5] = dict_char_count;
    }
    __syncthreads();
    if (t < 32)
    {
        dict_char_count = s->scratch_red[t];
        dict_char_count += SHFL_XOR(dict_char_count, 1);
        dict_char_count += SHFL_XOR(dict_char_count, 2);
        dict_char_count += SHFL_XOR(dict_char_count, 4);
        dict_char_count += SHFL_XOR(dict_char_count, 8);
        dict_char_count += SHFL_XOR(dict_char_count, 16);
    }
    if (t == 0)
    {
        stripes[stripe_id * num_columns + col_id].num_strings = num_strings - s->total_dupes;
        stripes[stripe_id * num_columns + col_id].dict_char_count = dict_char_count;
    }
}


/**
 * @brief Launches kernel for initializing dictionary chunks
 *
 * @param[in] chunks DictionaryChunk device array [rowgroup][column]
 * @param[in] num_columns Number of columns
 * @param[in] num_rowgroups Number of row groups
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t InitDictionaryIndices(DictionaryChunk *chunks, uint32_t num_columns, uint32_t num_rowgroups, cudaStream_t stream)
{
    dim3 dim_block(512, 1); // 512 threads per chunk
    dim3 dim_grid(num_columns, num_rowgroups);
    gpuInitDictionaryIndices <<< dim_grid, dim_block, 0, stream >>>(chunks, num_columns);
    return cudaSuccess;
}


struct nvstr_compare : public thrust::binary_function<uint32_t, uint32_t, bool>
{
    const nvstrdesc_s *str_data;
    nvstr_compare(const nvstrdesc_s *str_data_) : str_data(str_data_) {}
    inline __device__ bool operator()(const int a, const int b) const
    {
        return nvstr_is_greater(str_data[a].ptr, (uint32_t)str_data[a].count, str_data[b].ptr, (uint32_t)str_data[b].count);
    }
};


/**
 * @brief Launches kernel for building stripe dictionaries
 *
 * @param[in] stripes StripeDictionary device array [stripe][column]
 * @param[in] stripes_host StripeDictionary host array [stripe][column]
 * @param[in] chunks DictionaryChunk device array [rowgroup][column]
 * @param[in] num_stripes Number of stripes
 * @param[in] num_rowgroups Number of row groups
 * @param[in] num_columns Number of columns
 * @param[in] max_chunks_in_stripe Maximum number of rowgroups per stripe
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t BuildStripeDictionaries(StripeDictionary *stripes, StripeDictionary *stripes_host, DictionaryChunk *chunks,
                                    uint32_t num_stripes, uint32_t num_rowgroups, uint32_t num_columns, uint32_t max_chunks_in_stripe, cudaStream_t stream)
{
    dim3 dim_block(1024, 1); // 1024 threads per chunk
    dim3 dim_grid_compact(max_chunks_in_stripe, num_columns, num_stripes);
    dim3 dim_grid_build(num_columns, num_stripes);
    gpuCompactChunkDictionaries <<< dim_grid_compact, dim_block, 0, stream >>>(stripes, chunks, num_columns);
    for (uint32_t j = 0; j < num_stripes; j++)
    {
        for (uint32_t i = 0; i < num_columns; i++)
        {
            thrust::device_ptr<uint32_t> p = thrust::device_pointer_cast(stripes_host[j * num_columns + i].dict_data);
            const nvstrdesc_s *str_data = reinterpret_cast<const nvstrdesc_s *>(stripes_host[i].column_data_base);
        #if 1
            thrust::sort(thrust::device, p, p + stripes_host[j * num_columns + i].num_strings, nvstr_compare(str_data));
        #else
            // Requires the --expt-extended-lambda nvcc flag (same perf as above)
            thrust::sort(p, p + stripes_host[j * num_columns + i].num_strings,
                [str_data] __device__(const uint32_t &lhs, const uint32_t &rhs) {
                return nvstr_is_greater(str_data[lhs].ptr, (uint32_t)str_data[lhs].count, str_data[rhs].ptr, (uint32_t)str_data[rhs].count);
            });
        #endif
        }
    }
    gpuBuildStripeDictionaries <<< dim_grid_build, dim_block, 0, stream >>>(stripes, num_columns);
    return cudaSuccess;
}


} // namespace gpu
} // namespace orc
} // namespace io
} // namespace cudf
