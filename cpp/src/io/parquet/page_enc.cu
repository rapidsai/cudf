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
#include "parquet_gpu.h"
#include <io/utilities/block_utils.cuh>

namespace cudf {
namespace io {
namespace parquet {
namespace gpu {

struct frag_init_state_s
{
    EncColumnDesc col;
    PageFragment frag;
    volatile uint32_t scratch_red[32];
};


/**
 * @brief Initializes encoder page fragments
 *
 * @param[in] frag Fragment array [fragment_id][column_id]
 * @param[in] col_desc Column description array [column_id]
 * @param[in] num_fragments Number of fragments per column
 * @param[in] num_columns Number of columns
 *
 **/
// blockDim {512,1,1}
__global__ void __launch_bounds__(512)
gpuInitPageFragments(PageFragment *frag, const EncColumnDesc *col_desc, int32_t num_fragments, int32_t num_columns, uint32_t fragment_size, uint32_t max_num_rows)
{
    __shared__ __align__(16) frag_init_state_s state_g;

    frag_init_state_s * const s = &state_g;
    uint32_t t = threadIdx.x;
    uint32_t start_row, nrows, dtype_len, dtype;

    if (t < sizeof(EncColumnDesc) / sizeof(uint32_t))
    {
        reinterpret_cast<uint32_t *>(&s->col)[t] = reinterpret_cast<const uint32_t *>(&col_desc[blockIdx.x])[t];
    }
    __syncthreads();
    start_row = blockIdx.y * fragment_size;
    if (!t)
    {
        s->col.num_rows = min(s->col.num_rows, max_num_rows);
        s->frag.num_rows = min(fragment_size, max_num_rows - min(start_row, max_num_rows));
        s->frag.non_nulls = 0;
        s->frag.fragment_data_size = 0;
    }
    dtype = s->col.physical_type;
    dtype_len = (dtype == INT64 || dtype == INT64) ? 8 : (dtype == BOOLEAN) ? 1 : 4;
    __syncthreads();
    nrows = s->frag.num_rows;
    for (uint32_t i = 0; i < nrows; i += 512)
    {
        const uint32_t *valid = s->col.valid_map_base;
        uint32_t row = start_row + i + t;
        uint32_t is_valid = (row < s->col.num_rows) ? (valid[row >> 5] >> (row & 0x1f)) & 1 : (valid) ? 1 : 0;
        uint32_t valid_warp = BALLOT(is_valid);
        uint32_t len;
        if (is_valid) {
            len = dtype_len;
            if (dtype == BYTE_ARRAY) {
                len += (uint32_t)reinterpret_cast<const nvstrdesc_s *>(s->col.column_data_base)[row].count;
            }
        } else {
            len = 0;
        }
        len = WarpReduceSum32(len);
        if (!(t & 0x1f)) {
            s->scratch_red[(t >> 5) + 0] = __popc(valid_warp);
            s->scratch_red[(t >> 5) + 16] = len;
        }
        __syncthreads();
        if (t < 32) {
            uint32_t non_nulls = WarpReduceSum16((t < 16) ? s->scratch_red[t] : 0);
            len = WarpReduceSum16((t < 16) ? s->scratch_red[t + 16] : 0);
            if (!t) {
                s->frag.non_nulls = s->frag.non_nulls + non_nulls;
                s->frag.fragment_data_size += len;
            }
        }
    }
    __syncthreads();
    if (t < sizeof(PageFragment) / sizeof(uint32_t))
    {
        reinterpret_cast<uint32_t *>(&frag[blockIdx.y * num_columns + blockIdx.x])[t] = reinterpret_cast<uint32_t *>(&s->frag)[t];
    }
}



/**
 * @brief Launches kernel for initializing encoder page fragments
 *
 * @param[in] frag Fragment array [fragment_id][column_id]
 * @param[in] col_desc Column description array [column_id]
 * @param[in] num_fragments Number of fragments per column
 * @param[in] num_columns Number of columns
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t InitPageFragments(PageFragment *frag, const EncColumnDesc *col_desc, int32_t num_fragments, int32_t num_columns, uint32_t fragment_size, uint32_t num_rows, cudaStream_t stream)
{
    dim3 dim_grid(num_columns, num_fragments);  // 1 threadblock per fragment
    gpuInitPageFragments <<< dim_grid, 512, 0, stream >>> (frag, col_desc, num_fragments, num_columns, fragment_size, num_rows);
    return cudaSuccess;
}


} // namespace gpu
} // namespace parquet
} // namespace io
} // namespace cudf
