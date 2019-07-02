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

namespace cudf {
namespace io {
namespace avro {
namespace gpu {

/**
 * @brief Launches kernel for decoding column data
 *
 * @param[in] blocks Data block descriptions
 * @param[in] schema Schema description
 * @param[in] global_dictionary Global dictionary entries
 * @param[in] num_blocks Number of blocks
 * @param[in] schema_len Number of entries in schema
 * @param[in] num_dictionary_entries Number of entries in global dictionary
 * @param[in] max_rows Maximum number of rows to load
 * @param[in] first_row Crop all rows below first_row
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t __host__ DecodeAvroColumnData(block_desc_s *blocks, schemadesc_s *schema, nvstrdesc_s *global_dictionary, uint32_t num_blocks,
    uint32_t schema_len, uint32_t num_dictionary_entries, size_t max_rows, size_t first_row, cudaStream_t stream)
{
    dim3 dim_block(32*16, 1);   // 16 warps per threadblock
    dim3 dim_grid((num_blocks + 0xf) >> 4, 1); // 1 warp per datablock, 16 datablocks per threadblock
    //gpuDecodeAvroColumnData <<< dim_grid, dim_block, 0, stream >>>(blocks, schema, global_dictionary, num_blocks, schema_len, num_dictionary_entries, max_rows, first_row);
    return cudaSuccess;
}


}}}} // cudf::io::avro::gpu namespace

