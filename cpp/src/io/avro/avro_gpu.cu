/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

using cudf::detail::device_span;

namespace cudf {
namespace io {
namespace avro {
namespace gpu {
#define NWARPS 16
#define MAX_SHARED_SCHEMA_LEN 1000

/*
 * Avro varint encoding - see
 * https://avro.apache.org/docs/1.2.0/spec.html#binary_encoding
 */
static inline int64_t __device__ avro_decode_zigzag_varint(const uint8_t *&cur, const uint8_t *end)
{
  uint64_t u = 0;
  if (cur < end) {
    u = *cur++;
    if (u > 0x7f) {
      uint64_t scale = 128;
      u &= 0x7f;
      while (cur < end) {
        uint32_t c = *cur++;
        u += (c & 0x7f) * scale;
        scale <<= 7;
        if (c < 0x80) break;
      }
    }
  }
  return (int64_t)((u >> 1u) ^ -(int64_t)(u & 1));
}

/**
 * @brief Decode a row of values given an avro schema
 *
 * @param[in] schema Schema description
 * @param[in] schema_g Global schema in device mem
 * @param[in] schema_len Number of schema entries
 * @param[in] row Current row
 * @param[in] max_rows Total number of rows
 * @param[in] cur Current input data pointer
 * @param[in] end End of input data
 * @param[in] global_Dictionary Global dictionary entries
 *
 * @return data pointer at the end of the row (start of next row)
 *
 **/
static const uint8_t *__device__ avro_decode_row(const schemadesc_s *schema,
                                                 schemadesc_s *schema_g,
                                                 uint32_t schema_len,
                                                 size_t row,
                                                 size_t max_rows,
                                                 const uint8_t *cur,
                                                 const uint8_t *end,
                                                 device_span<nvstrdesc_s> global_dictionary)
{
  uint32_t array_start = 0, array_repeat_count = 0;
  int array_children = 0;
  for (uint32_t i = 0; i < schema_len;) {
    uint32_t kind = schema[i].kind;
    int skip      = 0;

    if (kind == type_union) {
      int skip_after;
      if (cur >= end) break;
      skip       = (*cur++) >> 1;  // NOTE: Assumes 1-byte union member
      skip_after = schema[i].count - skip - 1;
      ++i;
      while (skip > 0 && i < schema_len) {
        if (schema[i].kind >= type_record) { skip += schema[i].count; }
        ++i;
        --skip;
      }
      if (i >= schema_len || skip_after < 0) break;
      kind = schema[i].kind;
      skip = skip_after;
    }

    void *dataptr = schema[i].dataptr;
    switch (kind) {
      case type_null:
        if (dataptr != nullptr && row < max_rows) {
          atomicAnd(static_cast<uint32_t *>(dataptr) + (row >> 5), ~(1 << (row & 0x1f)));
          atomicAdd(&schema_g[i].count, 1);
        }
        break;

      case type_int:
      case type_long:
      case type_bytes:
      case type_string:
      case type_enum: {
        int64_t v = avro_decode_zigzag_varint(cur, end);
        if (kind == type_int) {
          if (dataptr != nullptr && row < max_rows) {
            static_cast<int32_t *>(dataptr)[row] = static_cast<int32_t>(v);
          }
        } else if (kind == type_long) {
          if (dataptr != nullptr && row < max_rows) { static_cast<int64_t *>(dataptr)[row] = v; }
        } else {  // string or enum
          size_t count    = 0;
          const char *ptr = 0;
          if (kind == type_enum) {  // dictionary
            size_t idx = schema[i].count + v;
            if (idx < global_dictionary.size()) {
              ptr   = global_dictionary[idx].ptr;
              count = global_dictionary[idx].count;
            }
          } else if (v >= 0 && cur + v <= end) {  // string
            ptr   = reinterpret_cast<const char *>(cur);
            count = (size_t)v;
            cur += count;
          }
          if (dataptr != nullptr && row < max_rows) {
            static_cast<nvstrdesc_s *>(dataptr)[row].ptr   = ptr;
            static_cast<nvstrdesc_s *>(dataptr)[row].count = count;
          }
        }
      } break;

      case type_float:
        if (dataptr != nullptr && row < max_rows) {
          uint32_t v;
          if (cur + 3 < end) {
            v = unaligned_load32(cur);
            cur += 4;
          } else {
            v = 0;
          }
          static_cast<uint32_t *>(dataptr)[row] = v;
        } else {
          cur += 4;
        }
        break;

      case type_double:
        if (dataptr != nullptr && row < max_rows) {
          uint64_t v;
          if (cur + 7 < end) {
            v = unaligned_load64(cur);
            cur += 8;
          } else {
            v = 0;
          }
          static_cast<uint64_t *>(dataptr)[row] = v;
        } else {
          cur += 8;
        }
        break;

      case type_boolean:
        if (dataptr != nullptr && row < max_rows) {
          uint8_t v                            = (cur < end) ? *cur : 0;
          static_cast<uint8_t *>(dataptr)[row] = (v) ? 1 : 0;
        }
        cur++;
        break;

      case type_array: {
        int32_t array_block_count = avro_decode_zigzag_varint(cur, end);
        if (array_block_count < 0) {
          avro_decode_zigzag_varint(cur, end);  // block size in bytes, ignored
          array_block_count = -array_block_count;
        }
        array_start        = i;
        array_repeat_count = array_block_count;
        array_children     = 1;
        if (array_repeat_count == 0) {
          skip += schema[i].count;  // Should always be 1
        }
      } break;
    }
    if (array_repeat_count != 0) {
      array_children--;
      if (schema[i].kind >= type_record) { array_children += schema[i].count; }
    }
    i++;
    while (skip > 0 && i < schema_len) {
      if (schema[i].kind >= type_record) { skip += schema[i].count; }
      ++i;
      --skip;
    }
    // If within an array, check if we reached the last item
    if (array_repeat_count != 0 && array_children <= 0 && cur < end) {
      if (!--array_repeat_count) {
        i = array_start;  // Restart at the array parent
      } else {
        i              = array_start + 1;  // Restart after the array parent
        array_children = schema[array_start].count;
      }
    }
  }
  return cur;
}

/**
 * @brief Decode column data
 *
 * @param[in] blocks Data block descriptions
 * @param[in] schema Schema description
 * @param[in] global_Dictionary Global dictionary entries
 * @param[in] avro_data Raw block data
 * @param[in] num_blocks Number of blocks
 * @param[in] schema_len Number of entries in schema
 * @param[in] min_row_size Minimum size in bytes of a row
 * @param[in] max_rows Maximum number of rows to load
 * @param[in] first_row Crop all rows below first_row
 *
 **/
// blockDim {32,NWARPS,1}
extern "C" __global__ void __launch_bounds__(NWARPS * 32, 2)
  gpuDecodeAvroColumnData(block_desc_s *blocks,
                          schemadesc_s *schema_g,
                          device_span<nvstrdesc_s> global_dictionary,
                          const uint8_t *avro_data,
                          uint32_t num_blocks,
                          uint32_t schema_len,
                          uint32_t min_row_size,
                          size_t max_rows,
                          size_t first_row)
{
  __shared__ __align__(8) schemadesc_s g_shared_schema[MAX_SHARED_SCHEMA_LEN];
  __shared__ __align__(8) block_desc_s blk_g[NWARPS];

  schemadesc_s *schema;
  block_desc_s *const blk = &blk_g[threadIdx.y];
  uint32_t block_id       = blockIdx.x * NWARPS + threadIdx.y;
  size_t cur_row;
  uint32_t rows_remaining;
  const uint8_t *cur, *end;

  // Fetch schema into shared mem if possible
  if (schema_len <= MAX_SHARED_SCHEMA_LEN) {
    for (int i = threadIdx.y * 32 + threadIdx.x; i < schema_len; i += NWARPS * 32) {
      g_shared_schema[i] = schema_g[i];
    }
    __syncthreads();
    schema = g_shared_schema;
  } else {
    schema = schema_g;
  }
  if (block_id < num_blocks and threadIdx.x == 0) { *blk = blocks[block_id]; }
  __syncthreads();
  if (block_id >= num_blocks) { return; }
  cur_row        = blk->first_row;
  rows_remaining = blk->num_rows;
  cur            = avro_data + blk->offset;
  end            = cur + blk->size;
  while (rows_remaining > 0 && cur < end) {
    uint32_t nrows;
    const uint8_t *start = cur;

    if (cur_row > first_row + max_rows) break;
    if (cur + min_row_size * rows_remaining == end) {
      nrows = min(rows_remaining, 32);
      cur += threadIdx.x * min_row_size;
    } else {
      nrows = 1;
    }
    if (threadIdx.x < nrows) {
      cur = avro_decode_row(schema,
                            schema_g,
                            schema_len,
                            cur_row - first_row + threadIdx.x,
                            max_rows,
                            cur,
                            end,
                            global_dictionary);
    }
    if (nrows <= 1) {
      cur = start + SHFL0(static_cast<uint32_t>(cur - start));
    } else {
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
 * @param[in] max_rows Maximum number of rows to load
 * @param[in] first_row Crop all rows below first_row
 * @param[in] min_row_size Minimum size in bytes of a row
 * @param[in] stream CUDA stream to use, default 0
 */
void __host__ DecodeAvroColumnData(block_desc_s *blocks,
                                   schemadesc_s *schema,
                                   device_span<nvstrdesc_s> global_dictionary,
                                   const uint8_t *avro_data,
                                   uint32_t num_blocks,
                                   uint32_t schema_len,
                                   size_t max_rows,
                                   size_t first_row,
                                   uint32_t min_row_size,
                                   cudaStream_t stream)
{
  // NWARPS warps per threadblock
  dim3 const dim_block(32, NWARPS);
  // 1 warp per datablock, NWARPS datablocks per threadblock
  dim3 const dim_grid((num_blocks + NWARPS - 1) / NWARPS, 1);

  gpuDecodeAvroColumnData<<<dim_grid, dim_block, 0, stream>>>(blocks,
                                                              schema,
                                                              global_dictionary,
                                                              avro_data,
                                                              num_blocks,
                                                              schema_len,
                                                              min_row_size,
                                                              max_rows,
                                                              first_row);
}

}  // namespace gpu
}  // namespace avro
}  // namespace io
}  // namespace cudf
