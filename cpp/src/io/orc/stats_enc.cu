/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <io/utilities/block_utils.cuh>

namespace cudf {
namespace io {
namespace orc {
namespace gpu {

/**
 * @brief Initializes statistics groups
 *
 * @param[out] groups Statistics groups
 * @param[in] cols Column descriptors
 * @param[in] num_columns Number of columns
 * @param[in] num_rowgroups Number of rowgroups
 * @param[in] row_index_stride Rowgroup size in rows
 *
 **/
// blockDim {128,1,1}
__global__ void __launch_bounds__(128)
gpuInitStatisticsGroups(statistics_group *groups, const stats_column_desc *cols,
                        uint32_t num_columns, uint32_t num_rowgroups, uint32_t row_index_stride) {
  __shared__ __align__(4) volatile statistics_group grp_g[4];
  uint32_t col_id = blockIdx.y;
  uint32_t ck_id = (blockIdx.x * 4) + (threadIdx.x >> 5);
  uint32_t t = threadIdx.x & 0x1f;
  volatile statistics_group *grp = &grp_g[threadIdx.x >> 5];
  if (ck_id < num_rowgroups) {
    if (!t) {
      uint32_t num_rows = cols[col_id].num_rows;
      grp->col = &cols[col_id];
      grp->start_row = ck_id * row_index_stride;
      grp->num_rows = min(num_rows - min(ck_id * row_index_stride, num_rows), row_index_stride);
      __threadfence_block();
    }
    SYNCWARP();
    if (t < sizeof(statistics_group) / sizeof(uint32_t)) {
      reinterpret_cast<uint32_t *>(&groups[col_id * num_rowgroups + ck_id])[t] = reinterpret_cast<volatile uint32_t *>(grp)[t];
    }
  } 
}


/**
 * @brief Get the buffer size and offsets of encoded statistics
 *
 * @param[in,out] groups Statistics merge groups
 * @param[in] statistics_count Number of statistics buffers
 *
 **/
// blockDim {1024,1,1}
__global__ void __launch_bounds__(1024, 1)
gpuInitStatisticsBufferSize(statistics_merge_group *groups, const statistics_chunk *chunks, uint32_t statistics_count) {
  __shared__ volatile uint32_t scratch_red[32];
  __shared__ volatile uint32_t stats_size;
  uint32_t t = threadIdx.x;
  if (!t) {
    stats_size = 0;
  }
  __syncthreads();
  for (uint32_t start = 0; start < statistics_count; start += 1024) {
    uint32_t stats_len = 0, stats_pos;
    uint32_t idx = start + t;
    if (idx < statistics_count) {
      const stats_column_desc *col = groups[idx].col;
      statistics_dtype dtype = col->stats_dtype;
      switch(dtype) {
      case dtype_bool8:
        stats_len = 12 + 1 + 1 + 10;
        break;
      case dtype_int8:
      case dtype_int16:
      case dtype_int32:
      case dtype_date32:
      case dtype_int64:
      case dtype_timestamp64:
        stats_len = 12 + 1 + 3 * (1 + 10);
        break;
      case dtype_float32:
      case dtype_float64:
        stats_len = 12 + 1 + 3 * (1 + 8);
        break;
      case dtype_decimal64:
      case dtype_decimal128:
        stats_len = 12 + 2 + 3 * (1 + 40);
        break;
      case dtype_string:
        stats_len = 12 + 5 + 3 * (1 + 10) + chunks[idx].min_value.str_val.length + chunks[idx].max_value.str_val.length;
        break;
      default: break;
      }
    }
    stats_pos = WarpReducePos32(stats_len, t);
    if ((t & 0x1f) == 0x1f) {
      scratch_red[t >> 5] = stats_pos;
    }
    __syncthreads();
    if (t < 32) {
      scratch_red[t] = WarpReducePos32(scratch_red[t], t);
    }
    __syncthreads();
    if (t >= 32) {
      stats_pos += scratch_red[(t >> 5) - 1];
    }
    stats_pos += stats_size;
    if (idx < statistics_count) {
      groups[idx].start_chunk = stats_pos - stats_len;
      groups[idx].num_chunks = stats_len;
    }
    __syncthreads();
    if (t == 1023) {
      stats_size = stats_pos;
    }
  }
}


struct stats_state_s {
  uint8_t *base;              ///< Output buffer start
  uint8_t *end;               ///< Output buffer end
  statistics_chunk ck;
  statistics_merge_group grp;
  stats_column_desc col;
  // ORC stats
  uint64_t numberOfValues;
  uint8_t hasNull;
};


// Protobuf varint encoding for unsigned int
__device__ inline uint8_t *pb_encode_uint(uint8_t *p, uint64_t v) {
  while (v > 0x7f) {
    *p++ = ((uint32_t)v | 0x80);
    v >>= 7;
  }
  *p++ = v;
  return p;
}

// Protobuf field encoding for unsigned int
__device__ inline uint8_t *pb_put_uint(uint8_t *p, uint32_t id, uint64_t v) {
  p[0] = id * 8 + PB_TYPE_VARINT; // NOTE: Assumes id < 16
  return pb_encode_uint(p + 1, v);
}

// Protobuf field encoding for signed int
__device__ inline uint8_t *pb_put_int(uint8_t *p, uint32_t id, int64_t v) {
  int64_t s = (v < 0);
  return pb_put_uint(p, id, (v ^ -s) * 2 + s);
}

// Protobuf field encoding for 'packed' unsigned int (single value)
__device__ inline uint8_t *pb_put_packed_uint(uint8_t *p, uint32_t id, uint64_t v) {
  uint8_t *p2 = pb_encode_uint(p + 2, v);
  p[0] = id * 8 + PB_TYPE_FIXEDLEN;
  p[1] = static_cast<uint8_t>(p2 - (p + 2));
  return p2;
}

// Protobuf field encoding for binary/string
__device__ inline uint8_t *pb_put_binary(uint8_t *p, uint32_t id, const void *bytes, uint32_t len) {
  p[0] = id * 8 + PB_TYPE_FIXEDLEN;
  p = pb_encode_uint(p + 1, len);
  memcpy(p, bytes, len);
  return p + len;
}

// Protobuf field encoding for 64-bit raw encoding (double)
__device__ inline uint8_t *pb_put_fixed64(uint8_t *p, uint32_t id, const void *raw64) {
  p[0] = id * 8 + PB_TYPE_FIXED64;
  memcpy(p + 1, raw64, 8);
  return p + 9;
}



/**
 * @brief Encode statistics in ORC protobuf format
 *
 * @param[in,out] groups Statistics merge groups
 * @param[in,out] chunks Statistics data
 * @param[in] statistics_count Number of statistics buffers
 *
 **/
// blockDim {128,1,1}
__global__ void __launch_bounds__(128)
gpuEncodeStatistics(uint8_t *blob_bfr, statistics_merge_group *groups, const statistics_chunk *chunks,
                    uint32_t statistics_count) {
  __shared__ __align__(8) stats_state_s state_g[4];
  uint32_t t = threadIdx.x & 0x1f;
  uint32_t idx = blockIdx.x * 4 + (threadIdx.x >> 5);
  stats_state_s * const s = &state_g[threadIdx.x >> 5];
  if (idx < statistics_count) {
    if (t < sizeof(statistics_chunk) / sizeof(uint32_t)) {
      reinterpret_cast<uint32_t *>(&s->ck)[t] = reinterpret_cast<const uint32_t *>(&chunks[idx])[t];
    }
    if (t < sizeof(statistics_merge_group) / sizeof(uint32_t)) {
      reinterpret_cast<uint32_t *>(&s->grp)[t] = reinterpret_cast<uint32_t *>(&groups[idx])[t];
    }
  }
  __syncthreads();
  if (idx < statistics_count) {
    if (t < sizeof(stats_column_desc) / sizeof(uint32_t)) {
      reinterpret_cast<uint32_t *>(&s->col)[t] = reinterpret_cast<const uint32_t *>(s->grp.col)[t];
    }
    if (t == 0) {
      s->base = blob_bfr + s->grp.start_chunk;
      s->end = blob_bfr + s->grp.start_chunk + s->grp.num_chunks;
    }
  }
  __syncthreads();
  // Encode and update actual bfr size
  if (idx < statistics_count && t == 0) {
    uint8_t *cur = pb_put_uint(s->base, 1, s->ck.non_nulls);
    uint8_t *fld_start = cur;
    switch(s->col.stats_dtype) {
    case dtype_int8:
    case dtype_int16:
    case dtype_int32:
    case dtype_int64:
      // IntegerStatistics = 2
      if (s->ck.has_minmax || s->ck.has_sum) {
        *cur = 2 * 8 + PB_TYPE_FIXEDLEN;
        cur += 2;
        if (s->ck.has_minmax) {
          cur = pb_put_int(cur, 1, s->ck.min_value.i_val);
          cur = pb_put_int(cur, 2, s->ck.max_value.i_val);
        }
        if (s->ck.has_sum) {
          cur = pb_put_int(cur, 3, s->ck.sum.i_val);
        }
        fld_start[1] = cur - (fld_start + 2);
      }
      break;
    case dtype_float32:
    case dtype_float64:
      // DoubleStatistics = 3
      if (s->ck.has_minmax) {
        *cur = 3 * 8 + PB_TYPE_FIXEDLEN;
        cur += 2;
        cur = pb_put_fixed64(cur, 1, &s->ck.min_value.fp_val);
        cur = pb_put_fixed64(cur, 2, &s->ck.max_value.fp_val);
        fld_start[1] = cur - (fld_start + 2);
      }
      break;
    case dtype_string:
      // StringStatistics = 4
      if (s->ck.has_minmax && s->ck.has_sum) {
        uint32_t sz = (pb_put_uint(cur, 3, s->ck.sum.i_val) - cur)
                    + (pb_put_uint(cur, 1, s->ck.min_value.str_val.length) - cur)
                    + (pb_put_uint(cur, 2, s->ck.max_value.str_val.length) - cur)
                    + s->ck.min_value.str_val.length + s->ck.max_value.str_val.length;
        cur[0] = 4 * 8 + PB_TYPE_FIXEDLEN;
        cur = pb_encode_uint(cur + 1, sz);
        cur = pb_put_binary(cur, 1, s->ck.min_value.str_val.ptr, s->ck.min_value.str_val.length);
        cur = pb_put_binary(cur, 2, s->ck.max_value.str_val.ptr, s->ck.max_value.str_val.length);
        cur = pb_put_uint(cur, 3, s->ck.sum.i_val);
      }
      break;
    case dtype_bool8:
      // BucketStatistics = 5
      if (s->ck.has_sum) { // Sum is equal to the number of 'true' values
        cur[0] = 5 * 8 + PB_TYPE_FIXEDLEN;
        cur = pb_put_packed_uint(cur + 2, 1, s->ck.sum.i_val);
        fld_start[1] = cur - (fld_start + 2);
      }
      break;
    case dtype_decimal64:
    case dtype_decimal128:
      // DecimalStatistics = 6
      if (s->ck.has_minmax) {
        // TODO: Decimal support (decimal min/max stored as strings)
      }
      break;
    case dtype_date32:
      // DateStatistics = 7
      if (s->ck.has_minmax) {
        cur[0] = 7 * 8 + PB_TYPE_FIXEDLEN;
        cur += 2;
        cur = pb_put_int(cur, 1, s->ck.min_value.i_val);
        cur = pb_put_int(cur, 2, s->ck.max_value.i_val);
        fld_start[1] = cur - (fld_start + 2);
      }      
      break;
    case dtype_timestamp64:
      // TimestampStatistics = 9
      if (s->ck.has_minmax) {
        cur[0] = 7 * 8 + PB_TYPE_FIXEDLEN;
        cur += 2;
        cur = pb_put_int(cur, 3, s->ck.min_value.i_val); // minimumUtc
        cur = pb_put_int(cur, 4, s->ck.max_value.i_val); // maximumUtc
        fld_start[1] = cur - (fld_start + 2);
      }      
      break;
    default:
      break;
    }
    groups[idx].num_chunks = static_cast<uint32_t>(cur - s->base);
  }
}


/**
 * @brief Launches kernels to initialize statistics collection
 *
 * @param[out] groups Statistics groups (rowgroup-level)
 * @param[in] cols Column descriptors
 * @param[in] num_columns Number of columns
 * @param[in] num_rowgroups Number of rowgroups
 * @param[in] row_index_stride Rowgroup size in rows
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t OrcInitStatisticsGroups(statistics_group *groups, const stats_column_desc *cols, uint32_t num_columns,
                                    uint32_t num_rowgroups, uint32_t row_index_stride, cudaStream_t stream)
{
    dim3 dim_groups((num_rowgroups+3) >> 2, num_columns);
    gpuInitStatisticsGroups <<< dim_groups, 128, 0, stream >>>(groups, cols, num_columns, num_rowgroups, row_index_stride);

    return cudaSuccess;
}


/**
 * @brief Launches kernels to return statistics buffer offsets and sizes
 *
 * @param[in,out] groups Statistics merge groups
 * @param[in] chunks Satistics chunks
 * @param[in] statistics_count Number of statistics buffers to encode
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t OrcInitStatisticsBufferSize(statistics_merge_group *groups, const statistics_chunk *chunks,
                                        uint32_t statistics_count, cudaStream_t stream)
{
    gpuInitStatisticsBufferSize <<< 1, 1024, 0, stream >>>(groups, chunks, statistics_count);
    return cudaSuccess;
}


/**
 * @brief Launches kernel to encode statistics in ORC protobuf format
 *
 * @param[out] blob_bfr Output buffer for statistics blobs
 * @param[in,out] groups Statistics merge groups
 * @param[in,out] chunks Statistics data
 * @param[in] statistics_count Number of statistics buffers
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t OrcEncodeStatistics(uint8_t *blob_bfr, statistics_merge_group *groups, const statistics_chunk *chunks,
                                uint32_t statistics_count, cudaStream_t stream)
{
    gpuEncodeStatistics <<< (statistics_count + 3) >> 2, 128, 0, stream >>>(blob_bfr, groups, chunks, statistics_count);
    return cudaSuccess;
}


} // namespace gpu
} // namespace orc
} // namespace io
} // namespace cudf
