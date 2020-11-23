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
#include <io/utilities/block_utils.cuh>
#include "orc_common.h"
#include "orc_gpu.h"

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
constexpr unsigned int init_threads_per_group = 32;
constexpr unsigned int init_groups_per_block  = 4;
constexpr unsigned int init_threads_per_block = init_threads_per_group * init_groups_per_block;

__global__ void __launch_bounds__(init_threads_per_block)
  gpu_init_statistics_groups(statistics_group *groups,
                             const stats_column_desc *cols,
                             uint32_t num_columns,
                             uint32_t num_rowgroups,
                             uint32_t row_index_stride)
{
  __shared__ __align__(4) statistics_group group_g[init_groups_per_block];
  uint32_t col_id         = blockIdx.y;
  uint32_t chunk_id       = (blockIdx.x * init_groups_per_block) + threadIdx.y;
  uint32_t t              = threadIdx.x;
  statistics_group *group = &group_g[threadIdx.y];
  if (chunk_id < num_rowgroups and t == 0) {
    uint32_t num_rows = cols[col_id].num_rows;
    group->col        = &cols[col_id];
    group->start_row  = chunk_id * row_index_stride;
    group->num_rows = min(num_rows - min(chunk_id * row_index_stride, num_rows), row_index_stride);
    groups[col_id * num_rowgroups + chunk_id] = *group;
  }
}

/**
 * @brief Get the buffer size and offsets of encoded statistics
 *
 * @param[in,out] groups Statistics merge groups
 * @param[in] statistics_count Number of statistics buffers
 *
 **/
constexpr unsigned int buffersize_reduction_dim = 32;
constexpr unsigned int buffersize_threads_per_block =
  buffersize_reduction_dim * buffersize_reduction_dim;
constexpr unsigned int pb_fld_hdrlen     = 1;
constexpr unsigned int pb_fld_hdrlen16   = 2;  // > 127-byte length
constexpr unsigned int pb_fld_hdrlen32   = 5;  // > 16KB length
constexpr unsigned int pb_fldlen_int64   = 10;
constexpr unsigned int pb_fldlen_float64 = 8;
constexpr unsigned int pb_fldlen_decimal = 40;  // Assume decimal2string fits in 40 characters
constexpr unsigned int pb_fldlen_bucket1 = 1 + pb_fldlen_int64;
constexpr unsigned int pb_fldlen_common  = 2 * pb_fld_hdrlen + pb_fldlen_int64;

__global__ void __launch_bounds__(buffersize_threads_per_block, 1)
  gpu_init_statistics_buffersize(statistics_merge_group *groups,
                                 const statistics_chunk *chunks,
                                 uint32_t statistics_count)
{
  __shared__ volatile uint32_t scratch_red[buffersize_reduction_dim];
  __shared__ volatile uint32_t stats_size;
  uint32_t tx = threadIdx.x;
  uint32_t ty = threadIdx.y;
  uint32_t t  = ty * buffersize_reduction_dim + tx;
  if (!t) { stats_size = 0; }
  __syncthreads();
  for (uint32_t start = 0; start < statistics_count; start += buffersize_threads_per_block) {
    uint32_t stats_len = 0, stats_pos;
    uint32_t idx       = start + t;
    if (idx < statistics_count) {
      const stats_column_desc *col = groups[idx].col;
      statistics_dtype dtype       = col->stats_dtype;
      switch (dtype) {
        case dtype_bool: stats_len = pb_fldlen_common + pb_fld_hdrlen + pb_fldlen_bucket1; break;
        case dtype_int8:
        case dtype_int16:
        case dtype_int32:
        case dtype_date32:
        case dtype_int64:
        case dtype_timestamp64:
          stats_len = pb_fldlen_common + pb_fld_hdrlen + 3 * (pb_fld_hdrlen + pb_fldlen_int64);
          break;
        case dtype_float32:
        case dtype_float64:
          stats_len = pb_fldlen_common + pb_fld_hdrlen + 3 * (pb_fld_hdrlen + pb_fldlen_float64);
          break;
        case dtype_decimal64:
        case dtype_decimal128:
          stats_len = pb_fldlen_common + pb_fld_hdrlen16 + 3 * (pb_fld_hdrlen + pb_fldlen_decimal);
          break;
        case dtype_string:
          stats_len = pb_fldlen_common + pb_fld_hdrlen32 + 3 * (pb_fld_hdrlen + pb_fldlen_int64) +
                      chunks[idx].min_value.str_val.length + chunks[idx].max_value.str_val.length;
          break;
        default: break;
      }
    }
    stats_pos = WarpReducePos32(stats_len, tx);
    if (tx == buffersize_reduction_dim - 1) { scratch_red[ty] = stats_pos; }
    __syncthreads();
    if (ty == 0) { scratch_red[tx] = WarpReducePos32(scratch_red[tx], tx); }
    __syncthreads();
    if (ty != 0) { stats_pos += scratch_red[ty - 1]; }
    stats_pos += stats_size;
    if (idx < statistics_count) {
      groups[idx].start_chunk = stats_pos - stats_len;
      groups[idx].num_chunks  = stats_len;
    }
    __syncthreads();
    if (t == buffersize_threads_per_block - 1) { stats_size = stats_pos; }
  }
}

struct stats_state_s {
  uint8_t *base;  ///< Output buffer start
  uint8_t *end;   ///< Output buffer end
  statistics_chunk chunk;
  statistics_merge_group group;
  stats_column_desc col;
  // ORC stats
  uint64_t numberOfValues;
  uint8_t hasNull;
};

/*
 * Protobuf encoding - see
 * https://developers.google.com/protocol-buffers/docs/encoding
 *
 */
// Protobuf varint encoding for unsigned int
__device__ inline uint8_t *pb_encode_uint(uint8_t *p, uint64_t v)
{
  while (v > 0x7f) {
    *p++ = ((uint32_t)v | 0x80);
    v >>= 7;
  }
  *p++ = v;
  return p;
}

// Protobuf field encoding for unsigned int
__device__ inline uint8_t *pb_put_uint(uint8_t *p, uint32_t id, uint64_t v)
{
  p[0] = id * 8 + PB_TYPE_VARINT;  // NOTE: Assumes id < 16
  return pb_encode_uint(p + 1, v);
}

// Protobuf field encoding for signed int
__device__ inline uint8_t *pb_put_int(uint8_t *p, uint32_t id, int64_t v)
{
  int64_t s = (v < 0);
  return pb_put_uint(p, id, (v ^ -s) * 2 + s);
}

// Protobuf field encoding for 'packed' unsigned int (single value)
__device__ inline uint8_t *pb_put_packed_uint(uint8_t *p, uint32_t id, uint64_t v)
{
  uint8_t *p2 = pb_encode_uint(p + 2, v);
  p[0]        = id * 8 + PB_TYPE_FIXEDLEN;
  p[1]        = static_cast<uint8_t>(p2 - (p + 2));
  return p2;
}

// Protobuf field encoding for binary/string
__device__ inline uint8_t *pb_put_binary(uint8_t *p, uint32_t id, const void *bytes, uint32_t len)
{
  p[0] = id * 8 + PB_TYPE_FIXEDLEN;
  p    = pb_encode_uint(p + 1, len);
  memcpy(p, bytes, len);
  return p + len;
}

// Protobuf field encoding for 64-bit raw encoding (double)
__device__ inline uint8_t *pb_put_fixed64(uint8_t *p, uint32_t id, const void *raw64)
{
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
 * ORC statistics format from https://orc.apache.org/specification/ORCv1/
 *
 * message ColumnStatistics {
 *  // the number of values
 *  optional uint64 numberOfValues = 1;
 *  // At most one of these has a value for any column
 *  optional IntegerStatistics intStatistics = 2;
 *  optional DoubleStatistics doubleStatistics = 3;
 *  optional StringStatistics stringStatistics = 4;
 *  optional BucketStatistics bucketStatistics = 5;
 *  optional DecimalStatistics decimalStatistics = 6;
 *  optional DateStatistics dateStatistics = 7;
 *  optional BinaryStatistics binaryStatistics = 8;
 *  optional TimestampStatistics timestampStatistics = 9;
 *  optional bool hasNull = 10;
 * }
 *
 **/
constexpr unsigned int encode_threads_per_chunk = 32;
constexpr unsigned int encode_chunks_per_block  = 4;
constexpr unsigned int encode_threads_per_block =
  encode_threads_per_chunk * encode_chunks_per_block;

__global__ void __launch_bounds__(encode_threads_per_block)
  gpu_encode_statistics(uint8_t *blob_bfr,
                        statistics_merge_group *groups,
                        const statistics_chunk *chunks,
                        uint32_t statistics_count)
{
  __shared__ __align__(8) stats_state_s state_g[encode_chunks_per_block];
  uint32_t t             = threadIdx.x;
  uint32_t idx           = blockIdx.x * encode_chunks_per_block + threadIdx.y;
  stats_state_s *const s = &state_g[threadIdx.y];

  // Encode and update actual bfr size
  if (idx < statistics_count && t == 0) {
    s->chunk           = chunks[idx];
    s->group           = groups[idx];
    s->col             = *(s->group.col);
    s->base            = blob_bfr + s->group.start_chunk;
    s->end             = blob_bfr + s->group.start_chunk + s->group.num_chunks;
    uint8_t *cur       = pb_put_uint(s->base, 1, s->chunk.non_nulls);
    uint8_t *fld_start = cur;
    switch (s->col.stats_dtype) {
      case dtype_int8:
      case dtype_int16:
      case dtype_int32:
      case dtype_int64:
        // intStatistics = 2
        // message IntegerStatistics {
        //  optional sint64 minimum = 1;
        //  optional sint64 maximum = 2;
        //  optional sint64 sum = 3;
        // }
        if (s->chunk.has_minmax || s->chunk.has_sum) {
          *cur = 2 * 8 + PB_TYPE_FIXEDLEN;
          cur += 2;
          if (s->chunk.has_minmax) {
            cur = pb_put_int(cur, 1, s->chunk.min_value.i_val);
            cur = pb_put_int(cur, 2, s->chunk.max_value.i_val);
          }
          if (s->chunk.has_sum) { cur = pb_put_int(cur, 3, s->chunk.sum.i_val); }
          fld_start[1] = cur - (fld_start + 2);
        }
        break;
      case dtype_float32:
      case dtype_float64:
        // doubleStatistics = 3
        // message DoubleStatistics {
        //  optional double minimum = 1;
        //  optional double maximum = 2;
        //  optional double sum = 3;
        // }
        if (s->chunk.has_minmax) {
          *cur = 3 * 8 + PB_TYPE_FIXEDLEN;
          cur += 2;
          cur          = pb_put_fixed64(cur, 1, &s->chunk.min_value.fp_val);
          cur          = pb_put_fixed64(cur, 2, &s->chunk.max_value.fp_val);
          fld_start[1] = cur - (fld_start + 2);
        }
        break;
      case dtype_string:
        // stringStatistics = 4
        // message StringStatistics {
        //  optional string minimum = 1;
        //  optional string maximum = 2;
        //  optional sint64 sum = 3; // sum will store the total length of all strings
        // }
        if (s->chunk.has_minmax && s->chunk.has_sum) {
          uint32_t sz = (pb_put_uint(cur, 3, s->chunk.sum.i_val) - cur) +
                        (pb_put_uint(cur, 1, s->chunk.min_value.str_val.length) - cur) +
                        (pb_put_uint(cur, 2, s->chunk.max_value.str_val.length) - cur) +
                        s->chunk.min_value.str_val.length + s->chunk.max_value.str_val.length;
          cur[0] = 4 * 8 + PB_TYPE_FIXEDLEN;
          cur    = pb_encode_uint(cur + 1, sz);
          cur    = pb_put_binary(
            cur, 1, s->chunk.min_value.str_val.ptr, s->chunk.min_value.str_val.length);
          cur = pb_put_binary(
            cur, 2, s->chunk.max_value.str_val.ptr, s->chunk.max_value.str_val.length);
          cur = pb_put_uint(cur, 3, s->chunk.sum.i_val);
        }
        break;
      case dtype_bool:
        // bucketStatistics = 5
        // message BucketStatistics {
        //  repeated uint64 count = 1 [packed=true];
        // }
        if (s->chunk.has_sum) {  // Sum is equal to the number of 'true' values
          cur[0]       = 5 * 8 + PB_TYPE_FIXEDLEN;
          cur          = pb_put_packed_uint(cur + 2, 1, s->chunk.sum.i_val);
          fld_start[1] = cur - (fld_start + 2);
        }
        break;
      case dtype_decimal64:
      case dtype_decimal128:
        // decimalStatistics = 6
        // message DecimalStatistics {
        //  optional string minimum = 1;
        //  optional string maximum = 2;
        //  optional string sum = 3;
        // }
        if (s->chunk.has_minmax) {
          // TODO: Decimal support (decimal min/max stored as strings)
        }
        break;
      case dtype_date32:
        // dateStatistics = 7
        // message DateStatistics { // min,max values saved as days since epoch
        //  optional sint32 minimum = 1;
        //  optional sint32 maximum = 2;
        // }
        if (s->chunk.has_minmax) {
          cur[0] = 7 * 8 + PB_TYPE_FIXEDLEN;
          cur += 2;
          cur          = pb_put_int(cur, 1, s->chunk.min_value.i_val);
          cur          = pb_put_int(cur, 2, s->chunk.max_value.i_val);
          fld_start[1] = cur - (fld_start + 2);
        }
        break;
      case dtype_timestamp64:
        // timestampStatistics = 9
        // message TimestampStatistics {
        //  optional sint64 minimum = 1; // min,max values saved as milliseconds since epoch
        //  optional sint64 maximum = 2;
        //  optional sint64 minimumUtc = 3; // min,max values saved as milliseconds since UNIX epoch
        //  optional sint64 maximumUtc = 4;
        // }
        if (s->chunk.has_minmax) {
          cur[0] = 9 * 8 + PB_TYPE_FIXEDLEN;
          cur += 2;
          cur          = pb_put_int(cur, 3, s->chunk.min_value.i_val);  // minimumUtc
          cur          = pb_put_int(cur, 4, s->chunk.max_value.i_val);  // maximumUtc
          fld_start[1] = cur - (fld_start + 2);
        }
        break;
      default: break;
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
 */
void orc_init_statistics_groups(statistics_group *groups,
                                const stats_column_desc *cols,
                                uint32_t num_columns,
                                uint32_t num_rowgroups,
                                uint32_t row_index_stride,
                                cudaStream_t stream)
{
  dim3 dim_grid((num_rowgroups + init_groups_per_block - 1) / init_groups_per_block, num_columns);
  dim3 dim_block(init_threads_per_group, init_groups_per_block);
  gpu_init_statistics_groups<<<dim_grid, dim_block, 0, stream>>>(
    groups, cols, num_columns, num_rowgroups, row_index_stride);
}

/**
 * @brief Launches kernels to return statistics buffer offsets and sizes
 *
 * @param[in,out] groups Statistics merge groups
 * @param[in] chunks Statistics chunks
 * @param[in] statistics_count Number of statistics buffers to encode
 * @param[in] stream CUDA stream to use, default 0
 */
void orc_init_statistics_buffersize(statistics_merge_group *groups,
                                    const statistics_chunk *chunks,
                                    uint32_t statistics_count,
                                    cudaStream_t stream)
{
  dim3 dim_block(buffersize_reduction_dim, buffersize_reduction_dim);
  gpu_init_statistics_buffersize<<<1, dim_block, 0, stream>>>(groups, chunks, statistics_count);
}

/**
 * @brief Launches kernel to encode statistics in ORC protobuf format
 *
 * @param[out] blob_bfr Output buffer for statistics blobs
 * @param[in,out] groups Statistics merge groups
 * @param[in,out] chunks Statistics data
 * @param[in] statistics_count Number of statistics buffers
 */
void orc_encode_statistics(uint8_t *blob_bfr,
                           statistics_merge_group *groups,
                           const statistics_chunk *chunks,
                           uint32_t statistics_count,
                           cudaStream_t stream)
{
  unsigned int num_blocks =
    (statistics_count + encode_chunks_per_block - 1) / encode_chunks_per_block;
  dim3 dim_block(encode_threads_per_chunk, encode_chunks_per_block);
  gpu_encode_statistics<<<num_blocks, dim_block, 0, stream>>>(
    blob_bfr, groups, chunks, statistics_count);
}

}  // namespace gpu
}  // namespace orc
}  // namespace io
}  // namespace cudf
