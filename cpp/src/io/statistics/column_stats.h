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
#pragma once
#include <stdint.h>

namespace cudf {
namespace io {

enum statistics_dtype {
  dtype_none,
  dtype_bool,
  dtype_int8,
  dtype_int16,
  dtype_int32,
  dtype_date32,
  dtype_int64,
  dtype_timestamp64,
  dtype_decimal64,
  dtype_decimal128,
  dtype_float32,
  dtype_float64,
  dtype_string,
};

struct stats_column_desc {
  statistics_dtype stats_dtype;    //!< physical data type of column
  uint32_t num_rows;               //!< number of rows in column
  const uint32_t *valid_map_base;  //!< base of valid bit map for this column (null if not present)
  const void *column_data_base;    //!< base ptr to column data
  int32_t ts_scale;  //!< timestamp scale (>0: multiply by scale, <0: divide by -scale)
};

struct string_stats {
  const char *ptr;  //!< ptr to character data
  uint32_t length;  //!< length of string
};

union statistics_val {
  string_stats str_val;  //!< string columns
  double fp_val;         //!< float columns
  int64_t i_val;         //!< integer columns
  struct {
    uint64_t lo64;
    int64_t hi64;
  } i128_val;  //!< decimal128 columns
};

struct statistics_chunk {
  uint32_t non_nulls;        //!< number of non-null values in chunk
  uint32_t null_count;       //!< number of null values in chunk
  statistics_val min_value;  //!< minimum value in chunk
  statistics_val max_value;  //!< maximum value in chunk
  union {
    double fp_val;  //!< Sum for fp types
    int64_t i_val;  //!< Sum for integer types or string lengths
  } sum;
  uint8_t has_minmax;  //!< Nonzero if min_value and max_values are valid
  uint8_t has_sum;     //!< Nonzero if sum is valid
};

struct statistics_group {
  const stats_column_desc *col;  //!< Column information
  uint32_t start_row;            //!< Start row of this group
  uint32_t num_rows;             //!< Number of rows in group
};

struct statistics_merge_group {
  const stats_column_desc *col;  //!< Column information
  uint32_t start_chunk;          //!< Start chunk of this group
  uint32_t num_chunks;           //!< Number of chunks in group
};

/**
 * @brief Launches kernel to gather column statistics
 *
 * @param[out] chunks Statistics results [num_chunks]
 * @param[in] groups Statistics row groups [num_chunks]
 * @param[in] num_chunks Number of chunks & rowgroups
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t GatherColumnStatistics(statistics_chunk *chunks,
                                   const statistics_group *groups,
                                   uint32_t num_chunks,
                                   cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Launches kernel to merge column statistics
 *
 * @param[out] chunks_out Statistics results [num_chunks]
 * @param[out] chunks_in Input statistics
 * @param[in] groups Statistics groups [num_chunks]
 * @param[in] num_chunks Number of chunks & groups
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t MergeColumnStatistics(statistics_chunk *chunks_out,
                                  const statistics_chunk *chunks_in,
                                  const statistics_merge_group *groups,
                                  uint32_t num_chunks,
                                  cudaStream_t stream = (cudaStream_t)0);

}  // namespace io
}  // namespace cudf
