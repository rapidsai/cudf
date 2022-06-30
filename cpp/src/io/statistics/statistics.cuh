/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

/**
 * @file statistics.cuh
 * @brief Common structures and utility functions for statistics
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/string_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

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
  statistics_dtype stats_dtype;  //!< physical data type of column
  uint32_t num_rows;             //!< number of rows in column
  uint32_t num_values;  //!< Number of data values in column. Different from num_rows in case of
                        //!< nested columns
  int32_t ts_scale;     //!< timestamp scale (>0: multiply by scale, <0: divide by -scale)

  column_device_view const* leaf_column;    //!< Pointer to leaf column
  column_device_view const* parent_column;  //!< Pointer to parent column; nullptr if not list type
};

struct string_stats {
  const char* ptr;  //!< ptr to character data
  uint32_t length;  //!< length of string
  __host__ __device__ __forceinline__ volatile string_stats& operator=(
    const string_view& val) volatile
  {
    ptr    = val.data();
    length = val.size_bytes();
    return *this;
  }
  __host__ __device__ __forceinline__ operator string_view() volatile
  {
    return string_view(ptr, static_cast<size_type>(length));
  }
  __host__ __device__ __forceinline__ operator string_view() const
  {
    return string_view(ptr, static_cast<size_type>(length));
  }
  __host__ __device__ __forceinline__ operator string_view()
  {
    return string_view(ptr, static_cast<size_type>(length));
  }
};

union statistics_val {
  string_stats str_val;  //!< string columns
  double fp_val;         //!< float columns
  int64_t i_val;         //!< integer columns
  uint64_t u_val;        //!< unsigned integer columns
  __int128_t d128_val;   //!< decimal128 columns
};

struct statistics_chunk {
  uint32_t non_nulls;        //!< number of non-null values in chunk
  uint32_t null_count;       //!< number of null values in chunk
  statistics_val min_value;  //!< minimum value in chunk
  statistics_val max_value;  //!< maximum value in chunk
  statistics_val sum;        //!< sum of chunk
  uint8_t has_minmax;        //!< Nonzero if min_value and max_values are valid
  uint8_t has_sum;           //!< Nonzero if sum is valid
};

struct statistics_group {
  const stats_column_desc* col;  //!< Column information
  uint32_t start_row;            //!< Start row of this group
  uint32_t num_rows;             //!< Number of rows in group
};

struct statistics_merge_group {
  data_type col_dtype;           //!< Column data type
  statistics_dtype stats_dtype;  //!< Statistics data type for this column
  uint32_t start_chunk;          //!< Start chunk of this group
  uint32_t num_chunks;           //!< Number of chunks in group
};

}  // namespace io
}  // namespace cudf
