/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include "cudf.h"

#include <cstddef>

/**---------------------------------------------------------------------------*
 * @file types.hpp
 * @brief Type declarations for libcudf.
 *
 *---------------------------------------------------------------------------**/

/**---------------------------------------------------------------------------*
 * @brief Forward declaration of cudaStream_t
 *---------------------------------------------------------------------------**/
using cudaStream_t = struct CUstream_st*;

namespace bit_mask {
using bit_mask_t = uint32_t;
}

namespace cudf {

// Forward declaration
struct table;
class column;

using size_type = int32_t;
using bitmask_type = uint32_t;

enum data_type {
  INVALID = 0,
  INT8,
  INT16,
  INT32,
  INT64,
  FLOAT32,
  FLOAT64,
  BOOL8,
  DATE32,
  DATE64,
  TIMESTAMP_NS,
  TIMESTAMP_US,
  TIMESTAMP_MS,
  TIMESTAMP_S,
  CATEGORY,
  STRING,
  STRING_CATEGORY
};

}  // namespace cudf
