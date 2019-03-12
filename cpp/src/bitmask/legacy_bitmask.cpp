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

#include "legacy_bitmask.hpp"
#include <cudf.h>
#include <utilities/cudf_utils.h>

namespace {

// Buffers are padded to 64-byte boundaries (for SIMD) static
static constexpr int32_t kArrowAlignment = 64;

// Tensors are padded to 64-byte boundaries static
static constexpr int32_t kTensorAlignment = 64;

// Align on 8-byte boundaries in IPC static
static constexpr int32_t kArrowIpcAlignment = 8;

// Align on 4-byte boundaries in CUDF static
static constexpr int32_t kCudfIpcAlignment = 4;

// todo, enable arrow ipc utils, and remove this method
static gdf_size_type PaddedLength(int64_t nbytes,
                                  int32_t alignment = kArrowAlignment) {
  return ((nbytes + alignment - 1) / alignment) * alignment;
}

}  // namespace

// Calculates number of bytes for valid bitmask for a column of a specified
// size
gdf_size_type gdf_valid_allocation_size(gdf_size_type column_size) {
  static_assert(sizeof(gdf_valid_type) == 1,
                "gdf_valid_type assumed to be 1 byte");
  return PaddedLength(
      (column_size + (GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE,
      kArrowAlignment);
}

gdf_size_type gdf_num_bitmask_elements(gdf_size_type column_size) {
  return ((column_size + (GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE);
}