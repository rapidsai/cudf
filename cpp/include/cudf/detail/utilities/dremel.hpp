/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>

#include <rmm/device_uvector.hpp>

namespace cudf::detail {

struct dremel_device_view {
  size_type* offsets;
  uint8_t* rep_levels;
  uint8_t* def_levels;
  size_type leaf_data_size;
  uint8_t max_def_level;
};

/**
 * @brief Dremel data that describes one nested type column
 *
 * @see get_dremel_data()
 */
struct dremel_data {
  rmm::device_uvector<size_type> dremel_offsets;
  rmm::device_uvector<uint8_t> rep_level;
  rmm::device_uvector<uint8_t> def_level;

  size_type leaf_data_size;
  uint8_t max_def_level;

  operator dremel_device_view()
  {
    return dremel_device_view{
      dremel_offsets.data(), rep_level.data(), def_level.data(), leaf_data_size, max_def_level};
  }
};

/**
 * @brief Get the dremel offsets and repetition and definition levels for a LIST column
 *
 * Dremel offsets are the per row offsets into the repetition and definition level arrays for a
 * column.
 * Example:
 * ```
 * col            = {{1, 2, 3}, { }, {5, 6}}
 * dremel_offsets = { 0,         3,   4,  6}
 * rep_level      = { 0, 1, 1,   0,   0, 1}
 * def_level      = { 1, 1, 1,   0,   1, 1}
 * ```
 * @param col Column of LIST type
 * @param level_nullability Pre-determined nullability at each list level. Empty means infer from
 * `col`
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return A struct containing dremel data
 */
dremel_data get_dremel_data(column_view h_col,
                            std::vector<uint8_t> const& nullability,
                            rmm::cuda_stream_view stream);

}  // namespace cudf::detail
