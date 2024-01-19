/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "orc_gpu.hpp"

#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf::io::orc::detail {

using cudf::detail::hostdevice_2dvector;
using cudf::detail::hostdevice_vector;

/**
 * @brief Struct to store file-level data that remains constant for all chunks being read.
 */
struct file_intermediate_data {
  std::vector<std::vector<rmm::device_buffer>> lvl_stripe_data;
  std::vector<std::vector<rmm::device_uvector<uint32_t>>> null_count_prefix_sums;

  std::vector<hostdevice_2dvector<gpu::ColumnDesc>> lvl_chunks;
  //  hostdevice_2dvector<gpu::RowGroup> row_groups;
  std::vector<hostdevice_vector<data_type>> lvl_col_types;  // type of each column in each level
  hostdevice_vector<size_type> lvl_num_cols;                // number of columns in each level
  hostdevice_vector<size_type> lvl_offsets;  // prefix sum of number of columns in each level

  int64_t rows_to_skip;
  size_type rows_to_read;
  std::vector<metadata::stripe_source_mapping> selected_stripes;

  std::vector<uint32_t> stripe_sizes;  // Number of rows at root level in each stripe.
};

/**
 * @brief Struct to identify the range for each chunk of rows during a chunked reading pass.
 */
struct row_range {
  int64_t start_rows;
  size_type end_rows;
};

/**
 * @brief Struct to store all data necessary for chunked reading.
 */
struct chunk_read_info {
  explicit chunk_read_info(std::size_t chunk_size_limit_ = 0) : chunk_size_limit{chunk_size_limit_}
  {
  }

  std::size_t chunk_size_limit;  // Maximum size (in bytes) of an output chunk, or 0 for no limit

  // The range of rows for output chunks.
  std::vector<row_range> chunk_ranges;
  std::size_t current_chunk_idx{0};
};

}  // namespace cudf::io::orc::detail
