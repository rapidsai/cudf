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

#include <tuple>
#include <unordered_map>

namespace cudf::io::orc::detail {

struct stripe_level_index {
  std::size_t stripe_idx;
  std::size_t level;
};
struct stripe_level_comp_info {
  std::size_t num_compressed_blocks;
  std::size_t num_uncompressed_blocks;
  std::size_t total_decomp_size;
};
struct stripe_level_equal {
  bool operator()(stripe_level_index const& lhs, stripe_level_index const& rhs) const
  {
    return lhs.stripe_idx == rhs.stripe_idx && lhs.level == rhs.level;
  }
};
struct stripe_level_hash {
  std::size_t operator()(stripe_level_index const& index) const
  {
    auto const hasher = std::hash<size_t>{};
    return hasher(index.stripe_idx) ^ hasher(index.level);
  }
};

/**
 * @brief Struct to store file-level data that remains constant for all chunks being read.
 */
struct file_intermediate_data {
  std::
    unordered_map<stripe_level_index, stripe_level_comp_info, stripe_level_hash, stripe_level_equal>
      compinfo_map;
  bool compinfo_ready{false};

  std::vector<std::vector<rmm::device_buffer>> lvl_stripe_data;
  std::vector<std::vector<rmm::device_uvector<uint32_t>>> null_count_prefix_sums;

  int64_t rows_to_skip;
  size_type rows_to_read;
  std::vector<metadata::stripe_source_mapping> selected_stripes;
};

}  // namespace cudf::io::orc::detail
