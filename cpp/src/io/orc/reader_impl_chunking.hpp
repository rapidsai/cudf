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

/**
 * @brief Struct that maps ORC streams to columns
 */
struct orc_stream_info {
  explicit orc_stream_info(uint64_t offset_,
                           std::size_t dst_pos_,
                           uint32_t length_,
                           uint32_t stripe_idx_,
                           std::size_t level_,
                           uint32_t orc_col_idx_,
                           StreamKind kind_)
    : offset(offset_),
      dst_pos(dst_pos_),
      length(length_),
      stripe_idx(stripe_idx_),
      level(level_),
      orc_col_idx(orc_col_idx_),
      kind(kind_)
  {
#ifdef PRINT_DEBUG
    printf("   construct stripe id [%d, %d, %d, %d]\n",
           (int)stripe_idx,
           (int)level,
           (int)orc_col_idx,
           (int)kind);
#endif
  }
  uint64_t offset;      // offset in file
  std::size_t dst_pos;  // offset in memory relative to start of compressed stripe data
  std::size_t length;   // length in file
  uint32_t stripe_idx;  // stripe processing index, not stripe index in source
  std::size_t level;    // TODO
  uint32_t orc_col_idx;
  StreamKind kind;
};

// unify this with orc_stream_info
struct stream_id_info {
  std::size_t stripe_idx;
  std::size_t level;
  uint32_t orc_col_idx;
  StreamKind kind;
};
struct stripe_level_comp_info {
  std::size_t num_compressed_blocks{0};
  std::size_t num_uncompressed_blocks{0};
  std::size_t total_decomp_size{0};
};
struct stream_id_equal {
  bool operator()(stream_id_info const& lhs, stream_id_info const& rhs) const
  {
    return lhs.stripe_idx == rhs.stripe_idx && lhs.level == rhs.level &&
           lhs.orc_col_idx == rhs.orc_col_idx && lhs.kind == rhs.kind;
  }
};
struct stream_id_hash {
  std::size_t operator()(stream_id_info const& index) const
  {
    auto const hasher = std::hash<size_t>{};
    return hasher(index.stripe_idx) ^ hasher(index.level) ^
           hasher(static_cast<std::size_t>(index.orc_col_idx)) ^
           hasher(static_cast<std::size_t>(index.kind));
  }
};

/**
 * @brief Struct to store file-level data that remains constant for all chunks being read.
 */
struct file_intermediate_data {
  std::unordered_map<stream_id_info, stripe_level_comp_info, stream_id_hash, stream_id_equal>
    compinfo_map;
  bool compinfo_ready{false};

  std::vector<std::vector<std::size_t>> lvl_stripe_sizes;
  std::vector<std::vector<rmm::device_buffer>> lvl_stripe_data;
  std::vector<std::vector<rmm::device_uvector<uint32_t>>> null_count_prefix_sums;
  std::vector<cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>> lvl_data_chunks;
  std::vector<std::vector<orc_stream_info>> lvl_stream_info;

  // Each read correspond to one or more consecutive stream combined.
  struct read_info {
    read_info(uint64_t offset_,
              std::size_t length_,
              std::size_t dst_pos_,
              std::size_t source_idx_,
              std::size_t stripe_idx_,
              std::size_t level_)
      : offset(offset_),
        length(length_),
        dst_pos(dst_pos_),
        source_idx(source_idx_),
        stripe_idx(stripe_idx_),
        level(level_)
    {
    }
    uint64_t offset;
    std::size_t length;
    std::size_t dst_pos;
    std::size_t source_idx;
    std::size_t stripe_idx;
    std::size_t level;
  };
  std::vector<read_info> read_info;

  int64_t rows_to_skip;
  size_type rows_to_read;
  std::vector<metadata::OrcStripeInfo> selected_stripes;
};

}  // namespace cudf::io::orc::detail
