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

#include "io/utilities/hostdevice_vector.hpp"
#include "orc_gpu.hpp"

#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <unordered_map>

namespace cudf::io::orc::detail {

/**
 * @brief Struct that store source information of an ORC streams.
 */
struct stream_source_info {
  std::size_t stripe_idx;  // global stripe id throughout all data sources
  std::size_t level;       // level of the nested column
  uint32_t orc_col_idx;    // orc column id
  StreamKind kind;         // stream kind

  struct hash {
    std::size_t operator()(stream_source_info const& id) const
    {
      auto const hasher = std::hash<size_t>{};
      return hasher(id.stripe_idx) ^ hasher(id.level) ^
             hasher(static_cast<std::size_t>(id.orc_col_idx)) ^
             hasher(static_cast<std::size_t>(id.kind));
    }
  };
  struct equal_to {
    bool operator()(stream_source_info const& lhs, stream_source_info const& rhs) const
    {
      return lhs.stripe_idx == rhs.stripe_idx && lhs.level == rhs.level &&
             lhs.orc_col_idx == rhs.orc_col_idx && lhs.kind == rhs.kind;
    }
  };
};

/**
 * @brief Map to lookup a value from stream source.
 */
template <typename T>
using stream_source_map =
  std::unordered_map<stream_source_info, T, stream_source_info::hash, stream_source_info::equal_to>;

/**
 * @brief Struct that store information of an ORC stream.
 */
struct orc_stream_info {
  explicit orc_stream_info(uint64_t offset_,
                           std::size_t dst_pos_,
                           uint32_t length_,
                           stream_source_info const& source_)
    : offset(offset_), dst_pos(dst_pos_), length(length_), source(source_)
  {
  }
  // Data info:
  uint64_t offset;      // offset in data source
  std::size_t dst_pos;  // offset to store data in memory relative to start of raw stripe data
  std::size_t length;   // stream length to read

  // Store source of the stream in the stripe, so we can look up where this stream comes from.
  stream_source_info source;
};

/**
 * @brief Compression information for a stripe at a specific nested level.
 */
struct stripe_level_comp_info {
  std::size_t num_compressed_blocks{0};
  std::size_t num_uncompressed_blocks{0};
  std::size_t total_decomp_size{0};
};

/**
 * @brief Struct representing a range of data.
 */
struct range {
  std::size_t begin;
  std::size_t end;
};

/**
 * @brief Struct storing intermediate processing data loaded from data sources.
 */
struct file_intermediate_data {
  int64_t rows_to_skip;
  int64_t rows_to_read;
  std::vector<metadata::orc_stripe_info> selected_stripes;

  // Return true if no rows or stripes to read.
  bool has_no_data() const { return rows_to_read == 0 || selected_stripes.empty(); }

  // Store the compression information for each data stream.
  stream_source_map<stripe_level_comp_info> compinfo_map;

  // The buffers to store raw data read from disk, initialized for each reading stripe chunks.
  // After decoding, such buffers can be released.
  // This can only be implemented after chunked output is ready.
  std::vector<std::vector<rmm::device_buffer>> lvl_stripe_data;

  // Store the size of each stripe at each nested level.
  // This is used to initialize the stripe_data buffers.
  std::vector<std::vector<std::size_t>> lvl_stripe_sizes;

  // Store information to identify where to read a chunk of data from source.
  // Each read corresponds to one or more consecutive streams combined.
  struct stream_data_read_info {
    stream_data_read_info(uint64_t offset_,
                          std::size_t dst_pos_,
                          std::size_t length_,
                          std::size_t source_idx_,
                          std::size_t stripe_idx_,
                          std::size_t level_)
      : offset(offset_),
        dst_pos(dst_pos_),
        length(length_),
        source_idx(source_idx_),
        stripe_idx(stripe_idx_),
        level(level_)
    {
    }

    uint64_t offset;         // offset in data source
    std::size_t dst_pos;     // offset to store data in memory relative to start of raw stripe data
    std::size_t length;      // data length to read
    std::size_t source_idx;  // the data source id
    std::size_t stripe_idx;  // global stripe index
    std::size_t level;       // nested level
  };

  // Identify what data to read from source.
  std::vector<stream_data_read_info> data_read_info;

  // For each stripe, we perform a number of read for its streams.
  // Those reads are identified by a chunk of consecutive read info, stored in data_read_info.
  std::vector<range> stripe_data_read_ranges;

  // Store info for each ORC stream at each nested level.
  std::vector<std::vector<orc_stream_info>> lvl_stream_info;

  // At each nested level, the streams for each stripe are stored consecutively in lvl_stream_info.
  // This is used to identify the range of streams for each stripe from that vector.
  std::vector<std::vector<range>> lvl_stripe_stream_ranges;

  // TODO rename
  std::vector<std::vector<rmm::device_uvector<uint32_t>>> null_count_prefix_sums;

  // For data processing, decompression, and decoding.
  // Each 'chunk' of data here corresponds to an orc column, in a stripe, at a nested level.
  std::vector<cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>> lvl_data_chunks;

  bool global_preprocessed{false};
};

/**
 * @brief Struct collecting data necessary for chunked reading.
 */
struct chunk_read_data {
  explicit chunk_read_data(std::size_t output_size_limit_,
                           std::size_t data_read_limit_,
                           size_type output_row_granularity_)
    : output_size_limit{output_size_limit_},
      data_read_limit{data_read_limit_},
      output_row_granularity{output_row_granularity_}
  {
  }

  // TODO: const for 3 below?
  std::size_t const
    output_size_limit;  // maximum size (in bytes) of an output chunk, or 0 for no limit
  std::size_t const data_read_limit;       // approximate maximum size (in bytes) used for store
                                           // intermediate data, or 0 for no limit
  size_type const output_row_granularity;  // TODO
  static double constexpr load_limit_ratio{0.4};  // TODO

  // Chunks of stripes that can be load into memory such that their data size is within a size
  // limit.
  std::vector<range> load_stripe_ranges;
  std::size_t curr_load_stripe_range{0};
  bool more_stripe_to_load() const { return curr_load_stripe_range < load_stripe_ranges.size(); }

  // Chunks of stripes such that their decompression size is within a size limit.
  std::vector<range> decode_stripe_ranges;
  std::size_t curr_decode_stripe_range{0};
  bool more_stripe_to_decode() const
  {
    return curr_decode_stripe_range < decode_stripe_ranges.size();
  }

  // Chunk of rows in the internal decoded table to output for each `read_chunk()`.
  std::vector<range> output_table_ranges;
  std::size_t curr_output_table_range{0};
  std::unique_ptr<cudf::table> decoded_table;
  bool more_table_chunk_to_output() const
  {
    return curr_output_table_range < output_table_ranges.size();
  }

  // Only has more chunk to output if:
  bool has_next() const
  {
#ifdef LOCAL_TEST
    printf("compute has_next: %d, %d, %d\n",
           (int)more_stripe_to_load(),
           (int)more_stripe_to_decode(),
           (int)more_table_chunk_to_output());
#endif

    return more_stripe_to_load() || more_stripe_to_decode() || more_table_chunk_to_output();
  }
};

/**
 * @brief Struct to accumulate counts and sizes of some types such as stripes or rows.
 */
struct cumulative_size {
  std::size_t count{0};
  std::size_t size_bytes{0};
};

/**
 * @brief Struct to accumulate counts, sizes, and number of rows of some types such as stripes or
 * rows in tables.
 */
struct cumulative_size_and_row {
  std::size_t count{0};
  std::size_t size_bytes{0};
  std::size_t rows{0};
};

/**
 * @brief Functor to sum up cumulative data.
 */
struct cumulative_size_sum {
  __device__ cumulative_size operator()(cumulative_size const& a, cumulative_size const& b) const
  {
    return cumulative_size{a.count + b.count, a.size_bytes + b.size_bytes};
  }

  __device__ cumulative_size_and_row operator()(cumulative_size_and_row const& a,
                                                cumulative_size_and_row const& b) const
  {
    return cumulative_size_and_row{a.count + b.count, a.size_bytes + b.size_bytes, a.rows + b.rows};
  }
};

/**
 * @brief Find the splits of the input data such that each split range has cumulative size less than
 * a given `size_limit`.
 *
 * Note that the given limit is just a soft limit. The function will always output ranges that
 * have at least one count, even such ranges have sizes exceed the value of `size_limit`.
 *
 * @param cumulative_sizes The input cumulative sizes to compute split ranges
 * @param total_count The total count in the entire input
 * @param size_limit The given soft limit to compute splits
 * @return A vector of ranges as splits of the input
 */
template <typename T>
std::vector<range> find_splits(host_span<T const> cumulative_sizes,
                               std::size_t total_count,
                               std::size_t size_limit);

/**
 * @brief Expand a range of ranges into a simple range of data.
 *
 * @param input_ranges The list of all data ranges
 * @param selected_ranges A range of ranges from `input_ranges`
 * @return The range of data span by the selected range of ranges
 */
std::pair<int64_t, int64_t> get_range(std::vector<range> const& input_ranges,
                                      range const& selected_ranges);

/**
 * @brief Function that populates descriptors for either individual streams or chunks of column
 * data, but not both.
 *
 * This function is used in the global step, to gather information for streams of all stripes in
 * the data sources (when `stream_info` is present). Later on, it is used again to populate column
 * descriptors (`chunks` is present) during decompression and decoding. The two steps share
 * most of the execution path thus this function takes mutually exclusive parameters `stream_info`
 * or `chunks` depending on each use case.
 */
std::size_t gather_stream_info_and_column_desc(
  std::size_t stripe_processing_order,
  std::size_t level,
  orc::StripeInformation const* stripeinfo,
  orc::StripeFooter const* stripefooter,
  host_span<int const> orc2gdf,
  host_span<orc::SchemaType const> types,
  bool use_index,
  bool apply_struct_map,
  int64_t* num_dictionary_entries,
  std::size_t* stream_processing_order,
  std::optional<std::vector<orc_stream_info>*> const& stream_info,
  std::optional<cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>*> const& chunks);

}  // namespace cudf::io::orc::detail
