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

#include "reader_impl_helpers.hpp"

#include <cudf/types.hpp>

namespace cudf::io::parquet::detail {

/**
 * @brief Struct to store file-level data that remains constant for
 * all passes/chunks in the file.
 */
struct file_intermediate_data {
  // all row groups to read
  std::vector<row_group_info> row_groups{};

  // all chunks from the selected row groups.
  std::vector<ColumnChunkDesc> chunks{};

  // an array of offsets into _file_itm_data::global_chunks. Each pair of offsets represents
  // the start/end of the chunks to be loaded for a given pass.
  std::vector<std::size_t> input_pass_row_group_offsets{};

  // start row counts per input-pass. this includes all rows in the row groups of the pass and
  // is not capped by global_skip_rows and global_num_rows.
  std::vector<std::size_t> input_pass_start_row_count{};

  // number of rows to be read from each data source
  std::vector<std::size_t> num_rows_per_source{};

  // partial sum of the number of rows per data source
  std::vector<std::size_t> exclusive_sum_num_rows_per_source{};

  size_t _current_input_pass{0};  // current input pass index
  size_t _output_chunk_count{0};  // how many output chunks we have produced

  // skip_rows/num_rows values for the entire file.
  size_t global_skip_rows;
  size_t global_num_rows;

  [[nodiscard]] size_t num_passes() const
  {
    return input_pass_row_group_offsets.size() == 0 ? 0 : input_pass_row_group_offsets.size() - 1;
  }
};

/**
 * @brief Struct to identify a range of rows.
 */
struct row_range {
  size_t skip_rows;
  size_t num_rows;
};

/**
 * @brief Passes are broken down into subpasses based on temporary memory constraints.
 */
struct subpass_intermediate_data {
  rmm::device_buffer decomp_page_data;

  rmm::device_buffer level_decode_data{};
  cudf::detail::hostdevice_span<PageInfo> pages{};

  // optimization. if the single_subpass flag is set, it means we will only be doing
  // one subpass for the entire pass. this allows us to skip various pieces of work
  // during processing. notably, page_buf will not be allocated to hold a compacted
  // copy of the pages specific to the subpass.
  bool single_subpass{false};
  cudf::detail::hostdevice_vector<PageInfo> page_buf{};

  // for each page in the subpass, the index of our source page in the pass
  rmm::device_uvector<size_t> page_src_index{0, cudf::get_default_stream()};
  // for each column in the file (indexed by _input_columns.size())
  // the number of associated pages for this subpass
  std::vector<size_t> column_page_count;
  cudf::detail::hostdevice_vector<PageNestingInfo> page_nesting_info{};
  cudf::detail::hostdevice_vector<PageNestingDecodeInfo> page_nesting_decode_info{};

  std::vector<row_range> output_chunk_read_info;
  std::size_t current_output_chunk{0};

  // skip_rows and num_rows values for this particular subpass. in absolute row indices.
  size_t skip_rows;
  size_t num_rows;
};

/**
 * @brief Struct to store pass-level data that remains constant for a single pass.
 *
 * A pass is defined as a set of rowgroups read but not yet decompressed. This set of
 * rowgroups may represent less than all of the rowgroups to be read for the file.
 */
struct pass_intermediate_data {
  std::vector<rmm::device_buffer> raw_page_data;

  // rowgroup, chunk and page information for the current pass.
  bool has_compressed_data{false};
  std::vector<row_group_info> row_groups{};
  cudf::detail::hostdevice_vector<ColumnChunkDesc> chunks{};
  cudf::detail::hostdevice_vector<PageInfo> pages{};

  // base memory used for the pass itself (compressed data in the loaded chunks and any
  // decompressed dictionary pages)
  size_t base_mem_size{0};

  // offsets to each group of input pages (by column/schema, indexed by _input_columns.size())
  // so if we had 2 columns/schemas, with page keys
  //
  // 1 1 1 1 1 2 2 2
  //
  // page_offsets would be 0, 5, 8
  rmm::device_uvector<size_type> page_offsets{0, cudf::get_default_stream()};

  rmm::device_buffer decomp_dict_data{0, cudf::get_default_stream()};
  rmm::device_uvector<string_index_pair> str_dict_index{0, cudf::get_default_stream()};

  // cumulative strings column sizes.
  std::vector<size_t> cumulative_col_string_sizes{};

  int level_type_size{0};

  // skip_rows / num_rows for this pass.
  // NOTE: skip_rows is the absolute row index in the file.
  size_t skip_rows;
  size_t num_rows;
  // number of rows we have processed so far (out of num_rows). note that this
  // only includes the number of rows we have processed before starting the current
  // subpass. it does not get updated as a subpass iterates through output chunks.
  size_t processed_rows{0};

  // currently active subpass
  std::unique_ptr<subpass_intermediate_data> subpass{};
};

}  // namespace cudf::io::parquet::detail
