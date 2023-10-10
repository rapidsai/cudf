/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

  // all chunks from the selected row groups. We may end up reading these chunks progressively
  // instead of all at once
  std::vector<ColumnChunkDesc> chunks{};

  // an array of offsets into _file_itm_data::global_chunks. Each pair of offsets represents
  // the start/end of the chunks to be loaded for a given pass.
  std::vector<std::size_t> input_pass_row_group_offsets{};
  // row counts per input-pass
  std::vector<std::size_t> input_pass_row_count{};

  // skip_rows/num_rows values for the entire file. these need to be adjusted per-pass because we
  // may not be visiting every row group that contains these bounds
  size_t global_skip_rows;
  size_t global_num_rows;
};

/**
 * @brief Struct to identify the range for each chunk of rows during a chunked reading pass.
 */
struct chunk_read_info {
  size_t skip_rows;
  size_t num_rows;
};

/**
 * @brief Struct to store pass-level data that remains constant for a single pass.
 */
struct pass_intermediate_data {
  std::vector<std::unique_ptr<datasource::buffer>> raw_page_data;
  rmm::device_buffer decomp_page_data;

  // rowgroup, chunk and page information for the current pass.
  std::vector<row_group_info> row_groups{};
  cudf::detail::hostdevice_vector<ColumnChunkDesc> chunks{};
  cudf::detail::hostdevice_vector<PageInfo> pages_info{};
  cudf::detail::hostdevice_vector<PageNestingInfo> page_nesting_info{};
  cudf::detail::hostdevice_vector<PageNestingDecodeInfo> page_nesting_decode_info{};

  rmm::device_uvector<int32_t> page_keys{0, rmm::cuda_stream_default};
  rmm::device_uvector<int32_t> page_index{0, rmm::cuda_stream_default};
  rmm::device_uvector<string_index_pair> str_dict_index{0, rmm::cuda_stream_default};

  std::vector<chunk_read_info> output_chunk_read_info;
  std::size_t current_output_chunk{0};

  rmm::device_buffer level_decode_data{};
  int level_type_size{0};

  // skip_rows and num_rows values for this particular pass. these may be adjusted values from the
  // global values stored in file_intermediate_data.
  size_t skip_rows;
  size_t num_rows;
};

}  // namespace cudf::io::parquet::detail
