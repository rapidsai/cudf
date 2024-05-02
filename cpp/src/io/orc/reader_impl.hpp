/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "aggregate_orc_metadata.hpp"
#include "io/utilities/column_buffer.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/orc.hpp>
#include <cudf/io/orc.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace cudf::io::orc::detail {

struct reader_column_meta;
struct file_intermediate_data;

/**
 * @brief Implementation for ORC reader.
 */
class reader::impl {
 public:
  /**
   * @brief Constructor from a dataset source with reader options.
   *
   * @param sources Dataset sources
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit impl(std::vector<std::unique_ptr<datasource>>&& sources,
                orc_reader_options const& options,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr);

  /**
   * @brief Read an entire set or a subset of data and returns a set of columns
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows_opt Optional number of rows to read, or `std::nullopt` to read all rows
   * @param stripes Indices of individual stripes to load if non-empty
   * @return The set of columns along with metadata
   */
  table_with_metadata read(int64_t skip_rows,
                           std::optional<size_type> const& num_rows_opt,
                           std::vector<std::vector<size_type>> const& stripes);

 private:
  /**
   * @brief Perform all the necessary data preprocessing before creating an output table.
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows_opt Optional number of rows to read, or `std::nullopt` to read all rows
   * @param stripes Indices of individual stripes to load if non-empty
   */
  void prepare_data(int64_t skip_rows,
                    std::optional<size_type> const& num_rows_opt,
                    std::vector<std::vector<size_type>> const& stripes);

  /**
   * @brief Create the output table metadata from file metadata.
   *
   * @return Columns' metadata to output with the table read from file
   */
  table_metadata make_output_metadata();

  /**
   * @brief Read a chunk of data from the input source and return an output table with metadata.
   *
   * This function is called internally and expects all preprocessing steps have already been done.
   *
   * @return The output table along with columns' metadata
   */
  table_with_metadata read_chunk_internal();

  rmm::cuda_stream_view const _stream;
  rmm::device_async_resource_ref const _mr;

  // Reader configs
  data_type const _timestamp_type;  // Override output timestamp resolution
  bool const _use_index;            // Enable or disable attempt to use row index for parsing
  bool const _use_np_dtypes;        // Enable or disable the conversion to numpy-compatible dtypes
  std::vector<std::string> const _decimal128_columns;   // Control decimals conversion
  std::unique_ptr<reader_column_meta> const _col_meta;  // Track of orc mapping and child details

  // Intermediate data for internal processing.
  std::vector<std::unique_ptr<datasource>> const _sources;  // Unused but owns data for `_metadata`
  aggregate_orc_metadata _metadata;
  column_hierarchy const _selected_columns;  // Construct from `_metadata` thus declare after it
  std::unique_ptr<file_intermediate_data> _file_itm_data;
  std::unique_ptr<table_metadata> _output_metadata;
  std::vector<std::vector<cudf::io::detail::column_buffer>> _out_buffers;
};

}  // namespace cudf::io::orc::detail
