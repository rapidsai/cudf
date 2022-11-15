/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

/**
 * @file reader_impl.hpp
 * @brief cuDF-IO Parquet reader class implementation header
 */

#pragma once

#include "parquet_gpu.hpp"
#include "reader_impl_helpers.hpp"

#include <io/utilities/column_buffer.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/parquet.hpp>
#include <cudf/io/parquet.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace cudf::io::detail::parquet {

/**
 * @brief Implementation for Parquet reader
 */
class reader::impl {
 public:
  /**
   * @brief Constructor from an array of dataset sources with reader options.
   *
   * @param sources Dataset sources
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit impl(std::vector<std::unique_ptr<datasource>>&& sources,
                parquet_reader_options const& options,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr);

  /**
   * @brief Read an entire set or a subset of data and returns a set of columns
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows Number of rows to read
   * @param uses_custom_row_bounds Whether or not num_rows and skip_rows represents user-specific
   *        bounds
   * @param row_group_indices Lists of row groups to read, one per source
   *
   * @return The set of columns along with metadata
   */
  table_with_metadata read(size_type skip_rows,
                           size_type num_rows,
                           bool uses_custom_row_bounds,
                           host_span<std::vector<size_type> const> row_group_indices);

 private:
  /**
   * @brief Perform the necessary data preprocessing for parsing file later on.
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows Number of rows to read, or `-1` to read all rows
   * @param uses_custom_row_bounds Whether or not num_rows and skip_rows represents user-specific
   *        bounds
   * @param row_group_indices Lists of row groups to read (one per source), or empty if read all
   */
  void prepare_data(size_type skip_rows,
                    size_type num_rows,
                    bool uses_custom_row_bounds,
                    host_span<std::vector<size_type> const> row_group_indices);

  /**
   * @brief Load and decompress the input file(s) into memory.
   */
  void load_and_decompress_data(std::vector<row_group_info> const& row_groups_info,
                                size_type num_rows);

  /**
   * @brief Allocate nesting information storage for all pages and set pointers to it.
   *
   * One large contiguous buffer of PageNestingInfo structs is allocated and
   * distributed among the PageInfo structs.
   *
   * Note that this gets called even in the flat schema case so that we have a
   * consistent place to store common information such as value counts, etc.
   */
  void allocate_nesting_info();

  /**
   * @brief Read a chunk of data and return an output table.
   *
   * This function is called internally and expects all preprocessing steps have already been done.
   *
   * @param uses_custom_row_bounds Whether or not num_rows and skip_rows represents user-specific
   *        bounds
   * @return The output table along with columns' metadata
   */
  table_with_metadata read_chunk_internal(bool uses_custom_row_bounds);

  /**
   * @brief Finalize the output table by adding empty columns for the non-selected columns in
   * schema.
   *
   * @param out_metadata The output table metadata
   * @param out_columns The columns for building the output table
   * @return The output table along with columns' metadata
   */
  table_with_metadata finalize_output(table_metadata& out_metadata,
                                      std::vector<std::unique_ptr<column>>& out_columns);

  /**
   * @brief Allocate data bufers for the output columns.
   *
   * @param skip_rows Crop all rows below skip_rows
   * @param num_rows Maximum number of rows to read
   * @param uses_custom_row_bounds Whether or not num_rows and skip_rows represents user-specific
   *        bounds
   */
  void allocate_columns(size_t skip_rows, size_t num_rows, bool uses_custom_row_bounds);

  /**
   * @brief Converts the page data and outputs to columns.
   *
   * @param skip_rows Minimum number of rows from start
   * @param num_rows Number of rows to output
   */
  void decode_page_data(size_t skip_rows, size_t num_rows);

 private:
  rmm::cuda_stream_view _stream;
  rmm::mr::device_memory_resource* _mr = nullptr;

  std::vector<std::unique_ptr<datasource>> _sources;
  std::unique_ptr<aggregate_reader_metadata> _metadata;

  // input columns to be processed
  std::vector<input_column_info> _input_columns;

  // Buffers for generating output columns
  std::vector<column_buffer> _output_buffers;

  // _output_buffers associated schema indices
  std::vector<int> _output_column_schemas;

  bool _strings_to_categorical = false;
  std::optional<std::vector<reader_column_schema>> _reader_column_schema;
  data_type _timestamp_type{type_id::EMPTY};

  cudf::io::parquet::gpu::file_intermediate_data _file_itm_data;

  size_type _skip_rows{0};
  size_type _num_rows{0};
};

}  // namespace cudf::io::detail::parquet
