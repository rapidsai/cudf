/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
 * @file writer_impl.hpp
 * @brief cuDF-IO Parquet writer class implementation header
 */

#pragma once

#include "parquet.hpp"
#include "parquet_gpu.hpp"

#include <cudf/io/data_sink.hpp>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/detail/parquet.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cudf {
namespace io {
namespace detail {
namespace parquet {
// Forward internal classes
struct parquet_column_view;
struct aggregate_writer_metadata;

using namespace cudf::io::parquet;
using namespace cudf::io;
using cudf::detail::device_2dspan;
using cudf::detail::host_2dspan;
using cudf::detail::hostdevice_2dvector;

/**
 * @brief Implementation for parquet writer
 */
class writer::impl {
 public:
  /**
   * @brief Constructor with writer options.
   *
   * @param sink data_sink's for storing dataset
   * @param options Settings for controlling behavior
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit impl(std::vector<std::unique_ptr<data_sink>> sinks,
                parquet_writer_options const& options,
                SingleWriteMode mode,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr);

  /**
   * @brief Constructor with chunked writer options.
   *
   * @param sink data_sink's for storing dataset
   * @param options Settings for controlling behavior
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit impl(std::vector<std::unique_ptr<data_sink>> sinks,
                chunked_parquet_writer_options const& options,
                SingleWriteMode mode,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr);

  /**
   * @brief Destructor to complete any incomplete write and release resources.
   */
  ~impl();

  /**
   * @brief Initializes the states before writing.
   */
  void init_state();

  /**
   * @brief Writes a single subtable as part of a larger parquet file/table write,
   * normally used for chunked writing.
   *
   * @throws rmm::bad_alloc if there is insufficient space for temporary buffers
   *
   * @param[in] table The table information to be written
   * @param[in] partitions Optional partitions to divide the table into. If specified, must be same
   * size as number of sinks.
   */
  void write(table_view const& table, std::vector<partition_info> const& partitions);

  /**
   * @brief Finishes the chunked/streamed write process.
   *
   * @param[in] column_chunks_file_path Column chunks file path to be set in the raw output metadata
   * @return A parquet-compatible blob that contains the data for all rowgroups in the list only if
   * `column_chunks_file_path` is provided, else null.
   */
  std::unique_ptr<std::vector<uint8_t>> close(
    std::vector<std::string> const& column_chunks_file_path = {});

 private:
  // TODO : figure out if we want to keep this. It is currently unused.
  rmm::mr::device_memory_resource* _mr = nullptr;
  // Cuda stream to be used
  rmm::cuda_stream_view stream;

  Compression compression_             = Compression::UNCOMPRESSED;
  size_t max_row_group_size            = default_row_group_size_bytes;
  size_type max_row_group_rows         = default_row_group_size_rows;
  size_t max_page_size_bytes           = default_max_page_size_bytes;
  size_type max_page_size_rows         = default_max_page_size_rows;
  statistics_freq stats_granularity_   = statistics_freq::STATISTICS_NONE;
  dictionary_policy dict_policy_       = dictionary_policy::ALWAYS;
  size_t max_dictionary_size_          = default_max_dictionary_size;
  bool int96_timestamps                = false;
  int32_t column_index_truncate_length = default_column_index_truncate_length;
  std::optional<size_type> max_page_fragment_size_;
  // Overall file metadata.  Filled in during the process and written during write_chunked_end()
  std::unique_ptr<aggregate_writer_metadata> md;
  // File footer key-value metadata. Written during write_chunked_end()
  std::vector<std::map<std::string, std::string>> kv_md;
  // optional user metadata
  std::unique_ptr<table_input_metadata> table_meta;
  // to track if the output has been written to sink
  bool closed = false;
  // To track if the last write(table) call completed successfully
  bool last_write_successful = false;
  // current write position for rowgroups/chunks
  std::vector<std::size_t> current_chunk_offset;
  // special parameter only used by detail::write() to indicate that we are guaranteeing
  // a single table write.  this enables some internal optimizations.
  bool const single_write_mode = true;

  std::vector<std::unique_ptr<data_sink>> out_sink_;
};

}  // namespace parquet
}  // namespace detail
}  // namespace io
}  // namespace cudf
