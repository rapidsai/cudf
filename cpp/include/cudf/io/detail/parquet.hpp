/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
 * @file parquet.hpp
 */

#pragma once

#include <cudf/io/parquet.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace io {
namespace detail {
namespace parquet {
/**
 * @brief Class to read Parquet dataset data into columns.
 */
class reader {
 private:
  class impl;
  std::unique_ptr<impl> _impl;

 public:
  /**
   * @brief Constructor from an array of file paths
   *
   * @param filepaths Paths to the files containing the input dataset
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::vector<std::string> const& filepaths,
                  parquet_reader_options const& options,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Constructor from an array of datasources
   *
   * @param sources Input `datasource` objects to read the dataset from
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
                  parquet_reader_options const& options,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header
   */
  ~reader();

  /**
   * @brief Reads the dataset as per given options.
   *
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return The set of columns along with table metadata
   */
  table_with_metadata read(parquet_reader_options const& options,
                           rmm::cuda_stream_view stream = rmm::cuda_stream_default);
};

/**
 * @brief Class to write parquet dataset data into columns.
 */
class writer {
 private:
  class impl;
  std::unique_ptr<impl> _impl;

 public:
  /**
   * @brief Constructor for output to a file.
   *
   * @param sink The data sink to write the data to
   * @param options Settings for controlling writing behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit writer(std::unique_ptr<cudf::io::data_sink> sink,
                  parquet_writer_options const& options,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header
   */
  ~writer();

  /**
   * @brief Writes the dataset as per options provided.
   *
   * @param table Set of columns to output
   * @param metadata Table metadata and column names
   * @param return_filemetadata If true, return the raw file metadata
   * @param column_chunks_file_path Column chunks file path to be set in the raw output metadata
   * @param int96_timestamps If true, write timestamps as INT96 values
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  std::unique_ptr<std::vector<uint8_t>> write(
    table_view const& table,
    const table_metadata* metadata            = nullptr,
    bool return_filemetadata                  = false,
    const std::string column_chunks_file_path = "",
    bool int96_timestamps                     = false,
    rmm::cuda_stream_view stream              = rmm::cuda_stream_default);

  /**
   * @brief Begins the chunked/streamed write process.
   *
   * @param[in] pq_chunked_state Internal state maintained between chunks.
   */
  void write_chunked_begin(struct pq_chunked_state& state);

  /**
   * @brief Writes a single subtable as part of a larger parquet file/table write.
   *
   * @param[in] table The table information to be written
   * @param[in] pq_chunked_state Internal state maintained between chunks.
   */
  void write_chunk(table_view const& table, struct pq_chunked_state& state);

  /**
   * @brief Finishes the chunked/streamed write process.
   *
   * @param[in] pq_chunked_state Internal state maintained between chunks.
   * @param[in] return_filemetadata If true, return the raw file metadata
   * @param[in] column_chunks_file_path Column chunks file path to be set in the raw output metadata
   *
   * @return A parquet-compatible blob that contains the data for all rowgroups in the list
   */
  std::unique_ptr<std::vector<uint8_t>> write_chunked_end(
    struct pq_chunked_state& state,
    bool return_filemetadata                   = false,
    const std::string& column_chunks_file_path = "");

  /**
   * @brief Merges multiple metadata blobs returned by write_all into a single metadata blob
   *
   * @param[in] metadata_list List of input file metadata
   * @return A parquet-compatible blob that contains the data for all rowgroups in the list
   */
  static std::unique_ptr<std::vector<uint8_t>> merge_rowgroup_metadata(
    const std::vector<std::unique_ptr<std::vector<uint8_t>>>& metadata_list);
};

};  // namespace parquet
};  // namespace detail
};  // namespace io
};  // namespace cudf
