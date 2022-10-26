/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/io/detail/utils.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <string>
#include <vector>

namespace cudf {
namespace io {

// Forward declaration
class parquet_reader_options;
class parquet_writer_options;
class chunked_parquet_writer_options;

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
   * @brief Constructor from an array of datasources
   *
   * @param sources Input `datasource` objects to read the dataset from
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
                  parquet_reader_options const& options,
                  rmm::cuda_stream_view stream,
                  rmm::mr::device_memory_resource* mr);

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header
   */
  ~reader();

  /**
   * @brief Reads the dataset as per given options.
   *
   * @param options Settings for controlling reading behavior
   *
   * @return The set of columns along with table metadata
   */
  table_with_metadata read(parquet_reader_options const& options);
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
   * @param sinks The data sinks to write the data to
   * @param options Settings for controlling writing behavior
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit writer(std::vector<std::unique_ptr<data_sink>> sinks,
                  parquet_writer_options const& options,
                  SingleWriteMode mode,
                  rmm::cuda_stream_view stream,
                  rmm::mr::device_memory_resource* mr);

  /**
   * @brief Constructor for writer to handle chunked parquet options.
   *
   * @param sinks The data sinks to write the data to
   * @param options Settings for controlling writing behavior for chunked writer
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   *
   * @return A parquet-compatible blob that contains the data for all rowgroups in the list
   */
  explicit writer(std::vector<std::unique_ptr<data_sink>> sinks,
                  chunked_parquet_writer_options const& options,
                  SingleWriteMode mode,
                  rmm::cuda_stream_view stream,
                  rmm::mr::device_memory_resource* mr);

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header
   */
  ~writer();

  /**
   * @brief Writes a single subtable as part of a larger parquet file/table write.
   *
   * @param[in] table The table information to be written
   * @param[in] partitions Optional partitions to divide the table into. If specified, must be same
   * size as number of sinks.
   */
  void write(table_view const& table, std::vector<partition_info> const& partitions = {});

  /**
   * @brief Finishes the chunked/streamed write process.
   *
   * @param[in] column_chunks_file_path Column chunks file path to be set in the raw output metadata
   *
   * @return A parquet-compatible blob that contains the data for all rowgroups in the list only if
   * `column_chunks_file_path` is provided, else null.
   */
  std::unique_ptr<std::vector<uint8_t>> close(
    std::vector<std::string> const& column_chunks_file_path = {});

  /**
   * @brief Merges multiple metadata blobs returned by write_all into a single metadata blob
   *
   * @param[in] metadata_list List of input file metadata
   * @return A parquet-compatible blob that contains the data for all rowgroups in the list
   */
  static std::unique_ptr<std::vector<uint8_t>> merge_row_group_metadata(
    const std::vector<std::unique_ptr<std::vector<uint8_t>>>& metadata_list);
};

};  // namespace parquet
};  // namespace detail
};  // namespace io
};  // namespace cudf
