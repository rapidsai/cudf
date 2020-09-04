/*
 * Copyright (c) 2019i-2020, NVIDIA CORPORATION.
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
 * @file readers.hpp
 * @brief cuDF-IO reader classes API
 */

#pragma once

#include <cudf/io/orc.hpp>

//! cuDF interfaces
namespace cudf {
//! IO interfaces
namespace io {
namespace detail {
namespace orc {
/**
 * @brief Class to read ORC dataset data into columns.
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
                  orc_reader_options const& options,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  /**
   * @brief Constructor from an array of datasources
   *
   * @param sources Input `datasource` objects to read the dataset from
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
                  orc_reader_options const& options,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header
   */
  ~reader();

  /**
   * @brief Reads the entire dataset.
   *
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return The set of columns along with table metadata
   */
  table_with_metadata read(orc_reader_options const& options, cudaStream_t stream = 0);
};

/**
 * @brief Class to write ORC dataset data into columns.
 */
class writer {
 private:
  class impl;
  std::unique_ptr<impl> _impl;

 public:
  /**
   * @brief Constructor for output to a file.
   *
   * @param sinkp The data sink to write the data to
   * @param options Settings for controlling writing behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit writer(std::unique_ptr<cudf::io::data_sink> sink,
                  orc_writer_options const& options,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header
   */
  ~writer();

  /**
   * @brief Writes the entire dataset.
   *
   * @param table Set of columns to output
   * @param metadata Table metadata and column names
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  void write(table_view const& table,
             const table_metadata* metadata = nullptr,
             cudaStream_t stream            = 0);

  /**
   * @brief Begins the chunked/streamed write process.
   *
   * @param[in] state State information that crosses _begin() / write_chunked() / _end() boundaries.
   */
  void write_chunked_begin(struct orc_chunked_state& state);

  /**
   * @brief Writes a single subtable as part of a larger ORC file/table write.
   *
   * @param[in] table The table information to be written
   * @param[in] state State information that crosses _begin() / write_chunked() / _end() boundaries.
   */
  void write_chunked(table_view const& table, struct orc_chunked_state& state);

  /**
   * @brief Finishes the chunked/streamed write process.
   *
   * @param[in] state State information that crosses _begin() / write_chunked() / _end() boundaries.
   */
  void write_chunked_end(struct orc_chunked_state& state);
};
}  // namespace orc
}  // namespace detail
}  // namespace io
}  // namespace cudf
