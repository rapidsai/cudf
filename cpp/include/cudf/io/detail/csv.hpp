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

#pragma once

#include <cudf/io/csv.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace io {
namespace detail {
namespace csv {
/**
 * @brief Class to read CSV dataset data into columns.
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
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::vector<std::string> const& filepaths,
                  csv_reader_options const& options,
                  rmm::cuda_stream_view stream,
                  rmm::mr::device_memory_resource* mr);

  /**
   * @brief Constructor from an array of datasources
   *
   * @param sources Input `datasource` objects to read the dataset from
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
                  csv_reader_options const& options,
                  rmm::cuda_stream_view stream,
                  rmm::mr::device_memory_resource* mr);

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header
   */
  ~reader();

  /**
   * @brief Reads the entire dataset.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return The set of columns along with table metadata
   */
  table_with_metadata read(rmm::cuda_stream_view stream = rmm::cuda_stream_default);
};

class writer {
 public:
  class impl;

 private:
  std::unique_ptr<impl> _impl;

 public:
  /**
   * @brief Constructor for output to a file.
   *
   * @param sinkp The data sink to write the data to
   * @param options Settings for controlling writing behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  writer(std::unique_ptr<cudf::io::data_sink> sinkp,
         csv_writer_options const& options,
         rmm::cuda_stream_view stream,
         rmm::mr::device_memory_resource* mr);  // cannot provide definition here (because
                                                // _impl is incomplete hence unique_ptr has
                                                // not enough sizeof() info)

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
             rmm::cuda_stream_view stream   = rmm::cuda_stream_default);
};
}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace cudf
