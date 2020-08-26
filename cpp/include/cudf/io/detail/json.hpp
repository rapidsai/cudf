/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
 * @file json.hpp
 * @brief cuDF-IO reader classes API
 */

#pragma once

#include <cudf/io/json.hpp>

// Forward declarations
namespace arrow {
namespace io {
class RandomAccessFile;
}
}  // namespace arrow

//! cuDF interfaces
namespace cudf {
//! IO interfaces
namespace io {
namespace detail {
namespace json {

/**
 * @brief Class to read JSON dataset data into columns.
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
  explicit reader(std::vector<std::string> const &filepaths,
                  json_reader_options const &options,
                  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

  /**
   * @brief Constructor from an array of datasources
   *
   * @param sources Input `datasource` objects to read the dataset from
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::vector<std::unique_ptr<cudf::io::datasource>> &&sources,
                  json_reader_options const &options,
                  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header
   */
  ~reader();

  /*
   * @brief Reads and returns the entire data set.
   *
   * @return cudf::table object that contains the array of cudf::column.
   */
  table_with_metadata read_all(cudaStream_t stream = 0);

  /*
   * @brief Reads and returns all the rows within a byte range.
   *
   * The returned data includes the row that straddles the end of the range.
   * In other words, a row is included as long as the row begins within the byte
   * range.
   *
   * @param[in] offset Byte offset from the start
   * @param[in] size Number of bytes from the offset; set to 0 for all remaining
   *
   * @return cudf::table object that contains the array of cudf::column
   */
  table_with_metadata read_byte_range(size_t offset, size_t size, cudaStream_t stream = 0);
};

}  // namespace json
}  // namespace detail
}  // namespace io
}  // namespace cudf
