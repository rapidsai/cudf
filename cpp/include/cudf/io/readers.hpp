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
 * @file readers.hpp
 * @brief cuDF-IO reader classes API
 */

#pragma once

#include "types.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
namespace orc {
/**
 * @brief Options for the ORC reader.
 */
struct reader_options {
  std::vector<std::string> columns;
  bool use_index     = true;
  bool use_np_dtypes = true;
  data_type timestamp_type{type_id::EMPTY};
  bool decimals_as_float    = true;
  int forced_decimals_scale = -1;

  reader_options()                       = default;
  reader_options(reader_options const &) = default;

  /**
   * @brief Constructor to populate reader options.
   *
   * @param columns Set of columns to read; empty for all columns
   * @param use_index_lookup Whether to use row index for faster scanning
   * @param np_compat Whether to use numpy-compatible dtypes
   * @param timestamp_type Cast timestamp columns to a specific type
   */
  reader_options(std::vector<std::string> columns,
                 bool use_index_lookup,
                 bool np_compat,
                 data_type timestamp_type,
                 bool decimals_as_float_    = true,
                 int forced_decimals_scale_ = -1)
    : columns(std::move(columns)),
      use_index(use_index_lookup),
      use_np_dtypes(np_compat),
      timestamp_type(timestamp_type),
      decimals_as_float(decimals_as_float_),
      forced_decimals_scale(forced_decimals_scale_)
  {
  }
};

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
  explicit reader(std::vector<std::string> const &filepaths,
                  reader_options const &options,
                  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Constructor from an array of datasources
   *
   * @param sources Input `datasource` objects to read the dataset from
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::vector<std::unique_ptr<cudf::io::datasource>> &&sources,
                  reader_options const &options,
                  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());

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
  table_with_metadata read_all(cudaStream_t stream = 0);

  /**
   * @brief Reads and returns specific stripes.
   *
   * @param stripes Indices of the stripes to read
   * @param stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return The set of columns along with table metadata
   *
   * @throw cudf::logic_error if stripe index is out of range
   */
  table_with_metadata read_stripes(const std::vector<size_type> &stripes, cudaStream_t stream = 0);

  /**
   * @brief Reads and returns a range of rows.
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows Number of rows to read; use `0` for all remaining data
   * @param stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return The set of columns along with table metadata
   */
  table_with_metadata read_rows(size_type skip_rows, size_type num_rows, cudaStream_t stream = 0);
};

}  // namespace orc
}  // namespace detail
}  // namespace io
}  // namespace cudf
