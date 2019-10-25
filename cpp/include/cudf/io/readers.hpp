/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
//! In-development features
namespace experimental {
//! IO interfaces
namespace io {
//! Inner interfaces and implementations
namespace detail {

//! ORC format
namespace orc {

/**
 * @brief Options for the ORC reader.
 */
struct reader_options {
  std::vector<std::string> columns;
  bool use_index = true;
  bool use_np_dtypes = true;
  data_type timestamp_type{EMPTY};

  reader_options() = default;
  reader_options(reader_options const &) = default;

  /**
   * @brief Constructor to populate reader options.
   *
   * @param columns Set of columns to read; empty for all columns
   * @param use_index_lookup Whether to use row index for faster scanning
   * @param np_compat Whether to use numpy-compatible dtypes
   * @param timestamp_type Cast timestamp columns to a specific type
   */
  reader_options(std::vector<std::string> columns, bool use_index_lookup,
                 bool np_compat, data_type timestamp_type)
      : columns(std::move(columns)),
        use_index(use_index_lookup),
        use_np_dtypes(np_compat),
        timestamp_type(timestamp_type) {}
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
   * @brief Constructor for a filepath to dataset.
   *
   * @param filepath Path to whole dataset
   * @param options Settings for controlling reading behavior
   * @param mr Optional resource to use for device memory allocation
   */
  explicit reader(
      std::string filepath, reader_options const &options,
      rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

  /**
   * @brief Constructor for a memory buffer to dataset.
   *
   * @param buffer Pointer to whole dataset
   * @param length Host buffer size in bytes
   * @param options Settings for controlling reading behavior
   * @param mr Optional resource to use for device memory allocation
   */
  explicit reader(
      const char *buffer, size_t length, reader_options const &options,
      rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

  /**
   * @brief Constructor for an Arrow file to dataset.
   *
   * @param file Arrow file object of dataset
   * @param options Settings for controlling reading behavior
   * @param mr Optional resource to use for device memory allocation
   */
  explicit reader(
      std::shared_ptr<arrow::io::RandomAccessFile> file,
      reader_options const &options,
      rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header
   */
  ~reader();

  /**
   * @brief Reads the entire dataset.
   *
   * @param stream Optional stream to use for device memory alloc and kernels
   *
   * @return `table` The set of columns
   */
  table read_all(cudaStream_t stream = 0);

  /**
   * @brief Reads and returns a specific stripe.
   *
   * @param stripe Index of the stripe
   * @param stream Optional stream to use for device memory alloc and kernels
   *
   * @return `table` The set of columns
   */
  table read_stripe(size_type stripe, cudaStream_t stream = 0);

  /**
   * @brief Reads and returns a range of rows.
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows Number of rows to read; use `0` for all remaining data
   * @param stream Optional stream to use for device memory alloc and kernels
   *
   * @return `table` The set of columns
   */
  table read_rows(size_type skip_rows, size_type num_rows,
                  cudaStream_t stream = 0);
};

}  // namespace orc

//! Parquet format
namespace parquet {

/**
 * @brief Options for the Parquet reader.
 */
struct reader_options {
  std::vector<std::string> columns;
  bool strings_to_categorical = false;
  bool use_pandas_metadata = false;
  data_type timestamp_type{EMPTY};

  reader_options() = default;
  reader_options(reader_options const &) = default;

  /**
   * @brief Constructor to populate reader options.
   *
   * @param columns Set of columns to read; empty for all columns
   * @param strings_to_categorical Whether to return strings as category
   * @param use_pandas_metadata Whether to always load PANDAS index columns
   * @param timestamp_type Cast timestamp columns to a specific type
   */
  reader_options(std::vector<std::string> columns, bool strings_to_categorical,
                 bool use_pandas_metadata, data_type timestamp_type)
      : columns(std::move(columns)),
        strings_to_categorical(strings_to_categorical),
        use_pandas_metadata(use_pandas_metadata),
        timestamp_type(timestamp_type) {}
};

/**
 * @brief Class to read Parquet dataset data into columns.
 */
class reader {
 private:
  class impl;
  std::unique_ptr<impl> _impl;

 public:
  /**
   * @brief Constructor for a filepath to dataset.
   *
   * @param filepath Path to whole dataset
   * @param options Settings for controlling reading behavior
   * @param mr Optional resource to use for device memory allocation
   */
  explicit reader(
      std::string filepath, reader_options const &options,
      rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

  /**
   * @brief Constructor for a memory buffer to dataset.
   *
   * @param buffer Pointer to whole dataset
   * @param length Host buffer size in bytes
   * @param options Settings for controlling reading behavior
   * @param mr Optional resource to use for device memory allocation
   */
  explicit reader(
      const char *buffer, size_t length, reader_options const &options,
      rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

  /**
   * @brief Constructor for an Arrow file to dataset.
   *
   * @param file Arrow file object of dataset
   * @param options Settings for controlling reading behavior
   * @param mr Optional resource to use for device memory allocation
   */
  explicit reader(
      std::shared_ptr<arrow::io::RandomAccessFile> file,
      reader_options const &options,
      rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header
   */
  ~reader();

  /**
   * @brief Returns the PANDAS-specific index column derived from the metadata.
   *
   * @return std::string The name of the column if it exists
   */
  std::string get_pandas_index();

  /**
   * @brief Reads the entire dataset.
   *
   * @param stream Optional stream to use for device memory alloc and kernels
   *
   * @return `table` The set of columns
   */
  table read_all(cudaStream_t stream = 0);

  /**
   * @brief Reads a specific group of rows.
   *
   * @param row_group Index of the row group
   * @param stream Optional stream to use for device memory alloc and kernels
   *
   * @return `table` The set of columns
   */
  table read_row_group(size_type row_group, cudaStream_t stream = 0);

  /**
   * @brief Reads a range of rows.
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows Number of rows to read; use `0` for all remaining data
   * @param stream Optional stream to use for device memory alloc and kernels
   *
   * @return `table` The set of columns
   */
  table read_rows(size_type skip_rows, size_type num_rows,
                  cudaStream_t stream = 0);
};

}  // namespace parquet

}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
