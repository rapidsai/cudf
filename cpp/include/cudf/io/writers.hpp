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
 * @brief cuDF-IO writer classes API
 */

#pragma once

#include "types.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <utility>

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
 * @brief Options for the ORC writer.
 */
struct writer_options {
  compression_type compression = compression_type::AUTO;

  writer_options() = default;
  writer_options(writer_options const&) = default;

  /**
   * @brief Constructor to populate writer options.
   *
   * @param format Compression format to use
   */
  explicit writer_options(compression_type format) : compression(format) {}
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
   * @param filepath Path to the output file
   * @param options Settings for controlling writing behavior
   * @param mr Optional resource to use for device memory allocation
   */
  explicit writer(
      std::string filepath, writer_options const& options,
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
   * @param stream Optional stream to use for device memory alloc and kernels
   */
  void write_all(table_view const& table, const table_metadata *metadata = nullptr, cudaStream_t stream = 0);
};

}  // namespace orc


//! Parquet format
namespace parquet {

/**
 * @brief Options for the parquet writer.
 */
struct writer_options {
  /// Selects the compressor to use in parquet file
  compression_type compression = compression_type::AUTO;
  /// Select the statistics level to generate in the parquet file
  statistics_freq stats_granularity = statistics_freq::STATISTICS_ROWGROUP;

  writer_options() = default;
  writer_options(writer_options const&) = default;

  /**
   * @brief Constructor to populate writer options.
   *
   * @param format Compression format to use
   */
  explicit writer_options(compression_type format, statistics_freq stats_lvl) :
             compression(format), stats_granularity(stats_lvl) {}
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
   * @param filepath Path to the output file
   * @param options Settings for controlling writing behavior
   * @param mr Optional resource to use for device memory allocation
   */
  explicit writer(
      std::string filepath, writer_options const& options,
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
   * @param stream Optional stream to use for device memory alloc and kernels
   */
  void write_all(table_view const& table, const table_metadata *metadata = nullptr, cudaStream_t stream = 0);
};

}  // namespace parquet


}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
