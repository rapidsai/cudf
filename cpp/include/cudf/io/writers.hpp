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
 * @file readers.hpp
 * @brief cuDF-IO writer classes API
 */

#pragma once

#include "types.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf/io/data_sink.hpp>

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
  /// Selects the compression format to use in the ORC file
  compression_type compression = compression_type::AUTO;
  /// Enables writing column statistics in the ORC file
  bool enable_statistics = true;

  writer_options()                      = default;
  writer_options(writer_options const&) = default;

  /**
   * @brief Constructor to populate writer options.
   *
   * @param format Compression format to use
   */
  explicit writer_options(compression_type format, bool stats_en)
    : compression(format), enable_statistics(stats_en) {}
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
   * @param mr Optional resource to use for device memory allocation
   */
  explicit writer(std::unique_ptr<cudf::io::data_sink> sinkp,
                  writer_options const& options,
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
  void write_all(table_view const& table,
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

  writer_options()                      = default;
  writer_options(writer_options const&) = default;

  /**
   * @brief Constructor to populate writer options.
   *
   * @param format Compression format to use
   */
  explicit writer_options(compression_type format, statistics_freq stats_lvl)
    : compression(format), stats_granularity(stats_lvl) {}
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
   * @param mr Optional resource to use for device memory allocation
   */
  explicit writer(std::unique_ptr<cudf::io::data_sink> sink,
                  writer_options const& options,
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
   * @param return_filemetadata If true, return the raw file metadata
   * @param metadata_out_file_path Column chunks file path to be set in the raw output metadata
   * @param stream Optional stream to use for device memory alloc and kernels
   */
  std::unique_ptr<std::vector<uint8_t>> write_all(table_view const& table,
                                                  const table_metadata* metadata = nullptr,
                                                  bool return_filemetadata       = false,
                                                  const std::string metadata_out_file_path = "",
                                                  cudaStream_t stream                      = 0);

  /**
   * @brief Begins the chunked/streamed write process.
   *
   * @param[in] pq_chunked_state State information that crosses _begin() / write_chunked() / _end()
   * boundaries.
   */
  void write_chunked_begin(struct pq_chunked_state& state);

  /**
   * @brief Writes a single subtable as part of a larger parquet file/table write.
   *
   * @param[in] table The table information to be written
   * @param[in] pq_chunked_state State information that crosses _begin() / write_chunked() / _end()
   * boundaries.
   */
  void write_chunked(table_view const& table, struct pq_chunked_state& state);

  /**
   * @brief Finishes the chunked/streamed write process.
   *
   * @param[in] pq_chunked_state State information that crosses _begin() / write_chunked() / _end()
   * boundaries.
   */
  void write_chunked_end(struct pq_chunked_state& state);

  /**
   * @brief Merges multiple metadata blobs returned by write_all into a single metadata blob
   *
   * @param[in] metadata_list List of input file metadata
   * @return A parquet-compatible blob that contains the data for all rowgroups in the list
   */
  static std::unique_ptr<std::vector<uint8_t>> merge_rowgroup_metadata(
    const std::vector<std::unique_ptr<std::vector<uint8_t>>>& metadata_list);
};

}  // namespace parquet

}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
