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
//! IO interfaces
namespace io {
namespace detail {
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
    : compression(format), stats_granularity(stats_lvl)
  {
  }
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
                  writer_options const& options,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

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
   * @param stream CUDA stream used for device memory operations and kernel launches.
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
   * @param[in] return_filemetadata If true, return the raw file metadata
   * @param[in] metadata_out_file_path Column chunks file path to be set in the raw output metadata
   *
   * @return A parquet-compatible blob that contains the data for all rowgroups in the list
   */
  std::unique_ptr<std::vector<uint8_t>> write_chunked_end(
    struct pq_chunked_state& state,
    bool return_filemetadata                  = false,
    const std::string& metadata_out_file_path = "");

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

namespace csv {

/**
 * @brief Options for the CSV writer.
 * Also base class for `write_csv_args`
 */
struct writer_options {
  writer_options()                      = default;
  writer_options(writer_options const&) = default;

  virtual ~writer_options(void) = default;

  /**
   * @brief Constructor to populate writer options.
   *
   * @param na string to use for null entries
   * @param include_header flag that indicates whether to write headers to csv
   * @param rows_per_chunk maximum number of rows to process for each file write
   * @param line_terminator character to use for separating lines (default "\n")
   * @param delim character to use between each column entry (default ',')
   * @param true_v string to use for values !=0 in INT8 types (default 'true')
   * @param false_v string to use for values ==0 in INT8 types (default 'false')
   */
  writer_options(std::string const& na,
                 bool include_header,
                 int rows_per_chunk,
                 std::string line_terminator = std::string{"\n"},
                 char delim                  = ',',
                 std::string true_v          = std::string{"true"},
                 std::string false_v         = std::string{"false"})
    : na_rep_(na),
      include_header_(include_header),
      rows_per_chunk_(rows_per_chunk),
      line_terminator_(line_terminator),
      inter_column_delimiter_(delim),
      true_value_(true_v),
      false_value_(false_v)
  {
  }

  std::string const& na_rep(void) const { return na_rep_; }

  bool include_header(void) const { return include_header_; }

  int rows_per_chunk(void) const { return rows_per_chunk_; }

  std::string const& line_terminator(void) const { return line_terminator_; }

  char inter_column_delimiter(void) const { return inter_column_delimiter_; }

  std::string const& true_value(void) const { return true_value_; }

  std::string const& false_value(void) const { return false_value_; }

  // string to use for null entries:
  //
  std::string const na_rep_;

  // Indicates whether to write headers to csv:
  //
  bool include_header_;

  // maximum number of rows to process for each file write:
  //
  int rows_per_chunk_;

  // character to use for separating lines (default "\n"):
  //
  std::string const line_terminator_;

  // character to use between each column entry (default ','):
  //
  char inter_column_delimiter_;

  // string to use for values !=0 in INT8 types (default 'true'):
  //
  std::string const true_value_;

  // string to use for values ==0 in INT8 types (default 'false'):
  //
  std::string const false_value_;
};

/**
 * @brief Class to write CSV dataset data into columns.
 */
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
   * @param mr Device memory resource to use for device memory allocation
   */
  writer(std::unique_ptr<cudf::io::data_sink> sinkp,
         writer_options const& options,
         rmm::mr::device_memory_resource* mr =
           rmm::mr::get_current_device_resource());  // cannot provide definition here (because
                                                     // _impl is incomplete, hence unique_ptr has
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
  void write_all(table_view const& table,
                 const table_metadata* metadata = nullptr,
                 cudaStream_t stream            = 0);
};

}  // namespace csv

}  // namespace detail
}  // namespace io
}  // namespace cudf
