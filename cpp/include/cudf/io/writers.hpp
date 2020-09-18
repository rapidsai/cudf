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
