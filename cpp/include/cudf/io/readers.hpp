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
namespace avro {
/**
 * @brief Options for the Avro reader.
 */
struct reader_options {
  std::vector<std::string> columns;

  reader_options()                       = default;
  reader_options(reader_options const &) = default;

  /**
   * @brief Constructor to populate reader options.
   *
   * @param columns Set of columns to read; empty for all columns
   */
  reader_options(std::vector<std::string> columns) : columns(std::move(columns)) {}
};

/**
 * @brief Class to read Avro dataset data into columns.
 */
class reader {
 private:
  class impl;
  std::unique_ptr<impl> _impl;

 public:
  /**
   * @brief Constructor from a file path
   *
   * @param filepath Path to the file containing the whole dataset
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::string filepath,
                  reader_options const &options,
                  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

  /**
   * @brief Constructor from a datasource
   *
   * @param source Input datasource object to read the dataset from
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::unique_ptr<cudf::io::datasource> source,
                  reader_options const &options,
                  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

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
   * @brief Reads and returns a range of rows.
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows Number of rows to read; use `0` for all remaining data
   * @param metadata Optional location to return table metadata
   * @param stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return The set of columns along with table metadata
   */
  table_with_metadata read_rows(size_type skip_rows, size_type num_rows, cudaStream_t stream = 0);
};

}  // namespace avro

namespace json {
/**
 * @brief Options for the JSON reader.
 */
struct reader_options {
  bool lines = false;
  /// Specify the compression format of the source or infer from file extension
  compression_type compression = compression_type::AUTO;
  /// Per-column types; disables type inference on those columns
  std::vector<std::string> dtype;
  bool dayfirst = false;

  reader_options()                       = default;
  reader_options(reader_options const &) = default;

  /**---------------------------------------------------------------------------*
   * @brief Constructor to populate reader options.
   *
   * @param[in] lines Restrict to `JSON Lines` format rather than full JSON
   * @param[in] compression Compression type: "none", "infer", "gzip", "zip"
   * @param[in] dtype Ordered list of data types; deduced from dataset if empty
   *---------------------------------------------------------------------------**/
  reader_options(bool lines,
                 compression_type compression,
                 std::vector<std::string> dtype,
                 bool dayfirst)
    : lines(lines), compression(compression), dtype(std::move(dtype)), dayfirst(dayfirst)
  {
  }
};

/**
 * @brief Class to read JSON dataset data into columns.
 */
class reader {
 private:
  class impl;
  std::unique_ptr<impl> _impl;

 public:
  /**
   * @brief Constructor from a file path
   *
   * @param filepath Path to the file containing the whole dataset
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::string filepath,
                  reader_options const &options,
                  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

  /**
   * @brief Constructor from a datasource
   *
   * @param source Input datasource object to read the dataset from
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::unique_ptr<cudf::io::datasource> source,
                  reader_options const &options,
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

namespace csv {
/**
 * @brief Options for the CSV reader.
 */
struct reader_options {
  // Read settings

  /// Specify the compression format of the source or infer from file extension
  compression_type compression = compression_type::AUTO;
  /// Names of all the columns; if empty then names are auto-generated
  std::vector<std::string> names;
  /// If there is no header or names, prepend this to the column ID as the name
  std::string prefix;
  /// Whether to rename duplicate column names
  bool mangle_dupe_cols = true;

  // Filter settings

  /// Names of columns to read; empty is all columns
  std::vector<std::string> use_cols_names;
  /// Indexes of columns to read; empty is all columns
  std::vector<int> use_cols_indexes;
  /// Header row index
  size_type header = 0;

  // Parsing settings

  /// Line terminator
  char lineterminator = '\n';
  /// Field delimiter
  char delimiter = ',';
  /// Numeric data thousands seperator; cannot match delimiter
  char thousands = '\0';
  /// Decimal point character; cannot match delimiter
  char decimal = '.';
  /// Comment line start character
  char comment = '\0';
  /// Treat whitespace as field delimiter; overrides character delimiter
  bool delim_whitespace = false;
  /// Skip whitespace after the delimiter
  bool skipinitialspace = false;
  /// Ignore empty lines or parse line values as invalid
  bool skip_blank_lines = true;
  /// Treatment of quoting behavior
  quote_style quoting = quote_style::MINIMAL;
  /// Quoting character (if `quoting` is true)
  char quotechar = '\"';
  /// Whether a quote inside a value is double-quoted
  bool doublequote = true;
  /// Whether to parse dates as DD/MM versus MM/DD
  bool dayfirst = false;
  /// Names of columns to read as datetime
  std::vector<std::string> infer_date_names;
  /// Indexes of columns to read as datetime
  std::vector<int> infer_date_indexes;

  // Conversion settings

  /// Per-column types; disables type inference on those columns
  std::vector<std::string> dtype;
  /// User-extensible list of values to recognize as boolean true values
  std::vector<std::string> true_values{"True", "TRUE", "true"};
  /// User-extensible list of values to recognize as boolean false values
  std::vector<std::string> false_values{"False", "FALSE", "false"};
  /// User-extensible list of values to recognize as null values
  std::vector<std::string> na_values{"#N/A",
                                     "#N/A N/A",
                                     "#NA",
                                     "-1.#IND",
                                     "-1.#QNAN",
                                     "-NaN",
                                     "-nan",
                                     "1.#IND",
                                     "1.#QNAN",
                                     "N/A",
                                     "NA",
                                     "NULL",
                                     "NaN",
                                     "n/a",
                                     "nan",
                                     "null"};
  /// Cast timestamp columns to a specific type
  data_type timestamp_type{type_id::EMPTY};

  reader_options()                       = default;
  reader_options(reader_options const &) = default;
};

/**
 * @brief Class to read CSV dataset data into columns.
 */
class reader {
 private:
  class impl;
  std::unique_ptr<impl> _impl;

 public:
  /**
   * @brief Constructor from a file path
   *
   * @param filepath Path to the file containing the whole dataset
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::string filepath,
                  reader_options const &options,
                  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

  /**
   * @brief Constructor from a datasource
   *
   * @param source Input datasource object to read the dataset from
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::unique_ptr<cudf::io::datasource> source,
                  reader_options const &options,
                  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header
   */
  ~reader();

  /**
   * @brief Reads the entire dataset.
   *
   * @param metadata Optional location to return table metadata
   *
   * @return The set of columns along with table metadata
   */
  table_with_metadata read_all(cudaStream_t stream = 0);

  /**
   * @brief Reads all the rows within a byte range.
   *
   * The returned data includes the row that straddles the end of the range.
   * In other words, a row is included as long as the row begins within the byte
   * range.
   *
   * @param offset Byte offset from the start
   * @param size Number of bytes from the offset; set to 0 for all remaining
   * @param stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return The set of columns along with table metadata
   */
  table_with_metadata read_byte_range(size_t offset, size_t size, cudaStream_t stream = 0);

  /**
   * @brief Reads a range of rows.
   *
   * @param skip_rows Number of rows to skip from the start
   * @param skip_rows_end Number of rows to skip from the end
   * @param num_rows Number of rows to read; use `0` for all remaining data
   * @param stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return The set of columns along with table metadata
   */
  table_with_metadata read_rows(size_type skip_rows,
                                size_type skip_rows_end,
                                size_type num_rows,
                                cudaStream_t stream = 0);
};

}  // namespace csv

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
   * @brief Constructor from a file path
   *
   * @param filepath Path to the file containing the whole dataset
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::string filepath,
                  reader_options const &options,
                  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

  /**
   * @brief Constructor from a datasource
   *
   * @param source Input datasource object to read the dataset from
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::unique_ptr<cudf::io::datasource> source,
                  reader_options const &options,
                  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

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
   * @brief Reads and returns a specific stripe.
   *
   * @param stripe Index of the stripe
   * @param stripe_count Number of stripes to read
   * @param stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return The set of columns along with table metadata
   *
   * @throw cudf::logic_error if stripe index is out of range
   */
  table_with_metadata read_stripe(size_type stripe,
                                  size_type stripe_count = 1,
                                  cudaStream_t stream    = 0);

  /**
   * @brief Reads and returns specific stripes.
   *
   * @param stripe_list Indices of the stripes to read
   * @param stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return The set of columns along with table metadata
   *
   * @throw cudf::logic_error if stripe index is out of range
   */
  table_with_metadata read_stripes(const std::vector<size_type> &stripe_list,
                                   cudaStream_t stream = 0);

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

namespace parquet {
/**
 * @brief Options for the Parquet reader.
 */
struct reader_options {
  std::vector<std::string> columns;
  bool strings_to_categorical = false;
  bool use_pandas_metadata    = false;
  data_type timestamp_type{type_id::EMPTY};

  reader_options()                       = default;
  reader_options(reader_options const &) = default;

  /**
   * @brief Constructor to populate reader options.
   *
   * @param columns Set of columns to read; empty for all columns
   * @param strings_to_categorical Whether to return strings as category
   * @param use_pandas_metadata Whether to always load PANDAS index columns
   * @param timestamp_type Cast timestamp columns to a specific type
   */
  reader_options(std::vector<std::string> columns,
                 bool strings_to_categorical,
                 bool use_pandas_metadata,
                 data_type timestamp_type)
    : columns(std::move(columns)),
      strings_to_categorical(strings_to_categorical),
      use_pandas_metadata(use_pandas_metadata),
      timestamp_type(timestamp_type)
  {
  }
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
   * @brief Constructor from a file path
   *
   * @param filepath Path to the file containing the whole dataset
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::string filepath,
                  reader_options const &options,
                  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

  /**
   * @brief Constructor from a datasource
   *
   * @param source Input datasource object to read the dataset from
   * @param options Settings for controlling reading behavior
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::unique_ptr<cudf::io::datasource> source,
                  reader_options const &options,
                  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

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
   * @brief Reads a specific group of rows.
   *
   * @param row_group Index of the row group
   * @param row_group_count Number of row groups to read
   * @param stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return The set of columns along with table metadata
   *
   * @throw cudf::logic_error if row group index is out of range
   */
  table_with_metadata read_row_group(size_type row_group,
                                     size_type row_group_count = 1,
                                     cudaStream_t stream       = 0);

  /**
   * @brief Reads specific row groups.
   *
   * @param row_group_list Indices of the row groups
   * @param stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return The set of columns along with table metadata
   *
   * @throw cudf::logic_error if row group index is out of range
   */
  table_with_metadata read_row_groups(const std::vector<size_type> &row_group_list,
                                      cudaStream_t stream = 0);

  /**
   * @brief Reads a range of rows.
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows Number of rows to read; use `0` for all remaining data
   * @param stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return The set of columns along with table metadata
   */
  table_with_metadata read_rows(size_type skip_rows, size_type num_rows, cudaStream_t stream = 0);
};

}  // namespace parquet

}  // namespace detail
}  // namespace io
}  // namespace cudf
