/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
 * @file types.hpp
 * @brief cuDF-IO API type definitions
 */

#pragma once

#include <cudf/types.hpp>

#include <thrust/optional.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

// Forward declarations
namespace arrow {
namespace io {
class RandomAccessFile;
}
}  // namespace arrow

namespace cudf {
//! IO interfaces
namespace io {
class data_sink;
class datasource;
}  // namespace io
}  // namespace cudf

//! cuDF interfaces
namespace cudf {
//! IO interfaces
namespace io {
/**
 * @brief Compression algorithms
 */
enum class compression_type {
  NONE,    ///< No compression
  AUTO,    ///< Automatically detect or select compression format
  SNAPPY,  ///< Snappy format, using byte-oriented LZ77
  GZIP,    ///< GZIP format, using DEFLATE algorithm
  BZIP2,   ///< BZIP2 format, using Burrows-Wheeler transform
  BROTLI,  ///< BROTLI format, using LZ77 + Huffman + 2nd order context modeling
  ZIP,     ///< ZIP format, using DEFLATE algorithm
  XZ       ///< XZ format, using LZMA(2) algorithm
};

/**
 * @brief Data source or destination types
 */
enum class io_type {
  FILEPATH,          ///< Input/output is a file path
  HOST_BUFFER,       ///< Input/output is a buffer in host memory
  VOID,              ///< Input/output is nothing. No work is done. Useful for benchmarking
  USER_IMPLEMENTED,  ///< Input/output is handled by a custom user class
};

/**
 * @brief Behavior when handling quotations in field data
 */
enum class quote_style {
  MINIMAL,     ///< Quote only fields which contain special characters
  ALL,         ///< Quote all fields
  NONNUMERIC,  ///< Quote all non-numeric fields
  NONE         ///< Never quote fields; disable quotation parsing
};

/**
 * @brief Column statistics granularity type for parquet/orc writers
 */
enum statistics_freq {
  STATISTICS_NONE     = 0,  ///< No column statistics
  STATISTICS_ROWGROUP = 1,  ///< Per-Rowgroup column statistics
  STATISTICS_PAGE     = 2,  ///< Per-page column statistics
};

/**
 * @brief Detailed name information for output columns.
 *
 * The hierarchy of children matches the hierarchy of children in the output
 * cudf columns.
 */
struct column_name_info {
  std::string name;
  std::vector<column_name_info> children;
  column_name_info(std::string const& _name) : name(_name) {}
  column_name_info() = default;
};

/**
 * @brief Table metadata for io readers/writers (primarily column names)
 * For nested types (structs, maps, unions), the ordering of names in the column_names vector
 * corresponds to a pre-order traversal of the column tree.
 * In the example below (2 top-level columns: struct column "col1" and string column "col2"),
 *  column_names = {"col1", "s3", "f5", "f6", "f4", "col2"}.
 *
 *     col1     col2
 *      / \
 *     /   \
 *   s3    f4
 *   / \
 *  /   \
 * f5    f6
 */
struct table_metadata {
  std::vector<std::string> column_names;  //!< Names of columns contained in the table
  std::vector<column_name_info>
    schema_info;  //!< Detailed name information for the entire output hierarchy
  std::map<std::string, std::string> user_data;  //!< Format-dependent metadata of the first input
                                                 //!< file as key-values pairs (deprecated)
  std::vector<std::unordered_map<std::string, std::string>>
    per_file_user_data;  //!< Per file format-dependent metadata as key-values pairs
};

/**
 * @brief Table with table metadata used by io readers to return the metadata by value
 */
struct table_with_metadata {
  std::unique_ptr<table> tbl;
  table_metadata metadata;
};

/**
 * @brief Non-owning view of a host memory buffer
 *
 * Used to describe buffer input in `source_info` objects.
 */
struct host_buffer {
  const char* data = nullptr;
  size_t size      = 0;
  host_buffer()    = default;
  host_buffer(const char* data, size_t size) : data(data), size(size) {}
};

/**
 * @brief Source information for read interfaces
 */
struct source_info {
  std::vector<std::shared_ptr<arrow::io::RandomAccessFile>> _files;

  source_info() = default;

  explicit source_info(std::vector<std::string> const& file_paths) : _filepaths(file_paths) {}
  explicit source_info(std::string const& file_path) : _filepaths({file_path}) {}

  explicit source_info(std::vector<host_buffer> const& host_buffers)
    : _type(io_type::HOST_BUFFER), _buffers(host_buffers)
  {
  }
  explicit source_info(const char* host_data, size_t size)
    : _type(io_type::HOST_BUFFER), _buffers({{host_data, size}})
  {
  }

  explicit source_info(std::vector<cudf::io::datasource*> const& sources)
    : _type(io_type::USER_IMPLEMENTED), _user_sources(sources)
  {
  }
  explicit source_info(cudf::io::datasource* source)
    : _type(io_type::USER_IMPLEMENTED), _user_sources({source})
  {
  }

  [[nodiscard]] auto type() const { return _type; }
  [[nodiscard]] auto const& filepaths() const { return _filepaths; }
  [[nodiscard]] auto const& buffers() const { return _buffers; }
  [[nodiscard]] auto const& files() const { return _files; }
  [[nodiscard]] auto const& user_sources() const { return _user_sources; }

 private:
  io_type _type = io_type::FILEPATH;
  std::vector<std::string> _filepaths;
  std::vector<host_buffer> _buffers;
  std::vector<cudf::io::datasource*> _user_sources;
};

/**
 * @brief Destination information for write interfaces
 */
struct sink_info {
  sink_info() = default;
  sink_info(size_t num_sinks) : _num_sinks(num_sinks) {}

  explicit sink_info(std::vector<std::string> const& file_paths)
    : _type(io_type::FILEPATH), _num_sinks(file_paths.size()), _filepaths(file_paths)
  {
  }
  explicit sink_info(std::string const& file_path)
    : _type(io_type::FILEPATH), _filepaths({file_path})
  {
  }

  explicit sink_info(std::vector<std::vector<char>*> const& buffers)
    : _type(io_type::HOST_BUFFER), _num_sinks(buffers.size()), _buffers(buffers)
  {
  }
  explicit sink_info(std::vector<char>* buffer) : _type(io_type::HOST_BUFFER), _buffers({buffer}) {}

  explicit sink_info(std::vector<cudf::io::data_sink*> const& user_sinks)
    : _type(io_type::USER_IMPLEMENTED), _num_sinks(user_sinks.size()), _user_sinks(user_sinks)
  {
  }
  explicit sink_info(class cudf::io::data_sink* user_sink)
    : _type(io_type::USER_IMPLEMENTED), _user_sinks({user_sink})
  {
  }

  [[nodiscard]] auto type() const { return _type; }
  [[nodiscard]] auto num_sinks() const { return _num_sinks; }
  [[nodiscard]] auto const& filepaths() const { return _filepaths; }
  [[nodiscard]] auto const& buffers() const { return _buffers; }
  [[nodiscard]] auto const& user_sinks() const { return _user_sinks; }

 private:
  io_type _type     = io_type::VOID;
  size_t _num_sinks = 1;
  std::vector<std::string> _filepaths;
  std::vector<std::vector<char>*> _buffers;
  std::vector<cudf::io::data_sink*> _user_sinks;
};

class table_input_metadata;

class column_in_metadata {
  friend table_input_metadata;
  std::string _name = "";
  thrust::optional<bool> _nullable;
  bool _list_column_is_map  = false;
  bool _use_int96_timestamp = false;
  // bool _output_as_binary = false;
  thrust::optional<uint8_t> _decimal_precision;
  thrust::optional<int32_t> _parquet_field_id;
  std::vector<column_in_metadata> children;

 public:
  column_in_metadata() = default;
  column_in_metadata(std::string_view name) : _name{name} {}
  /**
   * @brief Get the children of this column metadata
   *
   * @return this for chaining
   */
  column_in_metadata& add_child(column_in_metadata const& child)
  {
    children.push_back(child);
    return *this;
  }

  /**
   * @brief Set the name of this column
   *
   * @return this for chaining
   */
  column_in_metadata& set_name(std::string const& name)
  {
    _name = name;
    return *this;
  }

  /**
   * @brief Set the nullability of this column
   *
   * Only valid in case of chunked writes. In single writes, this option is ignored.
   *
   * @return column_in_metadata&
   */
  column_in_metadata& set_nullability(bool nullable)
  {
    _nullable = nullable;
    return *this;
  }

  /**
   * @brief Specify that this list column should be encoded as a map in the written parquet file
   *
   * The column must have the structure list<struct<key, value>>. This option is invalid otherwise
   *
   * @return this for chaining
   */
  column_in_metadata& set_list_column_as_map()
  {
    _list_column_is_map = true;
    return *this;
  }

  /**
   * @brief Specifies whether this timestamp column should be encoded using the deprecated int96
   * physical type. Only valid for the following column types:
   * timestamp_s, timestamp_ms, timestamp_us, timestamp_ns
   *
   * @param req True = use int96 physical type. False = use int64 physical type
   * @return this for chaining
   */
  column_in_metadata& set_int96_timestamps(bool req)
  {
    _use_int96_timestamp = req;
    return *this;
  }

  /**
   * @brief Set the decimal precision of this column. Only valid if this column is a decimal
   * (fixed-point) type
   *
   * @param precision The integer precision to set for this decimal column
   * @return this for chaining
   */
  column_in_metadata& set_decimal_precision(uint8_t precision)
  {
    _decimal_precision = precision;
    return *this;
  }

  /**
   * @brief Set the parquet field id of this column.
   *
   * @param field_id The parquet field id to set
   * @return this for chaining
   */
  column_in_metadata& set_parquet_field_id(int32_t field_id)
  {
    _parquet_field_id = field_id;
    return *this;
  }

  /**
   * @brief Get reference to a child of this column
   *
   * @param i Index of the child to get
   * @return this for chaining
   */
  column_in_metadata& child(size_type i) { return children[i]; }

  /**
   * @brief Get const reference to a child of this column
   *
   * @param i Index of the child to get
   * @return this for chaining
   */
  [[nodiscard]] column_in_metadata const& child(size_type i) const { return children[i]; }

  /**
   * @brief Get the name of this column
   */
  [[nodiscard]] std::string get_name() const { return _name; }

  /**
   * @brief Get whether nullability has been explicitly set for this column.
   */
  [[nodiscard]] bool is_nullability_defined() const { return _nullable.has_value(); }

  /**
   * @brief Gets the explicitly set nullability for this column.
   * @throws If nullability is not explicitly defined for this column.
   *         Check using `is_nullability_defined()` first.
   */
  [[nodiscard]] bool nullable() const { return _nullable.value(); }

  /**
   * @brief If this is the metadata of a list column, returns whether it is to be encoded as a map.
   */
  [[nodiscard]] bool is_map() const { return _list_column_is_map; }

  /**
   * @brief Get whether to encode this timestamp column using deprecated int96 physical type
   */
  [[nodiscard]] bool is_enabled_int96_timestamps() const { return _use_int96_timestamp; }

  /**
   * @brief Get whether precision has been set for this decimal column
   */
  [[nodiscard]] bool is_decimal_precision_set() const { return _decimal_precision.has_value(); }

  /**
   * @brief Get the decimal precision that was set for this column.
   * @throws If decimal precision was not set for this column.
   *         Check using `is_decimal_precision_set()` first.
   */
  [[nodiscard]] uint8_t get_decimal_precision() const { return _decimal_precision.value(); }

  /**
   * @brief Get whether parquet field id has been set for this column.
   */
  [[nodiscard]] bool is_parquet_field_id_set() const { return _parquet_field_id.has_value(); }

  /**
   * @brief Get the parquet field id that was set for this column.
   * @throws If parquet field id was not set for this column.
   *         Check using `is_parquet_field_id_set()` first.
   */
  [[nodiscard]] int32_t get_parquet_field_id() const { return _parquet_field_id.value(); }

  /**
   * @brief Get the number of children of this column
   */
  [[nodiscard]] size_type num_children() const { return children.size(); }
};

class table_input_metadata {
 public:
  table_input_metadata() = default;  // Required by cython

  /**
   * @brief Construct a new table_input_metadata from a table_view.
   *
   * The constructed table_input_metadata has the same structure as the passed table_view
   *
   * @param table The table_view to construct metadata for
   */
  table_input_metadata(table_view const& table);

  std::vector<column_in_metadata> column_metadata;
};

/**
 * @brief Information used while writing partitioned datasets
 *
 * This information defines the slice of an input table to write to file. In partitioned dataset
 * writing, one partition_info struct defines one partition and corresponds to one output file
 */
struct partition_info {
  size_type start_row;
  size_type num_rows;

  partition_info() = default;
  partition_info(size_type start_row, size_type num_rows) : start_row(start_row), num_rows(num_rows)
  {
  }
};

}  // namespace io
}  // namespace cudf
