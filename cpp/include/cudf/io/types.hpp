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
 * @file types.hpp
 * @brief cuDF-IO API type definitions
 */

#pragma once

#include <cudf/types.hpp>
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
  STATISTICS_NONE     = 0,  //!< No column statistics
  STATISTICS_ROWGROUP = 1,  //!< Per-Rowgroup column statistics
  STATISTICS_PAGE     = 2,  //!< Per-page column statistics
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
  std::map<std::string, std::string> user_data;  //!< Format-dependent metadata as key-values pairs
};

/**
 * @brief Derived class of table_metadata which includes nullability information per column of
 * input.
 *
 * This information is used as an optimization for chunked writes. If the caller leaves
 * column_nullable uninitialized, the writer code will assume the worst case : that all columns are
 * nullable.
 *
 * If the column_nullable field is not empty, it is expected that it has a length equal to the
 * number of columns in the table being written.
 *
 * In the case where column nullability is known, pass `true` if the corresponding column could
 * contain nulls in one or more subtables to be written, otherwise `false`.
 *
 */
struct table_metadata_with_nullability : public table_metadata {
  std::vector<bool> column_nullable;  //!< Per-column nullability information.
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
  io_type type = io_type::FILEPATH;
  std::vector<std::string> filepaths;
  std::vector<host_buffer> buffers;
  std::vector<std::shared_ptr<arrow::io::RandomAccessFile>> files;
  std::vector<cudf::io::datasource*> user_sources;

  source_info() = default;

  explicit source_info(std::vector<std::string> const& file_paths)
    : type(io_type::FILEPATH), filepaths(file_paths)
  {
  }
  explicit source_info(std::string const& file_path)
    : type(io_type::FILEPATH), filepaths({file_path})
  {
  }

  explicit source_info(std::vector<host_buffer> const& host_buffers)
    : type(io_type::HOST_BUFFER), buffers(host_buffers)
  {
  }
  explicit source_info(const char* host_data, size_t size)
    : type(io_type::HOST_BUFFER), buffers({{host_data, size}})
  {
  }

  explicit source_info(std::vector<cudf::io::datasource*> const& sources)
    : type(io_type::USER_IMPLEMENTED), user_sources(sources)
  {
  }
  explicit source_info(cudf::io::datasource* source)
    : type(io_type::USER_IMPLEMENTED), user_sources({source})
  {
  }
};

/**
 * @brief Destination information for write interfaces
 */
struct sink_info {
  io_type type = io_type::VOID;
  std::string filepath;
  std::vector<char>* buffer      = nullptr;
  cudf::io::data_sink* user_sink = nullptr;

  sink_info() = default;

  explicit sink_info(const std::string& file_path) : type(io_type::FILEPATH), filepath(file_path) {}

  explicit sink_info(std::vector<char>* buffer) : type(io_type::HOST_BUFFER), buffer(buffer) {}

  explicit sink_info(class cudf::io::data_sink* user_sink_)
    : type(io_type::USER_IMPLEMENTED), user_sink(user_sink_)
  {
  }
};

}  // namespace io
}  // namespace cudf
