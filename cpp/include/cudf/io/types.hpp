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
 * @file types.hpp
 * @brief cuDF-IO API type definitions
 */

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <cudf/types.hpp>

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
  FILEPATH,                  ///< Input/output is a file path
  HOST_BUFFER,               ///< Input/output is a buffer in host memory,
  ARROW_RANDOM_ACCESS_FILE,  ///< Input/output is an arrow::io::RandomAccessFile
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
  STATISTICS_NONE = 0,     //!< No column statistics
  STATISTICS_ROWGROUP = 1, //!< Per-Rowgroup column statistics
  STATISTICS_PAGE = 2,     //!< Per-page column statistics
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
  std::vector<std::string> column_names; //!< Names of columns contained in the table
  std::map<std::string, std::string> user_data; //!< Format-dependent metadata as key-values pairs
};

/**
 * @brief Table with table metadata used by io readers to return the metadata by value
 */
struct table_with_metadata {
  std::unique_ptr<table> tbl;
  table_metadata metadata;
};


/**
 * @brief Source information for read interfaces
 */
struct source_info {
  io_type type = io_type::FILEPATH;
  std::string filepath;
  std::pair<const char*, size_t> buffer;
  std::shared_ptr<arrow::io::RandomAccessFile> file;

  explicit source_info(const std::string& file_path)
      : type(io_type::FILEPATH), filepath(file_path) {}

  explicit source_info(const char* host_buffer, size_t size)
      : type(io_type::HOST_BUFFER), buffer(host_buffer, size) {}

  explicit source_info(
      const std::shared_ptr<arrow::io::RandomAccessFile> arrow_file)
      : type(io_type::ARROW_RANDOM_ACCESS_FILE), file(arrow_file) {}
};

/**
 * @brief Destination information for write interfaces
 */
struct sink_info {
  io_type type = io_type::FILEPATH;
  std::string filepath;

  explicit sink_info(const std::string& file_path)
      : type(io_type::FILEPATH), filepath(file_path) {}
};

}  // namespace io
}  // namespace experimental
}  // namespace cudf
