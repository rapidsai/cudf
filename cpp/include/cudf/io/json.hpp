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

#pragma once

#include "types.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <string>
#include <vector>

namespace cudf {
namespace io {
/**
 * @addtogroup io_readers
 * @{
 * @file
 */

/**
 * @brief Builds settings to use for `read_json()`.
 */
class json_reader_options_builder;

/**
 * @brief Input arguments to the `read_json` interface.
 *
 * Available parameters and are closely patterned after PANDAS' `read_json` API.
 * Not all parameters are unsupported. If the matching PANDAS' parameter
 * has a default value of `None`, then a default value of `-1` or `0` may be
 * used as the equivalent.
 *
 * Parameters in PANDAS that are unavailable or in cudf:
 *
 * | Name                 | Description                                      |
 * | -------------------- | ------------------------------------------------ |
 * | `orient`             | currently fixed-format                           |
 * | `typ`                | data is always returned as a cudf::table         |
 * | `convert_axes`       | use column functions for axes operations instead |
 * | `convert_dates`      | dates are detected automatically                 |
 * | `keep_default_dates` | dates are detected automatically                 |
 * | `numpy`              | data is always returned as a cudf::table         |
 * | `precise_float`      | there is only one converter                      |
 * | `date_unit`          | only millisecond units are supported             |
 * | `encoding`           | only ASCII-encoded data is supported             |
 * | `chunksize`          | use `byte_range_xxx` for chunking instead        |
 */
class json_reader_options {
  source_info _source;

  // Data types of the column; empty to infer dtypes
  std::vector<std::string> _dtypes;
  // Specify the compression format of the source or infer from file extension
  compression_type _compression = compression_type::AUTO;

  // Read the file as a json object per line
  bool _lines = false;

  // Bytes to skip from the start
  size_t _byte_range_offset = 0;
  // Bytes to read; always reads complete rows
  size_t _byte_range_size = 0;

  // Whether to parse dates as DD/MM versus MM/DD
  bool _dayfirst = false;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read parquet file.
   */
  explicit json_reader_options(const source_info& src) : _source(src) {}

  friend json_reader_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  json_reader_options() = default;

  /**
   * @brief create json_reader_options_builder which will build json_reader_options.
   *
   * @param src source information used to read json file.
   * @returns builder to build the options.
   */
  static json_reader_options_builder builder(source_info const& src);

  /**
   * @brief Returns source info.
   */
  source_info const& get_source() const { return _source; }

  /**
   * @brief Returns data types of the columns.
   */
  std::vector<std::string> const& get_dtypes() const { return _dtypes; }

  /**
   * @brief Returns compression format of the source.
   */
  compression_type get_compression() const { return _compression; }

  /**
   * @brief Returns number of bytes to skip from source start.
   */
  size_t get_byte_range_offset() const { return _byte_range_offset; }

  /**
   * @brief Returns number of bytes to read.
   */
  size_t get_byte_range_size() const { return _byte_range_size; }

  /**
   * @brief Whether to read the file as a json object per line.
   */
  bool is_enabled_lines() const { return _lines; }

  /**
   * @brief Whether to parse dates as DD/MM versus MM/DD.
   */
  bool is_enabled_dayfirst() const { return _dayfirst; }

  /**
   * @brief Set data types for columns to be read.
   *
   * @param types Vector dtypes in string format.
   */
  void dtypes(std::vector<std::string> types) { _dtypes = std::move(types); }

  /**
   * @brief Set the compression type.
   *
   * @param comp_type The compression type used.
   */
  void compression(compression_type comp_type) { _compression = comp_type; }

  /**
   * @brief Set number of bytes to skip from source start.
   *
   * @param offset Number of bytes of offset.
   */
  void set_byte_range_offset(size_type offset) { _byte_range_offset = offset; }

  /**
   * @brief Set number of bytes to read.
   *
   * @param size Number of bytes to read.
   */
  void set_byte_range_size(size_type size) { _byte_range_size = size; }

  /**
   * @brief Set whether to read the file as a json object per line.
   *
   * @param val Boolean value to enable/disable the option to read each line as a json object.
   */
  void enable_lines(bool val) { _lines = val; }

  /**
   * @brief Set whether to parse dates as DD/MM versus MM/DD.
   *
   * @param val Boolean value to enable/disable day first parsing format.
   */
  void enable_dayfirst(bool val) { _dayfirst = val; }
};

class json_reader_options_builder {
  json_reader_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit json_reader_options_builder() = default;

  /**
   * @brief Constructor from source info.
   *
   * @param src The source information used to read avro file.
   */
  explicit json_reader_options_builder(source_info const& src) : options(src) {}

  /**
   * @brief Set data types for columns to be read.
   *
   * @param types Vector dtypes in string format.
   * @return this for chaining.
   */
  json_reader_options_builder& dtypes(std::vector<std::string> types)
  {
    options._dtypes = std::move(types);
    return *this;
  }

  /**
   * @brief Set the compression type.
   *
   * @param comp_type The compression type used.
   * @return this for chaining.
   */
  json_reader_options_builder& compression(compression_type comp_type)
  {
    options._compression = comp_type;
    return *this;
  }

  /**
   * @brief Set number of bytes to skip from source start.
   *
   * @param offset Number of bytes of offset.
   * @return this for chaining.
   */
  json_reader_options_builder& byte_range_offset(size_type offset)
  {
    options._byte_range_offset = offset;
    return *this;
  }

  /**
   * @brief Set number of bytes to read.
   *
   * @param size Number of bytes to read.
   * @return this for chaining
   */
  json_reader_options_builder& byte_range_size(size_type size)
  {
    options._byte_range_size = size;
    return *this;
  }

  /**
   * @brief Set whether to read the file as a json object per line.
   *
   * @param val Boolean value to enable/disable the option to read each line as a json object.
   * @return this for chaining.
   */
  json_reader_options_builder& lines(bool val)
  {
    options._lines = val;
    return *this;
  }

  /**
   * @brief Set whether to parse dates as DD/MM versus MM/DD.
   *
   * @param val Boolean value to enable/disable day first parsing format.
   * @return this for chaining.
   */
  json_reader_options_builder& dayfirst(bool val)
  {
    options._dayfirst = val;
    return *this;
  }

  /**
   * @brief move json_reader_options member once it's built.
   */
  operator json_reader_options&&() { return std::move(options); }

  /**
   * @brief move json_reader_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   */
  json_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads a JSON dataset into a set of columns.
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.json";
 *  cudf::read_json_options options = cudf::read_json_options::builder(cudf::source_info(filepath));
 *  ...
 *  auto result = cudf::read_json(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior.
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata.
 *
 * @return The set of columns along with metadata.
 */
table_with_metadata read_json(
  json_reader_options const& options,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace io
}  // namespace cudf
