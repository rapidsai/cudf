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
 * @file json.hpp
 * @brief cuDF-IO freeform API
 */

#pragma once

#include "types.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <string>
#include <vector>

//! cuDF interfaces
namespace cudf {
//! In-development features
namespace io {

/**
 * @brief Builds json_reader_options to use for `read_json()`
 */
class json_reader_options_builder;

/**
 * @brief Input arguments to the `read_json` interface
 *
 * @ingroup io_readers
 *
 * Available parameters and are closely patterned after PANDAS' `read_json` API.
 * Not all parameters are unsupported. If the matching PANDAS' parameter
 * has a default value of `None`, then a default value of `-1` or `0` may be
 * used as the equivalent.
 *
 * Parameters in PANDAS that are unavailable or in cudf:
 *
 * | Name | Description |
 * | ---- | ----------- |
 * | `orient`             | currently fixed-format |
 * | `typ`                | data is always returned as a cudf::table |
 * | `convert_axes`       | use column functions for axes operations instead |
 * | `convert_dates`      | dates are detected automatically |
 * | `keep_default_dates` | dates are detected automatically |
 * | `numpy`              | data is always returned as a cudf::table |
 * | `precise_float`      | there is only one converter |
 * | `date_unit`          | only millisecond units are supported |
 * | `encoding`           | only ASCII-encoded data is supported |
 * | `chunksize`          | use `byte_range_xxx` for chunking instead |
 *
 */
class json_reader_options {
  source_info _source;

  ///< Data types of the column; empty to infer dtypes
  std::vector<std::string> _dtypes;
  /// Specify the compression format of the source or infer from file extension
  compression_type _compression = compression_type::AUTO;

  ///< Read the file as a json object per line
  bool _lines = false;

  ///< Bytes to skip from the start
  size_t _byte_range_offset = 0;
  ///< Bytes to read; always reads complete rows
  size_t _byte_range_size = 0;

  /// Whether to parse dates as DD/MM versus MM/DD
  bool _dayfirst = false;

  explicit json_reader_options(const source_info& src) : _source(src) {}

  friend json_reader_options_builder;

 public:
  explicit json_reader_options() = default;

  /**
   * @brief enum class for json_reader_options boolean parameters
   */

  enum class boolean_param_id : int8_t {
    LINES,     // Read the file as a json object per line
    DAYFIRST,  // hether to parse dates as DD/MM versus MM/DD
  };

  /**
   * @brief enum class for json_reader_options size_type parameters
   */
  enum class size_type_param_id : int8_t {
    BYTE_RANGE_OFFSET,  // Bytes to skip from the start
    BYTE_RANGE_SIZE     // Bytes to read; always reads complete rows
  };

  /**
   * @brief create json_reader_options_builder which will build json_reader_options
   *
   * @param src source information used to read json file
   * @returns json_reader_options_builder
   */
  static json_reader_options_builder builder(source_info const& src);

  /**
   * @brief Returns source info
   */
  source_info const& source() const { return _source; }

  /**
   * @brief Returns Data types of the column
   */
  std::vector<std::string>& dtypes() { return _dtypes; }

  /**
   * @brief Returns Data types of the column
   */
  std::vector<std::string> const& dtypes() const { return _dtypes; }

  /**
   * @brief Returns expected compression format of the source
   */
  compression_type compression() const { return _compression; }

  /**
   * @brief Returns boolean parameter values as per the corresponding enum
   */
  bool get(boolean_param_id param_id) const
  {
    switch (param_id) {
      case boolean_param_id::LINES: return _lines;
      case boolean_param_id::DAYFIRST: return _dayfirst;
      default: CUDF_FAIL("Unsupported boolean_param_id enum");
    }
  }

  /**
   * @brief Returns size_type parameter values as per the corresponding enum
   */
  size_type get(size_type_param_id param_id) const
  {
    switch (param_id) {
      case size_type_param_id::BYTE_RANGE_OFFSET: return _byte_range_offset;
      case size_type_param_id::BYTE_RANGE_SIZE: return _byte_range_size;
      default: CUDF_FAIL("Unsupported size_type_param_id enum");
    }
  }
};

class json_reader_options_builder {
  json_reader_options options;

 public:
  explicit json_reader_options_builder() = default;

  explicit json_reader_options_builder(source_info const& src) : options(src) {}

  /**
   * @brief Set dtypes for columns to be read
   */
  json_reader_options_builder& dtypes(std::vector<std::string> types)
  {
    options._dtypes = std::move(types);
    return *this;
  }

  /**
   * @brief Set compression type to json_reader_options
   */
  json_reader_options_builder& compression(compression_type comp_type)
  {
    options._compression = comp_type;
    return *this;
  }

  /**
   * @brief Set boolean class members
   */
  json_reader_options_builder& set(json_reader_options::boolean_param_id param_id, bool val)
  {
    switch (param_id) {
      case json_reader_options::boolean_param_id::LINES: options._lines = val; break;
      case json_reader_options::boolean_param_id::DAYFIRST: options._dayfirst = val; break;
      default: CUDF_FAIL("Unsupported boolean_param_id enum");
    }

    return *this;
  }

  /**
   * @brief Set size_type class members
   */
  json_reader_options_builder& set(json_reader_options::size_type_param_id param_id, size_type val)
  {
    switch (param_id) {
      case json_reader_options::size_type_param_id::BYTE_RANGE_OFFSET:
        options._byte_range_offset = val;
        break;
      case json_reader_options::size_type_param_id::BYTE_RANGE_SIZE:
        options._byte_range_size = val;
        break;
      default: CUDF_FAIL("Unsupported size_type_param_id enum");
    }

    return *this;
  }

  /**
   * @brief move json_reader_options member once options is built
   */
  operator json_reader_options &&() { return std::move(options); }

  /**
   * @brief move json_reader_options member once options is built
   */
  json_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads a JSON dataset into a set of columns
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.json";
 *  cudf::read_json_args args{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_json(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_json(
  json_reader_options const& args,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace io
}  // namespace cudf
