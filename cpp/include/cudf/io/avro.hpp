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
 * @file avro.hpp
 * @brief cuDF-IO freeform API
 */

#pragma once

#include "types.hpp"

#include <cudf/io/writers.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <string>
#include <vector>

//! cuDF interfaces
namespace cudf {
//! In-development features
namespace io {
/**
 * @brief To build options for `read_avro()`
 */
class avro_reader_options_builder;

/**
 * @brief Settings to use for `read_avro()`
 *
 * @ingroup io_readers
 */
class avro_reader_options {
  source_info _source;

  /// Names of column to read; empty is all
  std::vector<std::string> _columns;

  /// Rows to skip from the start; -1 is none
  size_type _skip_rows = -1;
  /// Rows to read; -1 is all
  size_type _num_rows = -1;

  explicit avro_reader_options(source_info const& src) : _source(src) {}

  friend avro_reader_options_builder;

 public:
  avro_reader_options() = default;

  /**
   * @brief enum class for avro_reader_options size_type parameters
   */
  enum class size_type_param_id : int8_t {
    SKIP_ROWS,  // Rows to skip from the start
    NUM_ROWS    // Rows to read
  };

  /**
   * @brief Returns source info
   */
  source_info const& source() const { return _source; }

  /**
   * @brief Returns names of column to be read
   */
  std::vector<std::string> columns() const { return _columns; }

  /**
   * @brief Returns size_type parameter values as per the corresponding enum
   */
  size_type get(size_type_param_id param_id) const
  {
    switch (param_id) {
      case size_type_param_id::SKIP_ROWS: return _skip_rows;
      case size_type_param_id::NUM_ROWS: return _num_rows;
      default: CUDF_FAIL("Unsupported size_type_param_id enum");
    }
  }

  /**
   * @brief create avro_reader_options_builder which will build avro_reader_options
   *
   * @param src source information used to read avro file
   * @returns avro_reader_options_builder
   */
  static avro_reader_options_builder builder(source_info const& src);
};

class avro_reader_options_builder {
  avro_reader_options options;

 public:
  explicit avro_reader_options_builder() = default;

  explicit avro_reader_options_builder(source_info const& src) : options(src) {}

  /**
   * @brief Set column names which needs to be read
   */
  avro_reader_options_builder& columns(std::vector<std::string> col_names)
  {
    options._columns = std::move(col_names);
    return *this;
  }

  /**
   * @brief Set size_type class members
   */
  avro_reader_options_builder& set(avro_reader_options::size_type_param_id param_id, size_type val)
  {
    switch (param_id) {
      case avro_reader_options::size_type_param_id::SKIP_ROWS: options._skip_rows = val; break;
      case avro_reader_options::size_type_param_id::NUM_ROWS: options._num_rows = val; break;
      default: CUDF_FAIL("Unsupported size_type_param_id enum");
    }

    return *this;
  }

  /**
   * @brief move avro_reader_options member once options is built
   */
  operator avro_reader_options &&() { return std::move(options); }

  /**
   * @brief move avro_reader_options member once options is built
   */
  avro_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads an Avro dataset into a set of columns
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.avro";
 *  cudf::avro_reader_options args{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_avro(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_avro(
  avro_reader_options const& args,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
}  // namespace io
}  // namespace cudf
