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

/**
 * @file avro.hpp
 * @brief cuDF-IO freeform API
 */

#pragma once

#include "types.hpp"

#include <rmm/mr/device/per_device_resource.hpp>

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cudf {
namespace io {
/**
 * @brief To build options for `read_avro()`.
 */
class avro_reader_options_builder;

/**
 * @brief Settings to use for `read_avro()`.
 *
 * @ingroup io_readers
 */
class avro_reader_options {
  source_info _source;

  // Names of column to read; empty is all
  std::vector<std::string> _columns;

  // Rows to skip from the start;
  size_type _skip_rows = 0;
  // Rows to read; -1 is all
  size_type _num_rows = -1;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read avro file.
   */
  explicit avro_reader_options(source_info const& src) : _source(src) {}

  friend avro_reader_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  avro_reader_options() = default;

  /**
   * @brief Returns source info.
   */
  source_info const& get_source() const { return _source; }

  /**
   * @brief Returns names of the columns to be read.
   */
  std::vector<std::string> get_columns() const { return _columns; }

  /**
   * @brief Returns number of rows to skip from the start.
   */
  size_type get_skip_rows() const { return _skip_rows; }

  /**
   * @brief Returns number of rows to read.
   */
  size_type get_num_rows() const { return _num_rows; }

  /**
   * @brief Set names of the column to be read.
   *
   * @param col_names Vector of column names.
   */
  void set_columns(std::vector<std::string> col_names) { _columns = std::move(col_names); }

  /**
   * @brief Sets number of rows to skip.
   *
   * @param val Number of rows to skip from start.
   * @return this for chaining.
   */
  void set_skip_rows(size_type val) { _skip_rows = val; }

  /**
   * @brief Sets number of rows to read.
   *
   * @param val Number of rows to read after skip.
   * @return this for chaining.
   */
  void set_num_rows(size_type val) { _num_rows = val; }

  /**
   * @brief create avro_reader_options_builder which will build avro_reader_options.
   *
   * @param src source information used to read avro file.
   * @returns builder to build reader options.
   */
  static avro_reader_options_builder builder(source_info const& src);
};

class avro_reader_options_builder {
  avro_reader_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  avro_reader_options_builder() = default;

  /**
   * @brief Constructor from source info.
   *
   * @param src The source information used to read avro file.
   */
  explicit avro_reader_options_builder(source_info const& src) : options(src) {}

  /**
   * @brief Set names of the column to be read.
   *
   * @param col_names Vector of column names.
   * @return this for chaining.
   */
  avro_reader_options_builder& columns(std::vector<std::string> col_names)
  {
    options._columns = std::move(col_names);
    return *this;
  }

  /**
   * @brief Sets number of rows to skip.
   *
   * @param val Number of rows to skip from start.
   * @return this for chaining.
   */
  avro_reader_options_builder& skip_rows(size_type val)
  {
    options._skip_rows = val;
    return *this;
  }

  /**
   * @brief Sets number of rows to read.
   *
   * @param val Number of rows to read after skip.
   * @return this for chaining.
   */
  avro_reader_options_builder& num_rows(size_type val)
  {
    options._num_rows = val;
    return *this;
  }

  /**
   * @brief move avro_reader_options member once it's built.
   */
  operator avro_reader_options &&() { return std::move(options); }

  /**
   * @brief move avro_reader_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   */
  avro_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads an Avro dataset into a set of columns.
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.avro";
 *  cudf::avro_reader_options options =
 * cudf::avro_reader_options::builder(cudf::source_info(filepath));
 *  ...
 *  auto result = cudf::read_avro(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior.
 * @param mr Device memory resource used to allocate device memory of the table in the returned.
 * table_with_metadata
 *
 * @return The set of columns along with metadata.
 */
table_with_metadata read_avro(
  avro_reader_options const& options,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
}  // namespace io
}  // namespace cudf
