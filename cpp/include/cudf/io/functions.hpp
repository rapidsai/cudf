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
 * @file functions.hpp
 * @brief cuDF-IO freeform API
 */

#include "types.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <string>
#include <unordered_map>
#include <vector>

//! cuDF interfaces
namespace cudf {
//! In-development features
namespace experimental {
//! IO interfaces
namespace io {

/**
 * @brief Settings to use for `read_avro()`
 */
struct read_avro_args {
  source_info source;

  /// Names of column to read; empty is all
  std::vector<std::string> columns;

  /// Rows to skip from the start; -1 is none
  size_type skip_rows = -1;
  /// Rows to read; -1 is all
  size_type num_rows = -1;

  explicit read_avro_args(source_info const& src) : source(src) {}
};

/**
 * @brief Reads an Avro dataset into a set of columns
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  #include <cudf.h>
 *  ...
 *  std::string filepath = "dataset.avro";
 *  cudf::read_avro_args args{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_avro(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Optional resource to use for device memory allocation
 *
 * @return `table` The set of columns
 */
table read_avro(
    read_avro_args const& args,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Settings to use for `read_orc()`
 */
struct read_orc_args {
  source_info source;

  /// Names of column to read; empty is all
  std::vector<std::string> columns;

  /// Stripe to read; -1 is all
  size_type stripe = -1;
  /// Rows to skip from the start; -1 is none
  size_type skip_rows = -1;
  /// Rows to read; -1 is all
  size_type num_rows = -1;

  /// Whether to use row index to speed-up reading
  bool use_index = false;

  /// Whether to use numpy-compatible dtypes
  bool use_np_dtypes = true;
  /// Cast timestamp columns to a specific type
  data_type timestamp_type{EMPTY};

  explicit read_orc_args(source_info const& src) : source(src) {}
};

/**
 * @brief Reads an ORC dataset into a set of columns
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  #include <cudf.h>
 *  ...
 *  std::string filepath = "dataset.orc";
 *  cudf::read_orc_args args{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_orc(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Optional resource to use for device memory allocation
 *
 * @return `table` The set of columns
 */
table read_orc(read_orc_args const& args, rmm::mr::device_memory_resource* mr =
                                              rmm::mr::get_default_resource());

/**
 * @brief Settings to use for `write_orc()`
 */
struct write_orc_args {
  sink_info sink;

  /// Specify the compression format to use
  compression_type compression = compression_type::AUTO;
  /// Set of columns to output
  table_view table;

  explicit write_orc_args(sink_info const& snk, table_view const& table)
      : sink(snk), table(std::move(table)) {}
};

/**
 * @brief Writes a set of columns to ORC format
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  #include <cudf.h>
 *  ...
 *  std::string filepath = "dataset.orc";
 *  cudf::write_orc_args args{cudf::sink_info(filepath), table->view()};
 *  ...
 *  cudf::write_orc(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Optional resource to use for device memory allocation
 */
void write_orc(write_orc_args const& args, rmm::mr::device_memory_resource* mr =
                                               rmm::mr::get_default_resource());

/**
 * @brief Settings to use for `read_parquet()`
 */
struct read_parquet_args {
  source_info source;

  /// Names of column to read; empty is all
  std::vector<std::string> columns;

  /// Row group to read; -1 is all
  size_type row_group = -1;
  /// Rows to skip from the start; -1 is none
  size_type skip_rows = -1;
  /// Rows to read; -1 is all
  size_type num_rows = -1;

  /// Whether to store string data as categorical type
  bool strings_to_categorical = false;
  /// Whether to use PANDAS metadata to load columns
  bool use_pandas_metadata = true;
  /// Cast timestamp columns to a specific type
  data_type timestamp_type{EMPTY};

  explicit read_parquet_args(source_info const& src) : source(src) {}
};

/**
 * @brief Reads a Parquet dataset into a set of columns
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  #include <cudf.h>
 *  ...
 *  std::string filepath = "dataset.parquet";
 *  cudf::read_parquet_args args{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_parquet(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Optional resource to use for device memory allocation
 *
 * @return `table` The set of columns
 */
table read_parquet(
    read_parquet_args const& args,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace io
}  // namespace experimental
}  // namespace cudf
