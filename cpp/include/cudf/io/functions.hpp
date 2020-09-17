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
 * @file functions.hpp
 * @brief cuDF-IO freeform API
 */

#pragma once

#include <iostream>
#include "types.hpp"

#include <cudf/io/writers.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

//! cuDF interfaces
namespace cudf {
//! In-development features
namespace io {
/**
 * @brief Settings to use for `read_orc()`
 *
 * @ingroup io_readers
 */
struct read_orc_args {
  source_info source;

  /// Names of column to read; empty is all
  std::vector<std::string> columns;

  /// List of individual stripes to read (ignored if empty)
  std::vector<size_type> stripes;
  /// Rows to skip from the start; -1 is none
  size_type skip_rows = -1;
  /// Rows to read; -1 is all
  size_type num_rows = -1;

  /// Whether to use row index to speed-up reading
  bool use_index = true;

  /// Whether to use numpy-compatible dtypes
  bool use_np_dtypes = true;
  /// Cast timestamp columns to a specific type
  data_type timestamp_type{type_id::EMPTY};

  /// Whether to convert decimals to float64
  bool decimals_as_float = true;
  /// For decimals as int, optional forced decimal scale;
  /// -1 is auto (column scale), >=0: number of fractional digits
  int forced_decimals_scale = -1;

  read_orc_args() = default;

  explicit read_orc_args(source_info const& src) : source(src) {}
};

/**
 * @brief Reads an ORC dataset into a set of columns
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.orc";
 *  cudf::read_orc_args args{cudf::source_info(filepath)};
 *  ...
 *  auto result = cudf::read_orc(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata
 *
 * @return The set of columns
 */
table_with_metadata read_orc(
  read_orc_args const& args,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Settings to use for `write_orc()`
 *
 * @ingroup io_writers
 */
struct write_orc_args {
  /// Specify the sink to use for writer output
  sink_info sink;
  /// Specify the compression format to use
  compression_type compression;
  /// Enable writing column statistics
  bool enable_statistics;
  /// Set of columns to output
  table_view table;
  /// Optional associated metadata
  const table_metadata* metadata;

  write_orc_args() = default;

  explicit write_orc_args(sink_info const& snk,
                          table_view const& table_,
                          const table_metadata* metadata_ = nullptr,
                          compression_type compression_   = compression_type::AUTO,
                          bool stats_en                   = true)
    : sink(snk),
      table(table_),
      metadata(metadata_),
      compression(compression_),
      enable_statistics(stats_en)
  {
  }
};

/**
 * @brief Writes a set of columns to ORC format
 *
 * @ingroup io_writers
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  ...
 *  std::string filepath = "dataset.orc";
 *  cudf::write_orc_args args{cudf::sink_info(filepath), table->view()};
 *  ...
 *  cudf::write_orc(args);
 * @endcode
 *
 * @param args Settings for controlling reading behavior
 * @param mr Device memory resource to use for device memory allocation
 */
void write_orc(write_orc_args const& args,
               rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Settings to use for `write_orc_chunked()`
 *
 * @ingroup io_writers
 */
struct write_orc_chunked_args {
  /// Specify the sink to use for writer output
  sink_info sink;
  /// Specify the compression format to use
  compression_type compression;
  /// Enable writing column statistics
  bool enable_statistics;
  /// Optional associated metadata
  const table_metadata_with_nullability* metadata;

  explicit write_orc_chunked_args(sink_info const& sink_,
                                  const table_metadata_with_nullability* metadata_ = nullptr,
                                  compression_type compression_ = compression_type::AUTO,
                                  bool stats_en                 = true)
    : sink(sink_), metadata(metadata_), compression(compression_), enable_statistics(stats_en)
  {
  }
};

namespace detail {
namespace orc {
/**
 * @brief Forward declaration of anonymous chunked-writer state struct.
 */
struct orc_chunked_state;
};  // namespace orc
};  // namespace detail

/**
 * @brief Begin the process of writing an ORC file in a chunked/stream form.
 *
 * @ingroup io_writers
 *
 * The intent of the write_orc_chunked_ path is to allow writing of an
 * arbitrarily large / arbitrary number of rows to an ORC file in multiple passes.
 *
 * The following code snippet demonstrates how to write a single ORC file containing
 * one logical table by writing a series of individual cudf::tables.
 * @code
 *  ...
 *  std::string filepath = "dataset.orc";
 *  cudf::io::write_orc_chunked_args args{cudf::sink_info(filepath), table->view()};
 *  ...
 *  auto state = cudf::write_orc_chunked_begin(args);
 *    cudf::write_orc_chunked(table0, state);
 *    cudf::write_orc_chunked(table1, state);
 *    ...
 *  cudf_write_orc_chunked_end(state);
 * @endcode
 *
 * @param[in] args Settings for controlling writing behavior
 * @param[in] mr Device memory resource to use for device memory allocation
 *
 * @returns pointer to an anonymous state structure storing information about the chunked write.
 * this pointer must be passed to all subsequent write_orc_chunked() and write_orc_chunked_end()
 *          calls.
 */
std::shared_ptr<detail::orc::orc_chunked_state> write_orc_chunked_begin(
  write_orc_chunked_args const& args,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Write a single table as a subtable of a larger logical orc file/table.
 *
 * @ingroup io_writers
 *
 * All tables passed into multiple calls of this function must contain the same # of columns and
 * have columns of the same type.
 *
 * @param[in] table The table data to be written.
 * @param[in] state Opaque state information about the writer process. Must be the same pointer
 * returned from write_orc_chunked_begin()
 */
void write_orc_chunked(table_view const& table,
                       std::shared_ptr<detail::orc::orc_chunked_state> state);

/**
 * @brief Finish writing a chunked/stream orc file.
 *
 * @ingroup io_writers
 *
 * @param[in] state Opaque state information about the writer process. Must be the same pointer
 * returned from write_orc_chunked_begin()
 */
void write_orc_chunked_end(std::shared_ptr<detail::orc::orc_chunked_state>& state);
}  // namespace io
}  // namespace cudf
