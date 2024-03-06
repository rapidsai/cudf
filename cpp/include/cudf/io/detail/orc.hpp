/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/io/detail/utils.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cudf::io {

// Forward declaration
class orc_reader_options;
class orc_writer_options;
class chunked_orc_writer_options;

namespace orc::detail {

/**
 * @brief Class to read ORC dataset data into columns.
 */
class reader {
 protected:
  class impl;
  std::unique_ptr<impl> _impl;

  /**
   * @brief Default constructor, needed for subclassing.
   */
  reader();

 public:
  /**
   * @brief Constructor from an array of datasources
   *
   * @param sources Input `datasource` objects to read the dataset from
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
                  orc_reader_options const& options,
                  rmm::cuda_stream_view stream,
                  rmm::mr::device_memory_resource* mr);

  /**
   * @brief Destructor explicitly declared to avoid inlining in header
   */
  virtual ~reader();

  /**
   * @brief Reads the entire dataset.
   *
   * @param options Settings for controlling reading behavior
   * @return The set of columns along with table metadata
   */
  table_with_metadata read(orc_reader_options const& options);
};

/**
 * @brief The reader class that supports iterative reading from an array of data sources.
 *
 * This class intentionally subclasses the `reader` class with private inheritance to hide the
 * base class `reader::read()` API. As such, only chunked reading APIs are supported through it.
 */
class chunked_reader : private reader {
 public:
  /**
   * @brief Constructor from size limits and an array of data sources with reader options.
   *
   * The typical usage should be similar to this:
   * ```
   *  do {
   *    auto const chunk = reader.read_chunk();
   *    // Process chunk
   *  } while (reader.has_next());
   *
   * ```
   *
   * If `output_size_limit == 0` (i.e., no output limit) and `data_read_limit == 0` (no temporary
   * memory size limit), a call to `read_chunk()` will read the whole data source and return a table
   * containing all rows.
   *
   * The `output_size_limit` parameter controls the size of the output table to be returned per
   * `read_chunk()` call. If the user specifies a 100 MB limit, the reader will attempt to return
   * tables that have a total bytes size (over all columns) of 100 MB or less.
   * This is a soft limit and the code will not fail if it cannot satisfy the limit.
   *
   * The `data_read_limit` parameter controls how much temporary memory is used in the entire
   * process of loading, decompressing and decoding of data. Again, this is also a soft limit and
   * the reader will try to make the best effort.
   *
   * Finally, the parameter `output_row_granularity` controls the changes in row number of the
   * output chunk. For each call to `read_chunk()`, with respect to the given `data_read_limit`, a
   * subset of stripes may be loaded, decompressed and decoded into an intermediate table. The
   * reader will then subdivide that table into smaller tables for final output using
   * `output_row_granularity` as the subdivision step.
   *
   * @param output_size_limit Limit on total number of bytes to be returned per `read_chunk()` call,
   *        or `0` if there is no limit
   * @param data_read_limit Limit on temporary memory usage for reading the data sources,
   *        or `0` if there is no limit
   * @param output_row_granularity The granularity parameter used for subdividing the decoded
   *        table for final output
   * @param sources Input `datasource` objects to read the dataset from
   * @param options Settings for controlling reading behaviors
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit chunked_reader(std::size_t output_size_limit,
                          std::size_t data_read_limit,
                          size_type output_row_granularity,
                          std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
                          orc_reader_options const& options,
                          rmm::cuda_stream_view stream,
                          rmm::mr::device_memory_resource* mr);
  /**
   * @brief Constructor from size limits and an array of data sources with reader options.
   *
   * This constructor implicitly call the other constructor with `output_row_granularity` set to
   * 10'000 rows.
   *
   * @param output_size_limit Limit on total number of bytes to be returned per `read_chunk()` call,
   *        or `0` if there is no limit
   * @param data_read_limit Limit on temporary memory usage for reading the data sources,
   *        or `0` if there is no limit
   * @param sources Input `datasource` objects to read the dataset from
   * @param options Settings for controlling reading behaviors
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit chunked_reader(std::size_t output_size_limit,
                          std::size_t data_read_limit,
                          std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
                          orc_reader_options const& options,
                          rmm::cuda_stream_view stream,
                          rmm::mr::device_memory_resource* mr);

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header.
   *
   * Since the declaration of the internal `_impl` object does not exist in this header, this
   * destructor needs to be defined in a separate source file which can access to that object's
   * declaration.
   */
  ~chunked_reader();

  /**
   * @copydoc cudf::io::chunked_orc_reader::has_next
   */
  [[nodiscard]] bool has_next() const;

  /**
   * @copydoc cudf::io::chunked_orc_reader::read_chunk
   */
  [[nodiscard]] table_with_metadata read_chunk() const;
};

/**
 * @brief Class to write ORC dataset data into columns.
 */
class writer {
 private:
  class impl;
  std::unique_ptr<impl> _impl;

 public:
  /**
   * @brief Constructor for output to a file.
   *
   * @param sink The data sink to write the data to
   * @param options Settings for controlling writing behavior
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  explicit writer(std::unique_ptr<cudf::io::data_sink> sink,
                  orc_writer_options const& options,
                  cudf::io::detail::single_write_mode mode,
                  rmm::cuda_stream_view stream);

  /**
   * @brief Constructor with chunked writer options.
   *
   * @param sink The data sink to write the data to
   * @param options Settings for controlling writing behavior
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  explicit writer(std::unique_ptr<cudf::io::data_sink> sink,
                  chunked_orc_writer_options const& options,
                  cudf::io::detail::single_write_mode mode,
                  rmm::cuda_stream_view stream);

  /**
   * @brief Destructor explicitly declared to avoid inlining in header
   */
  ~writer();

  /**
   * @brief Writes a single subtable as part of a larger ORC file/table write.
   *
   * @param[in] table The table information to be written
   */
  void write(table_view const& table);

  /**
   * @brief Finishes the chunked/streamed write process.
   */
  void close();

  /**
   * @brief Skip work done in `close()`; should be called if `write()` failed.
   *
   * Calling skip_close() prevents the writer from writing the (invalid) file footer and the
   * postscript.
   */
  void skip_close();
};

}  // namespace orc::detail
}  // namespace cudf::io
