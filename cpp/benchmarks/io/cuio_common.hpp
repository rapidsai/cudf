/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/file_utilities.hpp>

#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>

#include <rmm/device_uvector.hpp>

// IO types supported in the benchmarks
enum class io_type {
  FILEPATH,       // Input/output are both files
  HOST_BUFFER,    // Input/output are both host buffers (pageable)
  PINNED_BUFFER,  // Input is a pinned host buffer, output is a host buffer (pageable)
  DEVICE_BUFFER,  // Input is a device buffer, output is a host buffer (pageable)
  VOID
};

std::string random_file_in_dir(std::string const& dir_path);

/**
 * @brief Class to create a coupled `source_info` and `sink_info` of given type.
 */
class cuio_source_sink_pair {
 public:
  cuio_source_sink_pair(io_type type);
  ~cuio_source_sink_pair()
  {
    // delete the temporary file
    std::remove(file_name.c_str());
  }
  // move constructor
  cuio_source_sink_pair(cuio_source_sink_pair&& ss)            = default;
  cuio_source_sink_pair& operator=(cuio_source_sink_pair&& ss) = default;

  /**
   * @brief Created a source info of the set type
   *
   * The `datasource` created using the returned `source_info` will read data from the same location
   * that the result of a @ref `make_sink_info` call writes to.
   *
   * @return The description of the data source
   */
  cudf::io::source_info make_source_info();

  /**
   * @brief Created a sink info of the set type
   *
   * The `data_sink` created using the returned `sink_info` will write data to the same location
   * that the result of a @ref `make_source_info` call reads from.
   *
   * `io_type::DEVICE_BUFFER` source/sink is an exception where a host buffer sink will be created.
   *
   * @return The description of the data sink
   */
  cudf::io::sink_info make_sink_info();

  [[nodiscard]] size_t size();

 private:
  static temp_directory const tmpdir;

  io_type const type;
  std::vector<char> h_buffer;
  cudf::detail::host_vector<char> pinned_buffer;
  rmm::device_uvector<std::byte> d_buffer;
  std::string const file_name;
  std::unique_ptr<cudf::io::data_sink> void_sink;
};

/**
 * @brief Column selection strategy.
 */
enum class column_selection { ALL, ALTERNATE, FIRST_HALF, SECOND_HALF };

/**
 * @brief Row selection strategy.
 *
 * Not all strategies are applicable to all readers.
 */
enum class row_selection { ALL, BYTE_RANGE, NROWS, SKIPFOOTER, STRIPES, ROW_GROUPS };

/**
 * @brief Modify data types such that total selected columns size is a fix fraction of the total
 * size.
 *
 * The data types are multiplied/rearranged such that the columns selected with the given column
 * selection enumerator add up to a fixed fraction of the total table size, regardless of the data
 * types.
 *
 * @param ids Array of column type IDs
 * @param cs The column selection enumerator
 *
 * @return The duplicated/rearranged array of type IDs
 */
std::vector<cudf::type_id> dtypes_for_column_selection(std::vector<cudf::type_id> const& ids,
                                                       column_selection col_sel);

/**
 * @brief Selects a subset of columns based on the input enumerator.
 */
std::vector<int> select_column_indexes(int num_cols, column_selection col_sel);

/**
 * @brief Selects a subset of columns from the array of names, based on the input enumerator.
 */
std::vector<std::string> select_column_names(std::vector<std::string> const& col_names,
                                             column_selection col_sel);

/**
 * @brief Returns file segments that belong to the given chunk if the file is split into a given
 * number of chunks.
 *
 * The segments could be Parquet row groups or ORC stripes.
 */
std::vector<cudf::size_type> segments_in_chunk(int num_segments, int num_chunks, int chunk);

/**
 * @brief Drops L3 cache if `CUDF_BENCHMARK_DROP_CACHE` environment variable is set.
 *
 * Has no effect if the environment variable is not set.
 * May require sudo access ro run successfully.
 *
 * @throw cudf::logic_error if the environment variable is set and the command fails
 */
void try_drop_l3_cache();

/**
 * @brief Convert a string to the corresponding io_type enum value.
 *
 * This function takes a string and returns the matching io_type enum value. It allows you to
 * convert a string representation of an io_type into its corresponding enum value.
 *
 * @param io_string The input string representing the io_type
 *
 * @return The io_type enum value
 */
io_type retrieve_io_type_enum(std::string_view io_string);

/**
 * @brief Convert a string to the corresponding compression_type enum value.
 *
 * This function takes a string and returns the matching compression_type enum value. It allows you
 * to convert a string representation of a compression_type into its corresponding enum value.
 *
 * @param compression_string The input string representing the compression_type
 *
 * @return The compression_type enum value
 */
cudf::io::compression_type retrieve_compression_type_enum(std::string_view compression_string);
