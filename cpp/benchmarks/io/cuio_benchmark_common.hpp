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

#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/types.hpp>

#include <cudf_test/file_utilities.hpp>

using cudf::io::io_type;

#define RD_BENCHMARK_DEFINE_ALL_SOURCES(benchmark, name, type_or_group)                  \
  benchmark(name##_file_input, type_or_group, static_cast<uint32_t>(io_type::FILEPATH)); \
  benchmark(name##_buffer_input, type_or_group, static_cast<uint32_t>(io_type::HOST_BUFFER));

#define WR_BENCHMARK_DEFINE_ALL_SINKS(benchmark, name, type_or_group)                          \
  benchmark(name##_file_output, type_or_group, static_cast<uint32_t>(io_type::FILEPATH));      \
  benchmark(name##_buffer_output, type_or_group, static_cast<uint32_t>(io_type::HOST_BUFFER)); \
  benchmark(name##_void_output, type_or_group, static_cast<uint32_t>(io_type::VOID));

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
   * The `data_sink` created using the returned `source_info` will write data to the same location
   * that the result of a @ref `make_source_info` call reads from.
   *
   * @return The description of the data sink
   */
  cudf::io::sink_info make_sink_info();

 private:
  static temp_directory const tmpdir;

  io_type const type;
  std::vector<char> buffer;
  std::string const file_name;
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
enum class row_selection { ALL, BYTE_RANGE, NROWS, SKIPFOOTER };

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
                                                       column_selection cs);

/**
 * @brief Select a subset of columns based on the input enumerator.
 */
std::vector<int> select_columns(column_selection cs, int num_cols);
