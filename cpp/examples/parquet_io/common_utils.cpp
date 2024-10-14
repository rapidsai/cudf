/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "common_utils.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/io/types.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <chrono>
#include <iomanip>
#include <string>

/**
 * @file common_utils.cpp
 * @brief Definitions for common utilities for `parquet_io` examples
 *
 */

std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(bool is_pool_used)
{
  auto cuda_mr = std::make_shared<rmm::mr::cuda_memory_resource>();
  if (is_pool_used) {
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      cuda_mr, rmm::percent_of_free_device_memory(50));
  }
  return cuda_mr;
}

cudf::io::column_encoding get_encoding_type(std::string name)
{
  using encoding_type = cudf::io::column_encoding;

  static std::unordered_map<std::string_view, encoding_type> const map = {
    {"DEFAULT", encoding_type::USE_DEFAULT},
    {"DICTIONARY", encoding_type::DICTIONARY},
    {"PLAIN", encoding_type::PLAIN},
    {"DELTA_BINARY_PACKED", encoding_type::DELTA_BINARY_PACKED},
    {"DELTA_LENGTH_BYTE_ARRAY", encoding_type::DELTA_LENGTH_BYTE_ARRAY},
    {"DELTA_BYTE_ARRAY", encoding_type::DELTA_BYTE_ARRAY},
  };

  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (map.find(name) != map.end()) { return map.at(name); }
  throw std::invalid_argument(name +
                              " is not a valid encoding type.\n\n"
                              "Available encoding types: DEFAULT, DICTIONARY, PLAIN,\n"
                              "DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY,\n"
                              "DELTA_BYTE_ARRAY\n\n");
}

cudf::io::compression_type get_compression_type(std::string name)
{
  using compression_type = cudf::io::compression_type;

  static std::unordered_map<std::string_view, compression_type> const map = {
    {"NONE", compression_type::NONE},
    {"AUTO", compression_type::AUTO},
    {"SNAPPY", compression_type::SNAPPY},
    {"LZ4", compression_type::LZ4},
    {"ZSTD", compression_type::ZSTD}};

  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (map.find(name) != map.end()) { return map.at(name); }
  throw std::invalid_argument(name +
                              " is not a valid compression type.\n\n"
                              "Available compression types: NONE, AUTO, SNAPPY,\n"
                              "LZ4, ZSTD\n\n");
}

bool get_boolean(std::string input)
{
  std::transform(input.begin(), input.end(), input.begin(), ::toupper);

  // Check if the input string matches to any of the following
  return input == "ON" or input == "TRUE" or input == "YES" or input == "Y" or input == "T";
}

void check_tables_equal(cudf::table_view const& lhs_table, cudf::table_view const& rhs_table)
{
  try {
    // Left anti-join the original and transcoded tables
    // identical tables should not throw an exception and
    // return an empty indices vector
    auto const indices = cudf::left_anti_join(lhs_table, rhs_table, cudf::null_equality::EQUAL);

    // No exception thrown, check indices
    auto const valid = indices->size() == 0;
    std::cout << "Tables identical: " << valid << "\n\n";
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl << std::endl;
    throw std::runtime_error("Tables identical: false\n\n");
  }
}

std::unique_ptr<cudf::table> concatenate_tables(std::vector<std::unique_ptr<cudf::table>> tables,
                                                rmm::cuda_stream_view stream)
{
  if (tables.size() == 1) { return std::move(tables[0]); }

  std::vector<cudf::table_view> table_views;
  table_views.reserve(tables.size());
  std::transform(
    tables.begin(), tables.end(), std::back_inserter(table_views), [&](auto const& tbl) {
      return tbl->view();
    });
  // Construct the final table
  return cudf::concatenate(table_views, stream);
}

std::string current_date_and_time()
{
  auto const time       = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  auto const local_time = *std::localtime(&time);
  // Stringstream to format the date and time
  std::stringstream ss;
  ss << std::put_time(&local_time, "%Y-%m-%d-%H-%M-%S");
  return ss.str();
}
