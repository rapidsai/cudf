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

#pragma once

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <chrono>
#include <iostream>
#include <optional>
#include <string>

/**
 * @brief Create memory resource for libcudf functions
 *
 * @param pool Whether to use a pool memory resource.
 * @return Memory resource instance
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

/**
 * @brief Get encoding type from the keyword
 *
 * @param name encoding keyword name
 * @return corresponding column encoding type
 */
[[nodiscard]] cudf::io::column_encoding get_encoding_type(std::string name)
{
  using encoding_type = cudf::io::column_encoding;

  static const std::unordered_map<std::string_view, cudf::io::column_encoding> map = {
    {"DEFAULT", encoding_type::USE_DEFAULT},
    {"DICTIONARY", encoding_type::DICTIONARY},
    {"PLAIN", encoding_type::PLAIN},
    {"DELTA_BINARY_PACKED", encoding_type::DELTA_BINARY_PACKED},
    {"DELTA_LENGTH_BYTE_ARRAY", encoding_type::DELTA_LENGTH_BYTE_ARRAY},
    {"DELTA_BYTE_ARRAY", encoding_type::DELTA_BYTE_ARRAY},
  };

  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (map.find(name) != map.end()) { return map.at(name); }
  throw std::invalid_argument("FATAL: " + std::string(name) +
                              " is not a valid encoding type.\n\n"
                              "Available encoding types: DEFAULT, DICTIONARY, PLAIN,\n"
                              "DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY,\n"
                              "DELTA_BYTE_ARRAY\n"
                              "\n"
                              "Exiting...\n");
}

/**
 * @brief Get compression type from the keyword
 *
 * @param name compression keyword name
 * @return corresponding compression type
 */
[[nodiscard]] cudf::io::compression_type get_compression_type(std::string name)
{
  using compression_type = cudf::io::compression_type;

  static const std::unordered_map<std::string_view, cudf::io::compression_type> map = {
    {"NONE", compression_type::NONE},
    {"AUTO", compression_type::AUTO},
    {"SNAPPY", compression_type::SNAPPY},
    {"LZ4", compression_type::LZ4},
    {"ZSTD", compression_type::ZSTD}};

  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (map.find(name) != map.end()) { return map.at(name); }
  throw std::invalid_argument("FATAL: " + std::string(name) +
                              " is not a valid compression type.\n\n"
                              "Available compression_type types: NONE, AUTO, SNAPPY,\n"
                              "LZ4, ZSTD\n"
                              "\n"
                              "Exiting...\n");
}

/**
 * @brief Get the optional page size stat frequency from they keyword
 *
 * @param use_stats keyword affirmation string such as: Y, T, YES, TRUE, ON
 * @return optional page statistics frequency set to full (STATISTICS_COLUMN)
 */
[[nodiscard]] std::optional<cudf::io::statistics_freq> get_page_size_stats(std::string use_stats)
{
  std::transform(use_stats.begin(), use_stats.end(), use_stats.begin(), ::toupper);

  // Check if the input string matches to any of the following
  if (not use_stats.compare("ON") or not use_stats.compare("TRUE") or
      not use_stats.compare("YES") or not use_stats.compare("Y") or not use_stats.compare("T")) {
    // Full column and offset indices - STATISTICS_COLUMN
    return std::make_optional(cudf::io::statistics_freq::STATISTICS_COLUMN);
  }

  return std::nullopt;
}

/**
 * @brief Light-weight timer for parquet reader and writer instrumentation
 *
 * Timer object constructed from std::chrono, instrumenting at microseconds
 * precision. Can display elapsed durations at milli and micro second
 * scales. Timer starts at object construction.
 */
class Timer {
 public:
  using micros = std::chrono::microseconds;
  using millis = std::chrono::milliseconds;

  Timer() { reset(); }
  void reset() { start_time = std::chrono::high_resolution_clock::now(); }
  auto elapsed() { return (std::chrono::high_resolution_clock::now() - start_time); }
  void print_elapsed_micros()
  {
    std::cout << "Elapsed Time: " << std::chrono::duration_cast<micros>(elapsed()).count()
              << "us\n\n";
  }
  void print_elapsed_millis()
  {
    std::cout << "Elapsed Time: " << std::chrono::duration_cast<millis>(elapsed()).count()
              << "ms\n\n";
  }

 private:
  using time_point_t = std::chrono::time_point<std::chrono::high_resolution_clock>;
  time_point_t start_time;
};
