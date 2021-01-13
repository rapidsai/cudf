/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
 * @file orc_metadata.hpp
 * @brief cuDF-IO freeform API
 */

#pragma once

#include <cudf/io/types.hpp>

#include <vector>

namespace cudf {
namespace io {

/**
 * @brief Holds column names and buffers containing raw file-level and stripe-level statistics.
 *
 * The buffers can be parsed using a Protobuf parser. Alternatively, use `parsed_orc_statistics` to
 * get the statistics parsed into a libcudf representation.
 *
 * The `column_names` and `column_stats` members contain one element per column. The `stripe_stats`
 * contains one element per stripe, where each element contains column statistics for each column.
 */
struct raw_orc_statistics {
  std::vector<std::string> column_names;
  std::vector<std::string> column_stats;
  std::vector<std::vector<std::string>> stripe_stats;
};

/**
 * @brief Reads file-level and stripe-level statistics of ORC dataset
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read statistics of a dataset
 * from a file:
 * @code
 *  auto result = cudf::read_raw_orc_statistics(cudf::source_info("dataset.orc"));
 * @endcode
 *
 * @param src_info Dataset source
 *
 * @return ORC file statistics and column names.
 */
raw_orc_statistics read_raw_orc_statistics(source_info const &src_info);

/**
 * @brief Base class for column statistics that include optional min and max values.
 *
 * Includes accessors for the min and max values.
 */
template <typename T>
struct minmax_statistics {
  std::unique_ptr<T> minimum;
  std::unique_ptr<T> maximum;

  auto has_minimum() const { return minimum != nullptr; }
  auto has_maximum() const { return maximum != nullptr; }
  auto *get_minimum() const { return minimum.get(); }
  auto *get_maximum() const { return maximum.get(); }
};

/**
 * @brief Base class for column statistics that include optional sum value.
 *
 * Includes accessors for the sum value.
 */
template <typename T>
struct sum_statistics {
  std::unique_ptr<T> sum;

  auto has_sum() const { return sum != nullptr; }
  auto *get_sum() const { return sum.get(); }
};

struct integer_statistics : minmax_statistics<int64_t>, sum_statistics<int64_t> {
};

struct double_statistics : minmax_statistics<double>, sum_statistics<double> {
};

// According to ORC specs, the sum should be signed, but pyarrow uses unsigned value
struct string_statistics : minmax_statistics<std::string>, sum_statistics<uint64_t> {
};

struct bucket_statistics {
  std::vector<uint64_t> count;

  auto *get_count(size_t index) const { return &count.at(index); }
};

struct decimal_statistics : minmax_statistics<std::string>, sum_statistics<std::string> {
};

struct date_statistics : minmax_statistics<int32_t> {
};

struct binary_statistics : sum_statistics<int64_t> {
};

struct timestamp_statistics : minmax_statistics<int64_t> {
  std::unique_ptr<int64_t> minimumUtc;
  std::unique_ptr<int64_t> maximumUtc;

  auto has_minimumUtc() const { return minimumUtc != nullptr; }
  auto has_maximumUtc() const { return maximumUtc != nullptr; }
  auto *get_minimumUtc() const { return minimumUtc.get(); }
  auto *get_maximumUtc() const { return maximumUtc.get(); }
};

/**
 * @brief Enumerator for types of column statistics that can be included in `column_statistics`.
 *
 * Different statistics types are generated for different column data types.
 */
enum class statistics_type {
  NONE,
  INT,
  DOUBLE,
  STRING,
  BUCKET,
  DECIMAL,
  DATE,
  BINARY,
  TIMESTAMP,
};

/**
 * @brief Contains per-column ORC statistics.
 *
 * `std::unique_ptr` is used to wrap the optional values.
 * At most one of the ***_statistics members has a non-null value. The `type` member can se used to
 * find the valid more easily.
 */
struct column_statistics {
  std::unique_ptr<uint64_t> numberOfValues;
  statistics_type type = statistics_type::NONE;
  std::unique_ptr<integer_statistics> intStatistics;
  std::unique_ptr<double_statistics> doubleStatistics;
  std::unique_ptr<string_statistics> stringStatistics;
  std::unique_ptr<bucket_statistics> bucketStatistics;
  std::unique_ptr<decimal_statistics> decimalStatistics;
  std::unique_ptr<date_statistics> dateStatistics;
  std::unique_ptr<binary_statistics> binaryStatistics;
  std::unique_ptr<timestamp_statistics> timestampStatistics;
  // TODO: hasNull (issue #7087)
};

/**
 * @brief Holds column names and parsed file-level and stripe-level statistics.
 *
 * The `column_names` and `column_stats` members contain one element per column. The `stripe_stats`
 * contains one element per stripe, where each element contains column statistics for each column.
 */
struct parsed_orc_statistics {
  std::vector<std::string> column_names;
  std::vector<column_statistics> column_stats;
  std::vector<std::vector<column_statistics>> stripe_stats;
};

parsed_orc_statistics read_parsed_orc_statistics(source_info const &src_info);

}  // namespace io
}  // namespace cudf
