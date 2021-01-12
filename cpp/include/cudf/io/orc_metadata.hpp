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
 * @file orc_metadata.hpp
 * @brief cuDF-IO freeform API
 */

#pragma once

#include <cudf/io/types.hpp>

#include <vector>

namespace cudf {
namespace io {

/**
 * @brief Reads file-level and stripe-level statistics of ORC dataset
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read statistics of a dataset
 * from a file:
 * @code
 *  std::string filepath = "dataset.orc";
 *  auto result = cudf::read_orc_statistics(cudf::source_info(filepath));
 * @endcode
 *
 * @param src_info Dataset source
 *
 * @return Decompressed ColumnStatistics blobs stored in a vector of vectors. The first element of
 * result vector, which is itself a vector, contains the name of each column. The second element
 * contains statistics of each column of the whole file. Remaining elements contain statistics of
 * each column for each stripe.
 */
std::vector<std::vector<std::string>> read_orc_statistics(source_info const& src_info);

template <typename T>
struct minmax_statistics {
  std::unique_ptr<T> minimum;
  std::unique_ptr<T> maximum;

  auto has_minimum() const { return minimum != nullptr; }
  auto has_maximum() const { return maximum != nullptr; }
  auto *get_minimum() const { return minimum.get(); }
  auto *get_maximum() const { return maximum.get(); }
};

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

struct string_statistics : minmax_statistics<std::string>, sum_statistics<int64_t> {
};

struct bucket_statistics {
  std::vector<uint64_t> count;

  auto *get_count(size_t index) const { return &count.at(index); }
};

struct decimal_statistics : minmax_statistics<std::string>, sum_statistics<std::string>{
};

struct date_statistics : minmax_statistics<int32_t>{
};

struct binary_statistics : sum_statistics<int64_t>{
};

struct timestamp_statistics : minmax_statistics<int64_t> {
  std::unique_ptr<int64_t> minimumUtc;
  std::unique_ptr<int64_t> maximumUtc; 

  auto has_minimumUtc() const { return minimumUtc != nullptr; }
  auto has_maximumUtc() const { return maximumUtc != nullptr; }
  auto *get_minimumUtc() const { return minimumUtc.get(); }
  auto *get_maximumUtc() const { return maximumUtc.get(); }
};

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
  // TODO: hasNull
};

void parse_orc_statistics(std::vector<std::vector<std::string>> const& blobs);

}  // namespace io
}  // namespace cudf
