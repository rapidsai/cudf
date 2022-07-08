/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <optional>
#include <variant>
#include <vector>

namespace cudf {
namespace io {

/**
 * @brief Holds column names and buffers containing raw file-level and stripe-level statistics.
 *
 * The buffers can be parsed using a Protobuf parser. Alternatively, use `parsed_orc_statistics` to
 * get the statistics parsed into a libcudf representation.
 *
 * The `column_names` and `file_stats` members contain one element per column. The `stripes_stats`
 * contains one element per stripe, where each element contains column statistics for each column.
 */
struct raw_orc_statistics {
  std::vector<std::string> column_names;                ///< Column names
  std::vector<std::string> file_stats;                  ///< File-level statistics for each column
  std::vector<std::vector<std::string>> stripes_stats;  ///< Stripe-level statistics for each column
};

/**
 * @brief Reads file-level and stripe-level statistics of ORC dataset.
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
 * @return Column names and encoded ORC statistics
 */
raw_orc_statistics read_raw_orc_statistics(source_info const& src_info);

/**
 * @brief Monostate type alias for the statistics variant.
 */
using no_statistics = std::monostate;

/**
 * @brief Base class for column statistics that include optional minimum and maximum.
 *
 * Includes accessors for the minimum and maximum values.
 */
template <typename T>
struct minmax_statistics {
  std::optional<T> minimum;  ///< Minimum value
  std::optional<T> maximum;  ///< Maximum value
};

/**
 * @brief Base class for column statistics that include an optional sum.
 *
 * Includes accessors for the sum value.
 */
template <typename T>
struct sum_statistics {
  std::optional<T> sum;  ///< Sum of values in column
};

/**
 * @brief Statistics for integral columns.
 */
struct integer_statistics : minmax_statistics<int64_t>, sum_statistics<int64_t> {
};

/**
 * @brief Statistics for floating point columns.
 */
struct double_statistics : minmax_statistics<double>, sum_statistics<double> {
};

/**
 * @brief Statistics for string columns.
 *
 * The `minimum` and `maximum` are the first and last elements, respectively, in lexicographical
 * order. The `sum` is the total length of elements in the column.
 * Note: According to ORC specs, the sum should be signed, but pyarrow uses unsigned value
 */
struct string_statistics : minmax_statistics<std::string>, sum_statistics<uint64_t> {
};

/**
 * @brief Statistics for boolean columns.
 *
 * The `count` array includes the count of `false` and `true` values.
 */
struct bucket_statistics {
  std::vector<uint64_t> count;  ///< Count of `false` and `true` values
};

/**
 * @brief Statistics for decimal columns.
 */
struct decimal_statistics : minmax_statistics<std::string>, sum_statistics<std::string> {
};

/**
 * @brief Statistics for date(time) columns.
 */
using date_statistics = minmax_statistics<int32_t>;

/**
 * @brief Statistics for binary columns.
 *
 * The `sum` is the total number of bytes across all elements.
 */
using binary_statistics = sum_statistics<int64_t>;

/**
 * @brief Statistics for timestamp columns.
 *
 * The `minimum` and `maximum` min/max elements in the column, as the number of milliseconds since
 * the UNIX epoch. The `minimum_utc` and `maximum_utc` are the same values adjusted to UTC.
 */
struct timestamp_statistics : minmax_statistics<int64_t> {
  std::optional<int64_t> minimum_utc;  ///< minimum in milliseconds
  std::optional<int64_t> maximum_utc;  ///< maximum in milliseconds
};

namespace orc {
// forward declare the type that ProtobufReader uses. The `cudf::io::column_statistics` objects,
// returned from `read_parsed_orc_statistics`, are constructed from
// `cudf::io::orc::column_statistics` objects that `ProtobufReader` initializes.
struct column_statistics;
}  // namespace orc

/**
 * @brief Contains per-column ORC statistics.
 *
 * All columns can have the `number_of_values` statistics. Depending on the data type, a column can
 * have additional statistics, accessible through `type_specific_stats` accessor.
 */
struct column_statistics {
  std::optional<uint64_t> number_of_values;  ///< number of statistics
  std::variant<no_statistics,
               integer_statistics,
               double_statistics,
               string_statistics,
               bucket_statistics,
               decimal_statistics,
               date_statistics,
               binary_statistics,
               timestamp_statistics>
    type_specific_stats;  ///< type-specific statistics

  /**
   * @brief Construct a new column statistics object
   *
   * @param detail_statistics The statistics to initialize the object with
   */
  column_statistics(cudf::io::orc::column_statistics&& detail_statistics);
};

/**
 * @brief Holds column names and parsed file-level and stripe-level statistics.
 *
 * The `column_names` and `file_stats` members contain one element per column. The `stripes_stats`
 * member contains one element per stripe, where each element contains column statistics for each
 * column.
 */
struct parsed_orc_statistics {
  std::vector<std::string> column_names;                      ///< column names
  std::vector<column_statistics> file_stats;                  ///< file-level statistics
  std::vector<std::vector<column_statistics>> stripes_stats;  ///< stripe-level statistics
};

/**
 * @brief Reads file-level and stripe-level statistics of ORC dataset.
 *
 * @ingroup io_readers
 *
 * @param src_info Dataset source
 *
 * @return Column names and decoded ORC statistics
 */
parsed_orc_statistics read_parsed_orc_statistics(source_info const& src_info);

}  // namespace io
}  // namespace cudf
