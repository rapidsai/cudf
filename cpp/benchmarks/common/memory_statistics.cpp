/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include "memory_statistics.hpp"

#include <cudf/column/column.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <numeric>

namespace {

/**
 * @brief Calculate the payload size of a string column.
 */
inline uint64_t required_bytes_string(const cudf::column_view& column)
{
  CUDF_EXPECTS(column.type().id() == cudf::type_id::STRING, "Input not a STRING column");
  cudf::strings_column_view input(column);

  uint64_t num_bytes = input.chars_size();
  if (column.nullable()) { num_bytes += cudf::bitmask_allocation_size_bytes(column.size()); }
  return num_bytes;
}

/**
 * @brief Calculate the payload size of a struct column.
 */
inline uint64_t required_bytes_struct(const cudf::column_view& column)
{
  CUDF_EXPECTS(column.type().id() == cudf::type_id::STRUCT, "Input not a STRUCT column");

  uint64_t num_bytes =
    std::accumulate(column.child_begin(), column.child_end(), 0, [](uint64_t acc, const auto& col) {
      return acc + required_bytes(col);
    });
  if (column.nullable()) { num_bytes += cudf::bitmask_allocation_size_bytes(column.size()); }
  return num_bytes;
}

/**
 * @brief Calculate the payload size of a list column.
 */
inline uint64_t required_bytes_list(const cudf::column_view& column)
{
  CUDF_EXPECTS(column.type().id() == cudf::type_id::LIST, "Input not a LIST column");

  uint64_t num_bytes =
    std::accumulate(column.child_begin(), column.child_end(), 0, [](uint64_t acc, const auto& col) {
      return acc + required_bytes(col);
    });
  if (column.nullable()) { num_bytes += cudf::bitmask_allocation_size_bytes(column.size()); }
  return num_bytes;
}

/**
 * @brief Calculate the payload size of a dict column.
 */
inline uint64_t required_bytes_dict(const cudf::column_view& column)
{
  CUDF_EXPECTS(column.type().id() == cudf::type_id::DICTIONARY32,
               "Input not a DICTIONARY32 column");

  cudf::dictionary_column_view input(column);
  uint64_t num_bytes = required_bytes(input.keys());
  num_bytes += required_bytes(input.indices());
  if (column.nullable()) { num_bytes += cudf::bitmask_allocation_size_bytes(column.size()); }
  return num_bytes;
}

/**
 * @brief Calculate the payload size of a column containing fixed width types.
 */
inline uint64_t required_bytes_fixed_width_type(const cudf::column_view& column)
{
  CUDF_EXPECTS(cudf::is_fixed_width(column.type()), "Invalid element type");

  uint64_t num_bytes = column.size() * cudf::size_of(column.type());
  if (column.nullable()) { num_bytes += cudf::bitmask_allocation_size_bytes(column.size()); }
  return num_bytes;
}

}  // namespace

uint64_t required_bytes(const cudf::column_view& column)
{
  uint64_t num_bytes = 0;

  switch (column.type().id()) {
    case cudf::type_id::STRING: num_bytes = required_bytes_string(column); break;
    case cudf::type_id::STRUCT: num_bytes = required_bytes_struct(column); break;
    case cudf::type_id::LIST: num_bytes = required_bytes_list(column); break;
    case cudf::type_id::DICTIONARY32: num_bytes = required_bytes_dict(column); break;
    default: num_bytes = required_bytes_fixed_width_type(column);
  }

  return num_bytes;
}

uint64_t required_bytes(const cudf::table_view& table)
{
  return std::accumulate(table.begin(), table.end(), 0, [](uint64_t acc, const auto& col) {
    return acc + required_bytes(col);
  });
}

uint64_t required_bytes(
  const cudf::host_span<cudf::groupby::aggregation_result>& aggregation_results)
{
  uint64_t read_bytes = 0;

  for (auto const& aggregation : aggregation_results) {  // vector of aggregation results
    for (auto const& col : aggregation.results) {        // vector of columns per result
      read_bytes += required_bytes(col->view());
    }
  }

  return read_bytes;
}
