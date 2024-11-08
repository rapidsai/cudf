/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "io/orc/aggregate_orc_metadata.hpp"
#include "io/orc/orc.hpp"
#include "io/utilities/column_buffer.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <vector>

namespace cudf::io::orc::detail {
using namespace cudf::io::detail;

/**
 * @brief Keeps track of orc mapping and child column details.
 */
struct reader_column_meta {
  // Mapping between column id in orc to processing order.
  std::vector<std::vector<size_type>> orc_col_map;

  // Number of rows in child columns.
  std::vector<int64_t> num_child_rows;

  // Consists of parent column valid_map and null count.
  std::vector<column_validity_info> parent_column_data;

  std::vector<size_type> parent_column_index;

  // Start row of child columns [stripe][column].
  std::vector<int64_t> child_start_row;

  // Number of rows of child columns [stripe][column].
  std::vector<int64_t> num_child_rows_per_stripe;

  struct row_group_meta {
    size_type num_rows;  // number of rows in a column in a row group
    int64_t start_row;   // start row in a column in a row group
  };

  // Row group metadata [rowgroup][column].
  std::vector<row_group_meta> rwgrp_meta;
};

/**
 * @brief Function that translates ORC data kind to cuDF type enum
 */
inline constexpr type_id to_cudf_type(orc::TypeKind kind,
                                      bool use_np_dtypes,
                                      type_id timestamp_type_id,
                                      type_id decimal_type_id)
{
  switch (kind) {
    case orc::BOOLEAN: return type_id::BOOL8;
    case orc::BYTE: return type_id::INT8;
    case orc::SHORT: return type_id::INT16;
    case orc::INT: return type_id::INT32;
    case orc::LONG: return type_id::INT64;
    case orc::FLOAT: return type_id::FLOAT32;
    case orc::DOUBLE: return type_id::FLOAT64;
    case orc::STRING:
    case orc::BINARY:
    case orc::VARCHAR:
    case orc::CHAR:
      // Variable-length types can all be mapped to STRING
      return type_id::STRING;
    case orc::TIMESTAMP:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_NANOSECONDS;
    case orc::DATE:
      // There isn't a (DAYS -> np.dtype) mapping
      return (use_np_dtypes) ? type_id::TIMESTAMP_MILLISECONDS : type_id::TIMESTAMP_DAYS;
    case orc::DECIMAL: return decimal_type_id;
    // Need to update once cuDF plans to support map type
    case orc::MAP:
    case orc::LIST: return type_id::LIST;
    case orc::STRUCT: return type_id::STRUCT;
    default: break;
  }

  return type_id::EMPTY;
}

/**
 * @brief Determines cuDF type of an ORC Decimal column.
 */
inline type_id to_cudf_decimal_type(host_span<std::string const> decimal128_columns,
                                    aggregate_orc_metadata const& metadata,
                                    int column_index)
{
  if (metadata.get_col_type(column_index).kind != DECIMAL) { return type_id::EMPTY; }

  if (std::find(decimal128_columns.begin(),
                decimal128_columns.end(),
                metadata.column_path(0, column_index)) != decimal128_columns.end()) {
    return type_id::DECIMAL128;
  }

  auto const precision = metadata.get_col_type(column_index)
                           .precision.value_or(cuda::std::numeric_limits<int64_t>::digits10);
  if (precision <= cuda::std::numeric_limits<int32_t>::digits10) { return type_id::DECIMAL32; }
  if (precision <= cuda::std::numeric_limits<int64_t>::digits10) { return type_id::DECIMAL64; }
  return type_id::DECIMAL128;
}

inline std::string get_map_child_col_name(std::size_t const idx)
{
  return (idx == 0) ? "key" : "value";
}

/**
 * @brief Create empty columns and respective schema information from the buffer.
 */
std::unique_ptr<column> create_empty_column(size_type orc_col_id,
                                            aggregate_orc_metadata const& metadata,
                                            host_span<std::string const> decimal128_columns,
                                            bool use_np_dtypes,
                                            data_type timestamp_type,
                                            column_name_info& schema_info,
                                            rmm::cuda_stream_view stream);

/**
 * @brief Assemble the buffer with child columns.
 */
column_buffer assemble_buffer(size_type orc_col_id,
                              std::size_t level,
                              reader_column_meta const& col_meta,
                              aggregate_orc_metadata const& metadata,
                              column_hierarchy const& selected_columns,
                              std::vector<std::vector<column_buffer>>& col_buffers,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

}  // namespace cudf::io::orc::detail
