/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/io/cuio_common.hpp>

#include <cudf/io/types.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

enum class data_type : int32_t {
  INTEGRAL        = static_cast<int32_t>(type_group_id::INTEGRAL),
  INTEGRAL_SIGNED = static_cast<int32_t>(type_group_id::INTEGRAL_SIGNED),
  FLOAT           = static_cast<int32_t>(type_group_id::FLOATING_POINT),
  DECIMAL         = static_cast<int32_t>(type_group_id::FIXED_POINT),
  TIMESTAMP       = static_cast<int32_t>(type_group_id::TIMESTAMP),
  DURATION        = static_cast<int32_t>(type_group_id::DURATION),
  STRING          = static_cast<int32_t>(cudf::type_id::STRING),
  LIST            = static_cast<int32_t>(cudf::type_id::LIST),
  STRUCT          = static_cast<int32_t>(cudf::type_id::STRUCT)
};

// NVBENCH_DECLARE_ENUM_TYPE_STRINGS macro must be used from global namespace scope
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  data_type,
  [](data_type value) {
    switch (value) {
      case data_type::INTEGRAL: return "INTEGRAL";
      case data_type::INTEGRAL_SIGNED: return "INTEGRAL_SIGNED";
      case data_type::FLOAT: return "FLOAT";
      case data_type::DECIMAL: return "DECIMAL";
      case data_type::TIMESTAMP: return "TIMESTAMP";
      case data_type::DURATION: return "DURATION";
      case data_type::STRING: return "STRING";
      case data_type::LIST: return "LIST";
      case data_type::STRUCT: return "STRUCT";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::io::io_type,
  [](auto value) {
    switch (value) {
      case cudf::io::io_type::FILEPATH: return "FILEPATH";
      case cudf::io::io_type::HOST_BUFFER: return "HOST_BUFFER";
      case cudf::io::io_type::VOID: return "VOID";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::io::compression_type,
  [](auto value) {
    switch (value) {
      case cudf::io::compression_type::SNAPPY: return "SNAPPY";
      case cudf::io::compression_type::NONE: return "NONE";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

enum class uses_index : bool { YES, NO };

enum class uses_numpy_dtype : bool { YES, NO };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  uses_index,
  [](auto value) {
    switch (value) {
      case uses_index::YES: return "YES";
      case uses_index::NO: return "NO";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  uses_numpy_dtype,
  [](auto value) {
    switch (value) {
      case uses_numpy_dtype::YES: return "YES";
      case uses_numpy_dtype::NO: return "NO";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  column_selection,
  [](auto value) {
    switch (value) {
      case column_selection::ALL: return "ALL";
      case column_selection::ALTERNATE: return "ALTERNATE";
      case column_selection::FIRST_HALF: return "FIRST_HALF";
      case column_selection::SECOND_HALF: return "SECOND_HALF";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  row_selection,
  [](auto value) {
    switch (value) {
      case row_selection::ALL: return "ALL";
      case row_selection::NROWS: return "NROWS";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::type_id,
  [](auto value) {
    switch (value) {
      case cudf::type_id::EMPTY: return "EMPTY";
      case cudf::type_id::TIMESTAMP_NANOSECONDS: return "TIMESTAMP_NANOSECONDS";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::io::statistics_freq,
  [](auto value) {
    switch (value) {
      case cudf::io::statistics_freq::STATISTICS_NONE: return "STATISTICS_NONE";
      case cudf::io::statistics_freq::STATISTICS_ROWGROUP: return "ORC_STATISTICS_STRIPE";
      case cudf::io::statistics_freq::STATISTICS_PAGE: return "ORC_STATISTICS_ROW_GROUP";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

enum class converts_strings : bool { YES, NO };

enum class uses_pandas_metadata : bool { YES, NO };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  converts_strings,
  [](auto value) {
    switch (value) {
      case converts_strings::YES: return "YES";
      case converts_strings::NO: return "NO";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  uses_pandas_metadata,
  [](auto value) {
    switch (value) {
      case uses_pandas_metadata::YES: return "YES";
      case uses_pandas_metadata::NO: return "NO";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })
