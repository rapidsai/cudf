/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

enum class join_t { CONDITIONAL, MIXED, HASH, SORT_MERGE };

// TODO: need to add list and struct
enum class data_type : int32_t {
  INT32           = static_cast<int32_t>(cudf::type_id::INT32),
  INT64           = static_cast<int32_t>(cudf::type_id::INT64),
  INTEGRAL        = static_cast<int32_t>(type_group_id::INTEGRAL),
  INTEGRAL_SIGNED = static_cast<int32_t>(type_group_id::INTEGRAL_SIGNED),
  FLOAT           = static_cast<int32_t>(type_group_id::FLOATING_POINT),
  BOOL8           = static_cast<int32_t>(cudf::type_id::BOOL8),
  DECIMAL         = static_cast<int32_t>(type_group_id::FIXED_POINT),
  STRING          = static_cast<int32_t>(cudf::type_id::STRING),
  LIST            = static_cast<int32_t>(cudf::type_id::LIST),
  STRUCT          = static_cast<int32_t>(cudf::type_id::STRUCT)
};

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::null_equality,
  [](auto value) {
    switch (value) {
      case cudf::null_equality::EQUAL: return "EQUAL";
      case cudf::null_equality::UNEQUAL: return "UNEQUAL";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  join_t,
  [](auto value) {
    switch (value) {
      case join_t::HASH: return "HASH";
      case join_t::SORT_MERGE: return "SORT_MERGE";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  data_type,
  [](data_type value) {
    switch (value) {
      case data_type::INT32: return "INT32";
      case data_type::INT64: return "INT64";
      case data_type::INTEGRAL: return "INTEGRAL";
      case data_type::INTEGRAL_SIGNED: return "INTEGRAL_SIGNED";
      case data_type::FLOAT: return "FLOAT";
      case data_type::BOOL8: return "BOOL8";
      case data_type::DECIMAL: return "DECIMAL";
      case data_type::STRING: return "STRING";
      case data_type::LIST: return "LIST";
      case data_type::STRUCT: return "STRUCT";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })
