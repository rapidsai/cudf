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

enum class data_type : int32_t {
  INT32   = static_cast<int32_t>(cudf::type_id::INT32),
  INT64   = static_cast<int32_t>(cudf::type_id::INT64),
  FLOAT32 = static_cast<int32_t>(cudf::type_id::FLOAT32),
  FLOAT64 = static_cast<int32_t>(cudf::type_id::FLOAT64),
  STRING  = static_cast<int32_t>(cudf::type_id::STRING),
  LIST    = static_cast<int32_t>(cudf::type_id::LIST),
  STRUCT  = static_cast<int32_t>(cudf::type_id::STRUCT),

  INTEGRAL       = static_cast<int32_t>(type_group_id::INTEGRAL),
  FLOATING_POINT = static_cast<int32_t>(type_group_id::FLOATING_POINT),
  NUMERIC        = static_cast<int32_t>(type_group_id::NUMERIC),
  DECIMAL        = static_cast<int32_t>(type_group_id::FIXED_POINT),
  COMPOUND       = static_cast<int32_t>(type_group_id::COMPOUND),
  NESTED         = static_cast<int32_t>(type_group_id::NESTED)
};

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::null_equality,
  [](auto value) {
    switch (value) {
      case cudf::null_equality::EQUAL: return "NULLS_EQUAL";
      case cudf::null_equality::UNEQUAL: return "NULLS_UNEQUAL";
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
      case join_t::CONDITIONAL: return "CONDITIONAL";
      case join_t::MIXED: return "MIXED";
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
      case data_type::FLOAT32: return "FLOAT32";
      case data_type::FLOAT64: return "FLOAT64";
      case data_type::STRING: return "STRING";
      case data_type::LIST: return "LIST";
      case data_type::STRUCT: return "STRUCT";

      case data_type::INTEGRAL: return "INTEGRAL";
      case data_type::FLOATING_POINT: return "FLOATING_POINT";
      case data_type::NUMERIC: return "NUMERIC";
      case data_type::DECIMAL: return "DECIMAL";
      case data_type::COMPOUND: return "COMPOUND";
      case data_type::NESTED: return "NESTED";

      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })
