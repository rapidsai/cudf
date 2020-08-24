/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/io/types.hpp>
#include <cudf/utilities/traits.hpp>

// used to make CUIO_BENCH_ALL_TYPES calls more readable
constexpr int UNCOMPRESSED = (int)cudf::io::compression_type::NONE;
constexpr int USE_SNAPPY   = (int)cudf::io::compression_type::SNAPPY;

#define CUIO_BENCH_ALL_TYPES(benchmark_define, compression)

enum class type_group_id : int32_t {
  INTEGRAL = static_cast<int32_t>(cudf::type_id::NUM_TYPE_IDS),
  FLOATING_POINT,
  TIMESTAMP,
  DURATION,
  FIXED_POINT,
  COMPOUND,
  NESTED,
};
std::vector<cudf::type_id> get_type_or_group(int32_t id)
{
  // identity transformation when passing a concrete type_id
  if (id < static_cast<int32_t>(cudf::type_id::NUM_TYPE_IDS))
    return {static_cast<cudf::type_id>(id)};

  // if the value is larger that type_id::NUM_TYPE_IDS, it's a group id
  type_group_id const group_id = static_cast<type_group_id>(id);

  using trait_fn       = bool (*)(cudf::data_type);
  trait_fn is_integral = [](cudf::data_type type) {
    return cudf::is_numeric(type) && !cudf::is_floating_point(type);
  };
  auto fn = [&]() -> trait_fn {
    switch (group_id) {
      case type_group_id::FLOATING_POINT: return cudf::is_floating_point;
      case type_group_id::INTEGRAL: return is_integral;
      case type_group_id::TIMESTAMP: return cudf::is_timestamp;
      case type_group_id::DURATION: return cudf::is_duration;
      case type_group_id::FIXED_POINT: return cudf::is_fixed_point;
      case type_group_id::COMPOUND: return cudf::is_compound;
      case type_group_id::NESTED: return cudf::is_nested;
      default: CUDF_FAIL("Invalid data type group");
    }
  }();
  std::vector<cudf::type_id> types;
  for (int type_int = 0; type_int < static_cast<int32_t>(cudf::type_id::NUM_TYPE_IDS); ++type_int) {
    auto const type = static_cast<cudf::type_id>(type_int);
    if (type != cudf::type_id::EMPTY && fn(cudf::data_type(type))) types.push_back(type);
  }
  return types;
}
