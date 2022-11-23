/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

namespace cudf {
namespace io {

/**
 * @brief Per-column histogram struct containing detected occurrences of each dtype
 */
struct column_type_histogram {
  cudf::size_type null_count{};
  cudf::size_type float_count{};
  cudf::size_type datetime_count{};
  cudf::size_type string_count{};
  cudf::size_type negative_small_int_count{};
  cudf::size_type positive_small_int_count{};
  cudf::size_type big_int_count{};
  cudf::size_type bool_count{};
  auto total_count() const
  {
    return null_count + float_count + datetime_count + string_count + negative_small_int_count +
           positive_small_int_count + big_int_count + bool_count;
  }
};

struct column_type_bool_any {
  enum class type {
    NULL_COUNT,
    FLOAT_COUNT,
    DATETIME_COUNT,
    STRING_COUNT,
    NEGATIVE_SMALL_INT_COUNT,
    POSITIVE_SMALL_INT_COUNT,
    BIG_INT_COUNT,
    BOOL_COUNT,
    VALID_COUNT,
    TOTAL_COUNT
  };
  uint32_t bitfield{};
  constexpr bool null_count() const
  {
    return bitfield & (1 << static_cast<uint32_t>(type::NULL_COUNT));
  }
  constexpr bool float_count() const
  {
    return bitfield & (1 << static_cast<uint32_t>(type::FLOAT_COUNT));
  }
  constexpr bool datetime_count() const
  {
    return bitfield & (1 << static_cast<uint32_t>(type::DATETIME_COUNT));
  }
  constexpr bool string_count() const
  {
    return bitfield & (1 << static_cast<uint32_t>(type::STRING_COUNT));
  }
  constexpr bool negative_small_int_count() const
  {
    return bitfield & (1 << static_cast<uint32_t>(type::NEGATIVE_SMALL_INT_COUNT));
  }
  constexpr bool positive_small_int_count() const
  {
    return bitfield & (1 << static_cast<uint32_t>(type::POSITIVE_SMALL_INT_COUNT));
  }
  constexpr bool big_int_count() const
  {
    return bitfield & (1 << static_cast<uint32_t>(type::BIG_INT_COUNT));
  }
  constexpr bool bool_count() const
  {
    return bitfield & (1 << static_cast<uint32_t>(type::BOOL_COUNT));
  }
  constexpr bool valid_count() const
  {
    return bitfield & (1 << static_cast<uint32_t>(type::VALID_COUNT));
  }
};
// ==null, OR => any null or all valid.
// ==null, AND=> all null or any valid.
// ==valid, OR=> any valid or all null.
// ==valid, AND=> all valid or any null.
/// null_count == size --> all null. (any way to infer this without size?)
/// null_count > 0  --> any null.
}  // namespace io
}  // namespace cudf
