/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/utilities/traits.hpp>

// return true if the aggregation is valid for the specified ColumnType
// valid aggregations may still be further specialized (eg, is_string_specialized)
template <typename ColumnType, cudf::aggregation::Kind op>
static constexpr bool is_rolling_supported()
{
  using namespace cudf;

  if (!cudf::detail::is_valid_aggregation<ColumnType, op>()) {
    return false;
  } else if (cudf::is_numeric<ColumnType>() or cudf::is_duration<ColumnType>()) {
    return (op == aggregation::SUM) or (op == aggregation::MIN) or (op == aggregation::MAX) or
           (op == aggregation::COUNT_VALID) or (op == aggregation::COUNT_ALL) or
           (op == aggregation::MEAN) or (op == aggregation::ROW_NUMBER) or
           (op == aggregation::LEAD) or (op == aggregation::LAG) or
           (op == aggregation::COLLECT_LIST);
  } else if (cudf::is_timestamp<ColumnType>()) {
    return (op == aggregation::MIN) or (op == aggregation::MAX) or
           (op == aggregation::COUNT_VALID) or (op == aggregation::COUNT_ALL) or
           (op == aggregation::ROW_NUMBER) or (op == aggregation::LEAD) or
           (op == aggregation::LAG) or (op == aggregation::COLLECT_LIST);
  } else if (cudf::is_fixed_point<ColumnType>()) {
    return (op == aggregation::SUM) or (op == aggregation::MIN) or (op == aggregation::MAX) or
           (op == aggregation::COUNT_VALID) or (op == aggregation::COUNT_ALL) or
           (op == aggregation::ROW_NUMBER) or (op == aggregation::LEAD) or
           (op == aggregation::LAG) or (op == aggregation::COLLECT_LIST);
  } else if (std::is_same<ColumnType, cudf::string_view>()) {
    return (op == aggregation::MIN) or (op == aggregation::MAX) or
           (op == aggregation::COUNT_VALID) or (op == aggregation::COUNT_ALL) or
           (op == aggregation::ROW_NUMBER) or (op == aggregation::COLLECT_LIST);

  } else if (std::is_same<ColumnType, cudf::list_view>()) {
    return (op == aggregation::COUNT_VALID) or (op == aggregation::COUNT_ALL) or
           (op == aggregation::ROW_NUMBER) or (op == aggregation::COLLECT_LIST);
  } else if (std::is_same<ColumnType, cudf::struct_view>()) {
    // TODO: Add support for COUNT_VALID, COUNT_ALL, ROW_NUMBER.
    return op == aggregation::COLLECT_LIST;
  } else {
    return false;
  }
}
