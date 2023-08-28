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

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

/**
 * @brief Calculate the number of bytes needed to completely read/write the provided column.
 *
 * The functions computes only the size of the payload of the column in bytes, it excludes
 * any metadata.
 *
 * @param column View of the input column
 * @returns Number of bytes needed to read or write the column.
 */
uint64_t required_bytes(const cudf::column_view& column);

/**
 * @brief Calculate the number of bytes needed to completely read/write the provided table.
 *
 * The functions computes only the size of the payload of the table in bytes, it excludes
 * any metadata.
 *
 * @param table View of the input table.
 * @returns Number of bytes needed to read or write the table.
 */
uint64_t required_bytes(const cudf::table_view& table);

/**
 * @brief Calculate the number of bytes needed to completely read/write the provided sequence of
 * aggregation results.
 *
 * The functions computes only the size of the payload of the aggregation results in bytes, it
 * excludes any metadata.
 *
 * @param aggregation_results Sequence of aggregation results from groupby execution.
 * @returns Number of bytes needed to read or write the aggregation results.
 */
uint64_t required_bytes(
  const cudf::host_span<cudf::groupby::aggregation_result>& aggregation_results);
