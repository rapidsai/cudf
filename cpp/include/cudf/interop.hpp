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

#include <arrow/api.h>
#include <cudf/column/column.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/types.hpp>

namespace cudf {
/**
 * @brief Create `arrow::Table` from cudf table `input`
 *
 * Converts the `cudf::table_view` to `arrow::Table` with the provided
 * metadata `column_names`.
 *
 * @throws cudf::logic_error if `column_names` size doesn't match with number of columns.
 *
 * @param input table_view that needs to be converted to arrow Table
 * @param column_names Vector of column names for metadata of arrow Table
 * @param ar_mr arrow memory pool to allocate memory for arrow Table
 * @return arrow Table generated from given input cudf table
 **/
std::shared_ptr<arrow::Table> cudf_to_arrow(
  table_view input,
  std::vector<std::string> const& column_names = {},
  arrow::MemoryPool* ar_mr                     = arrow::default_memory_pool());

/**
 * @brief Create `cudf::table` from given arrow Table input
 *
 * @param input_table  arrow:Table that needs to be converted to `cudf::table`
 * @param mr           Device memory resource used to allocate `cudf::table`
 * @return cudf table generated from given arrow Table.
 **/

std::unique_ptr<table> arrow_to_cudf(
  arrow::Table const& input_table,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace cudf
