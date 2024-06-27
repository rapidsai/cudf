/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "table_utilities.hpp"

#include <cudf/transform.hpp>
#include <cudf/reduction.hpp>

int64_t estimate_size(std::unique_ptr<cudf::column> column)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.emplace_back(std::move(column));
  cudf::table table{std::move(columns)};
  return estimate_size(table.view());
}

int64_t estimate_size(cudf::table_view const& view)
{
  // Compute the size in bits for each row.
  auto const row_sizes = cudf::row_bit_count(view);
  // Accumulate the row sizes to compute a sum.
  auto const agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  cudf::data_type sum_dtype{cudf::type_id::INT64};
  auto const total_size_scalar = cudf::reduce(*row_sizes, *agg, sum_dtype);
  auto const total_size_in_bits = static_cast<cudf::numeric_scalar<int64_t>*>(total_size_scalar.get())->value();
  // Convert the size in bits to the size in bytes.
  return static_cast<int64_t>(static_cast<double>(total_size_in_bits) / 8);
}
