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

#include <cudf/table/primitive_row_operators.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf {
bool is_primitive_row_op_compatible(cudf::table_view const& table)
{
  return table.num_columns() == 1 and cudf::is_numeric(table.column(0).type());
}
}  // namespace cudf
