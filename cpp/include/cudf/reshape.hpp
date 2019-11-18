/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <memory>
#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf {

namespace experimental {

/*
 * @brief Stacks rows of a table in to a single column.
 *
 * Interlaces all columns of a table in to a single column.

 * @note: The dtypeof all columns of @p in should be compatible, if not identical.
 * @param[in] table containing the columns to be stacked.
 * @return column containing all values from all input columns, interlaced.
 */
std::unique_ptr<column> stack(table_view const& in);

} // namespace experimental

} // namespace cudf
