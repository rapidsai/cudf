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

#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

namespace cudf {
namespace experimental {

/* @brief For each column, computes a value for a quantile by interpolating
 *        between the values on either side of the desired quantile.
 *
 * @param[in] input        Table containing columns used to compute quantile
 *                         values.
 * @param[in] q            Desired quantile in range [0, 1].
 * @param[in] interp       Strategy used to interpolate between the two values
 *                         on either side of the desired quantile.
 * @param[in] column_order Indicates the sortedness of columns.
 *
 * @returns Value of the desired quantile for each column, or null if the column
 *          has no valid elements.
 */

std::vector<std::unique_ptr<scalar>>
quantiles(table_view const& input,
          double q,
          interpolation interp = interpolation::LINEAR,
          std::vector<order_info> column_order = {});

} // namespace experimental
} // namespace cudf
