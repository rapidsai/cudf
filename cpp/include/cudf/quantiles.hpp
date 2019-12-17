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
 *        between the values on either side of a percent-based location within
 *        that column.
 *
 * @param[in] in             Table containing columns used to compute quantile
 *                           values.
 * @param[in] percent        Percent-based location of desired quantile values.
 * @param[in] interp         Strategy used to interpolate between the two values
 *                           on either side of the percent-based location.
 * @param[in] col_is_sorted  Indicates which columns in the table are sorted.
 * @param[in] col_order      Order of values in pre-sorted columns.
 * @param[in] col_null_order Precendence of nulls in pre-sorted columns.
 *
 * @returns Quantile values at the given percent-based location, or null if the
 *          table contains no rows.
 */

std::vector<std::unique_ptr<scalar>>
quantiles(table_view const& input,
          double percent,
          interpolation interp = interpolation::LINEAR,
          std::vector<order_info> col_order_info = {});

} // namespace experimental
} // namespace cudf
