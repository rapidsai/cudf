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

/* @brief Computes a value for a quantile by interpolating between the values on
 *        either side of the desired quantile.
 *
 * @param[in] input        Column used to compute quantile values.
 * @param[in] q            Desired quantile in range [0, 1].
 * @param[in] interp       Strategy used to interpolate between the two values
 *                         on either side of the desired quantile.
 * @param[in] column_order Indicates the sortedness of the column.
 *
 * @returns Value of the desired quantile, or null if the column has no valid
 *          elements.
 */

std::unique_ptr<scalar>
quantile(column_view const& input,
         double q,
         interpolation interp = interpolation::LINEAR,
         order_info column_order = {});

/* @brief Selects the rows corrosponding to a quantiles by interpolating between
 *        the rows on either side of the desired quantiles.
 *
 * @param input           Table used to compute quantile rows.
 * @param q               Desired quantiles in range [0, 1].
 * @param interp          Strategy used to select between the two rows on either
                          side of the desired quantile.
 * @param column_order    The desired sort order for each column.
 * @param null_precedence The desired order of null compared to other elements.
 */
std::unique_ptr<table>
quantiles(table_view const& input,
          std::vector<double> q,
          interpolation interp = interpolation::NEAREST,
          std::vector<order> const& column_order = {},
          std::vector<null_order> const& null_precedence = {},
          rmm::mr::device_memory_resource* mr =
            rmm::mr::get_default_resource())

} // namespace experimental
} // namespace cudf
