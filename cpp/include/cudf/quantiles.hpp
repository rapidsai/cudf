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

/* @brief Computes a quantile of a given column.
 *
 * @param[in] in            Table containing columns used to compute quantiles.
 * @param[in] quantile      Requested quantile in range [0, 1].
 * @param[in] interpolation Interpolation strategy for quantiles lying between
 *                          two values.
 * @param[in] is_sorted     Whether the input has been pre-sorted.
 * @param[in] order         Order of pre-sorted values.
 * @param[in] null_order    precendence of pre-sorted nulls.
 *
 * @returns Quantiles for each column. Elements are null if columns are empty.
 */

std::vector<std::unique_ptr<scalar>>
quantiles(table_view const& input,
          double q,
          interpolation interp = interpolation::LINEAR,
          bool col_is_sorted = false,
          std::vector<cudf::order> col_order = {},
          std::vector<cudf::null_order> col_null_order = {});

} // namespace experimental
} // namespace cudf
