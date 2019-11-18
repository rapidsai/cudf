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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf {
    
namespace experimental {

enum class quantile_interpolation {
  LINEAR = 0,     ///< (a + b) / 2
  MEAN,           ///< (b - a) * x + a, where x in range [0, 1]
  LOW,            ///< a
  HIGH,           ///< b
  ROUND,          ///< a or b, whichever is nearest
};

/* @brief Computes the exact quantile of any sorted arithmetic column.
 *
 * @param[in] in                     Column from which quantile is computed.
 * @param[in] quantile_interpolation Strategy to obtain a quantile which falls
                                     between two points.
 *
 * @returns The quantile within range [0, 1]
 */
std::unique_ptr<scalar> quantile(column_view const& in,
                                 double quantile,
                                 quantile_interpolation interpolation = quantile_interpolation::LINEAR);

} // namespace cudf

} // namespace experimental