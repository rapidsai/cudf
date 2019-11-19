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

#include <cudf/types.hpp>

namespace cudf {

/**
 * @brief Interpolation method to use, when the desired quantile lies between 
 * two data points i and j
 * 
 */
enum interpolation{
  LINEAR =0,      ///< Linear interpolation between i and j
  LOWER,          ///< Lower data point (i)
  HIGHER,         ///< Higher data point (j)
  MIDPOINT,       ///< (i + j)/2
  NEAREST,        ///< i or j, whichever is nearest
};

/**
 * @brief  Computes exact quantile
 *
 * Computes quantile using double precision. 
 * This function works only with arithmetic columns.
 *
 * @param[in] col_in input column
 * @param[in] interpolation Method to use for interpolating quantiles that lie between points
 * @param[in] q requested quantile in [0,1]
 * @param[out] result The value at the requested quantile
 * @param[in] ctxt Options
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error quantile_exact(gdf_column* col_in,
                         interpolation prec,
                         double q,
                         gdf_scalar*  result,
                         gdf_context* ctxt);

/**
 * @brief  Computes approximate quantile
 *
 * Computes quantile with the same type as @p col_in. 
 * This function works only with arithmetic columns.
 *
 * @param[in] col_in input column
 * @param[in] q requested quantile in [0,1]
 * @param[out] result  The value at the requested quantile, with the same type as @p col_in
 * @param[in] ctxt Options
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error quantile_approx(gdf_column* col_in,
                          double q,
                          gdf_scalar*  result,
                          gdf_context* ctxt);

/**
 * @brief Find values at given quantiles within groups
 * 
 * Computes groupby @p keys and finds values at each quantile specified in
 * @p quantiles in each group of each column in @p values. When the quantile
 * does not correspond to an exact index, but lies between index i and j, the 
 * result is an interpolation of values at index i and j, using the method 
 * specified in @p interpolation. Nulls are always ignored in @p values.
 * 
 * Returns the resulting quantile(s) for all groups in a single column. 
 * When more than one quantile is requested, each group's results are stored
 * contiguously in the same order specified in `quantiles`. 
 * 
 * Illustration:
 * ```
 * Let
 * keys   = {[ a, c, b, c, a],}
 * values = {[v1,v2,v3,v4,v5],}
 * quantiles = {q1, q2}
 * out_keys, out_values = group_quantiles(keys, values, quantiles)
 * 
 * out_keys = {[ a,      b,     c     ],}
 * out_vals = {[x1, x2, y1, y2, z1, z2],}
 * where
 * x1 = value at quantile q1 in group [v1,v5]
 * x2 = value at quantile q2 in group [v1,v5]
 * y1 = value at quantile q1 in group [v3]
 * y2 = value at quantile q2 in group [v3]
 * z1 = value at quantile q1 in group [v2,v4]
 * z2 = value at quantile q2 in group [v2,v4]
 * ```
 * 
 * @param keys Keys to group by
 * @param values Values to find the quantiles in
 * @param quantiles List of quantiles q where q is in [0,1]
 * @param interpolation Method to use for interpolating quantiles that lie between points
 * @param include_nulls Whether to consider rows in `keys` that contain `NULL` values. 
 * @return std::pair<cudf::table, cudf::table> First table contains the unique
 *  keys in @p keys. Second table contains per-group values at quantiles
 */
std::pair<cudf::table, cudf::table>
group_quantiles(cudf::table const& keys,
                cudf::table const& values,
                std::vector<double> const& quantiles,
                interpolation interpolation = LINEAR,
                bool include_nulls = false);


} // namespace cudf
