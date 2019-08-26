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

#include "cudf.h"
#include "types.hpp"

namespace cudf {

/**
 * @brief Interpolation method to use, when the desired quantile lies between 
 * two data points i and j
 * 
 */
enum quantile_method{
  QUANT_LINEAR =0,      ///< Linear interpolation between i and j
  QUANT_LOWER,          ///< Lower data point (i)
  QUANT_HIGHER,         ///< Higher data point (j)
  QUANT_MIDPOINT,       ///< (i + j)/2
  QUANT_NEAREST,        ///< i or j, whichever is nearest
  N_QUANT_METHODS,
};

/**
 * @brief  Computes exact quantile
 * computes quantile as double. This function works with arithmetic colum.
 *
 * @param[in] input column
 * @param[in] precision: type of quantile method calculation
 * @param[in] requested quantile in [0,1]
 * @param[out] result the result as double. The type can be changed in future
 * @param[in] struct with additional info
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error quantile_exact(gdf_column* col_in,
                         quantile_method prec,
                         double q,
                         gdf_scalar*  result,
                         gdf_context* ctxt);

/**
 * @brief  Computes approximate quantile
 * computes quantile with the same type as @p col_in.
 * This function works with arithmetic colum.
 *
 * @param[in] input column
 * @param[in] requested quantile in [0,1]
 * @param[out] result quantile, with the same type as @p col_in
 * @param[in] struct with additional info
 *
 * @returns GDF_SUCCESS upon successful compute, otherwise returns appropriate error code
 */
gdf_error quantile_approx(gdf_column* col_in,
                          double q,
                          gdf_scalar*  result,
                          gdf_context* ctxt);

/**
 * @brief Find value at given quantiles within groups
 * 
 * Computes groupby @p key_table and finds values at each quantile specified in
 * @p quantiles in each group of each column in @p val_table. When the quantile
 * does not correspond to an exact index, but lies between index i and j, the 
 * result is an interpolation of values at index i and j, using the method 
 * specified in @p interpolation.
 * 
 * When multiple quantiles are requested, the result contains values in a
 * strided format.
 * 
 * Illustration:
 * Let
 * key_table = {[ a, c, b, c, a],}
 * val_table = {[v1,v2,v3,v4,v5],}
 * quantiles = {q1, q2}
 * out_keys, out_values = group_quantiles(key_table, val_table, quantiles)
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
 * 
 * @param key_table Keys to group by
 * @param val_table Values to find the quantiles in
 * @param quantiles List of quantiles q where q is in [0,1]
 * @param interpolation interpolation method to use, when the desired quantile
 *  lies between two data points
 * @param include_nulls Whether to consider rows in `key_table` that contain `NULL` values. 
 * @return std::pair<cudf::table, cudf::table> First table contains the unique
 *  keys in @p key_table. Second table contains per-group values at quantiles
 */
std::pair<cudf::table, cudf::table>
group_quantiles(cudf::table const& keys,
                cudf::table const& values,
                std::vector<double> const& quantiles,
                quantile_method interpolation = QUANT_LINEAR,
                bool include_nulls = false);


} // namespace cudf
