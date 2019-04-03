/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#ifndef REDUCTION_HPP
#define REDUCTION_HPP

#include "cudf.h"
#include "types.hpp"

namespace cudf {
/** --------------------------------------------------------------------------*
 * @brief  Computes the reduction of the values in all rows of a column
 * This function does not detect overflows in reductions.
 * Using a higher precision `dtype` may prevent overflow.
 * Only `min` and `max` ops are supported for reduction of non-arithmetic
 * types (date32, timestamp, category...)
 *
 * @param[in] col Input column
 * @param[in] op  The operator applied by the reduction
 * @param[in] dtype The computation and output precision.
 *     `dtype` must be a data type that is convertible from the input dtype.
 *     If the input column has arithmetic type, any arithmetic type can be specified.
 *     If the input column has non-arithmetic type
 *     (date32, timestamp, category...), the same type must be specified.
 *
 * @returns  gdf_scalar the result value
 * If the reduction fails, the member is_valid of the output gdf_scalar
 * will contain `false`.
 * ----------------------------------------------------------------------------**/
gdf_scalar gdf_reduction(const gdf_column *col, gdf_reduction_op op,
                        gdf_dtype output_dtype);

/** --------------------------------------------------------------------------*
 * @brief  Computes the scan (a.k.a. prefix sum) of a column.
 * The null values are skipped for the operation, and if an input element
 * at `i` is null, then the output element at `i` will also be null.
 *
 * @param[in] input The input column for the san
 * @param[out] output The pre-allocated output column
 * @param[in] op The operation of the scan
 * @param[in] inclusive The flag for applying an inclusive scan if true,
 * an exclusive scan if false.
 * ----------------------------------------------------------------------------**/
void gdf_scan(const gdf_column *input, gdf_column *output,
                   gdf_scan_op op, bool inclusive);

}  // namespace cudf

#endif  // REDUCTION_HPP
