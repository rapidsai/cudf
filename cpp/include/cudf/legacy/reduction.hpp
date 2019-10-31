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

#include <cudf/cudf.h>

/**
 * @brief These enums indicate the supported operations of prefix scan that can be
 * performed on a column
 */
typedef enum {
  GDF_SCAN_SUM = 0,             ///< Computes the prefix scan of sum operation of all values for the column
  GDF_SCAN_MIN,                 ///< Computes the prefix scan of maximum operation of all values for the column
  GDF_SCAN_MAX,                 ///< Computes the prefix scan of maximum operation of all values for the column
  GDF_SCAN_PRODUCT,             ///< Computes the prefix scan of multiplicative product operation of all values for the column
} gdf_scan_op;

namespace cudf {

namespace reduction {
/**
 * @brief These enums indicate the supported reduction operations that can be
 * performed on a column
 */
enum operators {
  SUM = 0,        ///< Computes the sum of all values in the column
  MIN,            ///< Computes the minimum of all values in the column
  MAX,            ///< Computes the maximum of all values in the column
  PRODUCT,        ///< Computes the multiplicative product of all values in the column
  SUMOFSQUARES,   ///< Computes the sum of squares of the values in the column
  MEAN,           ///< Computes the arithmetic mean of the values in the column
  VAR,            ///< Computes the variance of the values in the column
  STD,            ///< Computes the standard deviation of the values in the column
  ANY,            ///< Computes to true if any of the values are non-zero/true
  ALL,            ///< Computes to true if all of the values are non-zero/true
};
}  // namespace reduction


/** --------------------------------------------------------------------------*
 * @brief  Computes the reduction of the values in all rows of a column
 * This function does not detect overflows in reductions.
 * Using a higher precision `dtype` may prevent overflow.
 * Only `min` and `max` ops are supported for reduction of non-arithmetic
 * types (date32, timestamp, category...).
 * The null values are skipped for the operation.
 * If the column is empty, the member is_valid of the output gdf_scalar
 * will contain `false`.
 *
 * @param[in] col Input column
 * @param[in] op  The operator applied by the reduction
 * @param[in] output_dtype  The computation and output precision.
 *     `dtype` must be a data type that is convertible from the input dtype.
 *     If the input column has arithmetic type or cudf::bool8 type,
 *     output_dtype can be any arithmetic type or cudf::bool8 type.
 *     For `mean`, `var` and `std` ops, a floating point type must be specified.
 *     If the input column has non-arithmetic type
 *     (date32, timestamp, category...), the same type must be specified.
 * @param[in] ddof Delta Degrees of Freedom: the divisor used in calculation of `std` and `var` is `N - ddof`, where `N` is the population size.`
 *
 * @returns  gdf_scalar the result value
 * If the reduction fails, the member is_valid of the output gdf_scalar
 * will contain `false`.
 * ----------------------------------------------------------------------------**/
gdf_scalar reduce(const gdf_column *col, cudf::reduction::operators op,
                        gdf_dtype output_dtype,
                        cudf::size_type ddof = 1);

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
void scan(const gdf_column *input, gdf_column *output,
                   gdf_scan_op op, bool inclusive);

}  // namespace cudf

#endif  // REDUCTION_HPP
