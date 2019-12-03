/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/cudf.h>
#include <cudf/scalar/scalar.hpp>

namespace cudf {
namespace experimental {

/**
 * @brief These enums indicate the supported operations of prefix scan that can
 * be performed on a column
 */
enum class scan_op {
  SUM = 0,  ///< Computes the prefix scan of     sum operation of all values for the column
  MIN,      ///< Computes the prefix scan of maximum operation of all values for the column
  MAX,      ///< Computes the prefix scan of maximum operation of all values for the column
  PRODUCT,  ///< Computes the prefix scan of multiplicative product operation of all values for the column
};

/**
 * @brief These enums indicate the supported reduction operations that can be
 * performed on a column
 */
enum class reduction_op {
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

/** --------------------------------------------------------------------------*
 * @brief  Computes the reduction of the values in all rows of a column.
 * This function does not detect overflows in reductions.
 * Using a higher precision `data_type` may prevent overflow.
 * Only `min` and `max` ops are supported for reduction of non-arithmetic
 * types (timestamp, string...).
 * The null values are skipped for the operation.
 * If the column is empty, the member `is_valid()` of the output scalar
 * will contain `false`.
 *
 * @throws `cudf::logic_error` if reduction is called for non-arithmetic output
 * type and operator other than `min` and `max`.
 * @throws `cudf::logic_error` if input column data type is not convertible to
 * output data type.
 * If the input column has arithmetic type, output_dtype can be any arithmetic
 * type. For `mean`, `var` and `std` ops, a floating point output type must be 
 * specified. If the input column has non-arithmetic type
 *   eg.(timestamp, string...), the same type must be specified.
 *
 * @param[in] col Input column view
 * @param[in] op  The operator applied by the reduction
 * @param[in] output_dtype  The computation and output precision.
 * @params[in] mr The resource to use for all allocations
 * @param[in] ddof Delta Degrees of Freedom: the divisor used in calculation of
 * `std` and `var` is `N - ddof`, where `N` is the population size.`
 * @returns  cudf::scalar the result value
 * If the reduction fails, the member is_valid of the output scalar
 * will contain `false`.
 * ----------------------------------------------------------------------------**/
std::unique_ptr<scalar> reduce(
    const column_view& col, reduction_op op, data_type output_dtype,
    cudf::size_type ddof = 1,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/** --------------------------------------------------------------------------*
 * @brief  Computes the scan of a column.
 * The null values are skipped for the operation, and if an input element
 * at `i` is null, then the output element at `i` will also be null.
 *
 * @throws `cudf::logic_error` if column datatype is not numeric type.
 *
 * @param[in] input The input column view for the scan
 * @param[in] op The operation of the scan
 * @param[in] inclusive The flag for applying an inclusive scan if true,
 *            an exclusive scan if false.
 * @params[in] mr The resource to use for all allocations
 * @returns unique pointer to new output column
 * ----------------------------------------------------------------------------**/
std::unique_ptr<column> scan(
    const column_view& input, scan_op op, bool inclusive,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace experimental
}  // namespace cudf

