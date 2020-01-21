/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <memory>
#include <cudf/types.hpp>

namespace cudf {
namespace experimental {

enum class unary_op {
  SIN,          // < Trigonometric sine
  COS,          // < Trigonometric cosine
  TAN,          // < Trigonometric tangent
  ARCSIN,       // < Trigonometric sine inverse
  ARCCOS,       // < Trigonometric cosine inverse
  ARCTAN,       // < Trigonometric tangent inverse
  EXP,          // < Exponential (base e, Euler number)
  LOG,          // < Natural Logarithm (base e)
  SQRT,         // < Square-root (x^0.5)
  CEIL,         // < Smallest integer value not less than arg
  FLOOR,        // < largest integer value not greater than arg
  ABS,          // < Absolute value
  BIT_INVERT,   // < Bitwise Not (~)
  NOT,          // < Logical Not (!)
};

/**
 * @brief Performs unary op on all values in column
 *
 * @param input A `column_view` as input
 * @param op operation to perform
 * @param mr Optional, The resource to use for all allocations
 *
 * @returns std::unique_ptr<cudf::column> Result of the operation
 */
std::unique_ptr<cudf::column> unary_operation(cudf::column_view const& input,
                                              cudf::experimental::unary_op op,
                                              rmm::mr::device_memory_resource* mr =
                                              rmm::mr::get_default_resource());

/**
 * @brief Creates a column of `BOOL8` elements where for every element in `input` `true`
 * indicates the value is null and `false` indicates the value is valid.
 *
 * @param input A `column_view` as input
 * @param mr Optional, The resource to use for all allocations
 *
 * @returns std::unique_ptr<cudf::column> A non-nulalble column of `BOOL8` elements with `true` representing `null` values.
 */
std::unique_ptr<cudf::column> is_null(cudf::column_view const& input,
                                      rmm::mr::device_memory_resource* mr =
                                              rmm::mr::get_default_resource());

/**
 * @brief Creates a column of `BOOL8` elements where for every element in `input` `true`
 * indicates the value is valid and `false` indicates the value is null.
 *
 * @param input A `column_view` as input
 * @param mr Optional, The resource to use for all allocations
 *
 * @returns std::unique_ptr<cudf::column> A non-nulalble column of `BOOL8` elements with `false` representing `null` values.
 */
std::unique_ptr<cudf::column> is_valid(cudf::column_view const& input,
                                       rmm::mr::device_memory_resource* mr =
                                              rmm::mr::get_default_resource());

/**
 * @brief  Casts data from dtype specified in input to dtype specified in output.
 * Supports only fixed-width types.
 *
 * @param column_view Input column
 * @param out_type Desired datatype of output column
 * @param mr Optional, The resource to use for all allocations
 *
 * @returns unique_ptr<column> Result of the cast operation
 * @throw cudf::logic_error if `out_type` is not a fixed-width type
 */
std::unique_ptr<column> cast(column_view const& input, data_type out_type,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Creates a column of `BOOL8` elements indicating the presence of `NaN` values
 * in a column of floating point values.
 * The output element at row `i` is `true` if the element in `input` at row i is `NAN`, else `false`
 *
 * @throws cudf::logic_error if `input` is a non-floating point type
 *
 * @param input A column of floating-point elements
 * @param mr Optional, The resource to use for allocating the device memory in the returned column.
 *
 * @returns unique_ptr<column> A non-nulalble column of `BOOL8` elements with `true`
 * representing `NAN` values
 */
std::unique_ptr<column> is_nan(cudf::column_view const& input,
                               rmm::mr::device_memory_resource* mr =
                                   rmm::mr::get_default_resource());

/**
 * @brief Creates a column of `BOOL8` elements indicating the absence of `NaN` values
 * in a column of floating point values.
 * The output element at row `i` is `false` if the element in `input` at row i is `NAN`, else `true`
 *
 * @throws cudf::logic_error if `input` is a non-floating point type
 *
 * @param input A column of floating-point elements
 * @param mr Optional, The resource to use for allocating the device memory in the returned column.
 *
 * @returns unique_ptr<column> A non-nulalble column of `BOOL8` elements with `false`
 * representing `NAN` values
 */
std::unique_ptr<column> is_not_nan(cudf::column_view const& input,
                                 rmm::mr::device_memory_resource* mr =
                                   rmm::mr::get_default_resource());

} // namespace experimental
} // namespace cudf
