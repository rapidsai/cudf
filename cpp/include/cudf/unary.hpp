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

namespace cudf
{

/**
 * @brief Types of unary operations that can be performed on data.
 */
enum unary_op{
  GDF_SIN,          ///< Trigonometric sine
  GDF_COS,          ///< Trigonometric cosine
  GDF_TAN,          ///< Trigonometric tangent
  GDF_ARCSIN,       ///< Trigonometric sine inverse
  GDF_ARCCOS,       ///< Trigonometric cosine inverse
  GDF_ARCTAN,       ///< Trigonometric tangent inverse
  GDF_EXP,          ///< Exponential (base e, Euler number)
  GDF_LOG,          ///< Natural Logarithm (base e)
  GDF_SQRT,         ///< Square-root (x^0.5)
  GDF_CEIL,         ///< Smallest integer value not less than arg
  GDF_FLOOR,        ///< largest integer value not greater than arg
  GDF_ABS,          ///< Absolute value
  GDF_BIT_INVERT,   ///< Bitwise Not (~)
  GDF_NOT,          ///< Logical Not (!)
  GDF_INVALID_UNARY ///< invalid operation
};


/**
 * @brief  Performs unary op on all values in column
 * 
 * @param[in] gdf_column of the input
 * @param[in] unary_op operation to perform
 *
 * @returns gdf_column Result of the operation
 */
gdf_column gdf_unaryop(gdf_column& input, unary_op op);


} // namespace cudf
