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

#ifndef CUDF_TRANSFORM_HPP
#define CUDF_TRANSFORM_HPP

#include "cudf.h"
#include <cudf/types.h>

namespace cudf{

/**---------------------------------------------------------------------------*
 * @brief Creates a new column by applying a unary function against every
 * element of an input column.
 *
 * Computes:
 * `out[i] = F(in[i])`
 *
 * @param input The input column to transform
 * @param ptx_unary_function The PTX of the unary function to apply
 * @return gdf_column The column resulting from applying the unary function to
 * every element of the input
 *---------------------------------------------------------------------------**/
gdf_column transform(const gdf_column& input,
                           const std::string& ptx_unary_function,
                           gdf_dtype output_type);

/**
 * @brief Performs a unary operation on every element of the input 
 * gdf_columns using a user-defined PTX function.
 *
 * The desired output type must be specified in out->dtype.
 *
 * If the valid field in the gdf_column output is not nullptr, then it will be
 * filled with the valid masks of input gdf_columns
 *
 * @param out (gdf_column) Output of the operation.
 * @param in (gdf_column) Input operand of the operation.
 * @param ptx String containing the PTX of a binary function to apply on `in`
 */
void transform(gdf_column*           out,
                     gdf_column*           in,
                     const std::string&    ptx,const std::string& output_type
                     );

} // namespace cudf


#endif


