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
#include "types.h"

namespace cudf {

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
gdf_column transform(const gdf_column &input,
                     const std::string &ptx_unary_function,
                     gdf_dtype output_type);

}  // namespace cudf

#endif
