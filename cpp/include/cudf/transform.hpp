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

/**
 * @brief Creates a new column by applying a unary function against every
 * element of an input column.
 *
 * Computes:
 * `out[i] = F(in[i])`
 *
 * Support all GDF data types except for GDF_CATEGORY or GDF_STRING.
 * For GDF_STRING_CATEGORY the UDF is only applied to the indices, after
 * which the underlying category is cleared and remapped.
 *
 * @param input               The input column to transform
 * @param unary_udf           The PTX/CUDA string of the unary function to apply
 * @param outout_type         The output type that is compatible with the output type in the PTX code
 * @param is_ptx              If true the UDF is treated as a piece of PTX code; if fasle the UDF is treated as a piece of CUDA code
 * @return gdf_column         The column resulting from applying the unary function to
 *                            every element of the input
 **/
gdf_column transform(const gdf_column &input,
                     const std::string &unary_udf,
                     gdf_dtype output_type, bool is_ptx);

/**
 * @brief Given a column with floating point values, generate a bitmask where every NaN
 * is indicated as the corresponding null bit.
 * 
 * @param input The input column to generate bitmask from
 * @return An `std::pair` of `bit_mask_t*`, the output bitmask, and its null count
*/
std::pair<bit_mask::bit_mask_t*, gdf_size_type> nans_to_nulls(gdf_column const& input);

}  // namespace cudf

#endif
