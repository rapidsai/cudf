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

#ifndef CUDF_REDUCTION_FUNCTIONS_CUH
#define CUDF_REDUCTION_FUNCTIONS_CUH

#include "reduction.cuh"

namespace cudf {
namespace reduction {

gdf_scalar sum(gdf_column const& col, gdf_dtype const output_dtype, cudaStream_t stream=0);
gdf_scalar min(gdf_column const& col, gdf_dtype const output_dtype, cudaStream_t stream=0);
gdf_scalar max(gdf_column const& col, gdf_dtype const output_dtype, cudaStream_t stream=0);
gdf_scalar any(gdf_column const& col, gdf_dtype const output_dtype, cudaStream_t stream=0);
gdf_scalar all(gdf_column const& col, gdf_dtype const output_dtype, cudaStream_t stream=0);
gdf_scalar product(gdf_column const& col, gdf_dtype const output_dtype, cudaStream_t stream=0);
gdf_scalar sum_of_squares(gdf_column const& col, gdf_dtype const output_dtype, cudaStream_t stream=0);

gdf_scalar mean(gdf_column const& col, gdf_dtype const output_dtype, cudaStream_t stream=0);
gdf_scalar variance(gdf_column const& col, gdf_dtype const output_dtype, cudf::size_type ddof, cudaStream_t stream=0);
gdf_scalar standard_deviation(gdf_column const& col, gdf_dtype const output_dtype, cudf::size_type ddof, cudaStream_t stream=0);

} // namespace reduction
} // namespace cudf
#endif

