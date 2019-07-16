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
// The translation unit for reduction `standard deviation`

#include "reduction_functions.cuh"
#include "compound.cuh"


// @param[in] ddof Delta Degrees of Freedom used for `std`, `var`.
//                 The divisor used in calculations is N - ddof, where N represents the number of elements.

gdf_scalar cudf::reduction::standard_deviation(gdf_column const& col, gdf_dtype const output_dtype, gdf_size_type ddof, cudaStream_t stream)
{
    using reducer = cudf::reduction::compound::element_type_dispatcher<cudf::reduction::op::standard_deviation>;
    return cudf::type_dispatcher(col.dtype, reducer(), col, output_dtype, ddof, stream);
}


