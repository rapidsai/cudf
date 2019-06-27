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

#include "reduction_functions.cuh"

namespace cudf{

gdf_scalar reduction(const gdf_column *col,
                  gdf_reduction_op op, gdf_dtype output_dtype,
                  gdf_size_type ddof)
{
    gdf_scalar scalar;
    scalar.dtype = output_dtype;
    scalar.is_valid = false; // the scalar is not valid for error case

    CUDF_EXPECTS(col != nullptr, "Input column is null");
    // check if input column is empty
    if( col->size <= col->null_count )return scalar;

    switch(op){
    case GDF_REDUCTION_SUM:
        cudf::reductions::sum(*col, scalar);
        break;
    case GDF_REDUCTION_MIN:
        cudf::reductions::min(*col, scalar);
        break;
    case GDF_REDUCTION_MAX:
        cudf::reductions::max(*col, scalar);
        break;
    case GDF_REDUCTION_PRODUCT:
        cudf::reductions::product(*col, scalar);
        break;
    case GDF_REDUCTION_SUMOFSQUARES:
        cudf::reductions::sum_of_squares(*col, scalar);
        break;

    case GDF_REDUCTION_MEAN:
        cudf::reductions::mean(*col, scalar);
        break;
    case GDF_REDUCTION_VAR:
        cudf::reductions::variance(*col, scalar, ddof);
        break;
    case GDF_REDUCTION_STD:
        cudf::reductions::standard_deviation(*col, scalar, ddof);
        break;
    default:
        CUDF_FAIL("Unsupported reduction operator");
    }

    return scalar;
}

}   // cudf namespace

