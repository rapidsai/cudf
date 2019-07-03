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

gdf_scalar reduce(const gdf_column *col,
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
    case SUM:
        cudf::reduction::sum(*col, scalar);
        break;
    case MIN:
        cudf::reduction::min(*col, scalar);
        break;
    case MAX:
        cudf::reduction::max(*col, scalar);
        break;
    case PRODUCT:
        cudf::reduction::product(*col, scalar);
        break;
    case SUMOFSQUARES:
        cudf::reduction::sum_of_squares(*col, scalar);
        break;

    case MEAN:
        cudf::reduction::mean(*col, scalar);
        break;
    case VAR:
        cudf::reduction::variance(*col, scalar, ddof);
        break;
    case STD:
        cudf::reduction::standard_deviation(*col, scalar, ddof);
        break;
    default:
        CUDF_FAIL("Unsupported reduction operator");
    }

    return scalar;
}

}   // cudf namespace

