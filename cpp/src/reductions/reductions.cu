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
                  cudf::reduction::operators op, gdf_dtype output_dtype,
                  cudf::size_type ddof)
{
    gdf_scalar scalar;
    scalar.dtype = output_dtype;
    scalar.is_valid = false; // the scalar is not valid for error case

    CUDF_EXPECTS(col != nullptr, "Input column is null");
    // check if input column is empty
    if( col->size <= col->null_count )return scalar;

    switch(op){
      case cudf::reduction::SUM:
        scalar = cudf::reduction::sum(*col, output_dtype);
        break;
    case cudf::reduction::MIN:
        scalar = cudf::reduction::min(*col, output_dtype);
        break;
    case cudf::reduction::MAX:
        scalar = cudf::reduction::max(*col, output_dtype);
        break;
    case cudf::reduction::ANY:
        scalar = cudf::reduction::any(*col, output_dtype);
        break;
    case cudf::reduction::ALL:
        scalar = cudf::reduction::all(*col, output_dtype);
        break;
    case cudf::reduction::PRODUCT:
        scalar = cudf::reduction::product(*col, output_dtype);
        break;
    case cudf::reduction::SUMOFSQUARES:
        scalar = cudf::reduction::sum_of_squares(*col, output_dtype);
        break;

    case cudf::reduction::MEAN:
        scalar = cudf::reduction::mean(*col, output_dtype);
        break;
    case cudf::reduction::VAR:
        scalar = cudf::reduction::variance(*col, output_dtype, ddof);
        break;
    case cudf::reduction::STD:
        scalar = cudf::reduction::standard_deviation(*col, output_dtype, ddof);
        break;
    default:
        CUDF_FAIL("Unsupported reduction operator");
    }

    return scalar;
}

}   // cudf namespace

