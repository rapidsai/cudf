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
                  gdf_reduction_op op, gdf_dtype output_dtype)
{
    gdf_scalar scalar;
    scalar.dtype = output_dtype;
    scalar.is_valid = false; // the scalar is not valid for error case

    CUDF_EXPECTS(col != nullptr, "Input column is null");
    // check if input column is empty
    if( col->size <= col->null_count )return scalar;

    switch(op){
#if 0
    case GDF_REDUCTION_SUM:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionSum>(), col, &scalar);
        break;
    case GDF_REDUCTION_MIN:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionMin>(), col, &scalar);
        break;
    case GDF_REDUCTION_MAX:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionMax>(), col, &scalar);
        break;
    case GDF_REDUCTION_PRODUCT:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionProduct>(), col, &scalar);
        break;
    case GDF_REDUCTION_SUMOFSQUARES:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionSumOfSquares>(), col, &scalar);
        break;
    case GDF_REDUCTION_MEAN:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionMean>(), col, &scalar);
        break;
    case GDF_REDUCTION_VAR:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionVar>(), col, &scalar);
        break;
    case GDF_REDUCTION_STD:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionStd>(), col, &scalar);
        break;
#else
    case GDF_REDUCTION_SUM:
        reduction_sum(col, &scalar);
        break;
    case GDF_REDUCTION_MIN:
        reduction_min(col, &scalar);
        break;
    case GDF_REDUCTION_MAX:
        reduction_max(col, &scalar);
        break;
    case GDF_REDUCTION_PRODUCT:
        reduction_prod(col, &scalar);
        break;
    case GDF_REDUCTION_SUMOFSQUARES:
        reduction_sumofsquares(col, &scalar);
        break;

    case GDF_REDUCTION_MEAN:
        reduction_mean(col, &scalar);
        break;
    case GDF_REDUCTION_VAR:
        reduction_var(col, &scalar);
        break;
    case GDF_REDUCTION_STD:
        reduction_std(col, &scalar);
        break;
#endif
    default:
        CUDF_FAIL("Unsupported reduction operator");
    }

    return scalar;
}

}   // cudf namespace

