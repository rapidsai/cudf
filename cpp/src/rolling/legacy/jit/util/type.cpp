/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#include "type.h"

namespace cudf {
namespace rolling {
namespace jit {

    /**---------------------------------------------------------------------------*
     * @brief Get the Operator Name
     * 
     * @param ope (enum) The aggregation operator as enum of type gdf_agg_op
     * @return std::string The name of the operator as string
     *---------------------------------------------------------------------------**/
    std::string get_operator_name(gdf_agg_op ope) {
        switch (ope) {
            case GDF_CUDA_GENERIC_AGG_OPS:
                return "cuda_generic_aggregator";
            case GDF_NUMBA_GENERIC_AGG_OPS:
                return "numba_generic_aggregator";
            default:
                return "None";
        }
    }
   
    /**---------------------------------------------------------------------------*
     * @brief Get the Function Name
     * 
     * @param ope (enum) The aggregation funtion as enum of type gdf_agg_op
     * @return std::string The name of the operator as string
     *---------------------------------------------------------------------------**/
    std::string get_function_name(gdf_agg_op ope) {
        switch (ope) {
            case GDF_CUDA_GENERIC_AGG_OPS:
                return "CUDA_GENERIC_AGGREGATOR";
            case GDF_NUMBA_GENERIC_AGG_OPS:
                return "NUMBA_GENERIC_AGGREGATOR";
            default:
                return "None";
        }
    }
} // namespace jit
} // namespace rolling
} // namespace cudf
