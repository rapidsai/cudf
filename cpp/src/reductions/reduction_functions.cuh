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

#include <cudf/cudf.h>
#include <cudf/reduction.hpp>
#include "reduction_operators.cuh"

#include <rmm/rmm.h>
#include <utilities/cudf_utils.h>
#include <utilities/error_utils.hpp>
#include <utilities/type_dispatcher.hpp>

#include <cub/device/device_reduce.cuh>

namespace cudf {
namespace reductions {

// compute reduction by the operator
template <typename Op, typename InputIterator, typename T_output>
void reduce(T_output* dev_result, InputIterator d_in, gdf_size_type num_items,
    T_output init, Op op, cudaStream_t stream)
{
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, dev_result, num_items,
        op, init, stream);
    // Allocate temporary storage
    RMM_TRY(RMM_ALLOC(&d_temp_storage, temp_storage_bytes, stream));

    // Run reduction
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, dev_result, num_items,
        op, init, stream);

    // Free temporary storage
    RMM_TRY(RMM_FREE(d_temp_storage, stream));
}

void sum(gdf_column const& col, gdf_scalar& scalar, cudaStream_t stream=0);
void min(gdf_column const& col, gdf_scalar& scalar, cudaStream_t stream=0);
void max(gdf_column const& col, gdf_scalar& scalar, cudaStream_t stream=0);
void product(gdf_column const& col, gdf_scalar& scalar, cudaStream_t stream=0);
void sum_of_squares(gdf_column const& col, gdf_scalar& scalar, cudaStream_t stream=0);

void mean(gdf_column const& col, gdf_scalar& scalar, cudaStream_t stream=0);
void variance(gdf_column const& col, gdf_scalar& scalar, gdf_size_type ddof = 1, cudaStream_t stream=0);
void standard_deviation(gdf_column const& col, gdf_scalar& scalar, gdf_size_type ddof = 1, cudaStream_t stream=0);

} // namespace reductions
} // namespace cudf
#endif
