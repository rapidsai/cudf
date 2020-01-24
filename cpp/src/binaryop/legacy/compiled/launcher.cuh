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

#ifndef COMPILED_BINARY_OPS_LAUNCHER_H
#define COMPILED_BINARY_OPS_LAUNCHER_H

#include <cudf/utilities/nvtx_utils.hpp>
#include <utilities/legacy/error_utils.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/cudf.h>

namespace cudf {
namespace binops {
namespace compiled {

template<typename T, typename Tout, typename F>
__global__
void gpu_binary_op(const T *lhs_data, const cudf::valid_type *lhs_valid,
                   const T *rhs_data, const cudf::valid_type *rhs_valid,
                   cudf::size_type size, Tout *results, F functor) {
    int tid = threadIdx.x;
    int blkid = blockIdx.x;
    int blksz = blockDim.x;
    int gridsz = gridDim.x;

    int start = tid + blkid * blksz;
    int step = blksz * gridsz;

    for (int i=start; i<size; i+=step) {
        results[i] = functor.apply(lhs_data[i], rhs_data[i]);
    }
}

template<typename T, typename Tout, typename F>
struct BinaryOp {
    static
    gdf_error launch(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {

        // Return successully right away for empty inputs
        if((0 == lhs->size) || (0 == rhs->size)){
          return GDF_SUCCESS;
        }

        GDF_REQUIRE(lhs->size == rhs->size, GDF_COLUMN_SIZE_MISMATCH);
        GDF_REQUIRE(lhs->size == output->size, GDF_COLUMN_SIZE_MISMATCH);
        GDF_REQUIRE(lhs->dtype == rhs->dtype, GDF_UNSUPPORTED_DTYPE);

        nvtx::range_push("CUDF_BINARY_OP", nvtx::BINARY_OP_COLOR);
        // find optimal blocksize
        int mingridsize, blocksize;
        CUDA_TRY(
            cudaOccupancyMaxPotentialBlockSize(&mingridsize, &blocksize,
                                               gpu_binary_op<T, Tout, F>)
        );
        // find needed gridsize
        int neededgridsize = (lhs->size + blocksize - 1) / blocksize;
        int gridsize = std::min(mingridsize, neededgridsize);

        F functor;
        gpu_binary_op<<<gridsize, blocksize>>>(
            // inputs
            (const T*)lhs->data, lhs->valid,
            (const T*)rhs->data, rhs->valid,
            lhs->size,
            // output
            (Tout*)output->data,
            // action
            functor
        );

        cudaDeviceSynchronize();

        nvtx::range_pop();

        CHECK_CUDA(0);
        return GDF_SUCCESS;
    }
};

} // namespace compiled
} // namespace binops
} // namespace cudf

#endif // COMPILED_BINARY_OPS_LAUNCHER_H
