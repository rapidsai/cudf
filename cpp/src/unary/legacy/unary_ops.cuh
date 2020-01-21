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

#ifndef UNARY_OPS_H
#define UNARY_OPS_H

#include <utilities/legacy/cudf_utils.h>
#include <cudf/utilities/error.hpp>
#include <utilities/legacy/column_utils.hpp>
#include <bitmask/legacy/legacy_bitmask.hpp>
#include <cudf/cudf.h>

namespace cudf {
namespace unary {

template<typename T, typename Tout, typename F>
__global__
void gpu_op_kernel(const T *data, cudf::size_type size,
                   Tout *results, F functor) {
    int tid = threadIdx.x;
    int blkid = blockIdx.x;
    int blksz = blockDim.x;
    int gridsz = gridDim.x;

    int start = tid + blkid * blksz;
    int step = blksz * gridsz;
    for (int i=start; i<size; i+=step) {
        results[i] = functor.apply(data[i]);
    }
}

template<typename T, typename Tout, typename F>
struct Launcher {
    static
    gdf_error launch(gdf_column const* input, gdf_column *output) {

        // Return immediately for empty inputs
        if((0==input->size))
        {
          return GDF_SUCCESS;
        }

        /* check for size of the columns */
        if (input->size != output->size) {
            return GDF_COLUMN_SIZE_MISMATCH;
        }

        // find optimal blocksize
        int mingridsize, blocksize;
        CUDA_TRY(
            cudaOccupancyMaxPotentialBlockSize(&mingridsize, &blocksize,
                                               gpu_op_kernel<T, Tout, F>)
        );
        // find needed gridsize
        int neededgridsize = (input->size + blocksize - 1) / blocksize;
        int gridsize = std::min(neededgridsize, mingridsize);

        F functor;
        gpu_op_kernel<<<gridsize, blocksize>>>(
            // input
            (const T*)input->data, input->size,
            // output
            (Tout*)output->data,
            // action
            functor
        );

        CHECK_CUDA(0);
        return GDF_SUCCESS;
    }
};

inline void handleChecksAndValidity(gdf_column const& input, gdf_column& output) {
    // Check for null data pointer
    validate(input);

    if ( not is_nullable(input) ) {
        if ( not is_nullable(output) ) {
            // if input column has no mask, then output column is allowed to have no mask
            output.null_count = 0;
        }
        else { // output.valid != nullptr
            CUDA_TRY( cudaMemset(output.valid, 0xff,
                                gdf_num_bitmask_elements( input.size )) );
            output.null_count = 0;
        }
    }
    else { // input.valid != nullptr
        CUDF_EXPECTS( is_nullable(output),
            "Input column has valid mask but output column does not");

        // Validity mask transfer
        CUDA_TRY( cudaMemcpy(output.valid, input.valid,
                             gdf_num_bitmask_elements( input.size ),
                             cudaMemcpyDeviceToDevice) );
        output.null_count = input.null_count;
    }
}

} // unary
} // cudf

#endif // UNARY_OPS_H
