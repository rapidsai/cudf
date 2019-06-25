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

#include <utilities/cudf_utils.h>
#include <utilities/error_utils.hpp>
#include <bitmask/legacy_bitmask.hpp>
#include <cudf/cudf.h>

namespace cudf {
namespace unary {

template<typename T, typename Tout, typename F>
__global__
void gpu_op_kernel(const T *data, gdf_size_type size,
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
    gdf_error launch(gdf_column *input, gdf_column *output) {

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

        CUDA_CHECK_LAST();
        return GDF_SUCCESS;
    }
};

inline void handleChecksAndValidity(gdf_column *input, gdf_column *output) {
    // Check for null pointers in input
    CUDF_EXPECTS((input != nullptr), "Pointer to input column is null");
    CUDF_EXPECTS((output != nullptr), "Pointer to output column is null");

    // Check for null data pointer
    CUDF_EXPECTS((input->data != nullptr),
        "Pointer to data in input column is null");
    CUDF_EXPECTS((output->data != nullptr),
        "Pointer to data in output column is null");

    // Check if input has valid mask if null_count > 0
    CUDF_EXPECTS((input->null_count == 0 || input->valid != nullptr),
        "Pointer to input column's valid mask is null but null count > 0");

    if (input->valid == nullptr) {
        if (output->valid == nullptr) {
            // if input column has no mask, then output column is allowed to have no mask
            output->null_count = 0;
        }
        else { // output->valid != nullptr
            CUDA_TRY( cudaMemset(output->valid, 0xff,
                                gdf_num_bitmask_elements( input->size )) );
            output->null_count = 0;
        }
    }
    else { // input->valid != nullptr
        CUDF_EXPECTS((output->valid != nullptr),
            "Input column has valid mask but output column does not");

        // Validity mask transfer
        CUDA_TRY( cudaMemcpy(output->valid, input->valid,
                             gdf_num_bitmask_elements( input->size ),
                             cudaMemcpyDeviceToDevice) );
        output->null_count = input->null_count;
    }
}

} // unary
} // cudf

#endif // UNARY_OPS_H
