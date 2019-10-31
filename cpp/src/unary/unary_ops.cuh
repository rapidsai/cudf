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
#include <cudf/utilities/error.hpp>
#include <utilities/column_utils.hpp>
#include <bitmask/legacy/legacy_bitmask.hpp>
#include <cudf/cudf.h>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>

namespace cudf {
namespace experimental {
namespace unary {

template<typename T, typename Tout, typename F>
__global__
void gpu_op_kernel(const T *data, cudf::size_type size,
                   Tout *results, F functor) {
    int tid    = threadIdx.x;
    int blkid  = blockIdx.x;
    int blksz  = blockDim.x;
    int gridsz = gridDim.x;

    int start = tid + blkid * blksz;
    int step  = blksz * gridsz;

    for (int i = start; i < size; i += step) {
        results[i] = functor.apply(data[i]);
    }
}

template<typename T, typename Tout, typename F>
struct Launcher {
    static
    gdf_error launch(cudf::column_view const& input,
                     cudf::mutable_column_view& output) {

        // Return immediately for empty inputs
        if (input.size() == 0)
          return GDF_SUCCESS;

        // check for size of the columns
        if (input.size() != output.size())
            return GDF_COLUMN_SIZE_MISMATCH;

        // find optimal blocksize
        int min_grid_size, block_size;
        CUDA_TRY(
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                               gpu_op_kernel<T, Tout, F>)
        );

        // find needed gridsize
        int needed_grid_size = (input.size() + block_size - 1) / block_size;
        int grid_size        = std::min(needed_grid_size, min_grid_size);

        auto device_input  = cudf::column_device_view::create(input);
        auto device_output = cudf::mutable_column_device_view::create(output);

        F functor;
        gpu_op_kernel<<<grid_size, block_size>>>(
            static_cast<const T*>(device_input->head()), device_input->size(),
            static_cast<Tout*>(device_output->head()),
            functor
        );

        CUDA_CHECK_LAST();
        return GDF_SUCCESS;
    }
};

inline void handleChecksAndValidity(column_view const& input, mutable_column_view& output) {

    if (not input.nullable()) {
        if (not output.nullable())
            CUDA_TRY( cudaMemset(output.null_mask(), 0xff, gdf_num_bitmask_elements(input.size())));

        output.set_null_count(0);
    }
    else { // input.valid != nullptr
        CUDF_EXPECTS(output.nullable(), "Input column has valid mask but output column does not");

        // Validity mask transfer
        CUDA_TRY( cudaMemcpy(output.null_mask(), input.null_mask(),
                             gdf_num_bitmask_elements(input.size()),
                             cudaMemcpyDeviceToDevice));

        output.set_null_count(input.null_count());
    }
}

} // unary
} // namespace experimental
} // cudf

#endif // UNARY_OPS_H
