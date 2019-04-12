/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Eyal Rozenberg <eyalroz@blazingdb.com>
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

#ifndef CUDF_FILL_CUH_
#define CUDF_FILL_CUH_

#include "simple_1d_grid.cuh"
#include <utilities/cuda_utils.hpp>
#include <utilities/column_utils.hpp>
#include <utilities/bit_util.cuh>
#include <utility>

namespace cudf {

namespace kernels {

template <typename T, typename Size>
__global__ void
memset(T* buffer, Size length, T value)
{
    Size pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (pos < length) { buffer[pos] = value; }
}

} // namespace kernels

namespace detail {

void CUDART_CB set_null_count(cudaStream_t stream, cudaError_t status, void *user_data)
{
    auto column = reinterpret_cast<gdf_column&>(user_data);
    column.null_count = column.size;
}

} // namespace detail

gdf_error set_uniform_validity(
    gdf_column&   column,
    cudaStream_t  stream,
    bool          make_elements_valid = true)
{
    if (make_elements_valid and not cudf::is_nullable(column)) { return GDF_SUCCESS; }
    if (not make_elements_valid and not cudf::is_nullable(column)) { return GDF_VALIDITY_MISSING; }
    auto validity_pseudocolumn_size =
        cudf::util::packed_bit_sequence_size_in_bytes<uint32_t, gdf_size_type>(column.size);

    enum { threads_per_block = 256 }; // TODO: Magic number... :-(
    cudf::util::cuda::simple_1d_grid grid_config{ validity_pseudocolumn_size, threads_per_block };
    const gdf_valid_type fill_value { make_elements_valid ? gdf_valid_type{~0} : gdf_valid_type{0} };
    kernels::memset<gdf_valid_type>
        <<<
            grid_config.num_threads_per_block,
            grid_config.num_blocks,
            cudf::util::cuda::no_dynamic_shared_memory,
            stream
        >>>
        (column.valid, validity_pseudocolumn_size, fill_value);
    CUDA_TRY ( cudaGetLastError() );
    cudaStreamAddCallback(stream, detail::set_null_count, &column, 0);
        // Notes: This is slightly risky, in that the caller must keep the gdf_column structure
        // alive until the stream makes the callback. Oh well.
    return GDF_SUCCESS;
}


// Notes:
// - Can be easily "upgraded" into a CUDF API function if you like, using
//   a short wrapper
// - The null count is set _prospectively, even though the API call might fail.
// - Perhaps there should really be a separate utility function for turning-on
//   long stretches of a packed bit sequence, or the entire sequence.
template <typename E>
gdf_error fill(
    gdf_column&   column,
    cudaStream_t  stream,
    E             value,
    bool          fill_with_nulls = false)
{
    auto validity_pseudocolumn_size =
        cudf::util::packed_bit_sequence_size_in_bytes<uint32_t, gdf_size_type>(column.size);
    if (cudf::is_nullable(column)) {
        if (fill_with_nulls) {
            auto make_all_elements_invalid = false;
            set_uniform_validity(column, stream, make_all_elements_invalid );
        }
        else {
            auto make_all_elements_invalid = true;
            set_uniform_validity(column, stream, make_all_elements_invalid );
        }
    }
    else {
        enum { threads_per_block = 256 }; // TODO: Magic number... :-(
        cudf::util::cuda::simple_1d_grid grid_config {column.size, threads_per_block};
        kernels::memset<E>
            <<<
                grid_config.num_threads_per_block,
                grid_config.num_blocks,
                cudf::util::cuda::no_dynamic_shared_memory,
                stream
            >>>
            (static_cast<E*>(column.data), column.size, value);
        CUDA_TRY ( cudaGetLastError() );
    }
    return GDF_SUCCESS;
}

} // namespace cudf

#endif // CUDF_FILL_CUH_
