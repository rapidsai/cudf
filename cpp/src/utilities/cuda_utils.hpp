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

#ifndef CUDF_UTILITIES_CUDA_UTIL_CUH_
#define CUDF_UTILITIES_CUDA_UTIL_CUH_

/**
 * @file Utility code for host-side interaction with CUDA by cuDF API
 * function implementations
 */

#include "integer_utils.hpp"

#include <cuda_runtime_api.h>

namespace cudf {

namespace util {

namespace cuda {

enum : unsigned {
    no_dynamic_shared_memory = 0 //!< Use this when launching kernels to indicate no use of dynamic memory
};

/**
 * A RAII gadget for top-level CUDF API function implementations, which may
 * create their own streams. This will ensure the stream is destroyed when the API function returns,
 * without you having to explicitly destroy it - and also when you exit due to an error.
 */
struct scoped_stream {
    cudaStream_t stream_ { nullptr };

    scoped_stream() {
        cudaStreamCreate(&stream_);
        assert(stream_ != nullptr and "Failed creating a CUDA stream");
    }
    operator cudaStream_t() { return stream_; }
    ~scoped_stream() {
        if (not std::uncaught_exception()) {
            cudaStreamSynchronize(stream_);
        }
        cudaStreamDestroy(stream_);
    }
};

template <typename T>
cudaError_t copy_single_value(
    T&             destination,
    const T&       source,
    cudaStream_t   stream)
{
    static_assert(std::is_trivially_copyable<T>::value, "Invalid type specified - it must be trivially copyable");
    cudaMemcpyAsync(&destination, &source, sizeof(T), cudaMemcpyDefault, stream);
    if (cudaPeekAtLastError() != cudaSuccess) { return cudaGetLastError(); }
    cudaStreamSynchronize(stream);
    return cudaGetLastError();
}

} // namespace cuda

// TODO: Use the cuda-api-wrappers library instead
inline constexpr auto form_naive_1d_grid(
    int overall_num_elements,
    int num_threads_per_block,
    int elements_per_thread = 1)
{
    struct one_dimensional_grid_params_t {
        int num_blocks;
        int num_threads_per_block;
    };
    auto num_blocks = util::div_rounding_up_safe(overall_num_elements, elements_per_thread * num_threads_per_block);
    return one_dimensional_grid_params_t { num_blocks, num_threads_per_block };
}


} // namespace util



} // namespace cudf

#endif // CUDF_UTILITIES_CUDA_UTIL_CUH_
