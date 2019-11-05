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

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/types.h>

namespace cudf {

namespace util {

namespace cuda {

/**
 * @brief A kernel grid configuration construction gadget for simple one-dimensional/linear
 * kernels, with protection against integer overflow.
 */
class grid_config_1d {
public:
    const int num_threads_per_block;
    const int num_blocks;

    /**
     * @param overall_num_elements The number of elements the kernel needs to handle/process,
     * in its main, one-dimensional/linear input (e.g. one or more cuDF columns)
     * @param num_threads_per_block The grid block size, determined according to the kernel's
     * specific features (amount of shared memory necessary, SM functional units use pattern
     * etc.); this can't be determined generically/automatically (as opposed to the number of
     * blocks)
     * @param elements_per_thread Typically, a single kernel thread processes more than a single
     * element; this affects the number of threads the grid must contain
     */
    grid_config_1d (
        cudf::size_type  overall_num_elements,
        int            num_threads_per_block_,
        cudf::size_type  elements_per_thread = 1
     ) :
        num_threads_per_block(num_threads_per_block_),
        num_blocks(util::div_rounding_up_safe(overall_num_elements, elements_per_thread * num_threads_per_block))
    { }
    grid_config_1d (const grid_config_1d&) = default;
    grid_config_1d (grid_config_1d&&) = default;
};

/**
 * A RAII gadget for top-level CUDF API function implementations, which may
 * create their own streams. This will ensure the stream is destroyed when the API function returns,
 * without you having to explicitly destroy it - and also when you exit due to an error.
 */
struct scoped_stream {
    cudaStream_t stream_ { nullptr };

    scoped_stream() {
        CUDA_TRY( cudaStreamCreate(&stream_) );
    }
    operator cudaStream_t() { return stream_; }
    ~scoped_stream() {
        if (not std::uncaught_exception()) {
            auto synch_result = cudaStreamSynchronize(stream_);
            if (synch_result == cudaSuccess) {
                 cudaStreamDestroy(stream_);
            }
        }
    }
};
} // namespace cuda

} // namespace util

} // namespace cudf

#endif // CUDF_UTILITIES_CUDA_UTIL_CUH_
