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

#ifndef CUDF_SIMPLE_1D_GRID_CUH_
#define CUDF_SIMPLE_1D_GRID_CUH_

#include <utilities/cuda_utils.hpp>

namespace cudf {
namespace util {
namespace cuda {


enum {
    no_dynamic_shared_memory = 0    //!< Used in kernel launches instead of the "magic number" 0,
                                    //!< as the amount of dynamic shared memory to allocate
};


/**
 * @brief Safely obtain a kernel grid configuration for a simple one-dimensional/linear kernel.
 *
 * @param overall_num_elements The number of elements the kernel needs to handle/process,
 * in its main, one-dimensional/linear input (e.g. one or more cuDF columns)
 * @param num_threads_per_block The grid block size, determined according to the kernel's
 * specific features (amount of shared memory necessary, SM functional units use pattern
 * etc.); this can't be determined generically/automatically (as opposed to the number of
 * blocks)
 * @param elements_per_thread Typically, a single kernel thread processes more than a single
 * element; this affects the number of threads the grid must contain
 *
 * @return An anonymous-type grid configuration, with a number of threads per block (the
 * one specified
 */
class simple_1d_grid {
public:
    const int num_blocks;
    const int num_threads_per_block;

    simple_1d_grid (
        gdf_size_type overall_num_elements,
        int num_threads_per_block_,
        gdf_size_type elements_per_thread = 1
     ) :
        num_threads_per_block(num_threads_per_block_),
        num_blocks(util::div_rounding_up_safe(overall_num_elements, elements_per_thread * num_threads_per_block_))
    { }
    simple_1d_grid (const simple_1d_grid&) = default;
    simple_1d_grid (simple_1d_grid&&) = default;
};

} // namespace cuda
} // namespace cudf
} // namespace util

#endif // CUDF_SIMPLE_1D_GRID_CUH_
